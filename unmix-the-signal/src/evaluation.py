"""
Evaluation and Metrics Module.
Handles ICA permutation and scale/sign ambiguities, calculates Pearson correlation,
SDR (Signal-to-Distortion Ratio), SNR, and comparative performance benchmarks.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import scipy.optimize


def match_sources_and_resolve_ambiguity(
    originals: np.ndarray,
    estimates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[int], List[float]]:
    """
    Solves ICA permutation and scale/sign ambiguities using the Hungarian algorithm
    (linear sum assignment) on absolute Pearson correlation coefficients.

    Args:
        originals: Clean reference sources of shape (n_sources, n_samples).
        estimates: Recovered sources from ICA of shape (n_components, n_samples).

    Returns:
        matched_estimates: Estimates reordered to match originals (n_sources, n_samples).
        aligned_estimates: Estimates scaled and sign-corrected for fair metric evaluation.
        permutation_order: Mapping indices [orig_0_match, orig_1_match, ...].
        signs: Sign corrections [+1.0 or -1.0].
    """
    n_sources = originals.shape[0]
    n_components = estimates.shape[0]

    # Compute correlation matrix C where C[i, j] = Pearson corr between orig[i] and est[j]
    corr_matrix = np.zeros((n_sources, n_components))

    for i in range(n_sources):
        orig_norm = np.linalg.norm(originals[i]) + 1e-12
        for j in range(n_components):
            est_norm = np.linalg.norm(estimates[j]) + 1e-12
            corr_matrix[i, j] = np.dot(originals[i], estimates[j]) / (orig_norm * est_norm)

    # Hungarian algorithm: maximize sum of absolute correlations
    cost_matrix = -np.abs(corr_matrix)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

    matched_estimates = np.zeros_like(originals)
    aligned_estimates = np.zeros_like(originals)
    permutation_order = []
    signs = []

    for orig_idx, est_idx in zip(row_ind, col_ind):
        permutation_order.append(int(est_idx))
        raw_est = estimates[est_idx]
        matched_estimates[orig_idx] = raw_est

        # Optimal projection scalar: alpha = <orig, est> / ||est||^2
        # Automatically handles scale and sign flip
        est_energy = np.dot(raw_est, raw_est) + 1e-12
        alpha = np.dot(originals[orig_idx], raw_est) / est_energy
        sign = 1.0 if alpha >= 0 else -1.0
        signs.append(sign)

        aligned_estimates[orig_idx] = alpha * raw_est

    return matched_estimates, aligned_estimates, permutation_order, signs


def calculate_sdr(
    reference: np.ndarray,
    estimate_aligned: np.ndarray,
    epsilon: float = 1e-12
) -> float:
    """
    Calculates Signal-to-Distortion Ratio (SDR) in decibels (dB):
      SDR = 10 * log10( ||s_ref||^2 / (||s_ref - s_est||^2 + epsilon) )
    """
    ref_energy = np.sum(reference ** 2)
    error = reference - estimate_aligned
    distortion_energy = np.sum(error ** 2) + epsilon

    sdr_db = 10.0 * np.log10(max(ref_energy, epsilon) / distortion_energy)
    return float(sdr_db)


def evaluate_separation(
    originals: np.ndarray,
    estimates: np.ndarray,
    whitened_cov: Optional[np.ndarray] = None,
    execution_time_sec: float = 0.0,
    n_iterations: int = 0
) -> Dict[str, Any]:
    """
    Full evaluation suite:
    - Resolves permutation, scale, and polarity
    - Pearson correlations for all pairs
    - SDR (Signal to Distortion Ratio) in dB
    - Covariance of whitened signals
    - Runtime and convergence stats

    Args:
        originals: Clean reference sources (n_sources, n_samples).
        estimates: Recovered sources (n_sources, n_samples).
        whitened_cov: Covariance matrix of whitened signals.
        execution_time_sec: Time taken for ICA to converge.
        n_iterations: Number of FastICA iterations.

    Returns:
        Structured evaluation metrics dictionary.
    """
    matched, aligned, perm, signs = match_sources_and_resolve_ambiguity(originals, estimates)
    n_sources = originals.shape[0]

    source_metrics = []
    correlations = []
    sdrs = []

    for i in range(n_sources):
        orig = originals[i]
        rec_aligned = aligned[i]

        # Pearson correlation
        orig_centered = orig - np.mean(orig)
        rec_centered = rec_aligned - np.mean(rec_aligned)
        denom = (np.linalg.norm(orig_centered) * np.linalg.norm(rec_centered)) + 1e-12
        r = float(np.dot(orig_centered, rec_centered) / denom)

        # SDR
        sdr = calculate_sdr(orig, rec_aligned)

        correlations.append(abs(r))
        sdrs.append(sdr)

        source_metrics.append({
            "source_index": i + 1,
            "assigned_component": perm[i] + 1,
            "sign_flipped": signs[i] < 0,
            "pearson_correlation": r,
            "absolute_correlation": abs(r),
            "sdr_db": sdr
        })

    # Whitened covariance check
    cov_error = None
    if whitened_cov is not None:
        cov_error = float(np.linalg.norm(whitened_cov - np.eye(whitened_cov.shape[0]), "fro"))

    report = {
        "source_metrics": source_metrics,
        "mean_correlation": float(np.mean(correlations)),
        "mean_sdr_db": float(np.mean(sdrs)),
        "permutation_order": perm,
        "signs": signs,
        "execution_time_sec": execution_time_sec,
        "n_iterations": n_iterations,
        "whitening_frobenius_error": cov_error,
        "matched_estimates": matched,
        "aligned_estimates": aligned
    }

    return report
