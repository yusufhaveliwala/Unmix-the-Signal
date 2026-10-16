"""
Audio Mixing Module.
Applies instantaneous linear mixing A * S, centers signals, checks condition number,
and allows controlled additive noise injection with specified SNR.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def create_mixing_matrix(
    n_mixtures: int = 2,
    n_sources: int = 2,
    custom_matrix: Optional[np.ndarray] = None,
    random_seed: Optional[int] = 42
) -> np.ndarray:
    """
    Creates a non-singular, well-conditioned mixing matrix A of shape (n_mixtures, n_sources).

    Default 2x2 mixing matrix:
      A = [[0.75, 0.35],
           [0.40, 0.80]]
      Condition number ~ 2.4 (stable, easily invertible, non-singular).
    """
    if custom_matrix is not None:
        A = np.array(custom_matrix, dtype=np.float64)
        if A.shape != (n_mixtures, n_sources):
            raise ValueError(f"Custom matrix shape {A.shape} must match ({n_mixtures}, {n_sources})")
    elif n_mixtures == 2 and n_sources == 2:
        A = np.array([
            [0.75, 0.35],
            [0.40, 0.80]
        ], dtype=np.float64)
    elif n_mixtures == 3 and n_sources == 3:
        A = np.array([
            [0.70, 0.25, 0.20],
            [0.30, 0.75, 0.25],
            [0.20, 0.30, 0.70]
        ], dtype=np.float64)
    else:
        rng = np.random.RandomState(random_seed)
        A = rng.uniform(0.3, 0.9, size=(n_mixtures, n_sources))
        # Ensure diagonal dominance to guarantee invertibility
        for i in range(min(n_mixtures, n_sources)):
            A[i, i] += 0.5

    # Check condition number
    cond = np.linalg.cond(A)
    if cond > 50.0:
        raise ValueError(f"Mixing matrix is ill-conditioned (condition number {cond:.2f} > 50).")

    return A


def mix_audio_signals(
    sources: np.ndarray,
    mixing_matrix: np.ndarray,
    add_noise_snr_db: Optional[float] = None,
    noise_seed: int = 999
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Mixes sources: X = A @ S + Noise, then centers the mixed signals.

    Args:
        sources: Array of shape (n_sources, n_samples) containing clean sources.
        mixing_matrix: Array of shape (n_mixtures, n_sources).
        add_noise_snr_db: If provided, adds Gaussian/ambient noise with specified SNR in dB.
        noise_seed: Random seed for noise.

    Returns:
        X_centered: Centered mixed signals of shape (n_mixtures, n_samples).
        X_raw: Uncentered mixed signals (with noise if specified).
        metadata: Details on mixing matrix, condition number, added noise.
    """
    n_sources, n_samples = sources.shape
    n_mixtures, n_s_matrix = mixing_matrix.shape

    if n_sources != n_s_matrix:
        raise ValueError(f"Source count {n_sources} does not match mixing matrix columns {n_s_matrix}")

    # Instantaneous linear mixture: X = A * S
    X_raw = mixing_matrix @ sources

    noise_added = False
    noise_power_actual = 0.0

    if add_noise_snr_db is not None:
        rng = np.random.RandomState(noise_seed)
        noise_added = True
        # Calculate signal power across all channels
        sig_power = np.mean(X_raw ** 2)
        # SNR_dB = 10 * log10(P_signal / P_noise) => P_noise = P_signal / (10^(SNR/10))
        target_noise_power = sig_power / (10.0 ** (add_noise_snr_db / 10.0))
        noise = rng.normal(0, np.sqrt(target_noise_power), size=X_raw.shape)
        X_raw = X_raw + noise
        noise_power_actual = float(np.mean(noise ** 2))

    # Center the mixed signals by subtracting channel-wise means
    channel_means = np.mean(X_raw, axis=1, keepdims=True)
    X_centered = X_raw - channel_means

    metadata = {
        "mixing_matrix": mixing_matrix.tolist(),
        "condition_number": float(np.linalg.cond(mixing_matrix)),
        "determinant": float(np.linalg.det(mixing_matrix)) if n_mixtures == n_sources else None,
        "n_sources": n_sources,
        "n_mixtures": n_mixtures,
        "n_samples": n_samples,
        "noise_added": noise_added,
        "target_snr_db": add_noise_snr_db,
        "channel_means": channel_means.flatten().tolist()
    }

    return X_centered.astype(np.float32), X_raw.astype(np.float32), metadata
