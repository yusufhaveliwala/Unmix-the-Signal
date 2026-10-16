"""
PCA Whitening Module implemented completely from scratch using NumPy.
Zero external whitening or ICA libraries used.
"""

from typing import Tuple, Dict, Any
import numpy as np


def pca_whitening(
    X: np.ndarray,
    epsilon: float = 1e-12,
    return_components: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies PCA Whitening to centered mixed signals from scratch.

    Mathematical Formulation:
      Given centered data X in R^{m x N} (m channels, N samples):
      1. Sample covariance matrix: Sigma = (1 / N) * X @ X.T
      2. Eigendecomposition: Sigma = E @ D @ E.T
         where E is the orthogonal matrix of eigenvectors,
         and D is the diagonal matrix of eigenvalues (variances along principal axes).
      3. Whitening transformation matrix:
         V = D^{-1/2} @ E.T
      4. Whitened data:
         Z = V @ X
      5. Dewhitening (inverse) transformation matrix:
         V_inv = E @ D^{1/2}

    Properties of Z:
      Cov(Z) = (1/N) * Z @ Z.T
             = (1/N) * (V X) @ (V X).T
             = V @ Sigma @ V.T
             = (D^{-1/2} E.T) @ (E D E.T) @ (E D^{-1/2})
             = D^{-1/2} @ D @ D^{-1/2} = I_m (Identity matrix)

    Args:
        X: Centered input signals of shape (m, N), where m is number of channels,
           N is number of time samples.
        epsilon: Small regularization floor to prevent division by zero for
                 zero/near-zero eigenvalues.
        return_components: Whether to return eigenvalues and eigenvectors.

    Returns:
        Z: Whitened signals of shape (m, N) with covariance ~ Identity.
        V: Whitening matrix of shape (m, m).
        V_inv: Dewhitening matrix of shape (m, m).
    """
    if X.ndim != 2:
        raise ValueError(f"Input X must be 2-dimensional (m_channels, n_samples). Got shape {X.shape}")

    m, n_samples = X.shape
    if n_samples < m:
        raise ValueError(f"Number of samples ({n_samples}) must exceed number of channels ({m}).")

    # Verify centering (mean should be zero within numerical tolerance)
    mean_val = np.mean(X, axis=1, keepdims=True)
    if np.max(np.abs(mean_val)) > 1e-5:
        # Subtract mean automatically if not already centered
        X_centered = X - mean_val
    else:
        X_centered = X.copy()

    # 1. Compute empirical covariance matrix
    # Sigma has shape (m, m)
    covariance_matrix = (X_centered @ X_centered.T) / float(n_samples)

    # 2. Eigenvalue decomposition for symmetric positive semi-definite matrix
    # eigh guarantees real eigenvalues sorted in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort in descending order of eigenvalue magnitude
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # 3. Handle small or negative eigenvalues safely
    eigenvalues_clamped = np.maximum(eigenvalues, epsilon)

    # 4. Construct Whitening Matrix V = D^{-1/2} * E.T
    d_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_clamped))
    V = d_inv_sqrt @ eigenvectors.T

    # Dewhitening matrix V_inv = E * D^{1/2}
    d_sqrt = np.diag(np.sqrt(eigenvalues_clamped))
    V_inv = eigenvectors @ d_sqrt

    # 5. Transform data to whitened coordinates
    Z = V @ X_centered

    return Z, V, V_inv


def verify_whitening(Z: np.ndarray, tolerance: float = 1e-3) -> Dict[str, Any]:
    """
    Verifies that the whitened data Z has approximately zero mean and identity covariance.

    Args:
        Z: Whitened signals of shape (m, N).
        tolerance: Maximum allowed Frobenius norm difference from identity.

    Returns:
        Dictionary containing covariance matrix, deviation from identity,
        and boolean pass flag.
    """
    m, n_samples = Z.shape
    mean = np.mean(Z, axis=1)
    cov = (Z @ Z.T) / float(n_samples)
    identity = np.eye(m)
    frobenius_error = float(np.linalg.norm(cov - identity, 'fro'))
    max_abs_error = float(np.max(np.abs(cov - identity)))

    is_valid = frobenius_error < tolerance

    return {
        "covariance_matrix": cov.tolist(),
        "frobenius_error": frobenius_error,
        "max_abs_error": max_abs_error,
        "mean_norm": float(np.linalg.norm(mean)),
        "is_whitened": is_valid
    }
