"""
FastICA (Independent Component Analysis) implemented from scratch in NumPy.
No external ICA libraries used (no sklearn.decomposition.FastICA).
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np


class FastICAFromScratch:
    """
    Independent Component Analysis using the FastICA fixed-point algorithm
    with symmetric orthogonal decorrelation.

    Formulation:
      Given whitened data Z in R^{m x N} where E{Z Z^T} = I:
      We seek an orthogonal unmixing matrix W in R^{m x m} such that
      S = W Z maximizes the non-Gaussianity of each component.

    Fixed-point update for row i of W:
      w_i^+ = E{z * g(w_i^T z)} - E{g'(w_i^T z)} * w_i
      where g(u) is the derivative of a non-quadratic contrast function G(u).

    Symmetric Decorrelation:
      W^+ = (W^+ (W^+)^T)^{-1/2} * W^+
      ensuring orthogonality among all components at each iteration.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        contrast_func: str = "tanh",
        alpha: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-5,
        random_state: Optional[int] = 42
    ):
        """
        Args:
            n_components: Number of independent components to recover.
                          If None, defaults to number of channels m.
            contrast_func: Contrast function: 'tanh', 'exp', or 'cube'.
            alpha: Scaling parameter for tanh contrast function.
            max_iter: Maximum number of fixed-point iterations.
            tol: Convergence tolerance.
            random_state: Seed for reproducible initialization.
        """
        self.n_components = n_components
        self.contrast_func = contrast_func
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Fitted parameters
        self.W_: Optional[np.ndarray] = None  # Orthogonal matrix in whitened space
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.convergence_history_: List[float] = []

    def _g_and_g_prime(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the nonlinear contrast function g(u) and its derivative g'(u).
        """
        if self.contrast_func == "tanh":
            # g(u) = tanh(alpha * u)
            # g'(u) = alpha * (1 - tanh^2(alpha * u))
            gu = np.tanh(self.alpha * u)
            gpu = self.alpha * (1.0 - gu ** 2)
        elif self.contrast_func == "exp":
            # g(u) = u * exp(-u^2 / 2)
            # g'(u) = (1 - u^2) * exp(-u^2 / 2)
            exp_term = np.exp(-0.5 * (u ** 2))
            gu = u * exp_term
            gpu = (1.0 - (u ** 2)) * exp_term
        elif self.contrast_func == "cube":
            # g(u) = u^3
            # g'(u) = 3 * u^2
            gu = u ** 3
            gpu = 3.0 * (u ** 2)
        else:
            raise ValueError(f"Unknown contrast function: {self.contrast_func}")

        return gu, gpu

    @staticmethod
    def _symmetric_decorrelation(W: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """
        Applies symmetric orthogonalization: W <- (W * W^T)^{-1/2} * W.

        Computed via eigendecomposition of symmetric matrix M = W @ W.T:
          M = U @ diag(eigenvals) @ U.T
          M^{-1/2} = U @ diag(1 / sqrt(max(eigenvals, epsilon))) @ U.T
          W_orth = M^{-1/2} @ W
        """
        M = W @ W.T
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # Protect against non-positive eigenvalues
        eigenvalues = np.maximum(eigenvalues, epsilon)
        inv_sqrt_diag = np.diag(1.0 / np.sqrt(eigenvalues))
        M_inv_sqrt = eigenvectors @ inv_sqrt_diag @ eigenvectors.T
        return M_inv_sqrt @ W

    def fit(self, Z: np.ndarray) -> "FastICAFromScratch":
        """
        Fits FastICA on whitened data Z of shape (m, N).

        Args:
            Z: Whitened data array of shape (m, N).

        Returns:
            self
        """
        m, n_samples = Z.shape
        n_comp = self.n_components if self.n_components is not None else m
        if n_comp > m:
            raise ValueError(f"n_components ({n_comp}) cannot exceed input channels ({m})")

        rng = np.random.RandomState(self.random_state)
        # Random initial orthogonal matrix W of shape (n_comp, m)
        W = rng.normal(size=(n_comp, m))
        W = self._symmetric_decorrelation(W)

        self.convergence_history_ = []
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            # Compute current component estimates: Y = W @ Z (shape: n_comp, N)
            Y = W @ Z

            # Non-linear contrast evaluations
            gu, gpu = self._g_and_g_prime(Y)

            # Fixed-point update:
            # First term: E{ g(W z) z^T } = (gu @ Z.T) / N
            term1 = (gu @ Z.T) / float(n_samples)

            # Second term: E{ g'(W z) } * W
            # diag_gpu has shape (n_comp, 1) representing mean of g'(Y_i)
            mean_gpu = np.mean(gpu, axis=1, keepdims=True)
            term2 = mean_gpu * W

            W_new = term1 - term2

            # Symmetric decorrelation
            W_new = self._symmetric_decorrelation(W_new)

            # Check convergence: difference between identity and |W_new @ W.T|
            # (Accounting for arbitrary sign flips of individual components)
            lim = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
            self.convergence_history_.append(float(lim))

            W = W_new
            self.n_iter_ = iteration

            if lim < self.tol:
                self.converged_ = True
                break

        self.W_ = W
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Recovers independent components from whitened data Z.

        Args:
            Z: Whitened signals of shape (m, N).

        Returns:
            S_est: Separated sources of shape (n_components, N).
        """
        if self.W_ is None:
            raise RuntimeError("FastICA has not been fitted yet. Call fit() first.")
        return self.W_ @ Z

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Fits on Z and returns separated sources S_est.
        """
        self.fit(Z)
        return self.transform(Z)


def separate_sources(
    X_mixed: np.ndarray,
    V_whitening: np.ndarray,
    contrast_func: str = "tanh",
    max_iter: int = 250,
    tol: float = 1e-5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    High-level convenience function:
    Applies FastICA on whitened signals Z = V @ X_mixed and reconstructs
    separated sources and total unmixing / estimated mixing matrices.

    Args:
        X_mixed: Centered mixed signals of shape (m, N).
        V_whitening: Whitening matrix of shape (m, m).
        contrast_func: 'tanh', 'exp', or 'cube'.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        random_state: Seed.

    Returns:
        Dictionary with separated sources, W_total, A_estimated, iterations,
        and convergence flag.
    """
    Z = V_whitening @ X_mixed

    ica = FastICAFromScratch(
        n_components=X_mixed.shape[0],
        contrast_func=contrast_func,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    S_est = ica.fit_transform(Z)

    # Total unmixing matrix: W_total = W @ V
    # S_est = W_total @ X_mixed
    W_total = ica.W_ @ V_whitening

    # Estimated mixing matrix: pseudo-inverse of W_total
    A_est = np.linalg.pinv(W_total)

    return {
        "separated_sources": S_est,
        "W_whitened": ica.W_,
        "W_total": W_total,
        "A_estimated": A_est,
        "n_iter": ica.n_iter_,
        "converged": ica.converged_,
        "convergence_history": ica.convergence_history_
    }
