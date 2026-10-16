"""
Automated unit tests for PCA Whitening module.
Uses Python's standard unittest framework for 100% dependency-free test execution.
"""

import unittest
import numpy as np
from src.whitening import pca_whitening, verify_whitening


class TestPCAWhitening(unittest.TestCase):

    def test_pca_whitening_identity_covariance(self):
        """Verify that PCA whitening produces an empirical covariance matrix equal to Identity."""
        rng = np.random.RandomState(42)
        X_orig = rng.normal(loc=2.0, scale=3.0, size=(3, 10000))
        mix_mat = np.array([
            [1.0, 0.8, 0.2],
            [0.5, 1.2, 0.7],
            [0.1, 0.4, 1.5]
        ])
        X = mix_mat @ X_orig

        Z, V, V_inv = pca_whitening(X)

        self.assertEqual(Z.shape, X.shape)
        self.assertEqual(V.shape, (3, 3))
        self.assertEqual(V_inv.shape, (3, 3))

        # Verify zero mean
        mean_z = np.mean(Z, axis=1)
        np.testing.assert_allclose(mean_z, np.zeros(3), atol=1e-5)

        # Verify covariance is Identity
        verification = verify_whitening(Z, tolerance=1e-2)
        self.assertTrue(verification["is_whitened"])
        self.assertLess(verification["frobenius_error"], 0.02)

    def test_pca_whitening_inversion(self):
        """Verify that dewhitening matrix reconstructs centered data: X_centered = V_inv @ Z."""
        rng = np.random.RandomState(123)
        X = rng.randn(2, 5000)
        X_centered = X - np.mean(X, axis=1, keepdims=True)

        Z, V, V_inv = pca_whitening(X_centered)
        X_reconstructed = V_inv @ Z

        np.testing.assert_allclose(X_centered, X_reconstructed, atol=1e-6)

    def test_pca_whitening_singular_protection(self):
        """Verify safe handling of near-zero eigenvalues without crashing or NaN."""
        X = np.zeros((2, 1000))
        X[0] = np.sin(np.linspace(0, 10, 1000))
        X[1] = 2.0 * X[0]  # exact collinearity

        Z, V, V_inv = pca_whitening(X, epsilon=1e-10)
        self.assertTrue(np.all(np.isfinite(Z)))
        self.assertTrue(np.all(np.isfinite(V)))
        self.assertTrue(np.all(np.isfinite(V_inv)))


if __name__ == '__main__':
    unittest.main()
