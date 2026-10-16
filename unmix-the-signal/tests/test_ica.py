"""
Automated unit tests for FastICA algorithm implemented from scratch.
Uses Python's standard unittest framework for 100% dependency-free test execution.
"""

import unittest
import numpy as np
from src.whitening import pca_whitening
from src.ica import FastICAFromScratch
from src.evaluation import match_sources_and_resolve_ambiguity


class TestFastICA(unittest.TestCase):

    def _generate_benchmark_independent_signals(self, n_samples: int = 10000):
        t = np.linspace(0, 50, n_samples)
        s1 = np.sin(2.0 * t)
        s2 = np.sign(np.sin(3.0 * t))
        sources = np.stack([s1, s2], axis=0)
        sources = sources / np.std(sources, axis=1, keepdims=True)
        return sources

    def test_fastica_convergence_and_separation(self):
        """Verify FastICA converges and successfully recovers mixed independent sources."""
        sources = self._generate_benchmark_independent_signals(10000)
        A = np.array([[0.6, 0.4], [0.3, 0.7]])
        X = A @ sources

        Z, V, _ = pca_whitening(X)

        ica = FastICAFromScratch(contrast_func="tanh", max_iter=200, tol=1e-5, random_state=42)
        S_est = ica.fit_transform(Z)

        self.assertTrue(ica.converged_)
        self.assertLess(ica.n_iter_, 100)

        # Verify orthogonal W in whitened space: W @ W.T = I
        orth_error = np.linalg.norm(ica.W_ @ ica.W_.T - np.eye(2), 'fro')
        self.assertLess(orth_error, 1e-4)

        # Resolve permutation and sign ambiguities
        matched, aligned, perm, signs = match_sources_and_resolve_ambiguity(sources, S_est)

        corr_1 = np.corrcoef(sources[0], aligned[0])[0, 1]
        corr_2 = np.corrcoef(sources[1], aligned[1])[0, 1]

        self.assertGreater(abs(corr_1), 0.95)
        self.assertGreater(abs(corr_2), 0.95)

    def test_fastica_contrast_functions(self):
        """Test all supported contrast functions ('tanh', 'exp', 'cube')."""
        sources = self._generate_benchmark_independent_signals(6000)
        A = np.array([[0.7, 0.3], [0.4, 0.8]])
        X = A @ sources
        Z, V, _ = pca_whitening(X)

        for func in ["tanh", "exp", "cube"]:
            ica = FastICAFromScratch(contrast_func=func, max_iter=250, tol=1e-4, random_state=123)
            S_est = ica.fit_transform(Z)
            self.assertTrue(ica.converged_)
            matched, aligned, _, _ = match_sources_and_resolve_ambiguity(sources, S_est)
            r0 = np.corrcoef(sources[0], aligned[0])[0, 1]
            r1 = np.corrcoef(sources[1], aligned[1])[0, 1]
            self.assertGreater(abs(r0), 0.90)
            self.assertGreater(abs(r1), 0.90)


if __name__ == '__main__':
    unittest.main()
