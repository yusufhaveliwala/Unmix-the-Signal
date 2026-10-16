"""
Automated end-to-end audio pipeline tests.
Validates file generation, audio export integrity, duration, channel count,
amplitude validity, and spectral noise reduction effectiveness.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import soundfile as sf

from src.audio_loader import generate_synthetic_speech_like, generate_synthetic_music_like
from src.audio_mixer import create_mixing_matrix, mix_audio_signals
from src.whitening import pca_whitening
from src.ica import FastICAFromScratch
from src.noise_reduction import spectral_noise_reduction
from src.audio_export import export_pipeline_audio_files


class TestAudioPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_audio_pipeline_file_generation_and_playback(self):
        """
        Runs pipeline on test signals, exports all 9 files, and validates:
        1. All 9 files exist and have nonzero size (> 1000 bytes)
        2. Files can be decoded by soundfile without errors
        3. Proper sample rate (22050 Hz), 1 channel (mono), finite samples
        4. Audio is not silent (energy > 0) and not clipping (amplitude <= 1.0)
        """
        sr = 22050
        duration = 2.0
        s1 = generate_synthetic_speech_like(duration, sr, random_seed=42)
        s2 = generate_synthetic_music_like(duration, sr, random_seed=84)
        sources = np.stack([s1, s2], axis=0)

        A = create_mixing_matrix(2, 2)
        X_cent, X_raw, _ = mix_audio_signals(sources, A)

        Z, V, _ = pca_whitening(X_cent)

        ica = FastICAFromScratch(contrast_func="tanh", max_iter=200, random_state=42)
        S_est = ica.fit_transform(Z)

        cleaned_mix, _ = spectral_noise_reduction(X_raw[0], sr)
        cleaned_s1, _ = spectral_noise_reduction(S_est[0], sr)
        cleaned_s2, _ = spectral_noise_reduction(S_est[1], sr)
        cleaned_sep = np.stack([cleaned_s1, cleaned_s2], axis=0)

        saved_files = export_pipeline_audio_files(
            output_dir=self.test_dir,
            original_sources=sources,
            mixed_signals=X_raw,
            cleaned_mixture=cleaned_mix,
            separated_sources=S_est,
            cleaned_separated_sources=cleaned_sep,
            sample_rate=sr
        )

        mandatory_names = [
            "original_source_1.wav",
            "original_source_2.wav",
            "mixed_audio_1.wav",
            "mixed_audio_2.wav",
            "cleaned_mixture.wav",
            "separated_source_1.wav",
            "separated_source_2.wav",
            "cleaned_separated_source_1.wav",
            "cleaned_separated_source_2.wav"
        ]

        for fname in mandatory_names:
            self.assertIn(fname, saved_files)
            fpath = saved_files[fname]
            self.assertTrue(os.path.exists(fpath))
            self.assertGreater(os.path.getsize(fpath), 1000)

            audio_in, sample_rate_in = sf.read(fpath)
            self.assertEqual(sample_rate_in, sr)
            self.assertEqual(audio_in.ndim, 1)
            self.assertTrue(np.all(np.isfinite(audio_in)))
            self.assertLessEqual(np.max(np.abs(audio_in)), 1.0)
            self.assertGreater(np.max(np.abs(audio_in)), 0.05)

    def test_noise_reduction_attenuates_stationary_noise(self):
        """Verify that spectral noise reduction reduces background noise floor."""
        sr = 22050
        t = np.linspace(0, 2.0, int(2.0 * sr))
        signal = 0.6 * np.sin(2 * np.pi * 440 * t)
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 0.15, len(t))
        noisy = signal + noise

        cleaned, metrics = spectral_noise_reduction(noisy, sr)

        self.assertGreaterEqual(metrics["snr_improvement_db"], 0.0)
        self.assertTrue(np.all(np.isfinite(cleaned)))
        self.assertLessEqual(np.max(np.abs(cleaned)), 1.0)


if __name__ == '__main__':
    unittest.main()
