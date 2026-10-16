# Test Inputs Directory

This directory is designated for supplying custom external audio test recordings.

## Expected Test File Specification:
1. `noisy_test_audio.wav` — A mono WAV recording containing a target acoustic signal mixed with broadband hiss, 60 Hz hum, rumble, or clicks.
2. `ica_two_channel_mixture.wav` — A stereo 2-channel WAV recording containing two spatial mixtures generated from underlying independent sources.
3. `clean_reference.wav` — (Optional) Clean reference signal for objective SNR and Pearson correlation calculations.
4. `noise_only_reference.wav` — (Optional) Isolated noise sample for profile estimation.

## How to Run:
Place any custom `.wav` files into this directory or upload them directly via the web application interface. The FastICA and spectral noise reduction pipeline will automatically detect, decode, center, whiten, and unmix the audio.
