"""
Audio Loading and Source Generation Module.
Provides audio loading, resampling, duration alignment, and synthesis of
high-fidelity non-Gaussian audio sources (speech-like formant synthesis,
polyphonic musical progression, and ambient texture).
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import soundfile as sf
import scipy.signal


def load_and_preprocess_audio(
    file_path: str,
    target_sr: int = 22050,
    target_duration_sec: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file, converts multi-channel to mono, resamples to target_sr,
    and optionally trims/pads to target_duration_sec.

    Args:
        file_path: Path to the audio file.
        target_sr: Desired sampling rate (default 22050 Hz).
        target_duration_sec: Target duration in seconds.

    Returns:
        audio: 1D numpy array of float32 samples in range [-1.0, 1.0].
        sample_rate: Actual sample rate (target_sr).
    """
    data, sr = sf.read(file_path, dtype="float32")

    # Convert to mono if stereo or multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = scipy.signal.resample_poly(data, up, down)
        sr = target_sr

    # Duration alignment
    if target_duration_sec is not None:
        target_len = int(target_duration_sec * target_sr)
        if len(data) > target_len:
            data = data[:target_len]
        elif len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)), mode="constant")

    # Normalize to avoid initial clipping
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data / max_val * 0.9

    return data.astype(np.float32), sr


def generate_synthetic_speech_like(
    duration_sec: float = 6.0,
    sample_rate: int = 22050,
    random_seed: int = 101
) -> np.ndarray:
    """
    Generates a realistic synthetic speech-like acoustic signal using
    harmonic glottal excitation, pitch contour (F0 ~ 130-180 Hz),
    and time-varying vowel formants (F1, F2, F3) with unvoiced fricative bursts.

    Clearly labeled as high-fidelity synthetic acoustic speech.
    """
    rng = np.random.RandomState(random_seed)
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Pitch contour F0(t) with natural intonation cadence
    f0 = 135.0 + 25.0 * np.sin(2 * np.pi * 0.8 * t) + 15.0 * np.cos(2 * np.pi * 2.1 * t)
    phase = np.cumsum(2 * np.pi * f0 / sample_rate)

    # Glottal pulse train: sum of harmonics with 1/n spectral decay
    glottal = np.zeros(n_samples)
    for harmonic in range(1, 18):
        amplitude = (1.0 / (harmonic ** 0.85)) * (0.9 + 0.1 * np.sin(harmonic * 1.5))
        glottal += amplitude * np.sin(harmonic * phase)

    # Modulate vocal tract formants (cycling through vowels /a/, /i/, /u/, /e/, /o/)
    # Resonances:
    # Formant 1: 300 to 800 Hz
    # Formant 2: 900 to 2200 Hz
    # Formant 3: 2500 to 3200 Hz
    vowel_mod = 0.5 * (1.0 + np.sin(2 * np.pi * 1.4 * t))
    f1_freq = 400.0 + 400.0 * vowel_mod
    f2_freq = 1100.0 + 1000.0 * (1.0 - vowel_mod)
    f3_freq = 2700.0 + 300.0 * np.sin(2 * np.pi * 0.7 * t)

    # Formant filtering using 2nd-order resonator bandpass simulation
    def apply_resonator(sig, center_freq_arr, bandwidth=80.0):
        out = np.zeros_like(sig)
        for i in range(2, len(sig)):
            cf = center_freq_arr[i]
            r = np.exp(-np.pi * bandwidth / sample_rate)
            theta = 2 * np.pi * cf / sample_rate
            a1 = -2 * r * np.cos(theta)
            a2 = r * r
            b0 = 1 - r
            out[i] = b0 * sig[i] - a1 * out[i - 1] - a2 * out[i - 2]
        return out

    formant1 = apply_resonator(glottal, f1_freq, bandwidth=100.0)
    formant2 = apply_resonator(glottal, f2_freq, bandwidth=140.0)
    formant3 = apply_resonator(glottal, f3_freq, bandwidth=200.0)

    voiced_speech = 1.0 * formant1 + 0.65 * formant2 + 0.35 * formant3

    # Add unvoiced fricative /s/, /sh/, /t/ consonants at phrase intervals
    noise = rng.normal(0, 1, n_samples)
    # Bandpass fricative noise (3 kHz to 7 kHz)
    sos_fricative = scipy.signal.butter(4, [3000, 7000], btype="bandpass", fs=sample_rate, output="sos")
    fricative_noise = scipy.signal.sosfilt(sos_fricative, noise)

    consonant_gate = np.maximum(0, np.sin(2 * np.pi * 3.2 * t) - 0.75) * 4.0
    speech_total = voiced_speech + 0.3 * fricative_noise * consonant_gate

    # Syllabic envelope rhythm (speaking pauses and stress)
    syllables = np.clip(0.3 + 0.7 * np.sin(2 * np.pi * 2.5 * t) ** 2, 0.05, 1.0)
    # Phrase boundaries
    phrasing = np.sin(np.pi * t / duration_sec) ** 0.5
    speech_total *= syllables * phrasing

    # Normalize
    speech_total = speech_total / np.max(np.abs(speech_total)) * 0.85
    return speech_total.astype(np.float32)


def generate_synthetic_music_like(
    duration_sec: float = 6.0,
    sample_rate: int = 22050,
    random_seed: int = 202
) -> np.ndarray:
    """
    Generates a musical acoustic audio track:
    An acoustic progression of guitar/piano arpeggio notes with rich harmonic
    series, exponential decay, pluck resonance, and vibrato.

    Clearly labeled as high-fidelity synthetic acoustic music.
    """
    rng = np.random.RandomState(random_seed)
    n_samples = int(duration_sec * sample_rate)
    music = np.zeros(n_samples, dtype=np.float64)

    # Musical arpeggio notes in D major / B minor:
    # D3 (146.8), F#3 (185.0), A3 (220.0), D4 (293.7), F#4 (370.0), A4 (440.0), B3 (246.9), E4 (329.6)
    notes = [146.83, 220.00, 293.66, 369.99, 440.00, 369.99, 293.66, 220.00,
             164.81, 246.94, 329.63, 392.00, 493.88, 392.00, 329.63, 246.94]

    note_duration = 0.35  # seconds per note
    n_notes = int(np.ceil(duration_sec / note_duration))

    for idx in range(n_notes):
        start_time = idx * note_duration
        start_sample = int(start_time * sample_rate)
        if start_sample >= n_samples:
            break

        freq = notes[idx % len(notes)]
        note_len = min(int(1.8 * sample_rate), n_samples - start_sample)  # let notes ring/decay
        t_note = np.linspace(0, note_len / sample_rate, note_len, endpoint=False)

        # Subtle pitch vibrato
        vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * t_note)
        note_phase = np.cumsum(2 * np.pi * freq * vib / sample_rate)

        # Karplus-Strong / acoustic string harmonic decay
        tone = np.zeros(note_len)
        harmonics = [1.0, 0.65, 0.40, 0.28, 0.18, 0.10, 0.06]
        for h_idx, h_amp in enumerate(harmonics, start=1):
            decay_rate = 2.2 * (h_idx ** 0.6)
            tone += h_amp * np.sin(h_idx * note_phase) * np.exp(-decay_rate * t_note)

        # Pluck attack transient
        attack_len = min(int(0.005 * sample_rate), note_len)
        attack_env = np.ones(note_len)
        if attack_len > 0:
            attack_env[:attack_len] = np.linspace(0, 1, attack_len)

        # Pluck noise burst
        pluck_noise = rng.normal(0, 0.2, note_len) * np.exp(-80.0 * t_note)

        note_signal = (tone * attack_env) + pluck_noise
        music[start_sample:start_sample + note_len] += note_signal

    # Global normalizer
    music = music / np.max(np.abs(music)) * 0.85
    return music.astype(np.float32)


def generate_synthetic_ambient_like(
    duration_sec: float = 6.0,
    sample_rate: int = 22050,
    random_seed: int = 303
) -> np.ndarray:
    """
    Generates a natural ambient acoustic sound (rain, wind, stream ambience)
    with low-frequency pink noise filtering and occasional droplet transients.
    """
    rng = np.random.RandomState(random_seed)
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # 1. Pink/Brown noise for ambient atmospheric wash
    white = rng.normal(0, 1, n_samples)
    # 1/f filter approximation
    b, a = scipy.signal.butter(1, 400 / (sample_rate / 2), btype="low")
    ambient_base = scipy.signal.lfilter(b, a, white)

    # Slow wind modulation
    wind_mod = 0.5 + 0.5 * (np.sin(2 * np.pi * 0.3 * t) * np.cos(2 * np.pi * 0.1 * t))
    ambient_base *= wind_mod

    # 2. Water droplets / rain pitter-patter
    n_drops = int(duration_sec * 30)
    drop_times = rng.uniform(0, duration_sec, n_drops)
    droplets = np.zeros(n_samples)

    for dt in drop_times:
        s_idx = int(dt * sample_rate)
        drop_freq = rng.uniform(1200, 2400)
        d_len = int(sample_rate * 0.04)
        if s_idx + d_len < n_samples:
            t_drop = np.linspace(0, 0.04, d_len, endpoint=False)
            d_wave = np.sin(2 * np.pi * drop_freq * t_drop) * np.exp(-90 * t_drop)
            droplets[s_idx:s_idx + d_len] += rng.uniform(0.1, 0.4) * d_wave

    ambient = 0.7 * ambient_base + 0.3 * droplets
    ambient = ambient / np.max(np.abs(ambient)) * 0.80
    return ambient.astype(np.float32)


def get_default_sources(
    duration_sec: float = 6.0,
    sample_rate: int = 22050,
    source_dir: Optional[str] = None
) -> List[Tuple[str, np.ndarray, str]]:
    """
    Returns list of (source_name, audio_samples, source_description).
    Checks if external source audio exists in source_dir; otherwise generates
    high-fidelity synthetic speech, music, and ambient sound.
    """
    sources = []

    if source_dir and os.path.exists(source_dir):
        # Look for existing wav files
        wavs = sorted([f for f in os.listdir(source_dir) if f.endswith(".wav")])
        for w in wavs:
            full_path = os.path.join(source_dir, w)
            data, _ = load_and_preprocess_audio(full_path, sample_rate, duration_sec)
            sources.append((os.path.splitext(w)[0], data, "Loaded clean audio source"))

    if len(sources) < 2:
        # Generate pristine acoustic sources
        s1 = generate_synthetic_speech_like(duration_sec, sample_rate, random_seed=101)
        s2 = generate_synthetic_music_like(duration_sec, sample_rate, random_seed=202)
        s3 = generate_synthetic_ambient_like(duration_sec, sample_rate, random_seed=303)

        sources = [
            ("speech", s1, "Synthetic Human Speech (Formant acoustic model with F0, F1-F3 resonances & consonants)"),
            ("music", s2, "Acoustic Guitar/Piano Arpeggio (Harmonic series with pluck envelope & decay)"),
            ("ambient", s3, "Ambient Rain & Wind (Low-frequency pink noise wash & droplet transients)")
        ]

    return sources
