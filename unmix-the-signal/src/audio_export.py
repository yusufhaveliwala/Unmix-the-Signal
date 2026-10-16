"""
Audio Export Module.
Safely exports real audio files as standard 16-bit PCM WAV files,
ensures non-clipping amplitude ranges [-1.0, 1.0], validates file size and integrity.
"""

import os
from typing import Dict, Any, List
import numpy as np
import soundfile as sf


def export_wav(
    file_path: str,
    audio_data: np.ndarray,
    sample_rate: int = 22050,
    subtype: str = "PCM_16",
    normalize_peak: float = 0.90
) -> Dict[str, Any]:
    """
    Exports 1D or 2D audio array to a standard playable WAV file.

    Args:
        file_path: Output file path.
        audio_data: Array of audio samples.
        sample_rate: Audio sampling frequency in Hz.
        subtype: WAV subtype (default 16-bit PCM).
        normalize_peak: Peak amplitude to scale to.

    Returns:
        Dictionary with file size, duration, channels, and integrity check.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    # Ensure float32 in [-1.0, 1.0]
    data = np.array(audio_data, dtype=np.float32)

    # Check finite
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.8, neginf=-0.8)

    # Normalize peak to prevent clipping
    max_amp = np.max(np.abs(data))
    if max_amp > 1e-6:
        data = (data / max_amp) * normalize_peak

    # Save via soundfile
    sf.write(file_path, data, sample_rate, subtype=subtype)

    # Verify file
    if not os.path.exists(file_path):
        raise IOError(f"Failed to create output audio file at {file_path}")

    file_size_bytes = os.path.getsize(file_path)
    if file_size_bytes == 0:
        raise ValueError(f"Output audio file {file_path} is empty (0 bytes).")

    info = sf.info(file_path)

    return {
        "file_path": file_path,
        "file_size_bytes": file_size_bytes,
        "duration_sec": info.duration,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "format": info.format
    }


def export_pipeline_audio_files(
    output_dir: str,
    original_sources: np.ndarray,
    mixed_signals: np.ndarray,
    cleaned_mixture: np.ndarray,
    separated_sources: np.ndarray,
    cleaned_separated_sources: np.ndarray,
    sample_rate: int = 22050
) -> Dict[str, str]:
    """
    Exports all mandatory pipeline audio outputs:
      - original_source_1.wav
      - original_source_2.wav
      - mixed_audio_1.wav
      - mixed_audio_2.wav
      - cleaned_mixture.wav
      - separated_source_1.wav
      - separated_source_2.wav
      - cleaned_separated_source_1.wav
      - cleaned_separated_source_2.wav
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    # 1. Original sources
    for i in range(original_sources.shape[0]):
        filename = f"original_source_{i + 1}.wav"
        dest = os.path.join(output_dir, filename)
        export_wav(dest, original_sources[i], sample_rate)
        saved_paths[filename] = dest

    # 2. Mixed recordings
    for i in range(mixed_signals.shape[0]):
        filename = f"mixed_audio_{i + 1}.wav"
        dest = os.path.join(output_dir, filename)
        export_wav(dest, mixed_signals[i], sample_rate)
        saved_paths[filename] = dest

    # 3. Cleaned mixture
    clean_mix_signal = cleaned_mixture[0] if cleaned_mixture.ndim > 1 else cleaned_mixture
    dest_clean_mix = os.path.join(output_dir, "cleaned_mixture.wav")
    export_wav(dest_clean_mix, clean_mix_signal, sample_rate)
    saved_paths["cleaned_mixture.wav"] = dest_clean_mix

    # 4. Separated sources
    for i in range(separated_sources.shape[0]):
        filename = f"separated_source_{i + 1}.wav"
        dest = os.path.join(output_dir, filename)
        export_wav(dest, separated_sources[i], sample_rate)
        saved_paths[filename] = dest

    # 5. Cleaned separated sources
    for i in range(cleaned_separated_sources.shape[0]):
        filename = f"cleaned_separated_source_{i + 1}.wav"
        dest = os.path.join(output_dir, filename)
        export_wav(dest, cleaned_separated_sources[i], sample_rate)
        saved_paths[filename] = dest

    return saved_paths
