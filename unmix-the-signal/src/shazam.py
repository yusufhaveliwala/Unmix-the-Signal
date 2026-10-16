"""
Mini-Shazam Audio Fingerprinting & Song Identification Engine (Stretch Goal).
Implements landmark-based constellation map hashing (Wang / Shazam algorithm).
Extracts spectral peak landmarks, generates hash tokens, and matches query audio
against an acoustic reference library.
Demonstrates song identification on mixed vs. ICA-separated audio tracks.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import scipy.signal


class MiniShazam:
    """
    Landmark-based audio fingerprinting engine inspired by the Wang/Shazam algorithm.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        # Database: hash -> list of (song_name, timestamp)
        self.database: Dict[str, List[Tuple[str, float]]] = {}
        self.song_metadata: Dict[str, Dict[str, Any]] = {}

    def extract_landmarks(
        self,
        audio: np.ndarray,
        n_fft: int = 1024,
        hop_length: int = 256,
        peak_threshold_percentile: float = 75.0,
        max_peaks_per_frame: int = 3
    ) -> List[Tuple[float, float]]:
        """
        Extracts prominent local spectral peaks (time, frequency) from audio spectrogram.
        """
        frequencies, times, Zxx = scipy.signal.stft(
            audio,
            fs=self.sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length
        )
        magnitude = np.abs(Zxx)
        # Power in dB
        power_db = 20.0 * np.log10(np.maximum(magnitude, 1e-6))

        # Dynamic threshold
        threshold = np.percentile(power_db, peak_threshold_percentile)

        peaks = []
        n_freqs, n_times = power_db.shape

        for t_idx in range(1, n_times - 1):
            frame_vals = power_db[:, t_idx]
            # Find frequency peaks in this frame
            local_maxima = (frame_vals[1:-1] > frame_vals[:-2]) & (frame_vals[1:-1] > frame_vals[2:])
            cand_indices = np.where(local_maxima)[0] + 1
            cand_indices = [idx for idx in cand_indices if frame_vals[idx] > threshold]

            # Pick top loudest peaks in this frame
            if cand_indices:
                cand_indices = sorted(cand_indices, key=lambda i: frame_vals[i], reverse=True)[:max_peaks_per_frame]
                for f_idx in cand_indices:
                    t_sec = float(times[t_idx])
                    f_hz = float(frequencies[f_idx])
                    peaks.append((t_sec, f_hz))

        return peaks

    def generate_fingerprints(
        self,
        landmarks: List[Tuple[float, float]],
        target_zone_dt_max: float = 1.0,
        target_zone_dt_min: float = 0.05
    ) -> List[Tuple[str, float]]:
        """
        Pairs peaks into combinatorial hash tokens:
        Hash = (f1_bin, f2_bin, delta_t_bin)
        """
        hashes = []
        n_landmarks = len(landmarks)

        for i in range(n_landmarks):
            t1, f1 = landmarks[i]
            # Look ahead for target zone points
            for j in range(i + 1, min(i + 15, n_landmarks)):
                t2, f2 = landmarks[j]
                dt = t2 - t1
                if target_zone_dt_min <= dt <= target_zone_dt_max:
                    # Quantize frequency bins (~50 Hz bins) and delta time (~0.05s)
                    f1_bin = int(round(f1 / 50.0))
                    f2_bin = int(round(f2 / 50.0))
                    dt_bin = int(round(dt / 0.05))
                    token = f"{f1_bin}:{f2_bin}:{dt_bin}"
                    hashes.append((token, t1))

        return hashes

    def index_song(self, song_id: str, title: str, artist: str, audio: np.ndarray):
        """
        Indexes a reference track into the fingerprint database.
        """
        landmarks = self.extract_landmarks(audio)
        fingerprints = self.generate_fingerprints(landmarks)

        for token, t1 in fingerprints:
            if token not in self.database:
                self.database[token] = []
            self.database[token].append((song_id, t1))

        self.song_metadata[song_id] = {
            "title": title,
            "artist": artist,
            "landmark_count": len(landmarks),
            "fingerprint_count": len(fingerprints)
        }

    def query(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Identifies audio by matching fingerprints against database and
        finding coherent time offset alignment.
        """
        landmarks = self.extract_landmarks(audio)
        query_hashes = self.generate_fingerprints(landmarks)

        # song_id -> list of time offsets (t_ref - t_query)
        matches: Dict[str, List[float]] = {}

        for token, t_query in query_hashes:
            if token in self.database:
                for song_id, t_ref in self.database[token]:
                    if song_id not in matches:
                        matches[song_id] = []
                    matches[song_id].append(t_ref - t_query)

        # Find the song with the largest coherent peak in time-offset histogram
        ranked_results = []
        for song_id, offsets in matches.items():
            if not offsets:
                continue
            # Bin offsets into 0.1s histogram
            offsets_rounded = [round(o, 1) for o in offsets]
            vals, counts = np.unique(offsets_rounded, return_counts=True)
            max_cluster_count = int(np.max(counts))

            ranked_results.append({
                "song_id": song_id,
                "title": self.song_metadata[song_id]["title"],
                "artist": self.song_metadata[song_id]["artist"],
                "coherent_matches": max_cluster_count,
                "total_hash_hits": len(offsets),
                "confidence_score": float(max_cluster_count / max(1, len(query_hashes)) * 100.0)
            })

        ranked_results = sorted(ranked_results, key=lambda x: x["coherent_matches"], reverse=True)

        top_match = ranked_results[0] if ranked_results else None
        return {
            "query_landmarks": len(landmarks),
            "query_hashes": len(query_hashes),
            "top_match": top_match,
            "candidates": ranked_results[:5]
        }
