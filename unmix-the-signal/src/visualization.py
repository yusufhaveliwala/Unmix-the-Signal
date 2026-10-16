"""
Visualization Module.
Generates publication-quality comparative plots for audio waveforms,
PCA vs ICA scatter distributions, time-frequency spectrograms,
convergence curves, and SNR vs SDR performance benchmarks.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal


def plot_waveform_comparison(
    originals: np.ndarray,
    mixtures: np.ndarray,
    separated: np.ndarray,
    cleaned: np.ndarray,
    sample_rate: int,
    output_path: str,
    time_window_sec: float = 1.0
):
    """
    Plots time-domain waveforms for Original Sources, Mixtures,
    Recovered Separated Sources, and Cleaned Separated Sources.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_samples = min(int(time_window_sec * sample_rate), originals.shape[1])
    t = np.linspace(0, n_samples / sample_rate, n_samples)

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Unmix the Signal (ICA): End-to-End Waveform Comparison", fontsize=15, fontweight="bold", y=0.98)

    # 1. Originals
    for i in range(2):
        axes[0, i].plot(t, originals[i, :n_samples], color="#1E88E5", lw=1.2)
        axes[0, i].set_title(f"Clean Original Source {i+1} (Ground Truth)", fontsize=11, fontweight="semibold")
        axes[0, i].set_ylabel("Amplitude")
        axes[0, i].grid(True, alpha=0.3)

    # 2. Mixtures
    for i in range(2):
        axes[1, i].plot(t, mixtures[i, :n_samples], color="#E53935", lw=1.2)
        axes[1, i].set_title(f"Mixed Recording {i+1} (Linear Superposition)", fontsize=11, fontweight="semibold")
        axes[1, i].set_ylabel("Amplitude")
        axes[1, i].grid(True, alpha=0.3)

    # 3. Separated via FastICA
    for i in range(2):
        axes[2, i].plot(t, separated[i, :n_samples], color="#43A047", lw=1.2)
        axes[2, i].set_title(f"Recovered Source {i+1} (FastICA Unmixed)", fontsize=11, fontweight="semibold")
        axes[2, i].set_ylabel("Amplitude")
        axes[2, i].grid(True, alpha=0.3)

    # 4. Cleaned Separated Sources
    for i in range(2):
        axes[3, i].plot(t, cleaned[i, :n_samples], color="#8E24AA", lw=1.2)
        axes[3, i].set_title(f"Cleaned Source {i+1} (Spectral Gating Applied)", fontsize=11, fontweight="semibold")
        axes[3, i].set_ylabel("Amplitude")
        axes[3, i].set_xlabel("Time (seconds)")
        axes[3, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pca_vs_ica_scatter(
    sources: np.ndarray,
    mixtures: np.ndarray,
    whitened: np.ndarray,
    separated: np.ndarray,
    output_path: str
):
    """
    Scatter plots illustrating why PCA fails to separate independent sources:
    - Clean sources: Non-Gaussian, independent distribution
    - Mixtures: Convolved/skewed joint distribution
    - Whitened (PCA): Uncorrelated, spherical covariance, but still rotated/mixed!
    - FastICA: Re-aligned along true independent component axes via non-Gaussianity!
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("Geometric Insight: Why PCA Only Decorrelates, While FastICA Recovers Independent Sources",
                 fontsize=14, fontweight="bold")

    # Downsample points for clear visualization
    subsample = 4
    s1, s2 = sources[0, ::subsample], sources[1, ::subsample]
    x1, x2 = mixtures[0, ::subsample], mixtures[1, ::subsample]
    z1, z2 = whitened[0, ::subsample], whitened[1, ::subsample]
    y1, y2 = separated[0, ::subsample], separated[1, ::subsample]

    # Panel 1: Original Sources
    axes[0].scatter(s1, s2, alpha=0.15, s=6, color="#1E88E5")
    axes[0].set_title("1. Clean Sources (s1 vs s2)\nStatistically Independent", fontsize=11, fontweight="semibold")
    axes[0].set_xlabel("Source 1")
    axes[0].set_ylabel("Source 2")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].axvline(0, color="gray", lw=0.7, ls="--")
    axes[0].grid(True, alpha=0.2)

    # Panel 2: Mixtures
    axes[1].scatter(x1, x2, alpha=0.15, s=6, color="#E53935")
    axes[1].set_title("2. Mixed Signals (x1 vs x2)\nCorrelated Linear Combination", fontsize=11, fontweight="semibold")
    axes[1].set_xlabel("Mixture 1")
    axes[1].set_ylabel("Mixture 2")
    axes[1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[1].axvline(0, color="gray", lw=0.7, ls="--")
    axes[1].grid(True, alpha=0.2)

    # Panel 3: Whitened (PCA)
    axes[2].scatter(z1, z2, alpha=0.15, s=6, color="#FB8C00")
    axes[2].set_title("3. PCA Whitened (z1 vs z2)\nUncorrelated (Cov = I), but Rotated!", fontsize=11, fontweight="semibold")
    axes[2].set_xlabel("Whitened 1")
    axes[2].set_ylabel("Whitened 2")
    axes[2].axhline(0, color="gray", lw=0.7, ls="--")
    axes[2].axvline(0, color="gray", lw=0.7, ls="--")
    axes[2].grid(True, alpha=0.2)

    # Panel 4: FastICA Separated
    axes[3].scatter(y1, y2, alpha=0.15, s=6, color="#43A047")
    axes[3].set_title("4. FastICA Recovered (y1 vs y2)\nRotated to Max Non-Gaussianity", fontsize=11, fontweight="semibold")
    axes[3].set_xlabel("Recovered Component 1")
    axes[3].set_ylabel("Recovered Component 2")
    axes[3].axhline(0, color="gray", lw=0.7, ls="--")
    axes[3].axvline(0, color="gray", lw=0.7, ls="--")
    axes[3].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_spectrogram_comparison(
    original: np.ndarray,
    mixture: np.ndarray,
    separated: np.ndarray,
    cleaned: np.ndarray,
    sample_rate: int,
    output_path: str
):
    """
    Plots time-frequency spectrograms comparing Original vs Mixed vs Separated vs Cleaned.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle("Time-Frequency Spectrogram Comparison (STFT dB Scale)", fontsize=14, fontweight="bold")

    signals = [
        ("Original Speech Source (Reference)", original, axes[0, 0]),
        ("Mixed Signal 1 (Speech + Music)", mixture, axes[0, 1]),
        ("Recovered Source 1 (FastICA)", separated, axes[1, 0]),
        ("Cleaned Recovered Source 1 (Spectral Gating)", cleaned, axes[1, 1])
    ]

    for title, sig, ax in signals:
        f, t, Sxx = scipy.signal.spectrogram(sig, fs=sample_rate, nperseg=512, noverlap=384)
        im = ax.pcolormesh(t, f, 10 * np.log10(np.maximum(Sxx, 1e-10)), shading="gouraud", cmap="inferno")
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 8000)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Power/Freq (dB/Hz)", pad=0.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_snr_vs_performance(
    snr_levels: List[float],
    correlations: List[float],
    sdrs: List[float],
    output_path: str
):
    """
    Plots separation performance (Pearson correlation & SDR) across noise levels.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "#1E88E5"
    ax1.set_xlabel("Input SNR (dB) [Higher = Less Noise]", fontsize=11)
    ax1.set_ylabel("Pearson Correlation |r|", color=color, fontsize=11)
    line1 = ax1.plot(snr_levels, correlations, marker="o", color=color, lw=2.2, label="Pearson Correlation")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#E53935"
    ax2.set_ylabel("Signal-to-Distortion Ratio (SDR in dB)", color=color2, fontsize=11)
    line2 = ax2.plot(snr_levels, sdrs, marker="s", color=color2, lw=2.2, ls="--", label="SDR (dB)")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")

    plt.title("FastICA Robustness: Separation Performance vs. Noise Level", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_convergence_curve(
    convergence_history: List[float],
    tol: float,
    output_path: str
):
    """
    Plots convergence error per iteration of FastICA fixed-point algorithm.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    iters = list(range(1, len(convergence_history) + 1))
    plt.semilogy(iters, convergence_history, marker="o", color="#43A047", lw=2)
    plt.axhline(tol, color="#E53935", ls="--", label=f"Tolerance ({tol})")
    plt.title("FastICA Fixed-Point Convergence History", fontsize=13, fontweight="bold")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\max |1 - |\mathrm{diag}(W^+ W^T)||$ (Log Scale)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
