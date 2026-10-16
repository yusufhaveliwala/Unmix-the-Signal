#!/usr/bin/env python3
"""
Unmix the Signal — Independent Component Analysis (ICA)
Main Pipeline Execution Script.
Executes end-to-end audio source separation, PCA whitening from scratch,
FastICA from scratch, spectral noise reduction, evaluation metrics, and plot generation.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
import numpy as np

# Ensure module imports work smoothly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.audio_loader import get_default_sources
from src.audio_mixer import create_mixing_matrix, mix_audio_signals
from src.whitening import pca_whitening, verify_whitening
from src.ica import FastICAFromScratch, separate_sources
from src.noise_reduction import spectral_noise_reduction, clean_multichannel_audio
from src.evaluation import evaluate_separation, match_sources_and_resolve_ambiguity
from src.audio_export import export_pipeline_audio_files, export_wav
from src.visualization import (
    plot_waveform_comparison,
    plot_pca_vs_ica_scatter,
    plot_spectrogram_comparison,
    plot_snr_vs_performance,
    plot_convergence_curve
)


def run_full_pipeline(
    duration_sec: float = 6.0,
    sample_rate: int = 22050,
    contrast_func: str = "tanh",
    output_dir: str = os.path.join(BASE_DIR, "outputs")
) -> Dict[str, Any]:
    """
    Executes the complete ICA audio separation pipeline and generates all outputs.
    """
    audio_out_dir = os.path.join(output_dir, "audio")
    plots_out_dir = os.path.join(output_dir, "plots")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(plots_out_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "original"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "mixtures"), exist_ok=True)

    print("=" * 75)
    print("UNMIX THE SIGNAL — INDEPENDENT COMPONENT ANALYSIS (ICA)")
    print("=" * 75)

    # 1. OBTAIN & LOAD CLEAN AUDIO SOURCES
    print("\n[Step 1/7] Obtaining & Preparing Clean Audio Sources...")
    source_tuples = get_default_sources(
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        source_dir=os.path.join(BASE_DIR, "data", "original")
    )
    # Take first 2 sources (speech and music)
    s1_name, s1_data, s1_desc = source_tuples[0]
    s2_name, s2_data, s2_desc = source_tuples[1]
    sources = np.stack([s1_data, s2_data], axis=0)  # Shape (2, N)
    print(f"  Source 1: {s1_name.upper()} ({s1_desc}) - {len(s1_data)} samples @ {sample_rate} Hz")
    print(f"  Source 2: {s2_name.upper()} ({s2_desc}) - {len(s2_data)} samples @ {sample_rate} Hz")

    # Save originals in data/original
    export_wav(os.path.join(BASE_DIR, "data", "original", "source_1.wav"), s1_data, sample_rate)
    export_wav(os.path.join(BASE_DIR, "data", "original", "source_2.wav"), s2_data, sample_rate)

    # 2. CREATE MIXING MATRIX & MIX SIGNALS
    print("\n[Step 2/7] Creating Custom Mixing Matrix and Generating Mixtures...")
    # Standard non-singular mixing matrix with distinct directional channels
    A = create_mixing_matrix(n_mixtures=2, n_sources=2, custom_matrix=np.array([[0.75, 0.35], [0.40, 0.80]]))
    print(f"  Mixing Matrix A:\n    [[{A[0,0]:.2f}, {A[0,1]:.2f}],\n     [{A[1,0]:.2f}, {A[1,1]:.2f}]]")
    print(f"  Condition Number: {np.linalg.cond(A):.3f} (well-conditioned)")

    X_centered, X_raw, mix_meta = mix_audio_signals(sources, A)
    # Save raw mixtures to data/mixtures
    export_wav(os.path.join(BASE_DIR, "data", "mixtures", "mix_1.wav"), X_raw[0], sample_rate)
    export_wav(os.path.join(BASE_DIR, "data", "mixtures", "mix_2.wav"), X_raw[1], sample_rate)

    # 3. MANUAL PCA WHITENING FROM SCRATCH
    print("\n[Step 3/7] Performing Manual PCA Whitening from Scratch (NumPy)...")
    t0_white = time.time()
    Z_whitened, V_whitening, V_dewhitening = pca_whitening(X_centered)
    white_time = time.time() - t0_white
    white_verification = verify_whitening(Z_whitened)

    print(f"  Whitening completed in {white_time*1000:.2f} ms")
    cov_z = np.array(white_verification["covariance_matrix"])
    print(f"  Whitened Covariance Matrix:\n    [[{cov_z[0,0]:.4f}, {cov_z[0,1]:.4f}],\n     [{cov_z[1,0]:.4f}, {cov_z[1,1]:.4f}]]")
    print(f"  Frobenius Error from Identity: {white_verification['frobenius_error']:.6e}")
    print(f"  Identity Covariance Verified: {white_verification['is_whitened']}")

    # 4. MANUAL FASTICA FROM SCRATCH
    print(f"\n[Step 4/7] Running FastICA from Scratch (Contrast Function: {contrast_func})...")
    t0_ica = time.time()
    ica = FastICAFromScratch(
        n_components=2,
        contrast_func=contrast_func,
        max_iter=300,
        tol=1e-5,
        random_state=42
    )
    S_est = ica.fit_transform(Z_whitened)
    ica_time = time.time() - t0_ica

    # Total unmixing matrix W_total = W * V
    W_total = ica.W_ @ V_whitening
    A_est = np.linalg.pinv(W_total)

    print(f"  FastICA Converged: {ica.converged_} in {ica.n_iter_} iterations ({ica_time*1000:.2f} ms)")
    print(f"  Final Orthogonal W in Whitened Space:\n    [[{ica.W_[0,0]:.4f}, {ica.W_[0,1]:.4f}],\n     [{ica.W_[1,0]:.4f}, {ica.W_[1,1]:.4f}]]")

    # 5. NOISE REDUCTION PIPELINE
    print("\n[Step 5/7] Executing Audio Noise Reduction & Spectral Cleaning...")
    # Clean the first mixture
    cleaned_mix_1, mix_clean_metrics = spectral_noise_reduction(X_raw[0], sample_rate)
    # Clean both separated sources
    cleaned_separated, sep_clean_metrics = clean_multichannel_audio(S_est, sample_rate)

    print(f"  Cleaned Mixture 1 SNR Improvement: +{mix_clean_metrics['snr_improvement_db']:.2f} dB")
    for idx, scm in enumerate(sep_clean_metrics):
        print(f"  Cleaned Separated Track {idx+1} SNR: +{scm['snr_improvement_db']:.2f} dB improvement")

    # 6. EVALUATION & METRICS
    print("\n[Step 6/7] Evaluating Separation Quality Against Reference Sources...")
    eval_report = evaluate_separation(
        originals=sources,
        estimates=S_est,
        whitened_cov=cov_z,
        execution_time_sec=ica_time,
        n_iterations=ica.n_iter_
    )

    for sm in eval_report["source_metrics"]:
        print(f"  -> Source {sm['source_index']} matched to Component {sm['assigned_component']}:")
        print(f"     Pearson Correlation |r|: {sm['absolute_correlation']:.4f} (sign flip: {sm['sign_flipped']})")
        print(f"     Signal-to-Distortion Ratio (SDR): {sm['sdr_db']:.2f} dB")

    print(f"  Overall Mean Correlation: {eval_report['mean_correlation']:.4f}")
    print(f"  Overall Mean SDR: {eval_report['mean_sdr_db']:.2f} dB")

    # 7. EXPORT AUDIO FILES
    print("\n[Step 7/7] Exporting Mandatory WAV Files & Generating Plots...")
    audio_paths = export_pipeline_audio_files(
        output_dir=audio_out_dir,
        original_sources=sources,
        mixed_signals=X_raw,
        cleaned_mixture=cleaned_mix_1,
        separated_sources=eval_report["matched_estimates"],
        cleaned_separated_sources=cleaned_separated,
        sample_rate=sample_rate
    )
    for fname, fpath in audio_paths.items():
        print(f"  [WAV] {fname:<30} ({os.path.getsize(fpath):,} bytes)")

    # GENERATE PLOTS
    p_wave = os.path.join(plots_out_dir, "waveforms_comparison.png")
    plot_waveform_comparison(sources, X_raw, eval_report["matched_estimates"], cleaned_separated, sample_rate, p_wave)

    p_scatter = os.path.join(plots_out_dir, "scatter_pca_vs_ica.png")
    plot_pca_vs_ica_scatter(sources, X_raw, Z_whitened, eval_report["matched_estimates"], p_scatter)

    p_spec = os.path.join(plots_out_dir, "spectrograms_comparison.png")
    plot_spectrogram_comparison(sources[0], X_raw[0], eval_report["matched_estimates"][0], cleaned_separated[0], sample_rate, p_spec)

    p_conv = os.path.join(plots_out_dir, "convergence_curve.png")
    plot_convergence_curve(ica.convergence_history_, ica.tol, p_conv)

    # RUN EXPERIMENT: SNR SWEEP
    print("\n[Experiment] Running SNR Robustness Benchmark Sweep...")
    snr_levels = [30.0, 20.0, 15.0, 10.0, 5.0, 0.0]
    snr_correlations = []
    snr_sdrs = []

    for snr_db in snr_levels:
        X_n_cent, _, _ = mix_audio_signals(sources, A, add_noise_snr_db=snr_db, noise_seed=int(snr_db * 10))
        Z_n, V_n, _ = pca_whitening(X_n_cent)
        ica_n = FastICAFromScratch(n_components=2, contrast_func=contrast_func, max_iter=250, tol=1e-5, random_state=42)
        S_n = ica_n.fit_transform(Z_n)
        rep = evaluate_separation(sources, S_n)
        snr_correlations.append(rep["mean_correlation"])
        snr_sdrs.append(rep["mean_sdr_db"])
        print(f"  Noise SNR {snr_db:4.1f} dB -> Recovered |r|: {rep['mean_correlation']:.4f}, SDR: {rep['mean_sdr_db']:.2f} dB")

    p_snr = os.path.join(plots_out_dir, "snr_vs_performance.png")
    plot_snr_vs_performance(snr_levels, snr_correlations, snr_sdrs, p_snr)

    # COMPILE COMPREHENSIVE SUMMARY JSON
    summary_data = {
        "status": "success",
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "n_samples": sources.shape[1],
        "contrast_function": contrast_func,
        "mixing_matrix": A.tolist(),
        "mixing_condition_number": float(np.linalg.cond(A)),
        "whitening": {
            "frobenius_error": white_verification["frobenius_error"],
            "covariance_matrix": white_verification["covariance_matrix"],
            "is_whitened": white_verification["is_whitened"]
        },
        "ica": {
            "converged": bool(ica.converged_),
            "n_iterations": int(ica.n_iter_),
            "execution_time_sec": float(ica_time),
            "convergence_history": ica.convergence_history_,
            "unmixing_matrix_whitened": ica.W_.tolist(),
            "total_unmixing_matrix": W_total.tolist(),
            "estimated_mixing_matrix": A_est.tolist()
        },
        "evaluation": {
            "mean_correlation": eval_report["mean_correlation"],
            "mean_sdr_db": eval_report["mean_sdr_db"],
            "sources": eval_report["source_metrics"],
            "permutation_order": eval_report["permutation_order"]
        },
        "noise_reduction": {
            "mixture_snr_improvement_db": mix_clean_metrics["snr_improvement_db"],
            "separated_tracks": sep_clean_metrics
        },
        "snr_experiment": {
            "snr_levels": snr_levels,
            "correlations": snr_correlations,
            "sdrs": snr_sdrs
        },
        "audio_files": {
            name: {
                "rel_path": f"outputs/audio/{name}",
                "size_bytes": os.path.getsize(path)
            }
            for name, path in audio_paths.items()
        },
        "plot_files": {
            "waveforms": "outputs/plots/waveforms_comparison.png",
            "scatter_pca_vs_ica": "outputs/plots/scatter_pca_vs_ica.png",
            "spectrograms": "outputs/plots/spectrograms_comparison.png",
            "convergence": "outputs/plots/convergence_curve.png",
            "snr_vs_performance": "outputs/plots/snr_vs_performance.png"
        }
    }

    # Save summary json
    summary_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[Done] Pipeline finished successfully! Summary saved to {summary_path}")
    print("=" * 75)
    return summary_data


if __name__ == "__main__":
    run_full_pipeline()
