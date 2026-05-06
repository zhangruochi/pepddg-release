#!/usr/bin/env python3
"""
Synthetic simulation to validate Theorems 1 and 2 (cross-channel fusion gain
and same-channel saturation bound).

Generates Figure: theory_validation.pdf
  Panel (a): Cross-channel fusion — observed rho_K vs bound, varying K
  Panel (b): Same-channel saturation — Delta rho from refining one channel
  Panel (c): Varying alpha at K=3 — observed vs bound

Usage:
  conda run -n research python research/pepddg_v5/scripts/theory_validation_simulation.py
"""

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

np.random.seed(42)

# --- Publication style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'neurips', 'figures')
os.makedirs(OUTDIR, exist_ok=True)


def _load_mainline_numbers() -> dict:
    base = Path(__file__).resolve().parents[1]
    candidates = []
    manifest = base / "results" / "v19_release_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            paper_dir = str(data.get("paper_artifact_dir", "")).strip()
            official = str(data.get("official_conclusion_dir", "")).strip()
            if paper_dir:
                pd = Path(paper_dir)
                candidates.append(pd / "paper_numbers_v19.json")
                candidates.append(pd / "paper_numbers_v17_2.json")
            if official:
                od = Path(official)
                candidates.append(od / "paper_numbers_v19.json")
                candidates.append(od / "paper_numbers_v17_2.json")
        except Exception:
            pass
    candidates.extend(
        [
            base / "results" / "v19_sanitized_baseline" / "paper_numbers_v19.json",
            base / "results" / "v17_2_oodguard" / "paper_numbers_v17_2.json",
        ]
    )
    for p in candidates:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    return {}


MAINLINE_NUMBERS = _load_mainline_numbers()


def generate_correlated_ranks(N, K, rho_k_list, alpha, n_trials=200):
    """
    Generate K channels with specified individual correlations rho_k with
    a latent variable y, and pairwise inter-channel correlation ~alpha.

    Returns: list of (observed_rho_K, bound) over n_trials.
    """
    results = []
    for _ in range(n_trials):
        # Latent true scores
        y = np.random.randn(N)

        # Generate channels with desired correlation to y and inter-channel alpha
        # Use: s_k = rho_k * y + sqrt(1-rho_k^2) * (sqrt(alpha)*z_common + sqrt(1-alpha)*z_k)
        # where z_common is shared noise, z_k is channel-specific noise
        z_common = np.random.randn(N)
        channels = []
        for k in range(K):
            rho_k = rho_k_list[k] if k < len(rho_k_list) else rho_k_list[-1]
            z_k = np.random.randn(N)
            noise_part = np.sqrt(max(alpha, 0)) * z_common + np.sqrt(max(1 - alpha, 0)) * z_k
            s_k = rho_k * y + np.sqrt(max(1 - rho_k**2, 0)) * noise_part
            channels.append(s_k)

        # Compute ranks
        from scipy.stats import rankdata
        y_rank = rankdata(y)
        channel_ranks = [rankdata(ch) for ch in channels]

        # Borda score
        borda = np.sum(channel_ranks, axis=0)

        # Observed rho_K
        rho_K_obs, _ = spearmanr(borda, y_rank)

        # Individual rho_k observed
        rho_k_obs = [spearmanr(channel_ranks[k], y_rank)[0] for k in range(K)]

        # Inter-channel correlations
        alpha_obs = []
        for i in range(K):
            for j in range(i+1, K):
                a, _ = spearmanr(channel_ranks[i], channel_ranks[j])
                alpha_obs.append(a)

        # Theoretical bound (uniform alpha)
        alpha_max = max(alpha_obs) if alpha_obs else 0
        rho_bar = np.mean(rho_k_obs)
        bound = rho_bar * np.sqrt(K / (1 + (K - 1) * alpha_max))

        # Theoretical bound (heterogeneous alpha)
        denom_sq = K + 2 * sum(alpha_obs)
        bound_het = sum(rho_k_obs) / np.sqrt(denom_sq) if denom_sq > 0 else 0

        results.append({
            'rho_K_obs': rho_K_obs,
            'bound_uniform': min(bound, 1.0),
            'bound_het': min(bound_het, 1.0),
            'rho_k_obs': rho_k_obs,
            'alpha_obs': alpha_obs,
        })

    return results


def panel_a_cross_channel(ax):
    """Panel (a): Cross-channel fusion gain — rho_K vs K."""
    N = 500
    rho_k = float(MAINLINE_NUMBERS.get("rho_phys", 0.482) + MAINLINE_NUMBERS.get("rho_struct", 0.469) + MAINLINE_NUMBERS.get("rho_mpnn", 0.504)) / 3.0
    alpha = float(MAINLINE_NUMBERS.get("alpha_max", 0.410))
    K_values = [1, 2, 3, 4, 5, 6, 7, 8]

    obs_means, obs_stds = [], []
    bound_means = []

    for K in K_values:
        rho_list = [rho_k] * K
        results = generate_correlated_ranks(N, K, rho_list, alpha, n_trials=200)
        obs = [r['rho_K_obs'] for r in results]
        bounds = [r['bound_uniform'] for r in results]
        obs_means.append(np.mean(obs))
        obs_stds.append(np.std(obs))
        bound_means.append(np.mean(bounds))

    # PepDDG empirical point
    pepddg_K = 3
    pepddg_obs = float(MAINLINE_NUMBERS.get("rho_main", 0.681))
    pepddg_bound = float(MAINLINE_NUMBERS.get("bound_uniform", 0.619))

    ax.errorbar(K_values, obs_means, yerr=obs_stds, fmt='o-', color='#2C3E50',
                label='Simulated $\\rho_K$', markersize=4, capsize=3, linewidth=1.2)
    ax.plot(K_values, bound_means, 's--', color='#E74C3C', label='Bound (Thm 1)',
            markersize=4, linewidth=1.2)
    ax.plot(pepddg_K, pepddg_obs, '*', color='#27AE60', markersize=12, zorder=5,
            label=f'PepDDG (obs={pepddg_obs:.3f})')
    ax.plot(pepddg_K, pepddg_bound, 'D', color='#F39C12', markersize=6, zorder=5,
            label=f'PepDDG bound ({pepddg_bound:.3f})')

    ax.set_xlabel('Number of channels $K$')
    ax.set_ylabel('Spearman $\\rho_K$')
    ax.set_title('(a) Cross-channel fusion gain')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.3, 0.85)
    ax.grid(True, alpha=0.3)


def panel_b_saturation(ax):
    """Panel (b): Same-channel saturation — Delta rho from refining one channel."""
    N = 500
    K = 3
    alpha = float(MAINLINE_NUMBERS.get("alpha_max", 0.410))
    base_rho = float(MAINLINE_NUMBERS.get("rho_phys", 0.482) + MAINLINE_NUMBERS.get("rho_struct", 0.469) + MAINLINE_NUMBERS.get("rho_mpnn", 0.504)) / 3.0

    # Vary the refined channel's rho from baseline to oracle.
    refined_rhos = np.linspace(base_rho, 1.0, 15)

    delta_obs_means, delta_obs_stds = [], []
    delta_bound_means = []

    # Baseline: all channels at base_rho
    baseline_results = generate_correlated_ranks(N, K, [base_rho]*K, alpha, n_trials=200)
    baseline_rho = np.mean([r['rho_K_obs'] for r in baseline_results])

    for rho_refined in refined_rhos:
        rho_list = [rho_refined, base_rho, base_rho]
        results = generate_correlated_ranks(N, K, rho_list, alpha, n_trials=200)
        obs = [r['rho_K_obs'] for r in results]

        delta_obs = [o - baseline_rho for o in obs]
        delta_obs_means.append(np.mean(delta_obs))
        delta_obs_stds.append(np.std(delta_obs))

        # Theoretical upper bound on improvement (Thm 2)
        sum_rho = rho_refined + 2 * base_rho
        rho_K_approx = baseline_rho  # approximate current aggregate
        bound = (rho_refined - base_rho) * rho_K_approx / (base_rho * K)
        delta_bound_means.append(bound)

    ax.errorbar(refined_rhos, delta_obs_means, yerr=delta_obs_stds,
                fmt='o-', color='#2C3E50', label='Simulated $\\Delta\\rho$',
                markersize=4, capsize=3, linewidth=1.2)
    ax.plot(refined_rhos, delta_bound_means, 's--', color='#E74C3C',
            label='Upper bound (Thm 2)', markersize=4, linewidth=1.2)

    # Mark oracle refinement
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('Oracle\n($\\rho_{k\'}=1$)', xy=(0.98, max(delta_obs_means)*0.8),
                fontsize=7, ha='right', color='gray')

    ax.set_xlabel('Refined channel $\\rho_{k\'}$')
    ax.set_ylabel('$\\Delta\\rho_K$ (improvement)')
    ax.set_title('(b) Same-channel saturation')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(0.42, 1.02)
    ax.grid(True, alpha=0.3)


def panel_c_varying_alpha(ax):
    """Panel (c): Effect of alpha on fusion gain at K=3."""
    N = 500
    K = 3
    rho_k = float(MAINLINE_NUMBERS.get("rho_phys", 0.482) + MAINLINE_NUMBERS.get("rho_struct", 0.469) + MAINLINE_NUMBERS.get("rho_mpnn", 0.504)) / 3.0

    alpha_values = np.linspace(0.0, 0.95, 20)

    obs_means, obs_stds = [], []
    bound_values = []

    for alpha in alpha_values:
        results = generate_correlated_ranks(N, K, [rho_k]*K, alpha, n_trials=200)
        obs = [r['rho_K_obs'] for r in results]
        obs_means.append(np.mean(obs))
        obs_stds.append(np.std(obs))

        # Theoretical bound
        bound = rho_k * np.sqrt(K / (1 + (K - 1) * alpha))
        bound_values.append(min(bound, 1.0))

    # PepDDG point
    pepddg_alpha = float(MAINLINE_NUMBERS.get("alpha_max", 0.410))
    pepddg_obs = float(MAINLINE_NUMBERS.get("rho_main", 0.681))
    pepddg_bound = float(MAINLINE_NUMBERS.get("bound_uniform", 0.619))

    ax.errorbar(alpha_values, obs_means, yerr=obs_stds, fmt='o-', color='#2C3E50',
                label='Simulated $\\rho_3$', markersize=3, capsize=2, linewidth=1.2)
    ax.plot(alpha_values, bound_values, '-', color='#E74C3C', label='Bound (Thm 1)',
            linewidth=1.5)
    ax.plot(pepddg_alpha, pepddg_obs, '*', color='#27AE60', markersize=12, zorder=5,
            label=f'PepDDG ($\\alpha$={pepddg_alpha})')

    ax.axhline(y=rho_k, color='gray', linestyle=':', alpha=0.5,
               label=f'Single channel ($\\rho$={rho_k})')

    ax.set_xlabel('Inter-channel correlation $\\alpha$')
    ax.set_ylabel('Spearman $\\rho_3$')
    ax.set_title('(c) Fusion gain vs. redundancy')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.35, 0.85)
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    print("Generating panel (a): cross-channel fusion gain...")
    panel_a_cross_channel(axes[0])

    print("Generating panel (b): same-channel saturation...")
    panel_b_saturation(axes[1])

    print("Generating panel (c): varying alpha...")
    panel_c_varying_alpha(axes[2])

    plt.tight_layout()

    outpath = os.path.join(OUTDIR, 'theory_validation.pdf')
    fig.savefig(outpath, format='pdf')
    print(f"Saved: {outpath}")

    # Also save PNG for quick preview
    outpath_png = os.path.join(OUTDIR, 'theory_validation.png')
    fig.savefig(outpath_png, format='png', dpi=150)
    print(f"Saved: {outpath_png}")

    plt.close()


if __name__ == '__main__':
    main()
