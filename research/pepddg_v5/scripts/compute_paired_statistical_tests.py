#!/usr/bin/env python
"""
A5: Paired statistical tests for all method comparisons in PepDDG Table 1.

For every method pair:
1. Target-stratified bootstrap paired Delta-rho test
2. Wilcoxon signed-rank on per-target rho vectors
3. Effect size (rank-biserial correlation)
4. Holm-Bonferroni correction for multiple comparisons

Output: CSV with all pairwise comparisons + significance summary.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


def compute_per_target_rho(df, score_col, target_col='target', ddg_col='ddg_exp', min_n=4):
    """Compute per-target Spearman rho for a given score column."""
    results = {}
    for target, grp in df.groupby(target_col):
        if len(grp) < min_n:
            continue
        if grp[score_col].nunique() < 2 or grp[ddg_col].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(grp[score_col], grp[ddg_col])
        results[target] = rho
    return results


def pooled_spearman(df, score_col, ddg_col='ddg_exp'):
    """Compute pooled Spearman rho across all mutations."""
    valid = df[[score_col, ddg_col]].dropna()
    if len(valid) < 3:
        return np.nan
    rho, _ = stats.spearmanr(valid[score_col], valid[ddg_col])
    return rho


def bootstrap_paired_delta_rho(df, score_col_a, score_col_b, target_col='target',
                                ddg_col='ddg_exp', n_boot=5000, seed=42):
    """Target-stratified bootstrap of paired Delta-rho.

    Returns: delta_rho, ci_lo, ci_hi, p_value (two-sided)
    """
    rng = np.random.RandomState(seed)
    targets = df[target_col].unique()
    n_targets = len(targets)

    # Observed delta
    rho_a = pooled_spearman(df, score_col_a, ddg_col)
    rho_b = pooled_spearman(df, score_col_b, ddg_col)
    delta_obs = rho_a - rho_b

    # Pre-group by target for fast resampling
    target_groups = {t: grp[[score_col_a, score_col_b, ddg_col]].values
                     for t, grp in df.groupby(target_col)}
    target_list = list(target_groups.keys())
    target_arrays = [target_groups[t] for t in target_list]

    # Bootstrap
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        boot_idx = rng.randint(0, n_targets, size=n_targets)
        arrays = [target_arrays[i] for i in boot_idx]
        boot_data = np.vstack(arrays)
        rho_a_boot, _ = stats.spearmanr(boot_data[:, 0], boot_data[:, 2])
        rho_b_boot, _ = stats.spearmanr(boot_data[:, 1], boot_data[:, 2])
        deltas[b] = rho_a_boot - rho_b_boot

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided p-value: fraction of bootstrap samples where delta has opposite sign
    p_value = np.mean(deltas * np.sign(delta_obs) <= 0) * 2
    p_value = min(p_value, 1.0)

    return delta_obs, ci_lo, ci_hi, p_value


def wilcoxon_signed_rank_test(rhos_a, rhos_b):
    """Wilcoxon signed-rank test on per-target rho vectors."""
    common = set(rhos_a.keys()) & set(rhos_b.keys())
    if len(common) < 5:
        return np.nan, len(common)

    diffs = [rhos_a[t] - rhos_b[t] for t in common]
    diffs = np.array(diffs)
    # Remove zeros
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 5:
        return np.nan, len(common)

    stat, p_value = stats.wilcoxon(nonzero, alternative='two-sided')
    return p_value, len(common)


def rank_biserial(rhos_a, rhos_b):
    """Rank-biserial correlation (effect size for Wilcoxon test)."""
    common = set(rhos_a.keys()) & set(rhos_b.keys())
    if len(common) < 3:
        return np.nan

    diffs = np.array([rhos_a[t] - rhos_b[t] for t in common])
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n = n_pos + n_neg
    if n == 0:
        return 0.0
    return (n_pos - n_neg) / n


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    corrected = np.ones(n)

    for rank, idx in enumerate(sorted_idx):
        corrected[idx] = min(p_values[idx] * (n - rank), 1.0)

    # Ensure monotonicity
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])

    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-csv', default='research/pepddg_v5/results/tier2_3view/cohort_3view.csv')
    parser.add_argument('--baselines-dir', default='research/pepddg_v5/results/baselines')
    parser.add_argument('--foldx-csv', default='research/pepddg_v5/results/foldx/foldx_predicted_features.csv')
    parser.add_argument('--rosetta-csv', default='research/pepddg_v5/results/rosetta/cartddg_predicted_final.csv')
    parser.add_argument('--esmif-csv', default='research/pepddg_v5/results/multifeature/esmif_features.csv')
    parser.add_argument('--esm3-csv', default='research/pepddg_v5/results/multifeature/esm3_features.csv')
    parser.add_argument('--esm2-csv', default='research/pepddg_v5/results/multifeature/esm2_features.csv')
    parser.add_argument('--output-dir', default='research/pepddg_v5/results/statistical_tests')
    parser.add_argument('--n-boot', type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load PepDDG cohort
    cohort = pd.read_csv(args.cohort_csv)
    print(f"Loaded cohort: {len(cohort)} mutations, {cohort['target'].nunique()} targets")

    # Methods and their score columns in the cohort
    # PepDDG channels are already in cohort
    methods_in_cohort = {
        'PepDDG (3-channel)': 'rankscore_3view',
        'Energetic only': 'rankscore_phys',
        'Geometric only': 'rankscore_struct',
        'Evolutionary only': 'rankscore_mpnn',
    }

    # Load external baselines and merge
    baselines_dir = Path(args.baselines_dir)

    # StaB-ddG
    stabddg = pd.read_csv(baselines_dir / 'stabddg_full_skempi.csv')
    stabddg = (
        stabddg.groupby(["target", "mut"], as_index=False)
        .agg(ddG_pred=("ddG_pred", "mean"))
        .reset_index(drop=True)
    )
    cohort = cohort.merge(
        stabddg[['target', 'mut', 'ddG_pred']].rename(columns={'ddG_pred': 'stabddg_pred'}),
        on=['target', 'mut'], how='left'
    )
    methods_in_cohort['StaB-ddG'] = 'stabddg_pred'

    # DiffAffinity
    diffaff = pd.read_csv(baselines_dir / 'diffaffinity_full_skempi.csv')
    diffaff = (
        diffaff.groupby(["target", "mut"], as_index=False)
        .agg(ddG_pred=("ddG_pred", "mean"))
        .reset_index(drop=True)
    )
    cohort = cohort.merge(
        diffaff[['target', 'mut', 'ddG_pred']].rename(columns={'ddG_pred': 'diffaff_pred'}),
        on=['target', 'mut'], how='left'
    )
    methods_in_cohort['DiffAffinity'] = 'diffaff_pred'

    # ESM-IF1
    try:
        esmif = pd.read_csv(args.esmif_csv)
        esmif = (
            esmif.groupby(["target", "mut"], as_index=False)
            .agg(esmif_neg_llr=("esmif_neg_llr", "mean"))
            .reset_index(drop=True)
        )
        cohort = cohort.merge(
            esmif[['target', 'mut', 'esmif_neg_llr']].rename(columns={'esmif_neg_llr': 'esmif_pred'}),
            on=['target', 'mut'], how='left'
        )
        methods_in_cohort['ESM-IF1'] = 'esmif_pred'
    except Exception as e:
        print(f"Warning: Could not load ESM-IF1: {e}")

    # ESM3 (both seq-only and seq+struct baselines)
    try:
        esm3 = pd.read_csv(args.esm3_csv)
        esm3 = (
            esm3.groupby(["target", "mut"], as_index=False)
            .agg(
                esm3_struct_neg_llr=("esm3_struct_neg_llr", "mean"),
                esm3_seq_neg_llr=("esm3_seq_neg_llr", "mean"),
            )
            .reset_index(drop=True)
        )
        cohort = cohort.merge(
            esm3[['target', 'mut', 'esm3_struct_neg_llr', 'esm3_seq_neg_llr']].rename(
                columns={
                    'esm3_struct_neg_llr': 'esm3_struct_pred',
                    'esm3_seq_neg_llr': 'esm3_seq_pred',
                }
            ),
            on=['target', 'mut'],
            how='left',
        )
        methods_in_cohort['ESM3 (seq+struct)'] = 'esm3_struct_pred'
        methods_in_cohort['ESM3 (seq-only)'] = 'esm3_seq_pred'
    except Exception as e:
        print(f"Warning: Could not load ESM3: {e}")

    # ESM-2 (deduplicate by (target,mut) to avoid accidental weighting)
    try:
        esm2 = pd.read_csv(args.esm2_csv)
        esm2 = (
            esm2.groupby(['target', 'mut'], as_index=False)
            .agg(esm2_pred=('esm2_neg_llr', 'mean'))
            .reset_index(drop=True)
        )
        cohort = cohort.merge(esm2, on=['target', 'mut'], how='left')
        methods_in_cohort['ESM-2 (650M)'] = 'esm2_pred'
    except Exception as e:
        print(f"Warning: Could not load ESM-2: {e}")

    # FoldX
    try:
        foldx = pd.read_csv(args.foldx_csv)
        foldx = (
            foldx.groupby(["target", "mut"], as_index=False)
            .agg(ddg_foldx=("ddg_foldx", "mean"))
            .reset_index(drop=True)
        )
        cohort = cohort.merge(
            foldx[['target', 'mut', 'ddg_foldx']].rename(columns={'ddg_foldx': 'foldx_pred'}),
            on=['target', 'mut'], how='left'
        )
        methods_in_cohort['FoldX BuildModel'] = 'foldx_pred'
        print(f"  FoldX: {foldx['ddg_foldx'].notna().sum()} predictions loaded")
    except Exception as e:
        print(f"Warning: Could not load FoldX: {e}")

    # Rosetta
    try:
        rosetta = pd.read_csv(args.rosetta_csv)
        rosetta = (
            rosetta.groupby(["target", "mut"], as_index=False)
            .agg(ddg_interface=("ddg_interface", "mean"))
            .reset_index(drop=True)
        )
        cohort = cohort.merge(
            rosetta[['target', 'mut', 'ddg_interface']].rename(columns={'ddg_interface': 'rosetta_pred'}),
            on=['target', 'mut'], how='left'
        )
        methods_in_cohort['Rosetta cartesian_ddg'] = 'rosetta_pred'
        print(f"  Rosetta: {rosetta['ddg_interface'].notna().sum()} predictions loaded")
    except Exception as e:
        print(f"Warning: Could not load Rosetta: {e}")

    # Compute per-target rho for each method
    per_target_rhos = {}
    pooled_rhos = {}
    for method_name, score_col in methods_in_cohort.items():
        valid = cohort.dropna(subset=[score_col, 'ddg_exp'])
        per_target_rhos[method_name] = compute_per_target_rho(valid, score_col)
        pooled_rhos[method_name] = pooled_spearman(valid, score_col)
        n_muts = valid[score_col].notna().sum()
        n_targets = len(per_target_rhos[method_name])
        print(f"  {method_name}: pooled rho={pooled_rhos[method_name]:.3f}, "
              f"N={n_muts}, targets_with_rho={n_targets}")

    # Pairwise comparisons: PepDDG vs each baseline
    method_names = list(methods_in_cohort.keys())
    reference = 'PepDDG (3-channel)'

    results = []
    all_p_values = []

    for method_b in method_names:
        if method_b == reference:
            continue

        score_a = methods_in_cohort[reference]
        score_b = methods_in_cohort[method_b]

        # Use common subset
        valid = cohort.dropna(subset=[score_a, score_b, 'ddg_exp'])

        # 1. Bootstrap paired delta-rho
        delta, ci_lo, ci_hi, p_boot = bootstrap_paired_delta_rho(
            valid, score_a, score_b, n_boot=args.n_boot
        )

        # 2. Wilcoxon signed-rank
        rhos_a = compute_per_target_rho(valid, score_a)
        rhos_b = compute_per_target_rho(valid, score_b)
        p_wilcox, n_common = wilcoxon_signed_rank_test(rhos_a, rhos_b)

        # 3. Effect size
        r_rb = rank_biserial(rhos_a, rhos_b)

        results.append({
            'Method A': reference,
            'Method B': method_b,
            'N_mutations': len(valid),
            'N_targets_common': n_common,
            'rho_A': pooled_spearman(valid, score_a),
            'rho_B': pooled_spearman(valid, score_b),
            'Delta_rho': delta,
            'Delta_CI_lo': ci_lo,
            'Delta_CI_hi': ci_hi,
            'p_bootstrap': p_boot,
            'p_wilcoxon': p_wilcox,
            'rank_biserial': r_rb,
        })
        all_p_values.append(p_boot)

    # 4. Holm-Bonferroni correction
    results_df = pd.DataFrame(results)
    p_boot_all = results_df['p_bootstrap'].values
    p_corrected = holm_bonferroni(p_boot_all)
    results_df['p_holm_bonferroni'] = p_corrected

    # Add significance stars
    def sig_stars(p):
        if pd.isna(p):
            return ''
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    results_df['significance'] = results_df['p_holm_bonferroni'].apply(sig_stars)

    # Save full results
    results_df.to_csv(out_dir / 'pairwise_statistical_tests.csv', index=False, float_format='%.4f')
    print(f"\nSaved to {out_dir / 'pairwise_statistical_tests.csv'}")

    # Print summary
    print("\n" + "=" * 100)
    print("PAIRWISE STATISTICAL TESTS: PepDDG vs Baselines")
    print("=" * 100)
    pepddg_rows = results_df[results_df['Method A'] == reference].copy()
    for _, row in pepddg_rows.iterrows():
        print(f"\n  PepDDG vs {row['Method B']}:")
        print(f"    Delta rho = {row['Delta_rho']:+.3f} [{row['Delta_CI_lo']:+.3f}, {row['Delta_CI_hi']:+.3f}]")
        print(f"    Bootstrap p = {row['p_bootstrap']:.4f}")
        print(f"    Wilcoxon p = {row['p_wilcoxon']:.4f}" if not pd.isna(row['p_wilcoxon']) else "    Wilcoxon: N/A")
        print(f"    Effect size (rank-biserial) = {row['rank_biserial']:+.3f}")
        print(f"    Holm-Bonferroni p = {row['p_holm_bonferroni']:.4f} {row['significance']}")

    # Save per-target rho table for appendix
    pt_records = []
    for method_name, rhos in per_target_rhos.items():
        for target, rho in rhos.items():
            pt_records.append({'Method': method_name, 'Target': target, 'rho': rho})
    pt_df = pd.DataFrame(pt_records)
    pt_df.to_csv(out_dir / 'per_target_rho_all_methods.csv', index=False, float_format='%.4f')
    print(f"\nSaved per-target rho to {out_dir / 'per_target_rho_all_methods.csv'}")

    # Generate LaTeX snippet for significance stars in Table 1
    print("\n" + "=" * 100)
    print("LaTeX snippets for Table 1 significance annotations")
    print("=" * 100)
    for _, row in pepddg_rows.iterrows():
        method = row['Method B']
        sig = row['significance']
        p_hb = row['p_holm_bonferroni']
        print(f"  {method}: p_HB={p_hb:.3f} → {sig}")


if __name__ == '__main__':
    main()
