#!/usr/bin/env python
"""
Compute stronger statistics for the short/cyclic PepDDG analysis.

Outputs:
  research/pepddg_v5/results/tier2_3view/short_cyclic_significance.csv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def pooled_rho(df: pd.DataFrame, col: str) -> float:
    return float(spearmanr(df["ddg_exp"].values, df[col].values).statistic)


def target_stratified_bootstrap_delta(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    targets = np.array(sorted(df["target"].unique()))
    by_target = {target: df[df["target"] == target] for target in targets}
    deltas = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_targets = rng.choice(targets, size=len(targets), replace=True)
        sampled = pd.concat([by_target[target] for target in sampled_targets], ignore_index=True)
        deltas[index] = pooled_rho(sampled, col_a) - pooled_rho(sampled, col_b)
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    p_one_sided = float((deltas <= 0.0).mean())
    return ci_lo, ci_hi, p_one_sided


def exact_signflip_pvalue(per_target_deltas: np.ndarray) -> tuple[float, float]:
    observed = float(np.mean(per_target_deltas))
    n_targets = len(per_target_deltas)
    null_means = np.empty(2**n_targets, dtype=float)
    for index, signs in enumerate(itertools.product([-1.0, 1.0], repeat=n_targets)):
        null_means[index] = float(np.mean(per_target_deltas * np.asarray(signs, dtype=float)))
    p_one_sided = float((null_means >= observed - 1e-12).mean())
    return observed, p_one_sided


def summarize_subset(name: str, df: pd.DataFrame, comparator: str, n_bootstrap: int) -> dict:
    rho_pepddg = pooled_rho(df, "rankscore_3view")
    rho_cmp = pooled_rho(df, comparator)
    delta = rho_pepddg - rho_cmp
    ci_lo, ci_hi, p_boot = target_stratified_bootstrap_delta(
        df, "rankscore_3view", comparator, n_bootstrap=n_bootstrap, seed=1
    )

    per_target = []
    for target, group in sorted(df.groupby("target"), key=lambda item: item[0]):
        delta_target = pooled_rho(group, "rankscore_3view") - pooled_rho(group, comparator)
        per_target.append(delta_target)
    per_target = np.asarray(per_target, dtype=float)
    mean_delta, p_signflip = exact_signflip_pvalue(per_target)
    wins = int((per_target > 0).sum())
    ties = int((np.abs(per_target) < 1e-12).sum())
    relative_gain = delta / rho_cmp if abs(rho_cmp) > 1e-12 else np.nan
    signflip_p_floor = 2.0 ** (-int(df["target"].nunique()))

    return {
        "subset": name,
        "comparator": comparator,
        "n_mutations": int(len(df)),
        "n_targets": int(df["target"].nunique()),
        "rho_pepddg_3view": rho_pepddg,
        "rho_comparator": rho_cmp,
        "delta_rho": delta,
        "delta_rho_ci_lo": ci_lo,
        "delta_rho_ci_hi": ci_hi,
        "relative_gain_vs_comparator": relative_gain,
        "bootstrap_p_one_sided": p_boot,
        "mean_per_target_delta": mean_delta,
        "median_per_target_delta": float(np.median(per_target)),
        "wins_over_targets": wins,
        "ties_over_targets": ties,
        "signflip_p_one_sided": p_signflip,
        "signflip_p_floor": signflip_p_floor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=Path("research/pepddg_v5/results/tier2_3view/cohort_3view.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("research/pepddg_v5/results/tier2_3view/short_cyclic_significance.csv"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument(
        "--min-peptide-length",
        type=int,
        default=5,
        help="Exclude targets with peptide_length < this threshold (default: 5).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.cohort_csv)
    df = df[df["peptide_length"].astype(float) >= float(args.min_peptide_length)].copy()
    short_df = df[df["peptide_length"].astype(float) < 20.0].copy()
    cyclic_targets = {"1SMF", "3EQS", "3EQY", "5XCO"}
    cyclic_df = short_df[short_df["target"].isin(cyclic_targets)].copy()

    rows = [
        summarize_subset("short_lt20", short_df, "rankscore_phys", args.n_bootstrap),
        summarize_subset("cyclic_subset", cyclic_df, "rankscore_phys", args.n_bootstrap),
    ]
    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(out_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
