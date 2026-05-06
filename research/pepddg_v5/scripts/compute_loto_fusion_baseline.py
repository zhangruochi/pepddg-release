#!/usr/bin/env python
"""
LOTO learned-fusion baseline for PepDDG-Bench (predicted structures).

This script fits a supervised Ridge regressor to fuse the three PepDDG views,
using leave-one-target-out (LOTO) cross-validation with nested alpha selection.
It is intended as a *sanity-check baseline* for whether learning continuous
fusion weights improves over training-free rank aggregation.

We use rank-transformed view features (global ranks) so the baseline operates in
the same "rank space" as PepDDG, while still learning weights from labels.

Usage:
  conda run -n research python research/pepddg_v5/scripts/compute_loto_fusion_baseline.py \
    --out-csv research/pepddg_v5/results/tier2_3view/loto_ridge_fusion.csv \
    --n-boot 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rankdata(a: np.ndarray) -> np.ndarray:
    """Rank with average ties (1..N)."""
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j < len(sorted_a) and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[order[i:j]] = float(np.mean(ranks[order[i:j]]))
        i = j
    return ranks


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge fit. Returns coefficients including intercept as last entry."""
    n, p = X.shape
    X1 = np.concatenate([X, np.ones((n, 1))], axis=1)
    I = np.eye(p + 1)
    I[-1, -1] = 0.0  # do not regularize intercept
    w = np.linalg.solve(X1.T @ X1 + alpha * I, X1.T @ y)
    return w


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return X1 @ w


def target_stratified_bootstrap_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    targets: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    uniq = np.unique(targets)
    groups: Dict[str, np.ndarray] = {t: np.where(targets == t)[0] for t in uniq}

    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[t] for t in sampled])
        r = float(spearmanr(y[idx], yhat[idx]).correlation)
        if not np.isnan(r):
            boot.append(r)
    boot = np.asarray(boot, dtype=float)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def mean_per_target_rho(
    y: np.ndarray,
    yhat: np.ndarray,
    targets: np.ndarray,
    min_muts: int = 3,
) -> float:
    rhos: List[float] = []
    for t in np.unique(targets):
        m = targets == t
        if int(m.sum()) < min_muts:
            continue
        r = float(spearmanr(y[m], yhat[m]).correlation)
        if not np.isnan(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan")


def loto_ridge_predict(
    X: np.ndarray,
    y: np.ndarray,
    targets: np.ndarray,
    alphas: List[float],
) -> np.ndarray:
    """
    Outer LOTO Ridge CV with nested inner LOTO alpha selection.

    For each held-out target:
      - Standardize features on training targets only
      - Pick alpha maximizing mean inner-LOTO Spearman on training targets
      - Fit ridge on full training set and predict on held-out target
    """
    uniq = np.unique(targets)
    yhat = np.zeros_like(y, dtype=float)

    for t in uniq:
        test = targets == t
        train = ~test

        Xtr = X[train]
        ytr = y[train]
        mu = Xtr.mean(axis=0)
        sig = Xtr.std(axis=0) + 1e-8
        Xtr_s = (Xtr - mu) / sig
        Xte_s = (X[test] - mu) / sig

        tr_targets = np.unique(targets[train])
        best_alpha = None
        best_score = -1e9

        for a in alphas:
            scores: List[float] = []
            for t2 in tr_targets:
                te2 = targets[train] == t2
                tr2 = ~te2
                if int(te2.sum()) < 3:
                    continue
                w = ridge_fit(Xtr_s[tr2], ytr[tr2], a)
                pred = ridge_predict(Xtr_s[te2], w)
                r = float(spearmanr(ytr[te2], pred).correlation)
                if not np.isnan(r):
                    scores.append(r)
            if scores:
                m = float(np.mean(scores))
                if m > best_score:
                    best_score = m
                    best_alpha = a

        if best_alpha is None:
            # Fallback: no usable inner folds (should not happen here).
            best_alpha = alphas[0]

        w = ridge_fit(Xtr_s, ytr, best_alpha)
        yhat[test] = ridge_predict(Xte_s, w)

    return yhat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-csv",
        type=str,
        default="research/pepddg_v5/results/tier2_3view/loto_ridge_fusion.csv",
        help="Output CSV path for summary metrics",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-peptide-length",
        type=int,
        default=5,
        help="Exclude targets with peptide_length < this threshold (default: 5).",
    )
    args = parser.parse_args()

    base = Path("research/pepddg_v5")
    cohort = pd.read_csv(base / "results/phase0/cohort_locked.csv")
    if "peptide_length" not in cohort.columns:
        raise ValueError("Expected `peptide_length` column in cohort_locked.csv")
    cohort = cohort[cohort["peptide_length"].astype(float) >= float(args.min_peptide_length)].copy()
    mpnn = pd.read_csv(base / "results/phase1/mpnn_features.csv")

    df = cohort.merge(
        mpnn[["target", "mut", "mpnn_neg_llr_complex", "mpnn_ddg_bind"]],
        on=["target", "mut"],
        how="left",
    )
    needed = [
        "target",
        "ddg_exp",
        "ddg_paired_dual_blend_iface",
        "struct_composite",
        "mpnn_neg_llr_complex",
        "mpnn_ddg_bind",
    ]
    df = df.dropna(subset=needed).reset_index(drop=True)

    # MPNN view score (sub-rank-sum of two MPNN components).
    df["mpnn_view_score"] = (
        rankdata(df["mpnn_neg_llr_complex"].values)
        + rankdata(df["mpnn_ddg_bind"].values)
    )

    # Rank-space view features (global ranks).
    df["r_phys"] = rankdata(df["ddg_paired_dual_blend_iface"].values)
    df["r_struct"] = rankdata(df["struct_composite"].values)
    df["r_mpnn"] = rankdata(df["mpnn_view_score"].values)

    X = df[["r_phys", "r_struct", "r_mpnn"]].values.astype(float)
    y = df["ddg_exp"].values.astype(float)
    targets = df["target"].values.astype(str)

    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    yhat = loto_ridge_predict(X, y, targets, alphas)

    rho = float(spearmanr(y, yhat).correlation)
    ci_lo, ci_hi = target_stratified_bootstrap_ci(
        y, yhat, targets, n_boot=args.n_boot, seed=args.seed
    )
    mean_pt = mean_per_target_rho(y, yhat, targets)

    out = pd.DataFrame(
        [
            {
                "method": "LOTO ridge fusion (rank features)",
                "rho": rho,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_mutations": int(len(df)),
                "n_targets": int(pd.Series(targets).nunique()),
                "mean_per_target_rho": mean_pt,
            }
        ]
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
