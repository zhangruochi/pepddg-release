#!/usr/bin/env python3
"""
PepDDG v5: Hyperparameter Sensitivity Analysis — Energetic Channel.

Analyses how the number of minimization restarts and the aggregation method
affect the pooled Spearman rho of the energetic channel (dual_blend_iface).

Key parameters varied:
  1. Number of restarts used: {1, 2, 3, 4, 5, 6, 7}
     (simulated by subsetting the first K restarts from the cached 7)
  2. Aggregation method: {median, mean, min, trimmed_mean_20, max_min_drop}
  3. DDG mode: paired (per-restart subtraction) vs unpaired (aggregate then subtract)

Per-restart JSON sidecars on NFS provide restart-level energies for all
targets (crystal and predicted structures).

Usage:
    conda activate research
    python sensitivity_energetic.py

Output:
    research/pepddg_v5/results/sensitivity/
      restart_sensitivity.csv          — full results table
      restart_sensitivity_summary.txt  — human-readable summary
      restart_sensitivity_plot.png     — visualization
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # research/pepddg_v5/scripts -> repo root

# Restart JSONs — set via env var PEPDDG_RESTARTS_ROOT (defaults assume a sibling
# `restarts/` cache laid out as <root>/{crystal,predicted}/restarts).
# These per-restart sidecars are produced by the upstream OpenMM minimization
# pipeline; the pooled energetic channel in main_eval_v19.csv is already the
# aggregate, so re-running this script is only needed for ablations.
import os
_RESTARTS_ROOT = Path(os.environ.get("PEPDDG_RESTARTS_ROOT", str(REPO_ROOT / "data" / "restarts_cache")))
NFS_CRYSTAL_RESTARTS = _RESTARTS_ROOT / "crystal" / "restarts"
NFS_PREDICTED_RESTARTS = _RESTARTS_ROOT / "predicted" / "restarts"

# Score CSVs (for cohort definition and ddg_exp)
CRYSTAL_SCORES = REPO_ROOT / "research" / "pepddg_v5" / "results" / "all_scores_crystal.csv"
PREDICTED_SCORES = REPO_ROOT / "research" / "pepddg_v5" / "results" / "all_scores_predicted.csv"

# Phase 0 locked cohort (for canonical filtering)
COHORT_CSV = REPO_ROOT / "research" / "pepddg_v5" / "results" / "phase0" / "cohort_locked.csv"

OUT_DIR = REPO_ROOT / "research" / "pepddg_v5" / "results" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Saturation targets to exclude (same as evaluate_all_modes.py)
SATURATION_TARGETS = {"1CHO_EFG_I", "1R0R_E_I", "3SGB_E_I", "1PPF_E_I"}

# Modes to evaluate (mapped to restart JSON keys)
MODE_TO_RESTART_KEY = {
    "bind_proxy": "dg_bind_kcal_mol_restarts",
    "xint_iface": "e_cross_interface_total_screened_kcal_mol_restarts",
}

# Blend formula for dual_blend_iface:
# 0.24 * xint_iface + 0.76 * (0.6 * xint_iface + 0.4 * bind_proxy)
# Simplifies to: (0.24 + 0.76*0.6) * xint_iface + (0.76*0.4) * bind_proxy
#              = 0.696 * xint_iface + 0.304 * bind_proxy
BLEND_W_XINT = 0.24 + 0.76 * 0.6  # = 0.696
BLEND_W_BIND = 0.76 * 0.4          # = 0.304


# ---------------------------------------------------------------------------
# Spearman correlation (self-contained, no scipy dependency)
# ---------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Rank with average ties."""
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j < len(sorted_a) and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = np.mean(ranks[order[i:j]])
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Spearman rho. Returns None if < 3 valid pairs."""
    if len(x) < 3:
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------
def agg_median(values: List[float]) -> float:
    return float(np.median(values))


def agg_mean(values: List[float]) -> float:
    return float(np.mean(values))


def agg_min(values: List[float]) -> float:
    return float(np.min(values))


def agg_trimmed_mean_20(values: List[float]) -> float:
    """Trimmed mean dropping 20% from each tail (or at least 1 if >= 5 values)."""
    n = len(values)
    if n < 3:
        return float(np.mean(values))
    trim = max(1, int(n * 0.2))
    sorted_v = sorted(values)
    return float(np.mean(sorted_v[trim : n - trim]))


def agg_max_min_drop(values: List[float]) -> float:
    """Mean after dropping the max and min values (robust estimator)."""
    n = len(values)
    if n <= 2:
        return float(np.mean(values))
    sorted_v = sorted(values)
    return float(np.mean(sorted_v[1:-1]))


AGGREGATION_METHODS = {
    "median": agg_median,
    "mean": agg_mean,
    "min": agg_min,
    "trimmed_mean_20": agg_trimmed_mean_20,
    "max_min_drop": agg_max_min_drop,
}

# ---------------------------------------------------------------------------
# Load restart data
# ---------------------------------------------------------------------------
def load_all_restarts(restarts_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all per-target restart JSONs into a dict keyed by target PDB ID."""
    data = {}
    for jf in sorted(restarts_dir.glob("*.json")):
        with open(jf) as f:
            jdata = json.load(f)
        target_full = jf.stem  # e.g. "1ACB_E_I"
        target_pdb = target_full.split("_")[0]
        data[target_pdb] = jdata
    return data


def get_valid_restart_values(
    arr: List, n_restarts: int
) -> List[float]:
    """Extract first n_restarts values, filtering out None/NaN/Inf."""
    result = []
    for i, v in enumerate(arr):
        if i >= n_restarts:
            break
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                result.append(fv)
        except (TypeError, ValueError):
            continue
    return result


# ---------------------------------------------------------------------------
# Compute DDG for a single mutation under given parameters
# ---------------------------------------------------------------------------
def compute_ddg_paired(
    wt_xint: List[float],
    wt_bind: List[float],
    mut_xint: List[float],
    mut_bind: List[float],
    n_restarts: int,
    agg_fn,
) -> Optional[float]:
    """Compute paired DDG for dual_blend_iface score.

    Paired: For each restart i, compute:
        ddg_xint[i] = mut_xint[i] - wt_xint[i]
        ddg_bind[i] = mut_bind[i] - wt_bind[i]
        ddg_blend[i] = W_XINT * ddg_xint[i] + W_BIND * ddg_bind[i]
    Then aggregate across restarts.
    """
    n = min(n_restarts, len(wt_xint), len(wt_bind), len(mut_xint), len(mut_bind))
    paired_blend_ddg = []
    for i in range(n):
        if i >= len(wt_xint) or i >= len(mut_xint):
            break
        if i >= len(wt_bind) or i >= len(mut_bind):
            break
        # Check all four values are valid
        vals = [wt_xint[i], wt_bind[i], mut_xint[i], mut_bind[i]]
        if any(v is None or not math.isfinite(v) for v in vals):
            continue
        ddg_xint_i = mut_xint[i] - wt_xint[i]
        ddg_bind_i = mut_bind[i] - wt_bind[i]
        ddg_blend_i = BLEND_W_XINT * ddg_xint_i + BLEND_W_BIND * ddg_bind_i
        paired_blend_ddg.append(ddg_blend_i)

    if len(paired_blend_ddg) < 1:
        return None
    return agg_fn(paired_blend_ddg)


def compute_ddg_unpaired(
    wt_xint: List[float],
    wt_bind: List[float],
    mut_xint: List[float],
    mut_bind: List[float],
    n_restarts: int,
    agg_fn,
) -> Optional[float]:
    """Compute unpaired DDG: aggregate each component separately, then subtract.

    DDG = blend(agg(mut_xint[:k]) - agg(wt_xint[:k]),
                agg(mut_bind[:k]) - agg(wt_bind[:k]))
    """
    def valid_subset(arr, k):
        result = []
        for i, v in enumerate(arr[:k]):
            if v is not None and math.isfinite(v):
                result.append(v)
        return result

    wt_x = valid_subset(wt_xint, n_restarts)
    wt_b = valid_subset(wt_bind, n_restarts)
    mut_x = valid_subset(mut_xint, n_restarts)
    mut_b = valid_subset(mut_bind, n_restarts)

    if not wt_x or not wt_b or not mut_x or not mut_b:
        return None

    dg_wt_xint = agg_fn(wt_x)
    dg_wt_bind = agg_fn(wt_b)
    dg_mut_xint = agg_fn(mut_x)
    dg_mut_bind = agg_fn(mut_b)

    ddg_xint = dg_mut_xint - dg_wt_xint
    ddg_bind = dg_mut_bind - dg_wt_bind
    return BLEND_W_XINT * ddg_xint + BLEND_W_BIND * ddg_bind


# ---------------------------------------------------------------------------
# Also compute individual component rhos (xint_iface, bind_proxy)
# ---------------------------------------------------------------------------
def compute_component_ddg_paired(
    wt_arr: List[float],
    mut_arr: List[float],
    n_restarts: int,
    agg_fn,
) -> Optional[float]:
    """Paired DDG for a single energy component."""
    n = min(n_restarts, len(wt_arr), len(mut_arr))
    paired = []
    for i in range(n):
        w, m = wt_arr[i], mut_arr[i]
        if w is not None and m is not None and math.isfinite(w) and math.isfinite(m):
            paired.append(m - w)
    if len(paired) < 1:
        return None
    return agg_fn(paired)


# ---------------------------------------------------------------------------
# Bootstrap CI for a given configuration
# ---------------------------------------------------------------------------
def target_stratified_bootstrap(
    df: pd.DataFrame,
    ddg_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Target-stratified bootstrap: returns (rho_obs, ci_lo, ci_hi)."""
    valid = df.dropna(subset=["ddg_exp", ddg_col])
    targets = valid["target"].unique()
    n_targets = len(targets)
    if n_targets < 2:
        return float("nan"), float("nan"), float("nan")

    # Pre-group
    target_exp = {}
    target_pred = {}
    for t in targets:
        mask = valid["target"].values == t
        target_exp[t] = valid.loc[mask, "ddg_exp"].values.astype(float)
        target_pred[t] = valid.loc[mask, ddg_col].values.astype(float)

    rng = np.random.RandomState(seed)
    rhos = np.empty(n_boot, dtype=float)
    n_valid = 0

    for _ in range(n_boot):
        boot_targets = rng.choice(targets, size=n_targets, replace=True)
        exp_arr = np.concatenate([target_exp[t] for t in boot_targets])
        pred_arr = np.concatenate([target_pred[t] for t in boot_targets])
        if len(exp_arr) < 3:
            continue
        r = spearman_rho(exp_arr, pred_arr)
        if r is not None:
            rhos[n_valid] = r
            n_valid += 1

    if n_valid == 0:
        return float("nan"), float("nan"), float("nan")

    rhos = rhos[:n_valid]
    return float(np.median(rhos)), float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


# ---------------------------------------------------------------------------
# Restart permutation test: does restart ORDER matter?
# ---------------------------------------------------------------------------
def restart_permutation_test(
    all_restarts: Dict[str, Dict],
    scores_df: pd.DataFrame,
    n_perm: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Permute restart indices within each mutation to test if order matters.

    If the restarts are exchangeable (order doesn't matter), shuffling should
    not systematically change rho.
    """
    rng = np.random.RandomState(seed)

    # Baseline: first-K restarts with median
    baseline_rho = _compute_rho_for_config(
        all_restarts, scores_df, n_restarts=7,
        agg_name="median", ddg_mode="paired"
    )

    perm_rhos = []
    for _ in range(n_perm):
        # Create shuffled restart data
        shuffled = {}
        for target, tdata in all_restarts.items():
            shuffled[target] = {"target": tdata.get("target"), "n_restarts": 7, "WT": {}}
            wt = tdata.get("WT", {})
            # Generate a single permutation for this target
            perm = rng.permutation(7)
            for key, arr in wt.items():
                if isinstance(arr, list) and len(arr) == 7:
                    shuffled[target]["WT"][key] = [arr[p] for p in perm]
                else:
                    shuffled[target]["WT"][key] = arr

            for mut_key, mut_data in tdata.items():
                if mut_key in ("target", "n_restarts", "WT"):
                    continue
                if not isinstance(mut_data, dict):
                    continue
                perm_mut = rng.permutation(7)
                shuffled[target][mut_key] = {}
                for key, arr in mut_data.items():
                    if isinstance(arr, list) and len(arr) == 7:
                        shuffled[target][mut_key][key] = [arr[p] for p in perm_mut]
                    else:
                        shuffled[target][mut_key][key] = arr

        r = _compute_rho_for_config(
            shuffled, scores_df, n_restarts=7,
            agg_name="median", ddg_mode="paired"
        )
        if r is not None:
            perm_rhos.append(r)

    return {
        "baseline_rho": baseline_rho,
        "perm_mean": float(np.mean(perm_rhos)) if perm_rhos else None,
        "perm_std": float(np.std(perm_rhos)) if perm_rhos else None,
        "perm_min": float(np.min(perm_rhos)) if perm_rhos else None,
        "perm_max": float(np.max(perm_rhos)) if perm_rhos else None,
        "n_perm": len(perm_rhos),
    }


# ---------------------------------------------------------------------------
# Main computation: rho for a given config
# ---------------------------------------------------------------------------
def _compute_rho_for_config(
    all_restarts: Dict[str, Dict],
    scores_df: pd.DataFrame,
    n_restarts: int,
    agg_name: str,
    ddg_mode: str,  # "paired" or "unpaired"
) -> Optional[float]:
    """Compute pooled Spearman rho for given hyperparameters."""
    agg_fn = AGGREGATION_METHODS[agg_name]

    xint_key = MODE_TO_RESTART_KEY["xint_iface"]
    bind_key = MODE_TO_RESTART_KEY["bind_proxy"]

    records = []
    for target, tdata in all_restarts.items():
        wt = tdata.get("WT", {})
        wt_xint = wt.get(xint_key, [])
        wt_bind = wt.get(bind_key, [])
        if not wt_xint or not wt_bind:
            continue

        for mut_key, mut_data in tdata.items():
            if mut_key in ("target", "n_restarts", "WT"):
                continue
            if not isinstance(mut_data, dict):
                continue

            mut_xint = mut_data.get(xint_key, [])
            mut_bind = mut_data.get(bind_key, [])
            if not mut_xint or not mut_bind:
                continue

            if ddg_mode == "paired":
                ddg = compute_ddg_paired(
                    wt_xint, wt_bind, mut_xint, mut_bind,
                    n_restarts, agg_fn
                )
            else:
                ddg = compute_ddg_unpaired(
                    wt_xint, wt_bind, mut_xint, mut_bind,
                    n_restarts, agg_fn
                )

            if ddg is not None:
                records.append({"target": target, "mut": mut_key, "ddg_pred": ddg})

    if not records:
        return None

    df_pred = pd.DataFrame(records)
    # Merge with experimental DDG
    merged = df_pred.merge(
        scores_df[["target", "mut", "ddg_exp"]],
        on=["target", "mut"],
        how="inner",
    )
    if len(merged) < 3:
        return None

    return spearman_rho(
        merged["ddg_exp"].values.astype(float),
        merged["ddg_pred"].values.astype(float),
    )


def compute_full_ddg_df(
    all_restarts: Dict[str, Dict],
    scores_df: pd.DataFrame,
    n_restarts: int,
    agg_name: str,
    ddg_mode: str,
) -> pd.DataFrame:
    """Compute DDG for all mutations, returning a DataFrame with per-target info."""
    agg_fn = AGGREGATION_METHODS[agg_name]
    xint_key = MODE_TO_RESTART_KEY["xint_iface"]
    bind_key = MODE_TO_RESTART_KEY["bind_proxy"]

    records = []
    for target, tdata in all_restarts.items():
        wt = tdata.get("WT", {})
        wt_xint = wt.get(xint_key, [])
        wt_bind = wt.get(bind_key, [])
        if not wt_xint or not wt_bind:
            continue

        for mut_key, mut_data in tdata.items():
            if mut_key in ("target", "n_restarts", "WT"):
                continue
            if not isinstance(mut_data, dict):
                continue

            mut_xint = mut_data.get(xint_key, [])
            mut_bind = mut_data.get(bind_key, [])
            if not mut_xint or not mut_bind:
                continue

            if ddg_mode == "paired":
                ddg = compute_ddg_paired(
                    wt_xint, wt_bind, mut_xint, mut_bind,
                    n_restarts, agg_fn
                )
            else:
                ddg = compute_ddg_unpaired(
                    wt_xint, wt_bind, mut_xint, mut_bind,
                    n_restarts, agg_fn
                )

            if ddg is not None:
                records.append({"target": target, "mut": mut_key, "ddg_pred": ddg})

    df_pred = pd.DataFrame(records)
    merged = df_pred.merge(
        scores_df[["target", "mut", "ddg_exp"]],
        on=["target", "mut"],
        how="inner",
    )
    return merged


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PepDDG v5: Energetic Channel Hyperparameter Sensitivity Analysis")
    print("=" * 70)

    # Load score CSVs for cohort definition (ddg_exp values)
    df_crystal = pd.read_csv(CRYSTAL_SCORES)
    df_predicted = pd.read_csv(PREDICTED_SCORES)

    # Filter: non-WT, no saturation targets, no errors
    for label, df in [("crystal", df_crystal), ("predicted", df_predicted)]:
        n_before = len(df)
        df = df[~df["is_wt"]].copy()
        df = df[~df["target"].isin(SATURATION_TARGETS)].copy()
        df = df.dropna(subset=["ddg_exp"]).copy()
        # Remove error rows
        if "error" in df.columns:
            df = df[df["error"].isna() | (df["error"] == "")].copy()
        df = df.drop_duplicates(subset=["target", "mut"]).copy()
        if label == "crystal":
            df_crystal = df
        else:
            df_predicted = df
        print(f"  {label}: {n_before} -> {len(df)} mutations "
              f"({df['target'].nunique()} targets)")

    # Load restart data
    print("\nLoading per-restart JSON data...")
    crystal_restarts = load_all_restarts(NFS_CRYSTAL_RESTARTS)
    predicted_restarts = load_all_restarts(NFS_PREDICTED_RESTARTS)
    print(f"  Crystal: {len(crystal_restarts)} targets loaded")
    print(f"  Predicted: {len(predicted_restarts)} targets loaded")

    # -----------------------------------------------------------------------
    # 1. Main sensitivity sweep
    # -----------------------------------------------------------------------
    n_restart_values = [1, 2, 3, 4, 5, 6, 7]
    agg_methods = ["median", "mean", "min", "trimmed_mean_20", "max_min_drop"]
    ddg_modes = ["paired", "unpaired"]
    structure_types = [
        ("crystal", crystal_restarts, df_crystal),
        ("predicted", predicted_restarts, df_predicted),
    ]

    results = []
    print("\nRunning sensitivity sweep...")
    total = len(n_restart_values) * len(agg_methods) * len(ddg_modes) * len(structure_types)
    count = 0

    for struct_type, restarts, scores_df in structure_types:
        for n_r in n_restart_values:
            for agg_name in agg_methods:
                for ddg_mode in ddg_modes:
                    count += 1
                    rho = _compute_rho_for_config(
                        restarts, scores_df, n_r, agg_name, ddg_mode
                    )
                    results.append({
                        "structure_type": struct_type,
                        "n_restarts": n_r,
                        "aggregation": agg_name,
                        "ddg_mode": ddg_mode,
                        "pooled_rho": rho,
                    })

                    if count % 20 == 0:
                        print(f"  [{count}/{total}] {struct_type}/{ddg_mode}/"
                              f"{agg_name}/k={n_r}: rho={rho:.4f}" if rho else
                              f"  [{count}/{total}] {struct_type}/{ddg_mode}/"
                              f"{agg_name}/k={n_r}: rho=N/A")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_DIR / "restart_sensitivity.csv", index=False)
    print(f"\nSaved full results to {OUT_DIR / 'restart_sensitivity.csv'}")

    # -----------------------------------------------------------------------
    # 2. Bootstrap CIs for the key configurations
    # -----------------------------------------------------------------------
    print("\nComputing bootstrap CIs for key configurations...")
    key_configs = [
        (7, "median", "paired"),   # baseline
        (5, "median", "paired"),
        (3, "median", "paired"),
        (1, "median", "paired"),
        (7, "mean", "paired"),
        (7, "min", "paired"),
        (7, "trimmed_mean_20", "paired"),
        (7, "median", "unpaired"),  # unpaired comparison
    ]

    bootstrap_results = []
    for struct_type, restarts, scores_df in structure_types:
        for n_r, agg_name, ddg_mode in key_configs:
            df_full = compute_full_ddg_df(restarts, scores_df, n_r, agg_name, ddg_mode)
            if len(df_full) < 10:
                continue
            rho_obs = spearman_rho(
                df_full["ddg_exp"].values, df_full["ddg_pred"].values
            )
            rho_boot, ci_lo, ci_hi = target_stratified_bootstrap(
                df_full, "ddg_pred", n_boot=2000
            )
            bootstrap_results.append({
                "structure_type": struct_type,
                "n_restarts": n_r,
                "aggregation": agg_name,
                "ddg_mode": ddg_mode,
                "rho": rho_obs,
                "bootstrap_median": rho_boot,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_mutations": len(df_full),
                "n_targets": df_full["target"].nunique(),
            })
            rho_str = f"{rho_obs:.4f}" if rho_obs else "N/A"
            ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if not math.isnan(ci_lo) else "[N/A]"
            print(f"  {struct_type} k={n_r} {agg_name} {ddg_mode}: "
                  f"rho={rho_str} 95%CI={ci_str}")

    df_boot = pd.DataFrame(bootstrap_results)
    df_boot.to_csv(OUT_DIR / "restart_sensitivity_bootstrap.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Per-component analysis (xint_iface and bind_proxy separately)
    # -----------------------------------------------------------------------
    print("\nPer-component sensitivity (xint_iface, bind_proxy, dual_blend)...")
    component_results = []

    for struct_type, restarts, scores_df in structure_types:
        for n_r in n_restart_values:
            for agg_name in ["median", "mean"]:
                agg_fn = AGGREGATION_METHODS[agg_name]
                xint_key = MODE_TO_RESTART_KEY["xint_iface"]
                bind_key = MODE_TO_RESTART_KEY["bind_proxy"]

                records_xint = []
                records_bind = []
                records_blend = []

                for target, tdata in restarts.items():
                    wt = tdata.get("WT", {})
                    wt_xint = wt.get(xint_key, [])
                    wt_bind = wt.get(bind_key, [])
                    if not wt_xint or not wt_bind:
                        continue

                    for mut_key, mut_data in tdata.items():
                        if mut_key in ("target", "n_restarts", "WT"):
                            continue
                        if not isinstance(mut_data, dict):
                            continue

                        mut_xint = mut_data.get(xint_key, [])
                        mut_bind = mut_data.get(bind_key, [])

                        # Paired DDG for each component
                        ddg_x = compute_component_ddg_paired(
                            wt_xint, mut_xint, n_r, agg_fn
                        )
                        ddg_b = compute_component_ddg_paired(
                            wt_bind, mut_bind, n_r, agg_fn
                        )
                        ddg_blend = compute_ddg_paired(
                            wt_xint, wt_bind, mut_xint, mut_bind,
                            n_r, agg_fn
                        )

                        base = {"target": target, "mut": mut_key}
                        if ddg_x is not None:
                            records_xint.append({**base, "ddg_pred": ddg_x})
                        if ddg_b is not None:
                            records_bind.append({**base, "ddg_pred": ddg_b})
                        if ddg_blend is not None:
                            records_blend.append({**base, "ddg_pred": ddg_blend})

                for comp_name, recs in [
                    ("xint_iface", records_xint),
                    ("bind_proxy", records_bind),
                    ("dual_blend_iface", records_blend),
                ]:
                    if not recs:
                        continue
                    df_c = pd.DataFrame(recs)
                    df_m = df_c.merge(
                        scores_df[["target", "mut", "ddg_exp"]],
                        on=["target", "mut"], how="inner"
                    )
                    if len(df_m) < 3:
                        continue
                    rho = spearman_rho(
                        df_m["ddg_exp"].values, df_m["ddg_pred"].values
                    )
                    component_results.append({
                        "structure_type": struct_type,
                        "component": comp_name,
                        "n_restarts": n_r,
                        "aggregation": agg_name,
                        "rho": rho,
                        "n_mutations": len(df_m),
                    })

    df_comp = pd.DataFrame(component_results)
    df_comp.to_csv(OUT_DIR / "component_sensitivity.csv", index=False)

    # -----------------------------------------------------------------------
    # 4. Restart permutation test
    # -----------------------------------------------------------------------
    print("\nRestart permutation test (does restart order matter?)...")
    for struct_type, restarts, scores_df in structure_types:
        perm_result = restart_permutation_test(
            restarts, scores_df, n_perm=200, seed=42
        )
        print(f"  {struct_type}: baseline rho={perm_result['baseline_rho']:.4f}, "
              f"perm mean={perm_result['perm_mean']:.4f} "
              f"+/- {perm_result['perm_std']:.4f}, "
              f"range=[{perm_result['perm_min']:.4f}, {perm_result['perm_max']:.4f}]")

    # -----------------------------------------------------------------------
    # 5. Per-restart variance analysis
    # -----------------------------------------------------------------------
    print("\nPer-restart variance analysis...")
    variance_records = []
    for struct_type, restarts, scores_df in structure_types:
        xint_key = MODE_TO_RESTART_KEY["xint_iface"]
        bind_key = MODE_TO_RESTART_KEY["bind_proxy"]

        for target, tdata in restarts.items():
            wt = tdata.get("WT", {})
            wt_xint = wt.get(xint_key, [])
            wt_bind = wt.get(bind_key, [])
            if len(wt_xint) != 7 or len(wt_bind) != 7:
                continue

            for mut_key, mut_data in tdata.items():
                if mut_key in ("target", "n_restarts", "WT"):
                    continue
                if not isinstance(mut_data, dict):
                    continue

                mut_xint = mut_data.get(xint_key, [])
                mut_bind = mut_data.get(bind_key, [])
                if len(mut_xint) != 7 or len(mut_bind) != 7:
                    continue

                # Compute per-restart paired DDG_blend values
                ddg_blend_per_restart = []
                for i in range(7):
                    vals = [wt_xint[i], wt_bind[i], mut_xint[i], mut_bind[i]]
                    if any(v is None or not math.isfinite(v) for v in vals):
                        continue
                    ddg_x = mut_xint[i] - wt_xint[i]
                    ddg_b = mut_bind[i] - wt_bind[i]
                    ddg_blend_per_restart.append(
                        BLEND_W_XINT * ddg_x + BLEND_W_BIND * ddg_b
                    )

                if len(ddg_blend_per_restart) >= 3:
                    variance_records.append({
                        "structure_type": struct_type,
                        "target": target,
                        "mut": mut_key,
                        "ddg_std": float(np.std(ddg_blend_per_restart)),
                        "ddg_iqr": float(
                            np.percentile(ddg_blend_per_restart, 75) -
                            np.percentile(ddg_blend_per_restart, 25)
                        ),
                        "ddg_range": float(
                            max(ddg_blend_per_restart) - min(ddg_blend_per_restart)
                        ),
                        "ddg_median": float(np.median(ddg_blend_per_restart)),
                        "ddg_cv": abs(
                            float(np.std(ddg_blend_per_restart)) /
                            float(np.mean(ddg_blend_per_restart))
                        ) if abs(np.mean(ddg_blend_per_restart)) > 1e-6 else float("nan"),
                        "n_valid_restarts": len(ddg_blend_per_restart),
                    })

    df_var = pd.DataFrame(variance_records)
    df_var.to_csv(OUT_DIR / "restart_variance.csv", index=False)

    # Summarize variance
    for st in ["crystal", "predicted"]:
        v = df_var[df_var["structure_type"] == st]
        if len(v) == 0:
            continue
        print(f"\n  {st} restart variance (N={len(v)} mutations):")
        print(f"    DDG std:   median={v['ddg_std'].median():.3f}, "
              f"mean={v['ddg_std'].mean():.3f}")
        print(f"    DDG IQR:   median={v['ddg_iqr'].median():.3f}, "
              f"mean={v['ddg_iqr'].mean():.3f}")
        print(f"    DDG range: median={v['ddg_range'].median():.3f}, "
              f"mean={v['ddg_range'].mean():.3f}")
        print(f"    DDG CV:    median={v['ddg_cv'].dropna().median():.3f}")

    # -----------------------------------------------------------------------
    # 6. Generate summary report
    # -----------------------------------------------------------------------
    _generate_summary_report(df_results, df_boot, df_comp, df_var)

    # -----------------------------------------------------------------------
    # 7. Generate plots
    # -----------------------------------------------------------------------
    _generate_plots(df_results, df_comp, df_var)

    print(f"\nAll results saved to {OUT_DIR}/")
    print("Done.")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def _generate_summary_report(
    df_results: pd.DataFrame,
    df_boot: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_var: pd.DataFrame,
):
    """Write a human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("PepDDG v5: Energetic Channel Sensitivity Analysis — Summary")
    lines.append("=" * 70)
    lines.append("")

    # Best config per structure type
    for st in ["crystal", "predicted"]:
        lines.append(f"\n--- {st.upper()} STRUCTURES ---")
        sub = df_results[df_results["structure_type"] == st].copy()
        sub = sub.dropna(subset=["pooled_rho"])

        # Best overall
        best = sub.loc[sub["pooled_rho"].idxmax()]
        lines.append(
            f"  Best config: k={int(best['n_restarts'])}, "
            f"agg={best['aggregation']}, mode={best['ddg_mode']}, "
            f"rho={best['pooled_rho']:.4f}"
        )

        # Table: n_restarts vs aggregation (paired mode, which is the default)
        lines.append(f"\n  Paired DDG rho by n_restarts x aggregation:")
        paired = sub[sub["ddg_mode"] == "paired"]
        pivot = paired.pivot_table(
            index="n_restarts", columns="aggregation",
            values="pooled_rho", aggfunc="first"
        )
        # Format nicely
        agg_order = ["median", "mean", "min", "trimmed_mean_20", "max_min_drop"]
        agg_present = [a for a in agg_order if a in pivot.columns]
        header = f"  {'k':>3}  " + "  ".join(f"{a:>16}" for a in agg_present)
        lines.append(header)
        lines.append("  " + "-" * len(header))
        for k in sorted(pivot.index):
            row_vals = []
            for a in agg_present:
                v = pivot.loc[k, a] if a in pivot.columns else None
                row_vals.append(f"{v:.4f}" if v is not None and not math.isnan(v) else "  N/A ")
            lines.append(f"  {int(k):>3}  " + "  ".join(f"{v:>16}" for v in row_vals))

        # Unpaired comparison
        lines.append(f"\n  Unpaired DDG rho by n_restarts (median only):")
        unpaired = sub[(sub["ddg_mode"] == "unpaired") & (sub["aggregation"] == "median")]
        for _, row in unpaired.iterrows():
            r = row["pooled_rho"]
            rstr = f"{r:.4f}" if r is not None and not math.isnan(r) else "N/A"
            lines.append(f"    k={int(row['n_restarts'])}: rho={rstr}")

        # Paired vs unpaired at k=7
        paired_7 = sub[
            (sub["ddg_mode"] == "paired") &
            (sub["aggregation"] == "median") &
            (sub["n_restarts"] == 7)
        ]
        unpaired_7 = sub[
            (sub["ddg_mode"] == "unpaired") &
            (sub["aggregation"] == "median") &
            (sub["n_restarts"] == 7)
        ]
        if len(paired_7) > 0 and len(unpaired_7) > 0:
            delta = paired_7.iloc[0]["pooled_rho"] - unpaired_7.iloc[0]["pooled_rho"]
            lines.append(f"\n  Paired vs Unpaired (k=7, median): "
                        f"delta rho = {delta:+.4f} "
                        f"({'paired better' if delta > 0 else 'unpaired better'})")

    # Bootstrap CIs
    lines.append("\n\n--- BOOTSTRAP 95% CONFIDENCE INTERVALS ---")
    for _, row in df_boot.iterrows():
        ci_str = f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
        lines.append(
            f"  {row['structure_type']:>9} k={int(row['n_restarts'])} "
            f"{row['aggregation']:<16} {row['ddg_mode']:<8}: "
            f"rho={row['rho']:.4f}  95%CI={ci_str}  "
            f"N={int(row['n_mutations'])}"
        )

    # Component analysis
    lines.append("\n\n--- COMPONENT ANALYSIS (paired, median) ---")
    comp_sub = df_comp[df_comp["aggregation"] == "median"]
    for st in ["crystal", "predicted"]:
        lines.append(f"\n  {st}:")
        st_sub = comp_sub[comp_sub["structure_type"] == st]
        for comp in ["xint_iface", "bind_proxy", "dual_blend_iface"]:
            c_sub = st_sub[st_sub["component"] == comp].sort_values("n_restarts")
            vals = []
            for _, row in c_sub.iterrows():
                r = row["rho"]
                rstr = f"{r:.4f}" if r is not None and not math.isnan(r) else "N/A"
                vals.append(f"k={int(row['n_restarts'])}:{rstr}")
            lines.append(f"    {comp:>22}: {', '.join(vals)}")

    # Variance summary
    lines.append("\n\n--- RESTART VARIANCE ---")
    for st in ["crystal", "predicted"]:
        v = df_var[df_var["structure_type"] == st]
        if len(v) == 0:
            continue
        lines.append(f"\n  {st} (N={len(v)} mutations):")
        lines.append(f"    Per-restart DDG std:   median={v['ddg_std'].median():.3f}, "
                    f"mean={v['ddg_std'].mean():.3f}")
        lines.append(f"    Per-restart DDG IQR:   median={v['ddg_iqr'].median():.3f}, "
                    f"mean={v['ddg_iqr'].mean():.3f}")
        lines.append(f"    Per-restart DDG range: median={v['ddg_range'].median():.3f}, "
                    f"mean={v['ddg_range'].mean():.3f}")
        lines.append(f"    Per-restart DDG CV:    median={v['ddg_cv'].dropna().median():.3f}")

        # High-variance mutations
        high_var = v.nlargest(5, "ddg_std")
        lines.append(f"    Top-5 highest variance mutations:")
        for _, row in high_var.iterrows():
            lines.append(
                f"      {row['target']}_{row['mut']}: "
                f"std={row['ddg_std']:.1f}, range={row['ddg_range']:.1f}, "
                f"median_ddg={row['ddg_median']:.1f}"
            )

    # Key takeaways
    lines.append("\n\n--- KEY TAKEAWAYS ---")

    # Determine stability across restarts
    for st in ["crystal", "predicted"]:
        sub = df_results[
            (df_results["structure_type"] == st) &
            (df_results["ddg_mode"] == "paired") &
            (df_results["aggregation"] == "median")
        ].sort_values("n_restarts")
        rhos = sub["pooled_rho"].values
        if len(rhos) >= 2:
            rho_1 = rhos[0]  # k=1
            rho_7 = rhos[-1]  # k=7
            delta = rho_7 - rho_1
            lines.append(
                f"  {st}: k=1 rho={rho_1:.4f}, k=7 rho={rho_7:.4f}, "
                f"delta={delta:+.4f} ({abs(delta/rho_7)*100:.1f}% relative)"
            )

    report = "\n".join(lines) + "\n"
    report_path = OUT_DIR / "restart_sensitivity_summary.txt"
    report_path.write_text(report)
    print(f"\nSaved summary report to {report_path}")
    print(report)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _generate_plots(
    df_results: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_var: pd.DataFrame,
):
    """Generate sensitivity analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ---- Plot 1: n_restarts vs rho (paired, all aggregations) - crystal ----
    ax = axes[0, 0]
    sub = df_results[
        (df_results["structure_type"] == "crystal") &
        (df_results["ddg_mode"] == "paired")
    ]
    for agg in ["median", "mean", "min", "trimmed_mean_20", "max_min_drop"]:
        d = sub[sub["aggregation"] == agg].sort_values("n_restarts")
        if len(d) > 0:
            ax.plot(d["n_restarts"], d["pooled_rho"], "o-", label=agg, markersize=5)
    ax.set_xlabel("Number of restarts (k)")
    ax.set_ylabel("Pooled Spearman rho")
    ax.set_title("Crystal: Paired DDG by Aggregation")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, 8))
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: n_restarts vs rho (paired, all aggregations) - predicted ----
    ax = axes[0, 1]
    sub = df_results[
        (df_results["structure_type"] == "predicted") &
        (df_results["ddg_mode"] == "paired")
    ]
    for agg in ["median", "mean", "min", "trimmed_mean_20", "max_min_drop"]:
        d = sub[sub["aggregation"] == agg].sort_values("n_restarts")
        if len(d) > 0:
            ax.plot(d["n_restarts"], d["pooled_rho"], "o-", label=agg, markersize=5)
    ax.set_xlabel("Number of restarts (k)")
    ax.set_ylabel("Pooled Spearman rho")
    ax.set_title("Predicted: Paired DDG by Aggregation")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, 8))
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Paired vs Unpaired ----
    ax = axes[0, 2]
    for st, marker in [("crystal", "o"), ("predicted", "s")]:
        for mode, ls in [("paired", "-"), ("unpaired", "--")]:
            d = df_results[
                (df_results["structure_type"] == st) &
                (df_results["ddg_mode"] == mode) &
                (df_results["aggregation"] == "median")
            ].sort_values("n_restarts")
            if len(d) > 0:
                ax.plot(d["n_restarts"], d["pooled_rho"], f"{marker}{ls}",
                       label=f"{st}/{mode}", markersize=5)
    ax.set_xlabel("Number of restarts (k)")
    ax.set_ylabel("Pooled Spearman rho")
    ax.set_title("Paired vs Unpaired (median)")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, 8))
    ax.grid(True, alpha=0.3)

    # ---- Plot 4: Component rho curves (predicted, median) ----
    ax = axes[1, 0]
    comp_sub = df_comp[
        (df_comp["structure_type"] == "predicted") &
        (df_comp["aggregation"] == "median")
    ]
    for comp in ["xint_iface", "bind_proxy", "dual_blend_iface"]:
        d = comp_sub[comp_sub["component"] == comp].sort_values("n_restarts")
        if len(d) > 0:
            ax.plot(d["n_restarts"], d["rho"], "o-", label=comp, markersize=5)
    ax.set_xlabel("Number of restarts (k)")
    ax.set_ylabel("Pooled Spearman rho")
    ax.set_title("Predicted: Component rho (paired, median)")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, 8))
    ax.grid(True, alpha=0.3)

    # ---- Plot 5: DDG variance distribution ----
    ax = axes[1, 1]
    for st in ["crystal", "predicted"]:
        v = df_var[df_var["structure_type"] == st]
        if len(v) > 0:
            ax.hist(v["ddg_std"].clip(upper=50), bins=50, alpha=0.5,
                   label=f"{st} (N={len(v)})")
    ax.set_xlabel("Per-restart DDG std (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Restart Variance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Plot 6: Heatmap of aggregation x n_restarts (predicted, paired) ----
    ax = axes[1, 2]
    sub = df_results[
        (df_results["structure_type"] == "predicted") &
        (df_results["ddg_mode"] == "paired")
    ]
    pivot = sub.pivot_table(
        index="aggregation", columns="n_restarts",
        values="pooled_rho", aggfunc="first"
    )
    agg_order = ["min", "mean", "trimmed_mean_20", "max_min_drop", "median"]
    agg_present = [a for a in agg_order if a in pivot.index]
    pivot = pivot.loc[agg_present]

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=pivot.values.min() - 0.02,
                   vmax=pivot.values.max() + 0.02)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Number of restarts (k)")
    ax.set_title("Predicted: rho Heatmap (paired)")
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                       fontsize=7, color="black")
    plt.colorbar(im, ax=ax, label="Spearman rho")

    plt.tight_layout()
    plot_path = OUT_DIR / "restart_sensitivity_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
