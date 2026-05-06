#!/usr/bin/env python3
"""Per-target error analysis for PepDDG v5.

Loads cohort_3view.csv and per_target_rho_all_methods.csv, computes per-target
statistics, classifies targets by performance tier, identifies top/bottom
targets, and outputs a summary CSV.

Output: research/pepddg_v5/results/statistical_tests/per_target_error_analysis.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]  # research/pepddg_v5
COHORT_PATH = BASE / "results" / "tier2_3view" / "cohort_3view.csv"
RHO_PATH = BASE / "results" / "statistical_tests" / "per_target_rho_all_methods.csv"
OUT_PATH = BASE / "results" / "statistical_tests" / "per_target_error_analysis.csv"

# ── Load data ──────────────────────────────────────────────────────────────
cohort = pd.read_csv(COHORT_PATH)
rho_all = pd.read_csv(RHO_PATH)

print(f"Cohort: {cohort.shape[0]} mutations, {cohort['target'].nunique()} targets")
print(f"Rho table: {rho_all.shape[0]} rows, methods: {rho_all['Method'].unique().tolist()}")
print()

# ── Per-target cohort statistics ───────────────────────────────────────────
per_target = cohort.groupby("target").agg(
    n_mutations=("mutation_id", "count"),
    peptide_length=("peptide_length", "first"),
    receptor_length=("receptor_length", "first"),
    ddg_exp_mean=("ddg_exp", "mean"),
    ddg_exp_std=("ddg_exp", "std"),
    ddg_exp_range=("ddg_exp", lambda x: x.max() - x.min()),
    n_stabilizing=("ddg_exp", lambda x: (x < 0).sum()),
    n_destabilizing=("ddg_exp", lambda x: (x > 0).sum()),
    n_near_neutral=("ddg_exp", lambda x: ((x >= -0.5) & (x <= 0.5)).sum()),
).reset_index()

per_target["frac_stabilizing"] = per_target["n_stabilizing"] / per_target["n_mutations"]
per_target["frac_near_neutral"] = per_target["n_near_neutral"] / per_target["n_mutations"]

# Mutation type analysis: which chains are mutated?
chain_stats = cohort.groupby("target").agg(
    mutation_chains=("chain_id", lambda x: ",".join(sorted(x.unique()))),
    n_unique_positions=("resnum", "nunique"),
).reset_index()
per_target = per_target.merge(chain_stats, on="target")

# ── Pivot rho values per method ────────────────────────────────────────────
# Methods of interest for PepDDG
pepddg_methods = {
    "PepDDG (3-channel)": "rho_3channel",
    "Energetic only": "rho_energetic",
    "Geometric only": "rho_geometric",
    "Evolutionary only": "rho_evolutionary",
}

# Also include comparison methods
comparison_methods = {
    "StaB-ddG": "rho_stabddg",
    "DiffAffinity": "rho_diffaffinity",
    "FoldX": "rho_foldx",
    "Rosetta": "rho_rosetta",
}

all_methods = {**pepddg_methods, **comparison_methods}

for method_name, col_name in all_methods.items():
    subset = rho_all[rho_all["Method"] == method_name][["Target", "rho"]].rename(
        columns={"Target": "target", "rho": col_name}
    )
    per_target = per_target.merge(subset, on="target", how="left")

# ── Classification tiers ───────────────────────────────────────────────────
def classify_rho(rho):
    """Classify rho into performance tiers."""
    if pd.isna(rho):
        return "no_data"
    if rho > 0.7:
        return "strong"
    elif rho > 0.5:
        return "moderate"
    elif rho > 0.2:
        return "weak"
    elif rho > 0.0:
        return "very_weak"
    else:
        return "anti_correlated"

per_target["tier_3channel"] = per_target["rho_3channel"].apply(classify_rho)

# Channel agreement: how many channels are positive (rho > 0)?
def count_positive_channels(row):
    channels = ["rho_energetic", "rho_geometric", "rho_evolutionary"]
    n_pos = sum(1 for c in channels if pd.notna(row[c]) and row[c] > 0)
    n_avail = sum(1 for c in channels if pd.notna(row[c]))
    return f"{n_pos}/{n_avail}"

per_target["positive_channels"] = per_target.apply(count_positive_channels, axis=1)

# Best and worst individual channel
def best_channel(row):
    channels = {
        "energetic": row.get("rho_energetic"),
        "geometric": row.get("rho_geometric"),
        "evolutionary": row.get("rho_evolutionary"),
    }
    valid = {k: v for k, v in channels.items() if pd.notna(v)}
    if not valid:
        return "N/A"
    return max(valid, key=valid.get)

def worst_channel(row):
    channels = {
        "energetic": row.get("rho_energetic"),
        "geometric": row.get("rho_geometric"),
        "evolutionary": row.get("rho_evolutionary"),
    }
    valid = {k: v for k, v in channels.items() if pd.notna(v)}
    if not valid:
        return "N/A"
    return min(valid, key=valid.get)

per_target["best_channel"] = per_target.apply(best_channel, axis=1)
per_target["worst_channel"] = per_target.apply(worst_channel, axis=1)

# Channel spread (max - min rho across channels)
def channel_spread(row):
    vals = [row.get(c) for c in ["rho_energetic", "rho_geometric", "rho_evolutionary"]
            if pd.notna(row.get(c))]
    if len(vals) < 2:
        return np.nan
    return max(vals) - min(vals)

per_target["channel_spread"] = per_target.apply(channel_spread, axis=1)

# ── Beats comparison methods? ──────────────────────────────────────────────
for comp_col in ["rho_stabddg", "rho_diffaffinity", "rho_foldx", "rho_rosetta"]:
    comp_name = comp_col.replace("rho_", "")
    per_target[f"beats_{comp_name}"] = (
        per_target["rho_3channel"] > per_target[comp_col]
    ).map({True: "yes", False: "no"})
    # Handle NaN
    mask = per_target["rho_3channel"].isna() | per_target[comp_col].isna()
    per_target.loc[mask, f"beats_{comp_name}"] = "N/A"

# ── Sort by 3-channel rho and display ──────────────────────────────────────
per_target_sorted = per_target.sort_values("rho_3channel", ascending=False, na_position="last")

# ── Summary statistics ─────────────────────────────────────────────────────
targets_with_rho = per_target.dropna(subset=["rho_3channel"])
print("=" * 80)
print("PER-TARGET ERROR ANALYSIS SUMMARY")
print("=" * 80)
print()

# Tier distribution
tier_counts = targets_with_rho["tier_3channel"].value_counts()
print("3-Channel Performance Tier Distribution:")
for tier in ["strong", "moderate", "weak", "very_weak", "anti_correlated"]:
    n = tier_counts.get(tier, 0)
    pct = 100 * n / len(targets_with_rho)
    print(f"  {tier:20s}: {n:2d} targets ({pct:.1f}%)")
print()

# Top 5
print("TOP 5 TARGETS (highest 3-channel rho):")
top5 = targets_with_rho.nlargest(5, "rho_3channel")
for _, row in top5.iterrows():
    print(f"  {row['target']:6s}  rho={row['rho_3channel']:+.3f}  N={int(row['n_mutations']):3d}  "
          f"pep_len={int(row['peptide_length']):3d}  tier={row['tier_3channel']}")
print()

# Bottom 5
print("BOTTOM 5 TARGETS (lowest 3-channel rho):")
bot5 = targets_with_rho.nsmallest(5, "rho_3channel")
for _, row in bot5.iterrows():
    channels = f"E={row['rho_energetic']:+.3f}" if pd.notna(row['rho_energetic']) else "E=N/A"
    if pd.notna(row.get('rho_geometric')):
        channels += f" G={row['rho_geometric']:+.3f}"
    else:
        channels += " G=N/A"
    if pd.notna(row.get('rho_evolutionary')):
        channels += f" V={row['rho_evolutionary']:+.3f}"
    else:
        channels += " V=N/A"
    print(f"  {row['target']:6s}  rho={row['rho_3channel']:+.3f}  N={int(row['n_mutations']):3d}  "
          f"pep_len={int(row['peptide_length']):3d}  {channels}  tier={row['tier_3channel']}")
print()

# Anti-correlated targets
anti = targets_with_rho[targets_with_rho["rho_3channel"] < 0]
if len(anti) > 0:
    print(f"ANTI-CORRELATED TARGETS ({len(anti)}):")
    for _, row in anti.iterrows():
        print(f"  {row['target']:6s}  rho={row['rho_3channel']:+.3f}  N={int(row['n_mutations']):3d}  "
              f"frac_stab={row['frac_stabilizing']:.2f}  frac_neutral={row['frac_near_neutral']:.2f}  "
              f"ddg_range={row['ddg_exp_range']:.2f}")
    print()

# Channel disagreement analysis
print("CHANNEL DISAGREEMENT (spread > 0.5):")
high_spread = targets_with_rho[targets_with_rho["channel_spread"] > 0.5].sort_values(
    "channel_spread", ascending=False
)
for _, row in high_spread.iterrows():
    e = f"{row['rho_energetic']:+.3f}" if pd.notna(row['rho_energetic']) else "N/A"
    g = f"{row['rho_geometric']:+.3f}" if pd.notna(row['rho_geometric']) else "N/A"
    v = f"{row['rho_evolutionary']:+.3f}" if pd.notna(row['rho_evolutionary']) else "N/A"
    print(f"  {row['target']:6s}  3ch={row['rho_3channel']:+.3f}  E={e}  G={g}  V={v}  "
          f"spread={row['channel_spread']:.3f}  best={row['best_channel']}")
print()

# Correlation analysis: what predicts per-target rho?
from scipy.stats import spearmanr

print("CORRELATES OF PER-TARGET 3-CHANNEL RHO:")
predictors = [
    ("n_mutations", "N mutations"),
    ("peptide_length", "Peptide length"),
    ("receptor_length", "Receptor length"),
    ("ddg_exp_mean", "Mean DDG"),
    ("ddg_exp_std", "DDG std dev"),
    ("ddg_exp_range", "DDG range"),
    ("frac_stabilizing", "Frac stabilizing"),
    ("frac_near_neutral", "Frac near-neutral"),
    ("n_unique_positions", "N unique positions"),
]

valid = targets_with_rho.dropna(subset=["rho_3channel"])
for col, label in predictors:
    mask = valid[col].notna()
    if mask.sum() < 5:
        continue
    r, p = spearmanr(valid.loc[mask, col], valid.loc[mask, "rho_3channel"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {label:25s}: rho={r:+.3f}  p={p:.4f} {sig}")
print()

# Comparison: PepDDG vs each baseline per target
print("HEAD-TO-HEAD PER-TARGET WINS (3-channel rho > competitor rho):")
for comp in ["rho_stabddg", "rho_diffaffinity", "rho_foldx", "rho_rosetta"]:
    both = targets_with_rho.dropna(subset=[comp])
    if len(both) == 0:
        continue
    wins = (both["rho_3channel"] > both[comp]).sum()
    ties = (both["rho_3channel"] == both[comp]).sum()
    losses = (both["rho_3channel"] < both[comp]).sum()
    comp_name = comp.replace("rho_", "")
    print(f"  vs {comp_name:15s}: {wins}W / {ties}T / {losses}L  (N={len(both)} targets)")
print()

# ── Save output CSV ────────────────────────────────────────────────────────
# Select and order columns for output
output_cols = [
    "target",
    "n_mutations",
    "peptide_length",
    "receptor_length",
    "ddg_exp_mean",
    "ddg_exp_std",
    "ddg_exp_range",
    "n_stabilizing",
    "n_destabilizing",
    "n_near_neutral",
    "frac_stabilizing",
    "frac_near_neutral",
    "mutation_chains",
    "n_unique_positions",
    "rho_3channel",
    "rho_energetic",
    "rho_geometric",
    "rho_evolutionary",
    "tier_3channel",
    "positive_channels",
    "best_channel",
    "worst_channel",
    "channel_spread",
    "rho_stabddg",
    "rho_diffaffinity",
    "rho_foldx",
    "rho_rosetta",
    "beats_stabddg",
    "beats_diffaffinity",
    "beats_foldx",
    "beats_rosetta",
]

out_df = per_target_sorted[output_cols].copy()
out_df.to_csv(OUT_PATH, index=False, float_format="%.4f")
print(f"Saved: {OUT_PATH}")
print(f"  {out_df.shape[0]} targets, {out_df.shape[1]} columns")
print()

# ── Final summary table (compact) ─────────────────────────────────────────
print("=" * 120)
print(f"{'Target':6s} {'N':>3s} {'PepL':>4s} {'DDGrng':>6s} {'%stab':>5s} {'%neut':>5s} "
      f"{'rho_3ch':>7s} {'rho_E':>6s} {'rho_G':>6s} {'rho_V':>6s} {'spread':>6s} {'tier':>15s}")
print("-" * 120)
for _, row in per_target_sorted.iterrows():
    rho3 = f"{row['rho_3channel']:+.3f}" if pd.notna(row['rho_3channel']) else "  N/A"
    rhoE = f"{row['rho_energetic']:+.3f}" if pd.notna(row['rho_energetic']) else "  N/A"
    rhoG = f"{row['rho_geometric']:+.3f}" if pd.notna(row['rho_geometric']) else "  N/A"
    rhoV = f"{row['rho_evolutionary']:+.3f}" if pd.notna(row['rho_evolutionary']) else "  N/A"
    sp = f"{row['channel_spread']:.3f}" if pd.notna(row['channel_spread']) else "  N/A"
    print(f"{row['target']:6s} {int(row['n_mutations']):3d} {int(row['peptide_length']):4d} "
          f"{row['ddg_exp_range']:6.2f} {row['frac_stabilizing']:5.2f} {row['frac_near_neutral']:5.2f} "
          f"{rho3} {rhoE} {rhoG} {rhoV} {sp} {row['tier_3channel']:>15s}")
print("=" * 120)
