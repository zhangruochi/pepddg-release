#!/usr/bin/env python
"""External gate-only evaluation for PepDDG v9.1 strict-3 model.

This script applies frozen weights from v9.1 main-cohort training to BPTI and OOD
datasets. It does not tune parameters on external data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


AA_VOLUME = {
    "A": 67.0,
    "R": 148.0,
    "N": 96.0,
    "D": 91.0,
    "C": 86.0,
    "Q": 114.0,
    "E": 109.0,
    "G": 0.0,
    "H": 118.0,
    "I": 124.0,
    "L": 124.0,
    "K": 135.0,
    "M": 124.0,
    "F": 135.0,
    "P": 90.0,
    "S": 73.0,
    "T": 93.0,
    "W": 163.0,
    "Y": 141.0,
    "V": 105.0,
}

AA_CHARGE = {
    "A": 0.0,
    "R": 1.0,
    "N": 0.0,
    "D": -1.0,
    "C": 0.0,
    "Q": 0.0,
    "E": -1.0,
    "G": 0.0,
    "H": 0.0,
    "I": 0.0,
    "L": 0.0,
    "K": 1.0,
    "M": 0.0,
    "F": 0.0,
    "P": 0.0,
    "S": 0.0,
    "T": 0.0,
    "W": 0.0,
    "Y": 0.0,
    "V": 0.0,
}


def _rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average")


def _safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    rho, _ = spearmanr(y, pred)
    return float(rho) if not np.isnan(rho) else np.nan


def _parse_mut_token(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    token = series.fillna("").astype(str)
    wt = token.str[0].where(token.str.len() >= 2, "").str.upper()
    mt = token.str[-1].where(token.str.len() >= 2, "").str.upper()
    return wt, mt


def _fill_mutation_deltas(df: pd.DataFrame, mut_col: str) -> pd.DataFrame:
    out = df.copy()
    if "wt_aa1" in out.columns and "mut_aa1" in out.columns:
        wt = out["wt_aa1"].fillna("").astype(str).str.upper()
        mt = out["mut_aa1"].fillna("").astype(str).str.upper()
    else:
        wt, mt = _parse_mut_token(out[mut_col])

    out["delta_volume"] = mt.map(AA_VOLUME).astype(float) - wt.map(AA_VOLUME).astype(float)
    out["delta_charge"] = mt.map(AA_CHARGE).astype(float) - wt.map(AA_CHARGE).astype(float)
    return out


def _target_quantile(df: pd.DataFrame, col: str) -> pd.Series:
    q = df.groupby("target")[col].rank(method="average", pct=True)
    return (q - q.min()) / (q.max() - q.min() + 1e-12)


def _build_base_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    req = [
        "ddg_exp",
        "ddg_xint_iface",
        "ddg_bind_proxy",
        "struct_composite",
        "mpnn_neg_llr_complex",
        "mpnn_ddg_bind",
        "target",
    ]
    out = out.dropna(subset=req).copy()
    out["phys_view_score"] = _rank(out["ddg_xint_iface"].values) + _rank(
        out["ddg_bind_proxy"].values
    )
    out["mpnn_view_score"] = _rank(out["mpnn_neg_llr_complex"].values) + _rank(
        out["mpnn_ddg_bind"].values
    )
    out["rankscore_phys"] = _rank(out["phys_view_score"].values)
    out["rankscore_struct_base"] = _rank(out["struct_composite"].values)
    out["rankscore_mpnn"] = _rank(out["mpnn_view_score"].values)
    out["rankscore_3view_base"] = (
        out["rankscore_phys"] + out["rankscore_struct_base"] + out["rankscore_mpnn"]
    )
    return out


def _build_chem_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["burial_proxy"] = out.get("burial_proxy", 0.0)
    out["burial_proxy"] = out["burial_proxy"].fillna(0.0).astype(float)
    out["size_mag"] = np.clip(out["delta_volume"].abs().astype(float), 0.0, 120.0)
    out["charge_mag"] = np.clip(out["delta_charge"].abs().astype(float), 0.0, 2.0)
    out["charge_burial_mag"] = out["charge_mag"] * (
        1.0 + np.clip(out["burial_proxy"].values, 0.0, 1.0)
    )
    out["q_size_mag"] = _target_quantile(out, "size_mag")
    out["q_charge_mag"] = _target_quantile(out, "charge_mag")
    out["q_charge_burial_mag"] = _target_quantile(out, "charge_burial_mag")
    out["chem_term"] = (
        out["q_size_mag"] + out["q_charge_mag"] + out["q_charge_burial_mag"]
    ) / 3.0
    out["chem_rank"] = _rank(out["chem_term"].values)
    return out


def _build_neutral_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    view_mat = np.vstack(
        [
            out["rankscore_phys"].values.astype(float),
            out["rankscore_struct_base"].values.astype(float),
            out["rankscore_mpnn"].values.astype(float),
        ]
    )
    out["view_dispersion"] = np.std(view_mat, axis=0)
    out["q_view_dispersion"] = _target_quantile(out, "view_dispersion")
    out["neutral_term"] = 1.0 - out["q_view_dispersion"]
    out["neutral_rank"] = _rank(out["neutral_term"].values)
    return out


def _build_surface_rank(
    df: pd.DataFrame,
    allow_fallback_zero: bool = True,
) -> tuple[pd.DataFrame, bool]:
    out = df.copy()
    used_fallback = False

    if "rank_masif" in out.columns:
        s = out["rank_masif"].astype(float)
        if s.notna().sum() > 0:
            s = s.fillna(s.median())
            out["surface_term_raw"] = s
            out["surface_rank"] = _rank(out["surface_term_raw"].values)
            return out, used_fallback

    if not allow_fallback_zero:
        raise ValueError("rank_masif missing and fallback disabled.")
    out["surface_term_raw"] = 0.0
    out["surface_rank"] = 0.0
    used_fallback = True
    return out, used_fallback


def _score_with_weights(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    struct_aug = (
        df["rankscore_struct_base"].values.astype(float)
        + alpha * df["chem_rank"].values.astype(float)
        + beta * df["neutral_rank"].values.astype(float)
        + gamma * df["surface_rank"].values.astype(float)
    )
    rank_struct = _rank(struct_aug)
    pred = (
        df["rankscore_phys"].values.astype(float)
        + rank_struct
        + df["rankscore_mpnn"].values.astype(float)
    )
    return pred, rank_struct


def _eval_dataset(
    df: pd.DataFrame,
    weights_by_variant: dict[str, dict[str, float]],
    mut_col: str,
) -> tuple[pd.DataFrame, dict, bool]:
    out = df.copy()
    out = _fill_mutation_deltas(out, mut_col=mut_col)
    out = _build_base_scores(out)
    out = _build_chem_rank(out)
    out = _build_neutral_rank(out)
    out, used_surface_fallback = _build_surface_rank(out, allow_fallback_zero=True)

    y = out["ddg_exp"].values.astype(float)
    base_pred = out["rankscore_3view_base"].values.astype(float)
    rho_base = _safe_spearman(y, base_pred)

    variant_metrics: dict[str, dict] = {}
    for variant, w in weights_by_variant.items():
        alpha = float(w["alpha"])
        beta = float(w["beta"])
        gamma = float(w["gamma"])
        pred, rank_struct = _score_with_weights(out, alpha=alpha, beta=beta, gamma=gamma)
        out[f"rankscore_struct_{variant}"] = rank_struct
        out[f"rankscore_3view_{variant}"] = pred
        rho_new = _safe_spearman(y, pred)
        variant_metrics[variant] = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "rho_base": float(rho_base),
            "rho_new": float(rho_new),
            "delta": float(rho_new - rho_base),
        }

    metrics = {
        "n_mutations": int(len(out)),
        "n_targets": int(out["target"].nunique()),
        "used_surface_fallback_zero": bool(used_surface_fallback),
        "variants": variant_metrics,
    }
    return out, metrics, used_surface_fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="research/pepddg_v5/results/v9_1_strict3",
    )
    parser.add_argument(
        "--weights-json",
        default="research/pepddg_v5/results/v9_1_strict3/weights_frozen.json",
    )
    parser.add_argument(
        "--main-metrics-json",
        default="research/pepddg_v5/results/v9_1_strict3/main_metrics.json",
    )
    parser.add_argument(
        "--bpti-csv",
        default="research/pepddg_v5/results/independent_validation/bpti_all_features.csv",
    )
    parser.add_argument(
        "--ood-csv",
        default="research/pepddg_v5/results/ood_bindinggym/unified_ood_all_targets.csv",
    )
    parser.add_argument(
        "--gate-delta-threshold",
        type=float,
        default=-0.01,
        help="External non-regression gate threshold on rho delta.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_info = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
    main_metrics = json.loads(Path(args.main_metrics_json).read_text(encoding="utf-8"))
    variants = weights_info["variants"]
    if "combined" not in variants:
        raise ValueError("weights-json must include variants.combined")

    bpti = pd.read_csv(args.bpti_csv)
    bpti = bpti[bpti["target"] != "1CBW"].copy()
    bpti_eval, bpti_metrics, _ = _eval_dataset(bpti, weights_by_variant=variants, mut_col="mut")

    ood = pd.read_csv(args.ood_csv)
    if "is_ood" in ood.columns:
        ood = ood[ood["is_ood"] == 1].copy()
    mut_col = "mutation_token" if "mutation_token" in ood.columns else "mut"
    ood_eval, ood_metrics, _ = _eval_dataset(ood, weights_by_variant=variants, mut_col=mut_col)

    gate_main_threshold = float(main_metrics.get("gate_main_rho_threshold", 0.635))
    main_combined_rho = float(main_metrics["variants"]["combined"]["rho_new"])
    main_gate = bool(main_combined_rho >= gate_main_threshold)

    bpti_delta = float(bpti_metrics["variants"]["combined"]["delta"])
    ood_delta = float(ood_metrics["variants"]["combined"]["delta"])
    bpti_gate = bool(bpti_delta >= args.gate_delta_threshold)
    ood_gate = bool(ood_delta >= args.gate_delta_threshold)
    all_gate = bool(main_gate and bpti_gate and ood_gate)

    gate_metrics = {
        "gate_delta_threshold": float(args.gate_delta_threshold),
        "main": {
            "rho_combined": main_combined_rho,
            "gate_threshold": gate_main_threshold,
            "gate_rho_ge_threshold": main_gate,
        },
        "bpti": {
            **bpti_metrics,
            "gate_combined_delta_ge_threshold": bpti_gate,
        },
        "ood": {
            **ood_metrics,
            "gate_combined_delta_ge_threshold": ood_gate,
        },
        "gate_pass_all": all_gate,
    }

    bpti_eval.to_csv(out_dir / "bpti_eval_v9_1.csv", index=False)
    ood_eval.to_csv(out_dir / "ood_eval_v9_1.csv", index=False)
    (out_dir / "external_gate_metrics.json").write_text(
        json.dumps(gate_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    report_lines = [
        "# PepDDG v9.1 External Gate Report",
        "",
        f"- gate delta threshold: {args.gate_delta_threshold:+.3f}",
        "",
        "## Main",
        f"- combined rho: {main_combined_rho:.6f}",
        f"- gate rho>={gate_main_threshold:.3f}: {'PASS' if main_gate else 'FAIL'}",
        "",
        "## BPTI",
        f"- N: {bpti_metrics['n_mutations']} (targets={bpti_metrics['n_targets']})",
        f"- surface fallback used: {bpti_metrics['used_surface_fallback_zero']}",
    ]
    for variant, metric in bpti_metrics["variants"].items():
        report_lines.extend(
            [
                f"- {variant}: rho_base={metric['rho_base']:.6f}, "
                f"rho_new={metric['rho_new']:.6f}, delta={metric['delta']:+.6f}",
            ]
        )
    report_lines.extend(
        [
            f"- combined gate: {'PASS' if bpti_gate else 'FAIL'}",
            "",
            "## OOD",
            f"- N: {ood_metrics['n_mutations']} (targets={ood_metrics['n_targets']})",
            f"- surface fallback used: {ood_metrics['used_surface_fallback_zero']}",
        ]
    )
    for variant, metric in ood_metrics["variants"].items():
        report_lines.extend(
            [
                f"- {variant}: rho_base={metric['rho_base']:.6f}, "
                f"rho_new={metric['rho_new']:.6f}, delta={metric['delta']:+.6f}",
            ]
        )
    report_lines.extend(
        [
            f"- combined gate: {'PASS' if ood_gate else 'FAIL'}",
            "",
            f"## Final: {'PASS' if all_gate else 'FAIL'}",
        ]
    )
    (out_dir / "final_gate_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 72)
    print("PepDDG v9.1 external gate evaluation")
    print("=" * 72)
    print(f"main combined rho: {main_combined_rho:.6f}")
    print(f"bpti combined Δ:   {bpti_delta:+.6f}")
    print(f"ood combined Δ:    {ood_delta:+.6f}")
    print(f"Final gate:        {'PASS' if all_gate else 'FAIL'}")
    print(f"Outputs:           {out_dir}")


if __name__ == "__main__":
    main()
