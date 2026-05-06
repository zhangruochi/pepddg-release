#!/usr/bin/env python
"""PepDDG v9.1 strict 3-view with near-neutral rescue and ablations.

Architecture remains strict 3-view:
  physics + structural + mpnn

v9.1 keeps chemistry in the structural view (v9) and adds two bounded
subterms inside the same structural channel:
  - neutral_rank: near-neutral proxy from cross-view agreement
  - surface_rank: surface compatibility from existing rank_masif

No new channel is introduced.
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


def _fill_mutation_deltas(df: pd.DataFrame, mut_col: str = "mut") -> pd.DataFrame:
    out = df.copy()
    if "wt_aa1" in out.columns and "mut_aa1" in out.columns:
        wt = out["wt_aa1"].fillna("").astype(str).str.upper()
        mt = out["mut_aa1"].fillna("").astype(str).str.upper()
    else:
        wt, mt = _parse_mut_token(out[mut_col])

    delta_vol_fb = mt.map(AA_VOLUME).astype(float) - wt.map(AA_VOLUME).astype(float)
    delta_chg_fb = mt.map(AA_CHARGE).astype(float) - wt.map(AA_CHARGE).astype(float)

    if "delta_volume" not in out.columns:
        out["delta_volume"] = delta_vol_fb
    else:
        out["delta_volume"] = out["delta_volume"].astype(float).fillna(delta_vol_fb)

    if "delta_charge" not in out.columns:
        out["delta_charge"] = delta_chg_fb
    else:
        out["delta_charge"] = out["delta_charge"].astype(float).fillna(delta_chg_fb)
    return out


def _target_quantile(df: pd.DataFrame, col: str) -> pd.Series:
    q = df.groupby("target")[col].rank(method="average", pct=True)
    return (q - q.min()) / (q.max() - q.min() + 1e-12)


def _build_base_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    req = [
        "ddg_exp",
        "ddg_paired_xint_iface",
        "ddg_paired_bind_proxy",
        "struct_composite",
        "mpnn_neg_llr_complex",
        "mpnn_ddg_bind",
    ]
    out = out.dropna(subset=req).copy()

    out["phys_view_score"] = _rank(out["ddg_paired_xint_iface"].values) + _rank(
        out["ddg_paired_bind_proxy"].values
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
    """Near-neutral proxy: cross-view agreement (lower dispersion => more neutral)."""
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
    allow_fallback_zero: bool = False,
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
        raise ValueError("rank_masif is required for v9.1 main-cohort training.")

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


def _evaluate_objective(
    y: np.ndarray,
    pred: np.ndarray,
    nn_mask: np.ndarray,
    rho_base_all: float,
    rho_base_nn: float,
    alpha: float,
    neutral_weight: float,
    alpha_prior: float,
    alpha_reg: float,
    rho_floor_delta: float,
) -> tuple[float, float, float, float]:
    rho_all = _safe_spearman(y, pred)
    if np.isnan(rho_all):
        return -np.inf, np.nan, np.nan, np.nan
    if rho_all < rho_base_all + rho_floor_delta:
        return -np.inf, rho_all, np.nan, np.nan

    if int(nn_mask.sum()) >= 5:
        rho_nn = _safe_spearman(y[nn_mask], pred[nn_mask])
    else:
        rho_nn = np.nan
    if np.isnan(rho_base_nn) or np.isnan(rho_nn):
        delta_nn = 0.0
    else:
        delta_nn = float(rho_nn - rho_base_nn)

    objective = (
        rho_all
        + neutral_weight * delta_nn
        - alpha_reg * abs(alpha - alpha_prior)
    )
    return float(objective), float(rho_all), float(rho_nn), float(delta_nn)


def _loto_weight_trace(
    df: pd.DataFrame,
    candidates: list[tuple[float, float, float]],
    neutral_threshold: float,
    neutral_weight: float,
    alpha_prior: float,
    alpha_reg: float,
    rho_floor_delta: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    y_col = "ddg_exp"
    base_col = "rankscore_3view_base"

    for held_out in sorted(df["target"].unique()):
        train = df[df["target"] != held_out].copy()
        y = train[y_col].values.astype(float)
        base = train[base_col].values.astype(float)
        rho_base_all = _safe_spearman(y, base)
        nn_mask = (train[y_col].abs().values < neutral_threshold)
        rho_base_nn = (
            _safe_spearman(y[nn_mask], base[nn_mask]) if int(nn_mask.sum()) >= 5 else np.nan
        )

        best: dict | None = None
        for alpha, beta, gamma in candidates:
            pred, _ = _score_with_weights(train, alpha=alpha, beta=beta, gamma=gamma)
            obj, rho_all, rho_nn, delta_nn = _evaluate_objective(
                y=y,
                pred=pred,
                nn_mask=nn_mask,
                rho_base_all=rho_base_all,
                rho_base_nn=rho_base_nn,
                alpha=alpha,
                neutral_weight=neutral_weight,
                alpha_prior=alpha_prior,
                alpha_reg=alpha_reg,
                rho_floor_delta=rho_floor_delta,
            )
            if np.isneginf(obj):
                continue
            l1 = abs(alpha) + abs(beta) + abs(gamma)
            cand = {
                "best_alpha_train": float(alpha),
                "best_beta_train": float(beta),
                "best_gamma_train": float(gamma),
                "best_train_rho": float(rho_all),
                "best_train_rho_neutral": float(rho_nn),
                "best_train_delta_neutral": float(delta_nn),
                "best_objective": float(obj),
                "best_l1": float(l1),
            }
            if best is None:
                best = cand
                continue
            if obj > best["best_objective"] + 1e-12:
                best = cand
                continue
            if abs(obj - best["best_objective"]) <= 1e-12:
                if l1 < best["best_l1"] - 1e-12:
                    best = cand
                    continue
                if abs(l1 - best["best_l1"]) <= 1e-12:
                    cur_tuple = (alpha, beta, gamma)
                    best_tuple = (
                        best["best_alpha_train"],
                        best["best_beta_train"],
                        best["best_gamma_train"],
                    )
                    if cur_tuple < best_tuple:
                        best = cand

        if best is None:
            best = {
                "best_alpha_train": 0.0,
                "best_beta_train": 0.0,
                "best_gamma_train": 0.0,
                "best_train_rho": float(rho_base_all),
                "best_train_rho_neutral": float(rho_base_nn),
                "best_train_delta_neutral": 0.0,
                "best_objective": float(rho_base_all),
                "best_l1": 0.0,
            }

        rows.append({"held_out_target": held_out, **best})

    return pd.DataFrame(rows)


def _freeze_weights(loto: pd.DataFrame) -> tuple[float, float, float]:
    return (
        float(np.median(loto["best_alpha_train"].values)),
        float(np.median(loto["best_beta_train"].values)),
        float(np.median(loto["best_gamma_train"].values)),
    )


def _per_target_delta(
    df: pd.DataFrame,
    pred_col: str,
    base_col: str = "rankscore_3view_base",
) -> pd.DataFrame:
    rows: list[dict] = []
    for target, g in df.groupby("target"):
        if len(g) < 5:
            continue
        r_base = _safe_spearman(g["ddg_exp"].values, g[base_col].values)
        r_new = _safe_spearman(g["ddg_exp"].values, g[pred_col].values)
        rows.append(
            {
                "target": target,
                "n": int(len(g)),
                "rho_base": float(r_base),
                "rho_new": float(r_new),
                "delta": float(r_new - r_base),
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def _regime_breakdown(
    df: pd.DataFrame,
    pred_col: str,
    neutral_threshold: float,
    base_col: str = "rankscore_3view_base",
) -> pd.DataFrame:
    rows: list[dict] = []
    regimes = [
        ("stabilizing", df["ddg_exp"] < 0.0),
        ("near_neutral", df["ddg_exp"].abs() < neutral_threshold),
        ("destabilizing", df["ddg_exp"] > 0.0),
    ]
    for name, mask in regimes:
        sub = df[mask].copy()
        if len(sub) < 5:
            continue
        r_base = _safe_spearman(sub["ddg_exp"].values, sub[base_col].values)
        r_new = _safe_spearman(sub["ddg_exp"].values, sub[pred_col].values)
        rows.append(
            {
                "regime": name,
                "n": int(len(sub)),
                "rho_base": float(r_base),
                "rho_new": float(r_new),
                "delta": float(r_new - r_base),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_metrics(
    df: pd.DataFrame,
    pred_base: np.ndarray,
    pred_new: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    targets = np.array(sorted(df["target"].unique()))
    idx = {t: np.where(df["target"].values == t)[0] for t in targets}

    rows: list[dict] = []
    y = df["ddg_exp"].values
    for i in range(n_bootstrap):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        sample_idx = np.concatenate([idx[t] for t in sampled])
        r_base = _safe_spearman(y[sample_idx], pred_base[sample_idx])
        r_new = _safe_spearman(y[sample_idx], pred_new[sample_idx])
        if np.isnan(r_base) or np.isnan(r_new):
            continue
        rows.append(
            {
                "iter": i,
                "rho_base": float(r_base),
                "rho_new": float(r_new),
                "delta": float(r_new - r_base),
            }
        )
    return pd.DataFrame(rows)


def _variant_metrics(
    df: pd.DataFrame,
    pred_col: str,
    neutral_threshold: float,
    gate_main_rho: float,
    base_col: str = "rankscore_3view_base",
) -> dict[str, float]:
    y = df["ddg_exp"].values
    base = df[base_col].values
    pred = df[pred_col].values
    rho_base = _safe_spearman(y, base)
    rho_new = _safe_spearman(y, pred)

    mask_nn = df["ddg_exp"].abs().values < neutral_threshold
    rho_base_nn = _safe_spearman(y[mask_nn], base[mask_nn]) if int(mask_nn.sum()) >= 5 else np.nan
    rho_new_nn = _safe_spearman(y[mask_nn], pred[mask_nn]) if int(mask_nn.sum()) >= 5 else np.nan

    mask_stab = df["ddg_exp"].values < 0.0
    rho_base_stab = (
        _safe_spearman(y[mask_stab], base[mask_stab]) if int(mask_stab.sum()) >= 5 else np.nan
    )
    rho_new_stab = (
        _safe_spearman(y[mask_stab], pred[mask_stab]) if int(mask_stab.sum()) >= 5 else np.nan
    )

    return {
        "rho_base": float(rho_base),
        "rho_new": float(rho_new),
        "delta_rho": float(rho_new - rho_base),
        "rho_base_neutral": float(rho_base_nn),
        "rho_new_neutral": float(rho_new_nn),
        "delta_neutral": float(rho_new_nn - rho_base_nn),
        "rho_base_stabilizing": float(rho_base_stab),
        "rho_new_stabilizing": float(rho_new_stab),
        "delta_stabilizing": float(rho_new_stab - rho_base_stab),
        "gate_main_rho_ge_threshold": bool(rho_new >= gate_main_rho),
    }


def _parse_grid(raw: str) -> np.ndarray:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("grid is empty")
    return np.array(sorted(set(vals)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-csv",
        default="research/pepddg_v5/results/v7/ensemble_4ch_peponly.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="research/pepddg_v5/results/v9_1_strict3",
    )
    parser.add_argument(
        "--alpha-grid",
        default="0.1,0.2,0.3,0.4",
        help="Comma-separated alpha candidates for chem_rank.",
    )
    parser.add_argument(
        "--beta-grid",
        default="0.0,0.1,0.2,0.3,0.4",
        help="Comma-separated beta candidates for neutral_rank.",
    )
    parser.add_argument(
        "--gamma-grid",
        default="0.0,0.05,0.1,0.15,0.2",
        help="Comma-separated gamma candidates for surface_rank.",
    )
    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=0.5,
        help="|ddg_exp| threshold for near-neutral regime.",
    )
    parser.add_argument(
        "--objective-neutral-weight",
        type=float,
        default=0.4,
        help="Weight on near-neutral delta in LOTO objective.",
    )
    parser.add_argument(
        "--alpha-prior",
        type=float,
        default=0.25,
        help="Regularization prior for alpha.",
    )
    parser.add_argument(
        "--alpha-reg",
        type=float,
        default=0.03,
        help="Penalty coefficient for |alpha-alpha_prior| in LOTO objective.",
    )
    parser.add_argument(
        "--rho-floor-delta",
        type=float,
        default=-0.01,
        help="Discard candidates if train rho < baseline_rho + rho_floor_delta.",
    )
    parser.add_argument(
        "--gate-main-rho",
        type=float,
        default=0.635,
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_grid = _parse_grid(args.alpha_grid)
    beta_grid = _parse_grid(args.beta_grid)
    gamma_grid = _parse_grid(args.gamma_grid)

    base = pd.read_csv(args.base_csv)
    base = _fill_mutation_deltas(base, mut_col="mut")
    base = _build_base_scores(base)
    base = _build_chem_rank(base)
    base = _build_neutral_rank(base)
    base, _ = _build_surface_rank(base, allow_fallback_zero=False)

    combined_candidates = [
        (float(a), float(b), float(g))
        for a in alpha_grid
        for b in beta_grid
        for g in gamma_grid
    ]
    neutral_only_candidates = [(0.0, float(b), 0.0) for b in beta_grid]
    surface_only_candidates = [(0.0, 0.0, float(g)) for g in gamma_grid]

    loto_combined = _loto_weight_trace(
        base,
        candidates=combined_candidates,
        neutral_threshold=args.neutral_threshold,
        neutral_weight=args.objective_neutral_weight,
        alpha_prior=args.alpha_prior,
        alpha_reg=args.alpha_reg,
        rho_floor_delta=args.rho_floor_delta,
    )
    loto_neutral = _loto_weight_trace(
        base,
        candidates=neutral_only_candidates,
        neutral_threshold=args.neutral_threshold,
        neutral_weight=args.objective_neutral_weight,
        alpha_prior=args.alpha_prior,
        alpha_reg=args.alpha_reg,
        rho_floor_delta=args.rho_floor_delta,
    )
    loto_surface = _loto_weight_trace(
        base,
        candidates=surface_only_candidates,
        neutral_threshold=args.neutral_threshold,
        neutral_weight=0.0,
        alpha_prior=args.alpha_prior,
        alpha_reg=0.0,
        rho_floor_delta=args.rho_floor_delta,
    )

    alpha_c, beta_c, gamma_c = _freeze_weights(loto_combined)
    alpha_n, beta_n, gamma_n = _freeze_weights(loto_neutral)
    alpha_s, beta_s, gamma_s = _freeze_weights(loto_surface)

    pred_combined, rank_struct_combined = _score_with_weights(
        base, alpha=alpha_c, beta=beta_c, gamma=gamma_c
    )
    pred_neutral, rank_struct_neutral = _score_with_weights(
        base, alpha=alpha_n, beta=beta_n, gamma=gamma_n
    )
    pred_surface, rank_struct_surface = _score_with_weights(
        base, alpha=alpha_s, beta=beta_s, gamma=gamma_s
    )

    base["rankscore_struct_v9_1"] = rank_struct_combined
    base["rankscore_3view_v9_1"] = pred_combined
    base["rankscore_struct_v9_1_neutral_only"] = rank_struct_neutral
    base["rankscore_3view_v9_1_neutral_only"] = pred_neutral
    base["rankscore_struct_v9_1_surface_only"] = rank_struct_surface
    base["rankscore_3view_v9_1_surface_only"] = pred_surface

    metrics_combined = _variant_metrics(
        base,
        pred_col="rankscore_3view_v9_1",
        neutral_threshold=args.neutral_threshold,
        gate_main_rho=args.gate_main_rho,
    )
    metrics_neutral = _variant_metrics(
        base,
        pred_col="rankscore_3view_v9_1_neutral_only",
        neutral_threshold=args.neutral_threshold,
        gate_main_rho=args.gate_main_rho,
    )
    metrics_surface = _variant_metrics(
        base,
        pred_col="rankscore_3view_v9_1_surface_only",
        neutral_threshold=args.neutral_threshold,
        gate_main_rho=args.gate_main_rho,
    )

    per_target = _per_target_delta(base, pred_col="rankscore_3view_v9_1")
    nonneg_frac = float((per_target["delta"] >= 0.0).mean()) if len(per_target) else np.nan

    regime_rows: list[pd.DataFrame] = []
    for variant, col in [
        ("neutral_only", "rankscore_3view_v9_1_neutral_only"),
        ("surface_only", "rankscore_3view_v9_1_surface_only"),
        ("combined", "rankscore_3view_v9_1"),
    ]:
        r = _regime_breakdown(base, pred_col=col, neutral_threshold=args.neutral_threshold)
        r["variant"] = variant
        regime_rows.append(r)
    regime = pd.concat(regime_rows, axis=0, ignore_index=True)

    boot = _bootstrap_metrics(
        df=base,
        pred_base=base["rankscore_3view_base"].values,
        pred_new=base["rankscore_3view_v9_1"].values,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    ci_lo = float(np.percentile(boot["delta"], 2.5))
    ci_med = float(np.median(boot["delta"]))
    ci_hi = float(np.percentile(boot["delta"], 97.5))
    p_le0 = float((boot["delta"] <= 0.0).mean())

    ablations = pd.DataFrame(
        [
            {
                "variant": "neutral_only",
                "alpha": alpha_n,
                "beta": beta_n,
                "gamma": gamma_n,
                **metrics_neutral,
            },
            {
                "variant": "surface_only",
                "alpha": alpha_s,
                "beta": beta_s,
                "gamma": gamma_s,
                **metrics_surface,
            },
            {
                "variant": "combined",
                "alpha": alpha_c,
                "beta": beta_c,
                "gamma": gamma_c,
                **metrics_combined,
            },
        ]
    ).sort_values("rho_new", ascending=False)

    baseline_snapshot = {
        "base_csv": args.base_csv,
        "n_mutations": int(len(base)),
        "n_targets": int(base["target"].nunique()),
        "rho_3view_base": float(metrics_combined["rho_base"]),
    }

    weights_frozen = {
        "selection_rule": "median of LOTO per-fold best weights",
        "objective": {
            "neutral_weight": args.objective_neutral_weight,
            "alpha_prior": args.alpha_prior,
            "alpha_reg": args.alpha_reg,
            "rho_floor_delta": args.rho_floor_delta,
            "neutral_threshold": args.neutral_threshold,
        },
        "grids": {
            "alpha_grid": alpha_grid.tolist(),
            "beta_grid": beta_grid.tolist(),
            "gamma_grid": gamma_grid.tolist(),
        },
        "variants": {
            "neutral_only": {"alpha": alpha_n, "beta": beta_n, "gamma": gamma_n},
            "surface_only": {"alpha": alpha_s, "beta": beta_s, "gamma": gamma_s},
            "combined": {"alpha": alpha_c, "beta": beta_c, "gamma": gamma_c},
        },
    }

    main_metrics = {
        "n_mutations": int(len(base)),
        "n_targets": int(base["target"].nunique()),
        "gate_main_rho_threshold": float(args.gate_main_rho),
        "per_target_nonneg_frac_combined": nonneg_frac,
        "combined_delta_ci": [ci_lo, ci_med, ci_hi],
        "combined_delta_p_le0": p_le0,
        "variants": {
            "neutral_only": metrics_neutral,
            "surface_only": metrics_surface,
            "combined": metrics_combined,
        },
    }

    (out_dir / "baseline_snapshot.json").write_text(
        json.dumps(baseline_snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    loto_combined.to_csv(out_dir / "combined_loto_trace.csv", index=False)
    loto_neutral.to_csv(out_dir / "neutral_only_loto_trace.csv", index=False)
    loto_surface.to_csv(out_dir / "surface_only_loto_trace.csv", index=False)
    (out_dir / "weights_frozen.json").write_text(
        json.dumps(weights_frozen, indent=2, sort_keys=True), encoding="utf-8"
    )
    base.to_csv(out_dir / "cohort_3view_v9_1.csv", index=False)
    ablations.to_csv(out_dir / "ablation_main_metrics.csv", index=False)
    per_target.to_csv(out_dir / "per_target_delta_combined.csv", index=False)
    regime.to_csv(out_dir / "regime_breakdown.csv", index=False)
    boot.to_csv(out_dir / "main_bootstrap_combined.csv", index=False)
    (out_dir / "main_metrics.json").write_text(
        json.dumps(main_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    report_lines = [
        "# PepDDG v9.1 Strict-3 Main Evaluation",
        "",
        f"- N mutations: {len(base)}",
        f"- N targets: {base['target'].nunique()}",
        f"- Baseline rho: {metrics_combined['rho_base']:.6f}",
        f"- Combined rho: {metrics_combined['rho_new']:.6f}",
        f"- Combined delta: {metrics_combined['delta_rho']:+.6f}",
        f"- Combined near-neutral delta: {metrics_combined['delta_neutral']:+.6f}",
        f"- Combined delta CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]",
        f"- p(delta<=0): {p_le0:.4f}",
        f"- per-target non-negative delta fraction: {nonneg_frac:.4f}",
        "",
        "## Frozen weights",
        f"- neutral_only: alpha={alpha_n:.4f}, beta={beta_n:.4f}, gamma={gamma_n:.4f}",
        f"- surface_only: alpha={alpha_s:.4f}, beta={beta_s:.4f}, gamma={gamma_s:.4f}",
        f"- combined: alpha={alpha_c:.4f}, beta={beta_c:.4f}, gamma={gamma_c:.4f}",
        "",
        "## Main gate",
        (
            f"- Combined gate (rho>={args.gate_main_rho:.3f}): "
            f"{'PASS' if metrics_combined['gate_main_rho_ge_threshold'] else 'FAIL'}"
        ),
        "",
    ]
    (out_dir / "main_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 72)
    print("PepDDG v9.1 strict-3 main evaluation")
    print("=" * 72)
    print(f"N mutations:              {len(base)}")
    print(f"N targets:                {base['target'].nunique()}")
    print(f"baseline rho:             {metrics_combined['rho_base']:.6f}")
    print(f"combined rho:             {metrics_combined['rho_new']:.6f}")
    print(f"combined delta:           {metrics_combined['delta_rho']:+.6f}")
    print(f"combined near-neutral Δ:  {metrics_combined['delta_neutral']:+.6f}")
    print(f"combined delta CI:        [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"alpha/beta/gamma*:        ({alpha_c:.4f}, {beta_c:.4f}, {gamma_c:.4f})")
    print(
        "main gate:                "
        f"{'PASS' if metrics_combined['gate_main_rho_ge_threshold'] else 'FAIL'}"
    )
    print(f"Outputs:                  {out_dir}")


if __name__ == "__main__":
    main()
