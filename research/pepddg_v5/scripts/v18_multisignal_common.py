#!/usr/bin/env python
"""Common utilities for v18 multisignal deterministic fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.pepddg_v5.scripts.v14_ab_common import add_ab_channels


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).values


def safe_spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    rho, _ = spearmanr(y, yhat)
    return float(rho) if not np.isnan(rho) else np.nan


def eval_prediction(df: pd.DataFrame, pred_col: str) -> dict:
    y = df["ddg_exp"].astype(float).values
    base = df["rankscore_3view_base"].astype(float).values
    pred = df[pred_col].astype(float).values
    rho_base = safe_spearman(y, base)
    rho_new = safe_spearman(y, pred)
    return {"rho_base": float(rho_base), "rho_new": float(rho_new), "delta": float(rho_new - rho_base)}


@dataclass(frozen=True)
class V18Config:
    name: str
    core_col: str
    w_local_reorg: float
    w_ddg_foldx: float
    w_ddg_rosetta_iface: float
    w_ddg_rosetta_total: float
    w_mmpbsa_ddg_ala: float
    w_ddg_xint_iface_total: float
    w_ddg_xint_local_screened: float
    w_ddg_xint_iface_heavy: float


DEFAULT_V18_CONFIG = V18Config(
    name="v18_multisignal_target068",
    core_col="rankscore_3view_v13_v13_cons",
    w_local_reorg=0.02866140197614674,
    w_ddg_foldx=0.41617,
    w_ddg_rosetta_iface=0.00425,
    w_ddg_rosetta_total=0.06859,
    w_mmpbsa_ddg_ala=0.00222,
    w_ddg_xint_iface_total=0.08560,
    w_ddg_xint_local_screened=0.14799,
    w_ddg_xint_iface_heavy=0.20389,
)


FEATURE_COLS = [
    "ddg_foldx",
    "ddg_rosetta_iface",
    "ddg_rosetta_total",
    "mmpbsa_ddg_ala",
    "ddg_xint_iface_total",
    "ddg_xint_local_screened",
    "ddg_xint_iface_heavy",
]


def _add_feature_rank_and_avail(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        out[col] = np.nan
    arr = pd.to_numeric(out[col], errors="coerce")
    med = float(arr.median()) if arr.notna().any() else 0.0
    out[f"{col}_q"] = _rank(arr.fillna(med).values)
    out[f"{col}_avail"] = arr.notna().astype(float).values
    return out


def _orient_feature_sign_by_main(main_df: pd.DataFrame, col: str) -> float:
    sub = main_df[["ddg_exp", col]].dropna()
    if len(sub) < 80:
        return 1.0
    r = safe_spearman(sub["ddg_exp"].values, sub[col].values)
    if np.isnan(r):
        return 1.0
    return 1.0 if r >= 0.0 else -1.0


def _merge_main_extras(main_df: pd.DataFrame) -> pd.DataFrame:
    out = main_df.copy()

    foldx = pd.read_csv("research/pepddg_v5/results/foldx/foldx_predicted_features.csv")[
        ["target", "mut", "ddg_foldx"]
    ].drop_duplicates(["target", "mut"])
    out = out.merge(foldx, on=["target", "mut"], how="left")

    rosetta = pd.read_csv("research/pepddg_v5/results/rosetta_ensemble_final/full_features_with_rosetta.csv")[
        ["target", "mut", "ddg_rosetta_iface", "ddg_rosetta_total", "mmpbsa_ddg_ala"]
    ].drop_duplicates(["target", "mut"])
    out = out.merge(rosetta, on=["target", "mut"], how="left", suffixes=("", "_r"))
    for col in ["ddg_rosetta_iface", "ddg_rosetta_total", "mmpbsa_ddg_ala"]:
        if f"{col}_r" in out.columns:
            out[col] = out[col].fillna(out[f"{col}_r"])
            out = out.drop(columns=[f"{col}_r"])

    hist = pd.read_csv("research/pepddg_v5/results/all_scores_predicted.csv")
    num_cols = [c for c in hist.columns if pd.api.types.is_numeric_dtype(hist[c])]
    hist = hist.groupby(["target", "mut"], as_index=False)[num_cols].mean()
    out = out.merge(
        hist[
            [
                "target",
                "mut",
                "ddg_xint_iface_total",
                "ddg_xint_local_screened",
                "ddg_xint_iface_heavy",
            ]
        ],
        on=["target", "mut"],
        how="left",
    )
    return out


def _merge_ood_extras(ood_df: pd.DataFrame) -> pd.DataFrame:
    out = ood_df.copy()
    if "ddg_foldx" not in out.columns:
        fxo = pd.read_csv("research/pepddg_v5/results/ood_bindinggym/ood_foldx_results.csv")[
            ["target", "mutation_token", "ddg_foldx"]
        ]
        out = out.merge(fxo, on=["target", "mutation_token"], how="left")
    return out


def prepare_v18_inputs(main_df: pd.DataFrame, bpti_df: pd.DataFrame, ood_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    main = _merge_main_extras(main_df)
    bpti = bpti_df.copy()
    ood = _merge_ood_extras(ood_df)

    # Ensure columns exist before channel construction.
    for df in [bpti, ood]:
        for col in ["ddg_rosetta_iface", "ddg_rosetta_total"]:
            if col not in df.columns:
                df[col] = np.nan
    if "mmpbsa_ddg_ala" not in bpti.columns:
        bpti["mmpbsa_ddg_ala"] = np.nan
    if "mmpbsa_ddg_ala" not in ood.columns:
        ood["mmpbsa_ddg_ala"] = np.nan
    if "ddg_foldx" not in bpti.columns:
        bpti["ddg_foldx"] = np.nan
    for df in [bpti, ood]:
        for col in ["ddg_xint_iface_total", "ddg_xint_local_screened", "ddg_xint_iface_heavy"]:
            if col not in df.columns:
                df[col] = np.nan

    main = add_ab_channels(main)
    bpti = add_ab_channels(bpti)
    ood = add_ab_channels(ood)

    # Orient signs to main-correlation direction.
    signs: dict[str, float] = {}
    for col in FEATURE_COLS:
        sign = _orient_feature_sign_by_main(main, col)
        signs[col] = sign
        main[col] = pd.to_numeric(main[col], errors="coerce") * sign
        bpti[col] = pd.to_numeric(bpti[col], errors="coerce") * sign
        ood[col] = pd.to_numeric(ood[col], errors="coerce") * sign

    for col in FEATURE_COLS:
        main = _add_feature_rank_and_avail(main, col)
        bpti = _add_feature_rank_and_avail(bpti, col)
        ood = _add_feature_rank_and_avail(ood, col)

    return main, bpti, ood, signs


def apply_v18_config(df: pd.DataFrame, cfg: V18Config) -> pd.DataFrame:
    out = df.copy()
    if cfg.core_col not in out.columns:
        raise ValueError(f"Missing core column: {cfg.core_col}")
    raw = out[cfg.core_col].astype(float).values + cfg.w_local_reorg * out["rank_delta_local_reorg_v14"].astype(float).values
    raw += cfg.w_ddg_foldx * out["ddg_foldx_q"].astype(float).values * out["ddg_foldx_avail"].astype(float).values
    raw += cfg.w_ddg_rosetta_iface * out["ddg_rosetta_iface_q"].astype(float).values * out["ddg_rosetta_iface_avail"].astype(float).values
    raw += cfg.w_ddg_rosetta_total * out["ddg_rosetta_total_q"].astype(float).values * out["ddg_rosetta_total_avail"].astype(float).values
    raw += cfg.w_mmpbsa_ddg_ala * out["mmpbsa_ddg_ala_q"].astype(float).values * out["mmpbsa_ddg_ala_avail"].astype(float).values
    raw += cfg.w_ddg_xint_iface_total * out["ddg_xint_iface_total_q"].astype(float).values * out["ddg_xint_iface_total_avail"].astype(float).values
    raw += cfg.w_ddg_xint_local_screened * out["ddg_xint_local_screened_q"].astype(float).values * out["ddg_xint_local_screened_avail"].astype(float).values
    raw += cfg.w_ddg_xint_iface_heavy * out["ddg_xint_iface_heavy_q"].astype(float).values * out["ddg_xint_iface_heavy_avail"].astype(float).values
    out["rankscore_3view_v18"] = _rank(raw)
    return out


def cfg_to_dict(cfg: V18Config) -> dict:
    return asdict(cfg)


__all__ = [
    "DEFAULT_V18_CONFIG",
    "FEATURE_COLS",
    "V18Config",
    "apply_v18_config",
    "cfg_to_dict",
    "eval_prediction",
    "prepare_v18_inputs",
    "safe_spearman",
]
