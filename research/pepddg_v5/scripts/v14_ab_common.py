#!/usr/bin/env python
"""Common utilities for v14 first-principles A/B channels and minimal fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).values


def _target_quantile(df: pd.DataFrame, col: str, target_col: str = "target") -> np.ndarray:
    if target_col not in df.columns:
        return _rank(df[col].values)
    out = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby(target_col):
        out.loc[g.index] = _rank(g[col].values)
    return out.values


def safe_spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    rho, _ = spearmanr(y, yhat)
    return float(rho) if not np.isnan(rho) else np.nan


def eval_prediction(df: pd.DataFrame, pred_col: str) -> dict:
    y = df["ddg_exp"].astype(float).values
    base = df["rankscore_3view_base"].astype(float).values
    pred = df[pred_col].astype(float).values
    rho_base = safe_spearman(y, base)
    rho_new = safe_spearman(y, pred)
    return {
        "rho_base": float(rho_base),
        "rho_new": float(rho_new),
        "delta": float(rho_new - rho_base),
    }


def gate_summary(
    main_metrics: dict,
    bpti_metrics: dict,
    ood_metrics: dict,
    *,
    gate_main_rho: float = 0.650,
    gate_external_delta: float = -0.010,
) -> dict:
    main_rho = float(main_metrics["rho_new"])
    bpti_delta = float(bpti_metrics["delta"])
    ood_delta = float(ood_metrics["delta"])

    main_gate = bool(main_rho >= gate_main_rho)
    bpti_gate = bool(bpti_delta >= gate_external_delta)
    ood_gate = bool(ood_delta >= gate_external_delta)

    gate_main_margin = float(main_rho - gate_main_rho)
    gate_bpti_margin = float(bpti_delta - gate_external_delta)
    gate_ood_margin = float(ood_delta - gate_external_delta)

    objective = (
        main_rho
        + 0.80 * min(0.0, gate_bpti_margin)
        + 0.60 * min(0.0, gate_ood_margin)
        + 0.30 * min(0.0, gate_main_margin)
    )

    return {
        "main_rho": main_rho,
        "main_delta": float(main_metrics["delta"]),
        "bpti_delta": bpti_delta,
        "ood_delta": ood_delta,
        "main_gate": main_gate,
        "bpti_gate": bpti_gate,
        "ood_gate": ood_gate,
        "gate_pass_all": bool(main_gate and bpti_gate and ood_gate),
        "gate_main_margin": gate_main_margin,
        "gate_bpti_margin": gate_bpti_margin,
        "gate_ood_margin": gate_ood_margin,
        "objective": float(objective),
    }


def _resolve_phys_cols(df: pd.DataFrame) -> tuple[str, str]:
    if "ddg_paired_xint_iface" in df.columns and "ddg_paired_bind_proxy" in df.columns:
        return "ddg_paired_xint_iface", "ddg_paired_bind_proxy"
    if "ddg_xint_iface" in df.columns and "ddg_bind_proxy" in df.columns:
        return "ddg_xint_iface", "ddg_bind_proxy"
    raise ValueError("Missing required physics columns for v14 channels.")


def add_ab_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Add channel A (delta_elec_desolv) and B (delta_local_reorg) feature stack."""
    out = df.copy()
    required = [
        "delta_charge",
        "delta_volume",
        "burial_proxy_v10",
        "q_view_dispersion",
        "rankscore_phys",
        "rankscore_struct_base",
        "rankscore_3view_base",
        "rankscore_3view_target065",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for v14 A/B channels: {missing}")

    phys_iface_col, phys_bind_col = _resolve_phys_cols(out)

    out["abs_charge"] = out["delta_charge"].abs().astype(float)
    out["abs_volume"] = out["delta_volume"].abs().astype(float)
    out["charge_buried_flag"] = (
        (out["abs_charge"] >= 1.0) & (out["burial_proxy_v10"].astype(float) >= 0.60)
    ).astype(float)
    out["large_volume_flag"] = (out["abs_volume"] >= 85.0).astype(float)

    # Channel A: electrostatic desolvation (first-principles proxy decomposition)
    if "term_charge_desolv_raw" in out.columns:
        elec_raw = out["term_charge_desolv_raw"].astype(float).values
    else:
        elec_raw = (
            np.clip(out["abs_charge"].values, 0.0, 2.0) / 2.0
        ) * (0.35 + 0.65 * out["burial_proxy_v10"].astype(float).values)
    out["q_elec_proxy_v14"] = _target_quantile(out.assign(_tmp=elec_raw), "_tmp")

    phys_split = np.abs(out[phys_bind_col].astype(float).values - out[phys_iface_col].astype(float).values)
    out["q_phys_split_v14"] = _target_quantile(out.assign(_tmp=phys_split), "_tmp")

    mmpbsa_term = np.zeros(len(out), dtype=float)
    if "mmpbsa_ddg_ala" in out.columns:
        mm = out["mmpbsa_ddg_ala"].astype(float)
        mm_abs = mm.abs()
        mm_q = _target_quantile(out.assign(_tmp=mm_abs.fillna(mm_abs.median() if mm_abs.notna().any() else 0.0)), "_tmp")
        mm_flag = mm.notna().astype(float).values
        mmpbsa_term = mm_q * mm_flag
    out["mmpbsa_available_v14"] = (mmpbsa_term > 0.0).astype(float)
    out["q_mmpbsa_v14"] = mmpbsa_term

    # Sign flip contributes to electrostatic mismatch when available.
    if "wt_charge" in out.columns and "mut_charge" in out.columns:
        sign_flip = (
            np.sign(out["wt_charge"].astype(float).values) * np.sign(out["mut_charge"].astype(float).values) < 0.0
        ).astype(float)
    else:
        sign_flip = (out["abs_charge"].values >= 1.0).astype(float)

    a_raw = (
        0.55 * out["q_elec_proxy_v14"].astype(float).values
        + 0.25 * out["q_phys_split_v14"].astype(float).values
        + 0.10 * out["q_mmpbsa_v14"].astype(float).values
        + 0.08 * out["charge_buried_flag"].astype(float).values
        + 0.02 * sign_flip * out["burial_proxy_v10"].astype(float).values
    )
    out["delta_elec_desolv_raw_v14"] = a_raw
    out["q_delta_elec_desolv_v14"] = _target_quantile(out.assign(_tmp=a_raw), "_tmp")
    out["rank_delta_elec_desolv_v14"] = _rank(out["q_delta_elec_desolv_v14"].values)

    # Channel B: local reorganization strain (first-principles proxy decomposition)
    if "term_packing_strain_raw" in out.columns:
        pack_raw = out["term_packing_strain_raw"].astype(float).values
    else:
        pack_raw = (
            np.clip(out["abs_volume"].values, 0.0, 120.0) / 120.0
        ) * (0.25 + 0.75 * out["burial_proxy_v10"].astype(float).values)
    out["q_pack_proxy_v14"] = _target_quantile(out.assign(_tmp=pack_raw), "_tmp")

    vol_norm = np.clip(out["abs_volume"].values, 0.0, 120.0) / 120.0
    phys_struct_gap = np.abs(out["rankscore_phys"].astype(float).values - out["rankscore_struct_base"].astype(float).values)
    out["q_phys_struct_gap_v14"] = _target_quantile(out.assign(_tmp=phys_struct_gap), "_tmp")

    buried_large = (
        (out["abs_volume"].values >= 70.0) & (out["burial_proxy_v10"].astype(float).values >= 0.55)
    ).astype(float)
    b_raw = (
        0.45 * out["q_pack_proxy_v14"].astype(float).values
        + 0.25 * out["q_view_dispersion"].astype(float).values
        + 0.15 * vol_norm
        + 0.15 * out["q_phys_struct_gap_v14"].astype(float).values
        + 0.20 * buried_large
    )
    out["delta_local_reorg_raw_v14"] = b_raw
    out["q_delta_local_reorg_v14"] = _target_quantile(out.assign(_tmp=b_raw), "_tmp")
    out["rank_delta_local_reorg_v14"] = _rank(out["q_delta_local_reorg_v14"].values)

    # Shared risk score for minimal conservative shrinkage.
    out["risk_ab_v14"] = np.clip(
        0.45 * out["q_view_dispersion"].astype(float).values
        + 0.25 * out["q_delta_elec_desolv_v14"].astype(float).values
        + 0.20 * out["q_delta_local_reorg_v14"].astype(float).values
        + 0.10 * out["charge_buried_flag"].astype(float).values,
        0.0,
        1.0,
    )
    return out


@dataclass(frozen=True)
class AConfig:
    name: str
    lambda_a: float
    shrink_scale: float
    risk_thr: float


@dataclass(frozen=True)
class BConfig:
    name: str
    lambda_b: float
    shrink_scale: float
    risk_thr: float


@dataclass(frozen=True)
class FusionConfig:
    name: str
    lambda_a: float
    lambda_b: float
    shrink_scale: float
    risk_thr: float


def _risk_shrink(risk: np.ndarray, risk_thr: float, shrink_scale: float) -> np.ndarray:
    z = np.clip(risk - risk_thr, 0.0, 1.0)
    return np.clip(shrink_scale * z / (1.0 - risk_thr + 1e-12), 0.0, 1.0)


def apply_a_variant(
    df: pd.DataFrame,
    cfg: AConfig,
    *,
    core_col: str = "rankscore_3view_target065",
    fallback_col: str = "rankscore_3view_base",
) -> pd.DataFrame:
    out = df.copy()
    if core_col not in out.columns:
        raise ValueError(f"Missing core_col for A variant: {core_col}")
    if fallback_col not in out.columns:
        raise ValueError(f"Missing fallback_col for A variant: {fallback_col}")
    raw = out[core_col].astype(float).values + cfg.lambda_a * out["rank_delta_elec_desolv_v14"].astype(float).values
    shrink = _risk_shrink(out["risk_ab_v14"].astype(float).values, cfg.risk_thr, cfg.shrink_scale)
    pred = (1.0 - shrink) * raw + shrink * out[fallback_col].astype(float).values
    out["pred_a_raw_v14"] = raw
    out["pred_a_shrink_v14"] = shrink
    out["rankscore_3view_a_v14"] = _rank(pred)
    return out


def apply_b_variant(
    df: pd.DataFrame,
    cfg: BConfig,
    *,
    core_col: str = "rankscore_3view_target065",
    fallback_col: str = "rankscore_3view_base",
) -> pd.DataFrame:
    out = df.copy()
    if core_col not in out.columns:
        raise ValueError(f"Missing core_col for B variant: {core_col}")
    if fallback_col not in out.columns:
        raise ValueError(f"Missing fallback_col for B variant: {fallback_col}")
    raw = out[core_col].astype(float).values + cfg.lambda_b * out["rank_delta_local_reorg_v14"].astype(float).values
    shrink = _risk_shrink(out["risk_ab_v14"].astype(float).values, cfg.risk_thr, cfg.shrink_scale)
    pred = (1.0 - shrink) * raw + shrink * out[fallback_col].astype(float).values
    out["pred_b_raw_v14"] = raw
    out["pred_b_shrink_v14"] = shrink
    out["rankscore_3view_b_v14"] = _rank(pred)
    return out


def apply_fusion_variant(
    df: pd.DataFrame,
    cfg: FusionConfig,
    *,
    core_col: str = "rankscore_3view_target065",
    fallback_col: str = "rankscore_3view_base",
) -> pd.DataFrame:
    out = df.copy()
    if core_col not in out.columns:
        raise ValueError(f"Missing core_col for fusion variant: {core_col}")
    if fallback_col not in out.columns:
        raise ValueError(f"Missing fallback_col for fusion variant: {fallback_col}")
    raw = (
        out[core_col].astype(float).values
        + cfg.lambda_a * out["rank_delta_elec_desolv_v14"].astype(float).values
        + cfg.lambda_b * out["rank_delta_local_reorg_v14"].astype(float).values
    )
    shrink = _risk_shrink(out["risk_ab_v14"].astype(float).values, cfg.risk_thr, cfg.shrink_scale)
    pred = (1.0 - shrink) * raw + shrink * out[fallback_col].astype(float).values
    out["pred_fusion_raw_v14"] = raw
    out["pred_fusion_shrink_v14"] = shrink
    out["rankscore_3view_fusion_v14"] = _rank(pred)
    return out


def cfg_to_dict(cfg: AConfig | BConfig | FusionConfig) -> dict:
    return asdict(cfg)


__all__ = [
    "AConfig",
    "BConfig",
    "FusionConfig",
    "add_ab_channels",
    "apply_a_variant",
    "apply_b_variant",
    "apply_fusion_variant",
    "cfg_to_dict",
    "eval_prediction",
    "gate_summary",
    "safe_spearman",
]
