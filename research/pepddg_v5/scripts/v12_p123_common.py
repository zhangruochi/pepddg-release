#!/usr/bin/env python
"""Common utilities for PepDDG v12 P1->P2->P3 deterministic experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from research.pepddg_v5.scripts.v11_hybrid_common import (
    RULE_CONFIGS_V11,
    apply_v11_hybrid_variant,
    build_base_scores,
    build_lme_hybrid_terms,
    build_mechanism_terms,
    fill_mutation_properties,
    safe_spearman,
)


@dataclass(frozen=True)
class V12P123Config:
    """Deterministic config for the P1->P2->P3 pipeline."""

    name: str

    # P1: reliability-adaptive fusion
    seed_scale: float
    lam_local: float
    lam_surface: float
    lam_lme: float
    rel_disp: float
    rel_risk: float
    rel_reg: float
    rel_min: float
    reg_burial: float

    # P2: conformer-consensus style uncertainty shrink
    cons_alpha: float
    cons_seed_mix: float

    # P3: regime risk guardrails
    risk_burial: float
    risk_volume: float
    risk_blend: float
    risk_base_mix: float


RULE_CONFIGS_V12: dict[str, V12P123Config] = {
    # Main-rho maximizing track (still external-risky in current audit).
    "v12_p123_mainmax": V12P123Config(
        name="v12_p123_mainmax",
        seed_scale=1.268562276697827,
        lam_local=1.046869442471342,
        lam_surface=0.4973066016219442,
        lam_lme=0.09196077826059919,
        rel_disp=0.3671378135348488,
        rel_risk=0.1251813457328879,
        rel_reg=0.13297849324128228,
        rel_min=0.8452245284270093,
        reg_burial=0.5880493040481846,
        cons_alpha=0.10873297810783145,
        cons_seed_mix=0.23272379640509044,
        risk_burial=0.6715978144378467,
        risk_volume=96.20420122448859,
        risk_blend=0.25201005143140676,
        risk_base_mix=0.012935631963615801,
    ),
    # Best trade-off from random frontier search.
    "v12_p123_tradeoff": V12P123Config(
        name="v12_p123_tradeoff",
        seed_scale=1.3318647865506898,
        lam_local=0.7531178107632626,
        lam_surface=0.7107829211884853,
        lam_lme=0.04441650230079203,
        rel_disp=0.3384872025569597,
        rel_risk=0.12163470643532562,
        rel_reg=0.17983592604154336,
        rel_min=0.7198664081870775,
        reg_burial=0.5461434475307587,
        cons_alpha=0.09263047760055933,
        cons_seed_mix=0.3663382839167245,
        risk_burial=0.6208786317126422,
        risk_volume=98.4500820763211,
        risk_blend=0.09062920729091896,
        risk_base_mix=0.4126099388248203,
    ),
    # External-safe control: force P3 to balanced variant globally.
    "v12_p123_safefallback": V12P123Config(
        name="v12_p123_safefallback",
        seed_scale=1.3318647865506898,
        lam_local=0.7531178107632626,
        lam_surface=0.7107829211884853,
        lam_lme=0.04441650230079203,
        rel_disp=0.3384872025569597,
        rel_risk=0.12163470643532562,
        rel_reg=0.17983592604154336,
        rel_min=0.7198664081870775,
        reg_burial=0.5461434475307587,
        cons_alpha=0.09263047760055933,
        cons_seed_mix=0.3663382839167245,
        risk_burial=0.0,
        risk_volume=0.0,
        risk_blend=1.0,
        risk_base_mix=0.0,
    ),
}


def cfg_to_dict(cfg: V12P123Config) -> dict:
    return asdict(cfg)


def build_v12_stage_inputs(
    df: pd.DataFrame,
    *,
    mut_col: str,
    physics_mode: str,
) -> pd.DataFrame:
    """Build shared input table for v12 stage evaluation."""
    out = fill_mutation_properties(df, mut_col=mut_col)
    out = build_base_scores(out, physics_mode=physics_mode)
    out = build_mechanism_terms(out, use_existing_burial=False)
    out = build_lme_hybrid_terms(out)

    target065 = apply_v11_hybrid_variant(out, RULE_CONFIGS_V11["v11_hyb_target065"])
    balanced = apply_v11_hybrid_variant(out, RULE_CONFIGS_V11["v11_hyb_balanced"])

    out["rankscore_3view_target065"] = target065["rankscore_3view_v11"].values
    out["rankscore_3view_balanced"] = balanced["rankscore_3view_v11"].values
    out["rankscore_3view_seed"] = target065["rankscore_3view_seed"].values
    out["local_apply_flag_target065"] = target065["local_apply_flag"].values
    return out


def apply_v12_p123_variant(df: pd.DataFrame, cfg: V12P123Config) -> pd.DataFrame:
    """Apply P1->P2->P3 deterministic scoring to a prepared dataframe."""
    out = df.copy()

    abs_charge = out["delta_charge"].abs().astype(float).values
    abs_volume = out["delta_volume"].abs().astype(float).values
    burial = out["burial_proxy_v10"].astype(float).values
    q_disp = out["q_view_dispersion"].astype(float).values
    q_risk = np.maximum(out["q_charge"].astype(float).values, out["q_pack"].astype(float).values)

    reg_flag = ((abs_charge >= 1.0) & (burial >= cfg.reg_burial)).astype(float)
    rel = 1.0 - cfg.rel_disp * q_disp - cfg.rel_risk * q_risk - cfg.rel_reg * reg_flag
    rel = np.clip(rel, cfg.rel_min, 1.0)

    local = out["rank_local_signed"].values * out["local_apply_flag_target065"].values
    pred_p1 = cfg.seed_scale * out["rankscore_3view_seed"].values + rel * (
        cfg.lam_local * local
        + cfg.lam_surface * out["rank_surface_anchor"].values
        + cfg.lam_lme * out["rank_lme_signed"].values
    )

    uncert = 0.5 * q_disp + 0.5 * q_risk
    pred_p2 = (1.0 - cfg.cons_alpha * uncert) * pred_p1 + cfg.cons_alpha * uncert * (
        cfg.cons_seed_mix * out["rankscore_3view_seed"].values
        + (1.0 - cfg.cons_seed_mix) * out["rankscore_3view_target065"].values
    )

    high_risk = ((abs_charge >= 1.0) & (burial >= cfg.risk_burial)) | (abs_volume >= cfg.risk_volume)
    risk_ref = (
        cfg.risk_base_mix * out["rankscore_3view_base"].values
        + (1.0 - cfg.risk_base_mix) * out["rankscore_3view_balanced"].values
    )
    pred_p3 = np.where(
        high_risk,
        (1.0 - cfg.risk_blend) * pred_p2 + cfg.risk_blend * risk_ref,
        pred_p2,
    )

    out["reliability_factor"] = rel
    out["risk_reg_flag"] = reg_flag
    out["risk_high_flag"] = high_risk.astype(float)
    out["rankscore_3view_p1"] = pred_p1
    out["rankscore_3view_p2"] = pred_p2
    out["rankscore_3view_p3"] = pred_p3
    return out


def stage_metric(df: pd.DataFrame, pred_col: str) -> dict:
    """Return pooled rho/delta for one prediction column."""
    y = df["ddg_exp"].values
    base = df["rankscore_3view_base"].values
    pred = df[pred_col].values
    rho_base = safe_spearman(y, base)
    rho_new = safe_spearman(y, pred)
    return {
        "rho_base": float(rho_base),
        "rho_new": float(rho_new),
        "delta": float(rho_new - rho_base),
    }


__all__ = [
    "RULE_CONFIGS_V12",
    "V12P123Config",
    "apply_v12_p123_variant",
    "build_v12_stage_inputs",
    "cfg_to_dict",
    "safe_spearman",
    "stage_metric",
]
