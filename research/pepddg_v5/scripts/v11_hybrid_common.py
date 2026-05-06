#!/usr/bin/env python
"""Common utilities for PepDDG v11 LME + hybrid deterministic experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from research.pepddg_v5.scripts.v10_zero_shot_common import (
    HYDRO_MAX,
    HYDRO_MIN,
    HYDRO_SPAN,
    RULE_CONFIGS_V10_1,
    GuardrailRuleConfig,
    apply_guardrail_rule_variant,
    build_base_scores,
    build_mechanism_terms,
    fill_mutation_properties,
    rank,
    safe_spearman,
    target_quantile,
)


@dataclass(frozen=True)
class V11HybridConfig:
    """Deterministic config for v11 LME + hybrid structural augmentation."""

    name: str
    seed_variant: str
    seed_scale: float
    blend_mode: str
    lambda_lme: float
    lambda_local_signed: float
    lambda_surface: float
    local_max_volume: float
    local_min_abs_charge: float
    local_max_view_disp: float
    local_min_burial: float
    use_surface_anchor: bool
    enable_fallback: bool
    fallback_charge_burial_cutoff: float
    fallback_volume_cutoff: float


RULE_CONFIGS_V11: dict[str, V11HybridConfig] = {
    # External-safe reference variant.
    "v11_lme_conservative": V11HybridConfig(
        name="v11_lme_conservative",
        seed_variant="v10_1_conservative",
        seed_scale=1.00,
        blend_mode="struct_aug",
        lambda_lme=0.30,
        lambda_local_signed=0.18,
        lambda_surface=0.08,
        local_max_volume=60.0,
        local_min_abs_charge=1.0,
        local_max_view_disp=0.70,
        local_min_burial=0.0,
        use_surface_anchor=True,
        enable_fallback=True,
        fallback_charge_burial_cutoff=0.80,
        fallback_volume_cutoff=85.0,
    ),
    # Balanced hybrid variant.
    "v11_hyb_balanced": V11HybridConfig(
        name="v11_hyb_balanced",
        seed_variant="v10_1_conservative",
        seed_scale=1.15,
        blend_mode="struct_aug",
        lambda_lme=0.28,
        lambda_local_signed=0.35,
        lambda_surface=0.25,
        local_max_volume=75.0,
        local_min_abs_charge=0.5,
        local_max_view_disp=0.85,
        local_min_burial=0.0,
        use_surface_anchor=True,
        enable_fallback=True,
        fallback_charge_burial_cutoff=0.85,
        fallback_volume_cutoff=90.0,
    ),
    # Main-rho target variant (aggressive on local + surface).
    "v11_hyb_target065": V11HybridConfig(
        name="v11_hyb_target065",
        seed_variant="v10_1_conservative",
        seed_scale=1.30,
        blend_mode="direct_3view",
        lambda_lme=0.10,
        lambda_local_signed=0.90,
        lambda_surface=0.70,
        local_max_volume=120.0,
        local_min_abs_charge=0.0,
        local_max_view_disp=1.0,
        local_min_burial=0.0,
        use_surface_anchor=True,
        enable_fallback=False,
        fallback_charge_burial_cutoff=1.1,
        fallback_volume_cutoff=999.0,
    ),
}


def _seed_cfg(name: str) -> GuardrailRuleConfig:
    if name not in RULE_CONFIGS_V10_1:
        raise KeyError(f"Unknown seed variant: {name}")
    return RULE_CONFIGS_V10_1[name]


def build_lme_hybrid_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic signed local-mechanism terms used by v11."""
    out = df.copy()

    burial = out["burial_proxy_v10"].astype(float).values
    delta_volume = out["delta_volume"].astype(float).values
    delta_charge = out["delta_charge"].astype(float).values

    wt_hn = ((out["wt_hydrophobicity"].astype(float) - HYDRO_MIN) / HYDRO_SPAN).clip(0.0, 1.0).values
    mt_hn = ((out["mut_hydrophobicity"].astype(float) - HYDRO_MIN) / HYDRO_SPAN).clip(0.0, 1.0).values

    signed_volume = np.clip(-delta_volume, -120.0, 120.0) / 120.0
    signed_charge = np.clip(-delta_charge, -2.0, 2.0) / 2.0
    signed_hydro = np.clip(wt_hn - mt_hn, -1.0, 1.0)

    out["term_lme_pack_signed_raw"] = signed_volume * (0.25 + 0.75 * burial)
    out["term_lme_charge_signed_raw"] = signed_charge * (0.30 + 0.70 * burial)
    out["term_lme_hydro_signed_raw"] = signed_hydro * (0.20 + 0.80 * burial)

    out["term_lme_signal_raw"] = (
        0.45 * out["term_lme_pack_signed_raw"].values
        + 0.35 * out["term_lme_charge_signed_raw"].values
        + 0.20 * out["term_lme_hydro_signed_raw"].values
    )
    out["q_lme_signed"] = target_quantile(out, "term_lme_signal_raw")
    out["rank_lme_signed"] = rank(out["q_lme_signed"].values)

    local_signed_raw = rank(-delta_volume) + rank(-delta_charge)
    out["local_signed_raw"] = local_signed_raw
    out["rank_local_signed"] = rank(local_signed_raw)

    if "rank_masif" in out.columns:
        surface_source = out["rank_masif"].astype(float).fillna(out["rankscore_struct_base"]).values
    else:
        # If MaSIF is unavailable (e.g., external sets), fallback to structural baseline.
        surface_source = out["rankscore_struct_base"].astype(float).values
    out["surface_anchor_raw"] = surface_source
    out["rank_surface_anchor"] = rank(surface_source)

    return out


def apply_v11_hybrid_variant(df: pd.DataFrame, cfg: V11HybridConfig) -> pd.DataFrame:
    """Apply deterministic v11 hybrid structural augmentation."""
    out = df.copy()

    seed_scored = apply_guardrail_rule_variant(out, _seed_cfg(cfg.seed_variant))
    out["rankscore_struct_seed"] = seed_scored["rankscore_struct_v10_1"].values
    out["rankscore_3view_seed"] = seed_scored["rankscore_3view_v10_1"].values

    abs_volume = out["delta_volume"].abs().astype(float).values
    abs_charge = out["delta_charge"].abs().astype(float).values
    burial = out["burial_proxy_v10"].astype(float).values
    q_disp = out["q_view_dispersion"].astype(float).values

    local_apply = (
        (abs_volume <= cfg.local_max_volume)
        & (abs_charge >= cfg.local_min_abs_charge)
        & (q_disp <= cfg.local_max_view_disp)
        & (burial >= cfg.local_min_burial)
    ).astype(float)
    out["local_apply_flag"] = local_apply

    local_rank_masked = rank(local_apply * out["rank_local_signed"].values)
    surface_rank = out["rank_surface_anchor"].values if cfg.use_surface_anchor else out["rankscore_struct_base"].values

    if cfg.blend_mode == "struct_aug":
        struct_raw = (
            cfg.seed_scale * out["rankscore_struct_seed"].values
            + cfg.lambda_lme * out["rank_lme_signed"].values
            + cfg.lambda_local_signed * local_rank_masked
            + cfg.lambda_surface * surface_rank
        )
        out["rankscore_struct_v11"] = rank(struct_raw)
        pred_pre = (
            out["rankscore_phys"].values + out["rankscore_struct_v11"].values + out["rankscore_mpnn"].values
        )
    elif cfg.blend_mode == "direct_3view":
        # Aggressive main-rho mode: hybridize at 3-view score level directly.
        direct_local = cfg.lambda_local_signed * local_rank_masked
        direct_surface = cfg.lambda_surface * surface_rank
        direct_lme = cfg.lambda_lme * out["rank_lme_signed"].values
        pred_pre = cfg.seed_scale * out["rankscore_3view_seed"].values + direct_local + direct_surface + direct_lme
        out["rankscore_struct_v11"] = out["rankscore_struct_seed"].values
    else:
        raise ValueError(f"Unknown blend_mode: {cfg.blend_mode}")

    if cfg.enable_fallback:
        fallback_flag = (
            ((abs_charge >= 1.0) & (burial >= cfg.fallback_charge_burial_cutoff))
            | (abs_volume >= cfg.fallback_volume_cutoff)
        ).astype(float)
        pred = np.where(fallback_flag > 0.0, out["rankscore_3view_base"].values, pred_pre)
    else:
        fallback_flag = np.zeros(len(out), dtype=float)
        pred = pred_pre

    out["fallback_flag"] = fallback_flag
    out["rankscore_3view_v11"] = pred
    return out


def cfg_to_dict(cfg: V11HybridConfig) -> dict:
    return asdict(cfg)


__all__ = [
    "RULE_CONFIGS_V11",
    "V11HybridConfig",
    "apply_v11_hybrid_variant",
    "build_base_scores",
    "build_lme_hybrid_terms",
    "build_mechanism_terms",
    "cfg_to_dict",
    "fill_mutation_properties",
    "safe_spearman",
]
