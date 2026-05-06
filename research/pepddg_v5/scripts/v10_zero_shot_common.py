#!/usr/bin/env python
"""Common utilities for PepDDG v10 zero-shot / training-free experiments."""

from __future__ import annotations

from dataclasses import dataclass

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

# Kyte-Doolittle hydrophobicity scale.
AA_HYDRO = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

HYDRO_MIN = -4.5
HYDRO_MAX = 4.5
HYDRO_SPAN = HYDRO_MAX - HYDRO_MIN


@dataclass(frozen=True)
class RuleConfig:
    """Deterministic rule config for structural-channel augmentation."""

    name: str
    lambda_mech: float
    lambda_focus_charge: float
    lambda_focus_packing: float
    charge_cutoff: float
    volume_cutoff: float
    burial_cutoff_charge: float
    burial_cutoff_packing: float


@dataclass(frozen=True)
class GuardrailRuleConfig:
    """Deterministic rule config with external-safe guardrails for v10.1."""

    name: str
    lambda_mech: float
    lambda_focus_charge: float
    lambda_focus_packing: float
    charge_cutoff: float
    volume_cutoff: float
    burial_cutoff_charge: float
    burial_cutoff_packing: float
    charge_volume_cap: float
    max_view_disp_for_focus_charge: float
    max_view_disp_for_focus_packing: float
    shrink_base: float
    shrink_disp_coeff: float
    shrink_risk_coeff: float
    shrink_min: float
    shrink_max: float


RULE_CONFIGS: dict[str, RuleConfig] = {
    "v10_conservative": RuleConfig(
        name="v10_conservative",
        lambda_mech=0.40,
        lambda_focus_charge=0.10,
        lambda_focus_packing=0.05,
        charge_cutoff=1.0,
        volume_cutoff=45.0,
        burial_cutoff_charge=0.55,
        burial_cutoff_packing=0.60,
    ),
    "v10_balanced": RuleConfig(
        name="v10_balanced",
        lambda_mech=0.60,
        lambda_focus_charge=0.20,
        lambda_focus_packing=0.10,
        charge_cutoff=1.0,
        volume_cutoff=35.0,
        burial_cutoff_charge=0.50,
        burial_cutoff_packing=0.55,
    ),
    "v10_aggressive": RuleConfig(
        name="v10_aggressive",
        lambda_mech=0.85,
        lambda_focus_charge=0.30,
        lambda_focus_packing=0.20,
        charge_cutoff=1.0,
        volume_cutoff=30.0,
        burial_cutoff_charge=0.45,
        burial_cutoff_packing=0.50,
    ),
}

RULE_CONFIGS_V10_1: dict[str, GuardrailRuleConfig] = {
    "v10_1_conservative": GuardrailRuleConfig(
        name="v10_1_conservative",
        lambda_mech=0.38,
        lambda_focus_charge=0.06,
        lambda_focus_packing=0.04,
        charge_cutoff=1.0,
        volume_cutoff=45.0,
        burial_cutoff_charge=0.58,
        burial_cutoff_packing=0.62,
        charge_volume_cap=55.0,
        max_view_disp_for_focus_charge=0.62,
        max_view_disp_for_focus_packing=0.78,
        shrink_base=0.86,
        shrink_disp_coeff=0.30,
        shrink_risk_coeff=0.12,
        shrink_min=0.42,
        shrink_max=0.95,
    ),
    "v10_1_balanced": GuardrailRuleConfig(
        name="v10_1_balanced",
        lambda_mech=0.52,
        lambda_focus_charge=0.10,
        lambda_focus_packing=0.07,
        charge_cutoff=1.0,
        volume_cutoff=38.0,
        burial_cutoff_charge=0.55,
        burial_cutoff_packing=0.58,
        charge_volume_cap=50.0,
        max_view_disp_for_focus_charge=0.58,
        max_view_disp_for_focus_packing=0.74,
        shrink_base=0.82,
        shrink_disp_coeff=0.36,
        shrink_risk_coeff=0.16,
        shrink_min=0.35,
        shrink_max=0.93,
    ),
    "v10_1_aggressive": GuardrailRuleConfig(
        name="v10_1_aggressive",
        lambda_mech=0.68,
        lambda_focus_charge=0.14,
        lambda_focus_packing=0.10,
        charge_cutoff=1.0,
        volume_cutoff=32.0,
        burial_cutoff_charge=0.52,
        burial_cutoff_packing=0.55,
        charge_volume_cap=45.0,
        max_view_disp_for_focus_charge=0.54,
        max_view_disp_for_focus_packing=0.70,
        shrink_base=0.78,
        shrink_disp_coeff=0.42,
        shrink_risk_coeff=0.20,
        shrink_min=0.30,
        shrink_max=0.90,
    ),
}


def rank(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=float), method="average")


def safe_spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    rho, _ = spearmanr(y, yhat)
    return float(rho) if not np.isnan(rho) else np.nan


def parse_mut_token(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    token = series.fillna("").astype(str)
    wt = token.str[0].where(token.str.len() >= 2, "").str.upper()
    mt = token.str[-1].where(token.str.len() >= 2, "").str.upper()
    return wt, mt


def target_quantile(df: pd.DataFrame, col: str) -> pd.Series:
    q = df.groupby("target")[col].rank(method="average", pct=True)
    return (q - q.min()) / (q.max() - q.min() + 1e-12)


def fill_mutation_properties(df: pd.DataFrame, mut_col: str = "mut") -> pd.DataFrame:
    out = df.copy()

    if "wt_aa1" in out.columns and "mut_aa1" in out.columns:
        wt = out["wt_aa1"].fillna("").astype(str).str.upper()
        mt = out["mut_aa1"].fillna("").astype(str).str.upper()
    else:
        wt, mt = parse_mut_token(out[mut_col])

    out["wt_aa1"] = wt
    out["mut_aa1"] = mt

    wt_vol = wt.map(AA_VOLUME).astype(float)
    mt_vol = mt.map(AA_VOLUME).astype(float)
    wt_chg = wt.map(AA_CHARGE).astype(float)
    mt_chg = mt.map(AA_CHARGE).astype(float)
    wt_hydro = wt.map(AA_HYDRO).astype(float)
    mt_hydro = mt.map(AA_HYDRO).astype(float)

    delta_vol_fb = mt_vol - wt_vol
    delta_chg_fb = mt_chg - wt_chg
    delta_hydro_fb = mt_hydro - wt_hydro

    if "delta_volume" not in out.columns:
        out["delta_volume"] = delta_vol_fb
    else:
        out["delta_volume"] = out["delta_volume"].astype(float).fillna(delta_vol_fb)

    if "delta_charge" not in out.columns:
        out["delta_charge"] = delta_chg_fb
    else:
        out["delta_charge"] = out["delta_charge"].astype(float).fillna(delta_chg_fb)

    if "delta_hydrophobicity" not in out.columns:
        out["delta_hydrophobicity"] = delta_hydro_fb
    else:
        out["delta_hydrophobicity"] = out["delta_hydrophobicity"].astype(float).fillna(delta_hydro_fb)

    out["wt_charge"] = wt_chg
    out["mut_charge"] = mt_chg
    out["wt_hydrophobicity"] = wt_hydro
    out["mut_hydrophobicity"] = mt_hydro
    return out


def build_base_scores(df: pd.DataFrame, physics_mode: str = "paired") -> pd.DataFrame:
    """Build baseline 3-view scores.

    physics_mode:
      - paired: ddg_paired_xint_iface + ddg_paired_bind_proxy
      - external: ddg_xint_iface + ddg_bind_proxy
    """
    out = df.copy()
    if physics_mode == "paired":
        p1 = "ddg_paired_xint_iface"
        p2 = "ddg_paired_bind_proxy"
    elif physics_mode == "external":
        p1 = "ddg_xint_iface"
        p2 = "ddg_bind_proxy"
    else:
        raise ValueError(f"Unknown physics_mode: {physics_mode}")

    req = [
        "ddg_exp",
        "target",
        p1,
        p2,
        "struct_composite",
        "mpnn_neg_llr_complex",
        "mpnn_ddg_bind",
    ]
    out = out.dropna(subset=req).copy()

    out["phys_view_score"] = rank(out[p1].values) + rank(out[p2].values)
    out["mpnn_view_score"] = rank(out["mpnn_neg_llr_complex"].values) + rank(
        out["mpnn_ddg_bind"].values
    )
    out["rankscore_phys"] = rank(out["phys_view_score"].values)
    out["rankscore_struct_base"] = rank(out["struct_composite"].values)
    out["rankscore_mpnn"] = rank(out["mpnn_view_score"].values)
    out["rankscore_3view_base"] = (
        out["rankscore_phys"] + out["rankscore_struct_base"] + out["rankscore_mpnn"]
    )
    return out


def build_burial_proxy(df: pd.DataFrame, prefer_existing: bool = True) -> pd.DataFrame:
    out = df.copy()
    if prefer_existing and "burial_proxy_v10" in out.columns:
        out["burial_proxy_v10"] = out["burial_proxy_v10"].astype(float).fillna(0.5).clip(0.0, 1.0)
        return out

    if "n_iface_contacts_8a" in out.columns and "n_neighbors_10a" in out.columns:
        c1 = pd.to_numeric(out["n_iface_contacts_8a"], errors="coerce").fillna(0.0)
        c2 = pd.to_numeric(out["n_neighbors_10a"], errors="coerce").fillna(0.0)
        out["_q_iface8"] = target_quantile(out.assign(_tmp=c1), "_tmp")
        out["_q_nei10"] = target_quantile(out.assign(_tmp=c2), "_tmp")
        out["burial_proxy_v10"] = 0.5 * (out["_q_iface8"] + out["_q_nei10"])
        out = out.drop(columns=["_q_iface8", "_q_nei10"])
    elif "n_iface_contacts_5a" in out.columns and "n_intra_contacts_8a" in out.columns:
        c1 = pd.to_numeric(out["n_iface_contacts_5a"], errors="coerce").fillna(0.0)
        c2 = pd.to_numeric(out["n_intra_contacts_8a"], errors="coerce").fillna(0.0)
        out["_q_iface5"] = target_quantile(out.assign(_tmp=c1), "_tmp")
        out["_q_intra8"] = target_quantile(out.assign(_tmp=c2), "_tmp")
        out["burial_proxy_v10"] = 0.5 * (out["_q_iface5"] + out["_q_intra8"])
        out = out.drop(columns=["_q_iface5", "_q_intra8"])
    else:
        out["burial_proxy_v10"] = target_quantile(out, "struct_composite")

    out["burial_proxy_v10"] = out["burial_proxy_v10"].astype(float).fillna(0.5).clip(0.0, 1.0)
    return out


def build_mechanism_terms(df: pd.DataFrame, use_existing_burial: bool = False) -> pd.DataFrame:
    out = df.copy()
    if use_existing_burial:
        out = build_burial_proxy(out, prefer_existing=True)
    else:
        out = build_burial_proxy(out, prefer_existing=False)

    abs_delta_volume_norm = np.clip(out["delta_volume"].abs().astype(float), 0.0, 120.0) / 120.0
    abs_delta_charge_norm = np.clip(out["delta_charge"].abs().astype(float), 0.0, 2.0) / 2.0

    wt_hn = (out["wt_hydrophobicity"].astype(float) - HYDRO_MIN) / HYDRO_SPAN
    mt_hn = (out["mut_hydrophobicity"].astype(float) - HYDRO_MIN) / HYDRO_SPAN
    wt_hn = wt_hn.clip(0.0, 1.0)
    mt_hn = mt_hn.clip(0.0, 1.0)

    hydro_loss = np.clip(wt_hn - mt_hn, 0.0, 1.0)  # less hydrophobic mutant
    hydro_gain = np.clip(mt_hn - wt_hn, 0.0, 1.0)  # more hydrophobic mutant
    burial = out["burial_proxy_v10"].astype(float).values

    # Buried sites penalize hydrophobicity loss; exposed sites penalize gain.
    hydro_mismatch = burial * hydro_loss.values + (1.0 - burial) * hydro_gain.values

    charge_sign_flip = (
        np.sign(out["wt_charge"].astype(float).values)
        * np.sign(out["mut_charge"].astype(float).values)
        < 0.0
    ).astype(float)

    term_packing_strain = abs_delta_volume_norm.values * (0.25 + 0.75 * burial)
    term_charge_desolv = (
        abs_delta_charge_norm.values * (0.35 + 0.65 * burial) + 0.25 * charge_sign_flip * burial
    )
    term_hydrophobic_mismatch = hydro_mismatch

    out["term_packing_strain_raw"] = term_packing_strain
    out["term_charge_desolv_raw"] = term_charge_desolv
    out["term_hydrophobic_mismatch_raw"] = term_hydrophobic_mismatch

    out["q_pack"] = target_quantile(out, "term_packing_strain_raw")
    out["q_charge"] = target_quantile(out, "term_charge_desolv_raw")
    out["q_hydro"] = target_quantile(out, "term_hydrophobic_mismatch_raw")

    out["rank_pack"] = rank(out["q_pack"].values)
    out["rank_charge"] = rank(out["q_charge"].values)
    out["rank_hydro"] = rank(out["q_hydro"].values)

    # Confidence-style continuous weighting of mechanism importance.
    w_pack = 0.5 + 0.5 * out["q_pack"].values
    w_charge = 0.5 + 0.5 * out["q_charge"].values
    w_hydro = 0.5 + 0.5 * out["q_hydro"].values
    denom = w_pack + w_charge + w_hydro + 1e-12
    mech_signal = (
        w_pack * out["rank_pack"].values
        + w_charge * out["rank_charge"].values
        + w_hydro * out["rank_hydro"].values
    ) / denom
    out["mech_signal"] = mech_signal
    out["mech_rank"] = rank(out["mech_signal"].values)

    # View disagreement proxy used in v10.1 external-safe gating.
    view_mat = np.vstack(
        [
            out["rankscore_phys"].values.astype(float),
            out["rankscore_struct_base"].values.astype(float),
            out["rankscore_mpnn"].values.astype(float),
        ]
    )
    out["view_dispersion"] = np.std(view_mat, axis=0)
    out["q_view_dispersion"] = target_quantile(out, "view_dispersion")
    return out


def apply_rule_variant(df: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    out = df.copy()
    abs_charge = out["delta_charge"].abs().astype(float).values
    abs_volume = out["delta_volume"].abs().astype(float).values
    burial = out["burial_proxy_v10"].astype(float).values

    focus_charge = (
        (abs_charge >= cfg.charge_cutoff) & (burial >= cfg.burial_cutoff_charge)
    ).astype(float)
    focus_packing = (
        (abs_volume >= cfg.volume_cutoff) & (burial >= cfg.burial_cutoff_packing)
    ).astype(float)

    out["focus_charge_flag"] = focus_charge
    out["focus_packing_flag"] = focus_packing
    out["focus_charge_rank"] = rank(focus_charge * out["rank_charge"].values)
    out["focus_packing_rank"] = rank(focus_packing * out["rank_pack"].values)

    struct_aug = (
        out["rankscore_struct_base"].values
        + cfg.lambda_mech * out["mech_rank"].values
        + cfg.lambda_focus_charge * out["focus_charge_rank"].values
        + cfg.lambda_focus_packing * out["focus_packing_rank"].values
    )
    out["rankscore_struct_v10"] = rank(struct_aug)
    out["rankscore_3view_v10"] = (
        out["rankscore_phys"].values
        + out["rankscore_struct_v10"].values
        + out["rankscore_mpnn"].values
    )
    return out


def external_safe_gate(df: pd.DataFrame, cfg: GuardrailRuleConfig) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic guardrails to avoid over-activation on external cohorts."""
    abs_charge = df["delta_charge"].abs().astype(float).values
    abs_volume = df["delta_volume"].abs().astype(float).values
    burial = df["burial_proxy_v10"].astype(float).values
    q_disp = df["q_view_dispersion"].astype(float).values

    safe_charge = (
        (abs_charge >= cfg.charge_cutoff)
        & (burial >= cfg.burial_cutoff_charge)
        & (abs_volume <= cfg.charge_volume_cap)
        & (q_disp <= cfg.max_view_disp_for_focus_charge)
    ).astype(float)
    safe_packing = (
        (abs_volume >= cfg.volume_cutoff)
        & (burial >= cfg.burial_cutoff_packing)
        & (q_disp <= cfg.max_view_disp_for_focus_packing)
    ).astype(float)
    return safe_charge, safe_packing


def shrinkage_factor(df: pd.DataFrame, cfg: GuardrailRuleConfig) -> np.ndarray:
    """Continuous shrinkage for structural augmentation magnitude."""
    q_disp = df["q_view_dispersion"].astype(float).values
    q_risk = np.maximum(df["q_charge"].astype(float).values, df["q_pack"].astype(float).values)
    raw = cfg.shrink_base - cfg.shrink_disp_coeff * q_disp - cfg.shrink_risk_coeff * q_risk
    return np.clip(raw, cfg.shrink_min, cfg.shrink_max)


def apply_guardrail_rule_variant(df: pd.DataFrame, cfg: GuardrailRuleConfig) -> pd.DataFrame:
    """Apply v10.1 guarded deterministic rules."""
    out = df.copy()
    safe_charge, safe_packing = external_safe_gate(out, cfg)
    shrink = shrinkage_factor(out, cfg)

    out["safe_focus_charge_flag"] = safe_charge
    out["safe_focus_packing_flag"] = safe_packing
    out["shrink_factor"] = shrink

    out["safe_focus_charge_rank"] = rank(safe_charge * out["rank_charge"].values)
    out["safe_focus_packing_rank"] = rank(safe_packing * out["rank_pack"].values)

    rule_delta = (
        cfg.lambda_mech * out["mech_rank"].values
        + cfg.lambda_focus_charge * out["safe_focus_charge_rank"].values
        + cfg.lambda_focus_packing * out["safe_focus_packing_rank"].values
    )
    struct_aug = out["rankscore_struct_base"].values + shrink * rule_delta

    out["rankscore_struct_v10_1"] = rank(struct_aug)
    out["rankscore_3view_v10_1"] = (
        out["rankscore_phys"].values
        + out["rankscore_struct_v10_1"].values
        + out["rankscore_mpnn"].values
    )
    return out


def regime_metrics(
    df: pd.DataFrame,
    pred_col: str,
    base_col: str = "rankscore_3view_base",
    neutral_thr: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict] = []
    regimes = [
        ("stabilizing", df["ddg_exp"] < 0.0),
        ("near_neutral", df["ddg_exp"].abs() < neutral_thr),
        ("destabilizing", df["ddg_exp"] > 0.0),
    ]
    for name, mask in regimes:
        sub = df.loc[mask].copy()
        if len(sub) < 5:
            continue
        rho_base = safe_spearman(sub["ddg_exp"].values, sub[base_col].values)
        rho_new = safe_spearman(sub["ddg_exp"].values, sub[pred_col].values)
        rows.append(
            {
                "regime": name,
                "n": int(len(sub)),
                "rho_base": float(rho_base),
                "rho_new": float(rho_new),
                "delta": float(rho_new - rho_base),
            }
        )
    return pd.DataFrame(rows)


def per_target_delta(
    df: pd.DataFrame,
    pred_col: str,
    base_col: str = "rankscore_3view_base",
) -> pd.DataFrame:
    rows: list[dict] = []
    for target, sub in df.groupby("target"):
        if len(sub) < 5:
            continue
        rho_base = safe_spearman(sub["ddg_exp"].values, sub[base_col].values)
        rho_new = safe_spearman(sub["ddg_exp"].values, sub[pred_col].values)
        rows.append(
            {
                "target": target,
                "n": int(len(sub)),
                "rho_base": float(rho_base),
                "rho_new": float(rho_new),
                "delta": float(rho_new - rho_base),
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def bootstrap_delta(
    df: pd.DataFrame,
    pred_new_col: str,
    pred_base_col: str = "rankscore_3view_base",
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    targets = np.array(sorted(df["target"].unique()))
    idx_map = {t: np.where(df["target"].values == t)[0] for t in targets}
    y = df["ddg_exp"].values.astype(float)
    base = df[pred_base_col].values.astype(float)
    new = df[pred_new_col].values.astype(float)

    rows: list[dict] = []
    for i in range(n_bootstrap):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        idx = np.concatenate([idx_map[t] for t in sampled])
        rb = safe_spearman(y[idx], base[idx])
        rn = safe_spearman(y[idx], new[idx])
        if np.isnan(rb) or np.isnan(rn):
            continue
        rows.append({"iter": i, "rho_base": rb, "rho_new": rn, "delta": rn - rb})
    return pd.DataFrame(rows)
