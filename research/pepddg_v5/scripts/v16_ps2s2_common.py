#!/usr/bin/env python
"""Common utilities for PepDDG v16: Physical 2.0 + Structure 2.0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DENYLIST_PATTERNS = [
    r"foldx",
    r"rosetta",
    r"cartddg",
    r"stabddg",
    r"esm2",
    r"esmif",
    r"esm3",
    r"saprot",
    r"diffaffinity",
]

HIST_PHYS_COLS = [
    "ddg_xint_iface_total",
    "ddg_xint_local_screened",
    "ddg_xint_iface_heavy",
]


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).values


def _target_rank(df: pd.DataFrame, values: np.ndarray, target_col: str = "target") -> np.ndarray:
    if target_col not in df.columns:
        return _rank(values)
    out = pd.Series(index=df.index, dtype=float)
    tmp = df[[target_col]].copy()
    tmp["_v"] = np.asarray(values, dtype=float)
    for _, g in tmp.groupby(target_col):
        out.loc[g.index] = _rank(g["_v"].values)
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


def _resolve_first(df: pd.DataFrame, choices: list[str], *, name: str) -> str:
    for c in choices:
        if c in df.columns:
            return c
    raise ValueError(f"Missing {name} columns; tried {choices}")


def _resolve_optional(df: pd.DataFrame, choices: list[str]) -> str | None:
    for c in choices:
        if c in df.columns:
            return c
    return None


def _safe_numeric(df: pd.DataFrame, col: str, *, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    fill = float(x.median()) if x.notna().any() else float(default)
    return x.fillna(fill).values.astype(float)


def _safe_binary(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    return x.notna().astype(float).values


def _col_rank(df: pd.DataFrame, col: str, *, targetwise: bool = True, abs_value: bool = False) -> np.ndarray:
    arr = _safe_numeric(df, col)
    if abs_value:
        arr = np.abs(arr)
    return _target_rank(df, arr) if targetwise else _rank(arr)


def get_v16_used_columns() -> list[str]:
    return sorted(
        set(
            [
                "ddg_paired_dual_blend_iface",
                "ddg_dual_blend_iface",
                "ddg_paired_xint_iface",
                "ddg_xint_iface",
                "ddg_paired_bind_proxy",
                "ddg_bind_proxy",
                "mmpbsa_ddg_ala",
                "struct_composite",
                "rankscore_struct_base",
                "rankscore_struct",
                "n_iface_contacts_8a",
                "n_neighbors_10a",
                "term_packing_strain_raw",
                "q_pack",
                "burial_proxy_v10",
                "delta_volume",
                "delta_charge",
                "rank_surface_anchor",
                "rank_masif",
                "surface_anchor_raw",
                "q_view_dispersion",
                "q_risk",
                "q_charge",
                "rankscore_3view_base",
                "rankscore_3view_v13_v13_cons",
                "target",
            ]
            + HIST_PHYS_COLS
        )
    )


def violates_denylist(columns: Iterable[str], patterns: list[str] | None = None) -> list[str]:
    pats = DENYLIST_PATTERNS if patterns is None else patterns
    bad: list[str] = []
    for c in columns:
        if any(re.search(p, c.lower()) for p in pats):
            bad.append(c)
    return sorted(set(bad))


def strict3_policy_audit(
    *,
    used_columns: list[str] | None = None,
    input_columns: list[str] | None = None,
) -> dict:
    used = get_v16_used_columns() if used_columns is None else sorted(set(used_columns))
    used_viol = violates_denylist(used)
    input_viol = violates_denylist(input_columns or [])
    return {
        "used_columns": used,
        "denylist_patterns": DENYLIST_PATTERNS,
        "used_columns_violations": used_viol,
        "input_columns_with_banned_patterns": input_viol,
        "pass": len(used_viol) == 0,
    }


def add_ps2s2_channels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    req = [
        "rankscore_3view_base",
        "delta_charge",
        "delta_volume",
        "q_view_dispersion",
    ]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for v16 channels: {missing}")

    phys_dual_col = _resolve_optional(out, ["ddg_paired_dual_blend_iface", "ddg_dual_blend_iface"])
    phys_iface_col = _resolve_first(out, ["ddg_paired_xint_iface", "ddg_xint_iface"], name="phys iface")
    phys_bind_col = _resolve_first(out, ["ddg_paired_bind_proxy", "ddg_bind_proxy"], name="phys bind")

    p_iface = _col_rank(out, phys_iface_col, targetwise=True)
    p_bind = _col_rank(out, phys_bind_col, targetwise=True)
    if phys_dual_col is not None:
        p_dual = _col_rank(out, phys_dual_col, targetwise=True)
    else:
        # Graceful fallback for cohorts without dual mode: consensus proxy from iface/bind.
        p_dual = _rank((p_iface + p_bind) / 2.0)
    p_mat = np.vstack([p_dual, p_iface, p_bind]).T
    p_mean = np.mean(p_mat, axis=1)
    p_disp = _target_rank(out, np.std(p_mat, axis=1))
    p_tail = _target_rank(out, np.max(np.abs(p_mat - p_mean[:, None]), axis=1))
    p_sign_cons = _target_rank(out, 1.0 - (np.std(np.sign(p_mat - 0.5), axis=1) > 0.0).astype(float))

    if "mmpbsa_ddg_ala" in out.columns:
        mm = pd.to_numeric(out["mmpbsa_ddg_ala"], errors="coerce")
        mm_q = _target_rank(out, mm.abs().fillna(mm.abs().median() if mm.notna().any() else 0.0).values)
        mm_avail = mm.notna().astype(float).values
    else:
        mm_q = np.zeros(len(out), dtype=float)
        mm_avail = np.zeros(len(out), dtype=float)
    out["mmpbsa_available_v16"] = mm_avail

    hist_terms = []
    for c in HIST_PHYS_COLS:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce")
            med = float(x.median()) if x.notna().any() else 0.0
            q = _target_rank(out, x.fillna(med).values)
            hist_terms.append(q * x.notna().astype(float).values)
        else:
            hist_terms.append(np.zeros(len(out), dtype=float))
    p_hist = np.mean(np.vstack(hist_terms).T, axis=1)

    if "term_lme_signal_raw" in out.columns:
        p_lme = _target_rank(out, _safe_numeric(out, "term_lme_signal_raw"))
    else:
        p_lme = p_mean
    if "term_charge_desolv_raw" in out.columns:
        p_charge_desolv = _target_rank(out, np.abs(_safe_numeric(out, "term_charge_desolv_raw")))
    else:
        p_charge_desolv = _target_rank(out, np.abs(_safe_numeric(out, "delta_charge")))

    phys2_raw = (
        0.28 * p_mean
        + 0.12 * (1.0 - p_disp)
        + 0.10 * p_tail
        + 0.10 * p_sign_cons
        + 0.14 * p_lme
        + 0.12 * p_charge_desolv
        + 0.08 * (mm_q * mm_avail)
        + 0.06 * p_hist
    )
    out["phys2_raw_v16"] = phys2_raw
    out["rank_phys2_v16"] = _rank(_target_rank(out, phys2_raw))

    struct_base_col = _resolve_first(out, ["struct_composite", "rankscore_struct_base", "rankscore_struct"], name="struct base")
    q_struct_base = _col_rank(out, struct_base_col, targetwise=True)

    if "n_iface_contacts_8a" in out.columns:
        q_contacts = _col_rank(out, "n_iface_contacts_8a", targetwise=True)
    else:
        q_contacts = q_struct_base
    if "n_neighbors_10a" in out.columns:
        q_neighbors = _col_rank(out, "n_neighbors_10a", targetwise=True)
    else:
        q_neighbors = q_struct_base

    if "term_packing_strain_raw" in out.columns:
        q_pack = _col_rank(out, "term_packing_strain_raw", targetwise=True)
    elif "q_pack" in out.columns:
        q_pack = _col_rank(out, "q_pack", targetwise=True)
    else:
        q_pack = np.zeros(len(out), dtype=float)

    burial = np.clip(_safe_numeric(out, "burial_proxy_v10", default=0.5), 0.0, 1.0)
    q_vol_buried = _target_rank(out, np.abs(_safe_numeric(out, "delta_volume")) * (0.30 + 0.70 * burial))
    q_charge_buried = _target_rank(out, np.abs(_safe_numeric(out, "delta_charge")) * (0.30 + 0.70 * burial))

    if "rank_surface_anchor" in out.columns:
        q_surface = _col_rank(out, "rank_surface_anchor", targetwise=True)
    elif "rank_masif" in out.columns:
        q_surface = _col_rank(out, "rank_masif", targetwise=True)
    elif "surface_anchor_raw" in out.columns:
        q_surface = _col_rank(out, "surface_anchor_raw", targetwise=True)
    else:
        q_surface = np.zeros(len(out), dtype=float)

    if "local_signed_raw" in out.columns:
        q_local = _col_rank(out, "local_signed_raw", targetwise=True)
    elif "rank_local_signed" in out.columns:
        q_local = _col_rank(out, "rank_local_signed", targetwise=True)
    else:
        q_local = q_struct_base

    struct2_raw = (
        0.22 * q_struct_base
        + 0.12 * q_contacts
        + 0.08 * q_neighbors
        + 0.15 * q_pack
        + 0.16 * q_local
        + 0.15 * q_surface
        + 0.07 * q_vol_buried
        + 0.05 * q_charge_buried
    )
    out["struct2_raw_v16"] = struct2_raw
    out["rank_struct2_v16"] = _rank(_target_rank(out, struct2_raw))

    q_view_disp = _col_rank(out, "q_view_dispersion", targetwise=False)
    if "q_risk" in out.columns:
        q_risk = _col_rank(out, "q_risk", targetwise=False)
    else:
        q_risk = np.maximum(_col_rank(out, "delta_charge", abs_value=True), _col_rank(out, "delta_volume", abs_value=True))
    charge_buried = ((np.abs(_safe_numeric(out, "delta_charge")) >= 1.0) & (burial >= 0.60)).astype(float)
    large_volume = (np.abs(_safe_numeric(out, "delta_volume")) >= 85.0).astype(float)

    out["risk_ps2s2_v16"] = np.clip(
        0.40 * q_view_disp + 0.25 * q_risk + 0.20 * charge_buried + 0.15 * large_volume,
        0.0,
        1.0,
    )
    out["charge_buried_flag_v16"] = charge_buried
    out["large_volume_flag_v16"] = large_volume
    return out


def _risk_shrink(risk: np.ndarray, risk_thr: float, shrink_scale: float) -> np.ndarray:
    z = np.clip(risk - risk_thr, 0.0, 1.0)
    return np.clip(shrink_scale * z / (1.0 - risk_thr + 1e-12), 0.0, 1.0)


@dataclass(frozen=True)
class V16Config:
    name: str
    core_col: str
    lambda_phys2: float
    lambda_struct2: float
    shrink_scale: float
    risk_thr: float


def apply_v16_variant(df: pd.DataFrame, cfg: V16Config, *, fallback_col: str = "rankscore_3view_base") -> pd.DataFrame:
    out = df.copy()
    if cfg.core_col not in out.columns:
        raise ValueError(f"Missing core column: {cfg.core_col}")
    if fallback_col not in out.columns:
        raise ValueError(f"Missing fallback column: {fallback_col}")
    if "rank_phys2_v16" not in out.columns or "rank_struct2_v16" not in out.columns:
        out = add_ps2s2_channels(out)

    raw = (
        out[cfg.core_col].astype(float).values
        + cfg.lambda_phys2 * out["rank_phys2_v16"].astype(float).values
        + cfg.lambda_struct2 * out["rank_struct2_v16"].astype(float).values
    )
    shrink = _risk_shrink(out["risk_ps2s2_v16"].astype(float).values, cfg.risk_thr, cfg.shrink_scale)
    pred = (1.0 - shrink) * raw + shrink * out[fallback_col].astype(float).values
    out["pred_raw_v16"] = raw
    out["pred_shrink_v16"] = shrink
    out["rankscore_3view_v16"] = _rank(pred)
    return out


def cfg_to_dict(cfg: V16Config) -> dict:
    return asdict(cfg)


__all__ = [
    "DENYLIST_PATTERNS",
    "HIST_PHYS_COLS",
    "V16Config",
    "add_ps2s2_channels",
    "apply_v16_variant",
    "cfg_to_dict",
    "eval_prediction",
    "get_v16_used_columns",
    "safe_spearman",
    "strict3_policy_audit",
    "violates_denylist",
]
