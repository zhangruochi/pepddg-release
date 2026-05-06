#!/usr/bin/env python
"""Common utilities for PepDDG v15 clean strict 3-channel fusion."""

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

# Allowed sub-signals inside the three channels.
# Each entry is a fallback chain for one conceptual subcomponent.
PHYS_COMPONENT_GROUPS = [
    ["ddg_paired_dual_blend_iface", "ddg_dual_blend_iface"],
    ["ddg_paired_xint_iface", "ddg_xint_iface"],
    ["ddg_paired_bind_proxy", "ddg_bind_proxy"],
]
STRUCT_COMPONENT_GROUPS = [
    ["struct_composite", "rankscore_struct_base", "rankscore_struct"],
    ["n_iface_contacts_8a"],
    ["n_neighbors_10a"],
    ["rank_masif", "surface_anchor_raw"],
]
MPNN_COMPONENT_GROUPS = [
    ["rankscore_mpnn", "mpnn_view_score"],
    ["mpnn_ddg_bind"],
    ["mpnn_neg_llr_complex"],
]
RISK_INPUT_COLS = [
    "delta_charge",
    "delta_volume",
    "q_view_dispersion",
]
BASELINE_COL = "rankscore_3view_base"
TARGET_COL = "target"


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).values


def _target_rank(df: pd.DataFrame, values: np.ndarray, target_col: str = TARGET_COL) -> np.ndarray:
    if target_col not in df.columns:
        return _rank(values)
    out = pd.Series(index=df.index, dtype=float)
    tmp = df[[target_col]].copy()
    tmp["_value"] = np.asarray(values, dtype=float)
    for _, g in tmp.groupby(target_col):
        out.loc[g.index] = _rank(g["_value"].values)
    return out.values


def _winsorize(values: np.ndarray, low_q: float = 0.10, high_q: float = 0.90) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(np.quantile(arr, low_q))
    hi = float(np.quantile(arr, high_q))
    return np.clip(arr, lo, hi)


def _std01(matrix: np.ndarray) -> np.ndarray:
    # Std of rank-like values in [0, 1], normalized by theoretical max std 0.5.
    s = np.std(matrix, axis=1)
    return np.clip(2.0 * s, 0.0, 1.0)


def _safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    x = pd.to_numeric(df[col], errors="coerce")
    med = float(x.median()) if x.notna().any() else 0.0
    return x.fillna(med).values.astype(float)


def _resolve_struct_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise ValueError(f"Missing structural component columns: {preferred}/{fallback}")


def get_v15_used_columns() -> list[str]:
    phys = [c for group in PHYS_COMPONENT_GROUPS for c in group]
    struct = [c for group in STRUCT_COMPONENT_GROUPS for c in group]
    mpnn = [c for group in MPNN_COMPONENT_GROUPS for c in group]
    return sorted(
        set(
            phys
            + struct
            + mpnn
            + RISK_INPUT_COLS
            + [BASELINE_COL, TARGET_COL]
        )
    )


def violates_denylist(columns: Iterable[str], patterns: list[str] | None = None) -> list[str]:
    pats = DENYLIST_PATTERNS if patterns is None else patterns
    bad: list[str] = []
    for c in columns:
        cl = c.lower()
        if any(re.search(p, cl) for p in pats):
            bad.append(c)
    return sorted(set(bad))


def strict3_policy_audit(
    *,
    used_columns: list[str] | None = None,
    input_columns: list[str] | None = None,
) -> dict:
    used = get_v15_used_columns() if used_columns is None else sorted(set(used_columns))
    used_violations = violates_denylist(used)
    input_bad = violates_denylist(input_columns or [])
    return {
        "used_columns": used,
        "denylist_patterns": DENYLIST_PATTERNS,
        "used_columns_violations": used_violations,
        "input_columns_with_banned_patterns": input_bad,
        "pass": len(used_violations) == 0,
    }


@dataclass(frozen=True)
class V15Clean3Config:
    name: str
    channel_winsor_low_q: float = 0.10
    channel_winsor_high_q: float = 0.90
    per_target_mix: float = 0.50
    use_interactions: bool = False
    use_per_target_fusion: bool = False


DEFAULT_V15_CONFIG = V15Clean3Config(name="v15_clean3_cab3_safe")
V15_R2_CONFIG = V15Clean3Config(name="v15_clean3_r2_interactions", use_interactions=True)
V15_R3_CONFIG = V15Clean3Config(
    name="v15_clean3_r3_dualscale",
    use_interactions=True,
    use_per_target_fusion=True,
    per_target_mix=0.50,
)


def _channel_consensus_score(channel_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate channel-internal components with robust consensus:
      score = rank(mean(winsorized_components) * (1 - normalized_std)).
    """
    win = np.vstack([_winsorize(channel_matrix[:, i]) for i in range(channel_matrix.shape[1])]).T
    mean_term = np.mean(win, axis=1)
    std_term = _std01(channel_matrix)
    raw = mean_term * (1.0 - std_term)
    return _rank(raw), std_term


def _resolve_component_columns(
    df: pd.DataFrame,
    groups: list[list[str]],
    *,
    min_required: int,
    channel_name: str,
) -> list[str]:
    selected: list[str] = []
    for chain in groups:
        found = next((c for c in chain if c in df.columns), None)
        if found is not None:
            selected.append(found)
    if len(selected) < min_required:
        raise ValueError(
            f"{channel_name} channel has only {len(selected)} resolved components; "
            f"need at least {min_required}. available={list(df.columns)}"
        )
    return selected


def build_channel_scores_v15(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = RISK_INPUT_COLS + [BASELINE_COL]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for v15: {missing}")

    phys_cols = _resolve_component_columns(out, PHYS_COMPONENT_GROUPS, min_required=2, channel_name="phys")
    struct_cols = _resolve_component_columns(out, STRUCT_COMPONENT_GROUPS, min_required=1, channel_name="struct")
    mpnn_cols = _resolve_component_columns(out, MPNN_COMPONENT_GROUPS, min_required=2, channel_name="mpnn")

    phys_matrix = np.vstack([_rank(_safe_col(out, c)) for c in phys_cols]).T
    struct_matrix = np.vstack([_rank(_safe_col(out, c)) for c in struct_cols]).T
    mpnn_matrix = np.vstack([_rank(_safe_col(out, c)) for c in mpnn_cols]).T

    u_phys, c_phys = _channel_consensus_score(phys_matrix)
    u_struct, c_struct = _channel_consensus_score(struct_matrix)
    u_mpnn, c_mpnn = _channel_consensus_score(mpnn_matrix)

    out["u_phys_v15"] = u_phys
    out["u_struct_v15"] = u_struct
    out["u_mpnn_v15"] = u_mpnn
    out["u_phys_consensus_std_v15"] = c_phys
    out["u_struct_consensus_std_v15"] = c_struct
    out["u_mpnn_consensus_std_v15"] = c_mpnn
    out["n_phys_components_v15"] = float(phys_matrix.shape[1])
    out["n_struct_components_v15"] = float(struct_matrix.shape[1])
    out["n_mpnn_components_v15"] = float(mpnn_matrix.shape[1])
    return out


def cab3_safe_blend(raw: np.ndarray, base: np.ndarray, risk: np.ndarray, d: np.ndarray) -> np.ndarray:
    shrink = np.clip(np.asarray(risk, dtype=float) * np.asarray(d, dtype=float), 0.0, 1.0)
    return (1.0 - shrink) * np.asarray(raw, dtype=float) + shrink * np.asarray(base, dtype=float)


def _risk_score(df: pd.DataFrame) -> np.ndarray:
    q_charge = _rank(np.abs(_safe_col(df, "delta_charge")))
    q_volume = _rank(np.abs(_safe_col(df, "delta_volume")))
    q_disp = _rank(_safe_col(df, "q_view_dispersion"))
    return np.clip((q_charge + q_volume + q_disp) / 3.0, 0.0, 1.0)


def _cab3_core(
    df: pd.DataFrame,
    *,
    use_interactions: bool,
    use_per_target_fusion: bool,
    per_target_mix: float,
) -> pd.DataFrame:
    out = build_channel_scores_v15(df)
    u_phys = out["u_phys_v15"].astype(float).values
    u_struct = out["u_struct_v15"].astype(float).values
    u_mpnn = out["u_mpnn_v15"].astype(float).values
    base = _rank(out[BASELINE_COL].astype(float).values)

    parts = [u_phys, u_struct, u_mpnn]
    if use_interactions:
        parts.extend(
            [
                _rank(u_phys * u_struct),
                _rank(u_phys * u_mpnn),
                _rank(u_struct * u_mpnn),
            ]
        )
    part_matrix = np.vstack(parts).T

    b = np.mean(part_matrix, axis=1)
    m = np.median(part_matrix, axis=1)
    d = _std01(part_matrix)
    raw = (1.0 - d) * b + d * m
    risk = _risk_score(out)
    final_global = cab3_safe_blend(raw=raw, base=base, risk=risk, d=d)

    out["cab3_b_v15"] = b
    out["cab3_m_v15"] = m
    out["cab3_d_v15"] = d
    out["cab3_risk_v15"] = risk
    out["cab3_raw_v15"] = raw
    out["cab3_final_global_v15"] = final_global

    if use_per_target_fusion:
        b_t = _target_rank(out, b)
        m_t = _target_rank(out, m)
        d_t = _target_rank(out, d)
        raw_t = (1.0 - d_t) * b_t + d_t * m_t
        base_t = _target_rank(out, out[BASELINE_COL].astype(float).values)
        risk_t = _target_rank(out, risk)
        final_t = cab3_safe_blend(raw=raw_t, base=base_t, risk=risk_t, d=d_t)
        final_mix = (1.0 - per_target_mix) * _rank(final_global) + per_target_mix * _rank(final_t)
        out["cab3_final_target_v15"] = final_t
        out["cab3_final_mix_v15"] = final_mix
        out["rankscore_3view_v15"] = _rank(final_mix)
    else:
        out["rankscore_3view_v15"] = _rank(final_global)
    return out


def apply_v15_variant(df: pd.DataFrame, cfg: V15Clean3Config) -> pd.DataFrame:
    return _cab3_core(
        df,
        use_interactions=cfg.use_interactions,
        use_per_target_fusion=cfg.use_per_target_fusion,
        per_target_mix=cfg.per_target_mix,
    )


def safe_spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    rho, _ = spearmanr(y, yhat)
    return float(rho) if not np.isnan(rho) else np.nan


def eval_prediction(df: pd.DataFrame, pred_col: str) -> dict:
    y = df["ddg_exp"].astype(float).values
    base = df[BASELINE_COL].astype(float).values
    pred = df[pred_col].astype(float).values
    rho_base = safe_spearman(y, base)
    rho_new = safe_spearman(y, pred)
    return {"rho_base": float(rho_base), "rho_new": float(rho_new), "delta": float(rho_new - rho_base)}


def cfg_to_dict(cfg: V15Clean3Config) -> dict:
    return asdict(cfg)


__all__ = [
    "BASELINE_COL",
    "DEFAULT_V15_CONFIG",
    "DENYLIST_PATTERNS",
    "TARGET_COL",
    "V15Clean3Config",
    "V15_R2_CONFIG",
    "V15_R3_CONFIG",
    "apply_v15_variant",
    "build_channel_scores_v15",
    "cab3_safe_blend",
    "cfg_to_dict",
    "eval_prediction",
    "get_v15_used_columns",
    "strict3_policy_audit",
    "violates_denylist",
]
