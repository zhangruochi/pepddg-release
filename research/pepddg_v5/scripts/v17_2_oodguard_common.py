#!/usr/bin/env python
"""Common utilities for PepDDG v17.2 OOD guard (strict clean-3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd


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


@dataclass(frozen=True)
class V172Config:
    name: str = "v17_2_oodguard"
    col_v17: str = "rankscore_3view_v17_mainboost"
    col_v16: str = "rankscore_3view_v16"
    col_v13: str = "rankscore_3view_v13_v13_cons"
    col_base: str = "rankscore_3view_base"

    # Softmax-safe anchor logits from long random search best feasible candidate.
    safe_logit_v16: float = -0.5551183562233368
    safe_logit_v13: float = -0.8155267324112118
    safe_logit_base: float = 0.8054099533563269

    # Gate parameters (feature order documented in apply_v172_variant).
    gate_b0: float = 3.0843185438077176
    gate_w_charge_buried: float = -5.738150176283619
    gate_w_vol_buried: float = 2.6600012887543
    gate_w_q_risk: float = 0.21158950011311006
    gate_w_q_disp: float = 4.528080729342378
    gate_w_shift16: float = -4.940554937592532
    gate_w_shift13: float = 5.780460033214702
    gate_w_shift0: float = -2.240742669411386
    gate_w_combo1: float = -0.7228019549598219
    gate_w_combo2: float = -2.517578000608598
    gate_temp: float = 0.16113318201154941
    gate_gamma: float = 1.3391373849206896
    gate_power: float = 1.0411515337175392
    gate_alpha_floor: float = 0.11369737979774718


def cfg_to_dict(cfg: V172Config) -> dict:
    return asdict(cfg)


def _rank(values: np.ndarray) -> np.ndarray:
    """Stable rank in [0,1] using mergesort order (no tie averaging)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(n, dtype=float)
    if n == 1:
        out[order] = 0.0
    else:
        out[order] = np.arange(n, dtype=float) / float(n - 1)
    return out


def _safe_numeric(df: pd.DataFrame, col: str, *, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    fill = float(x.median()) if x.notna().any() else float(default)
    return x.fillna(fill).values.astype(float)


def _softmax3(a: float, b: float, c: float) -> tuple[float, float, float]:
    arr = np.array([a, b, c], dtype=float)
    arr = arr - float(np.max(arr))
    ex = np.exp(arr)
    ex = ex / float(np.sum(ex))
    return float(ex[0]), float(ex[1]), float(ex[2])


def get_v172_used_columns() -> list[str]:
    return sorted(
        set(
            [
                "rankscore_3view_v17_mainboost",
                "rankscore_3view_v16",
                "rankscore_3view_v13_v13_cons",
                "rankscore_3view_v13",
                "rankscore_3view_base",
                "delta_charge",
                "delta_volume",
                "abs_charge",
                "burial_proxy_v10",
                "q_risk",
                "q_view_dispersion",
                "v17_term_charge_buried",
                "v17_term_vol_buried",
                "charge_buried",
                "vol_buried",
                "risk_charge_buried",
                "risk_large_volume",
            ]
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
    used = get_v172_used_columns() if used_columns is None else sorted(set(used_columns))
    used_viol = violates_denylist(used)
    input_viol = violates_denylist(input_columns or [])
    return {
        "used_columns": used,
        "denylist_patterns": DENYLIST_PATTERNS,
        "used_columns_violations": used_viol,
        "input_columns_with_banned_patterns": input_viol,
        "pass": len(used_viol) == 0,
    }


def _get_charge_buried(df: pd.DataFrame) -> np.ndarray:
    for col in ["v17_term_charge_buried", "charge_buried", "risk_charge_buried"]:
        if col in df.columns:
            return _safe_numeric(df, col, default=0.0)
    abs_charge = _safe_numeric(df, "abs_charge", default=np.nan)
    if np.isnan(abs_charge).all():
        abs_charge = np.abs(_safe_numeric(df, "delta_charge", default=0.0))
    burial = np.clip(_safe_numeric(df, "burial_proxy_v10", default=0.5), 0.0, 1.0)
    return abs_charge * burial


def _get_vol_buried(df: pd.DataFrame) -> np.ndarray:
    for col in ["v17_term_vol_buried", "vol_buried", "risk_large_volume"]:
        if col in df.columns:
            return _safe_numeric(df, col, default=0.0)
    abs_volume = np.abs(_safe_numeric(df, "delta_volume", default=0.0))
    burial = np.clip(_safe_numeric(df, "burial_proxy_v10", default=0.5), 0.0, 1.0)
    return abs_volume * burial


def apply_v172_variant(df: pd.DataFrame, cfg: V172Config) -> pd.DataFrame:
    out = df.copy()
    v13_col = cfg.col_v13 if cfg.col_v13 in out.columns else "rankscore_3view_v13"
    for col in [cfg.col_v17, cfg.col_v16, v13_col, cfg.col_base]:
        if col not in out.columns:
            raise ValueError(f"Missing required score column for v17.2: {col}")

    v17 = _rank(_safe_numeric(out, cfg.col_v17))
    v16 = _rank(_safe_numeric(out, cfg.col_v16))
    v13 = _rank(_safe_numeric(out, v13_col))
    base = _rank(_safe_numeric(out, cfg.col_base))

    z_charge_buried = _rank(_get_charge_buried(out))
    z_vol_buried = _rank(_get_vol_buried(out))
    z_q_risk = _rank(_safe_numeric(out, "q_risk", default=0.5))
    z_q_disp = _rank(_safe_numeric(out, "q_view_dispersion", default=0.5))
    z_shift16 = _rank(np.abs(v17 - v16))
    z_shift13 = _rank(np.abs(v17 - v13))
    z_shift0 = _rank(np.abs(v17 - base))
    # Must match v17.2 random-search definition: combo terms use ranked inputs.
    z_combo1 = _rank(z_charge_buried * z_shift0)
    z_combo2 = _rank(z_q_disp * z_shift16)

    gate_logit = (
        cfg.gate_b0
        + cfg.gate_w_charge_buried * z_charge_buried
        + cfg.gate_w_vol_buried * z_vol_buried
        + cfg.gate_w_q_risk * z_q_risk
        + cfg.gate_w_q_disp * z_q_disp
        + cfg.gate_w_shift16 * z_shift16
        + cfg.gate_w_shift13 * z_shift13
        + cfg.gate_w_shift0 * z_shift0
        + cfg.gate_w_combo1 * z_combo1
        + cfg.gate_w_combo2 * z_combo2
    )
    gate_temp = max(float(cfg.gate_temp), 1e-6)
    alpha0 = 1.0 / (1.0 + np.exp(-gate_logit / gate_temp))
    alpha = np.clip(
        cfg.gate_gamma * (alpha0 ** float(cfg.gate_power)) + cfg.gate_alpha_floor,
        0.0,
        1.0,
    )

    w16, w13, wb = _softmax3(cfg.safe_logit_v16, cfg.safe_logit_v13, cfg.safe_logit_base)
    safe = w16 * v16 + w13 * v13 + wb * base
    pred = v17 + alpha * (safe - v17)

    out["v172_z_charge_buried"] = z_charge_buried
    out["v172_z_vol_buried"] = z_vol_buried
    out["v172_z_q_risk"] = z_q_risk
    out["v172_z_q_disp"] = z_q_disp
    out["v172_z_shift16"] = z_shift16
    out["v172_z_shift13"] = z_shift13
    out["v172_z_shift0"] = z_shift0
    out["v172_z_combo1"] = z_combo1
    out["v172_z_combo2"] = z_combo2
    out["v172_gate_alpha0"] = alpha0
    out["v172_gate_alpha"] = alpha
    out["v172_safe_w_v16"] = w16
    out["v172_safe_w_v13"] = w13
    out["v172_safe_w_base"] = wb
    out["rankscore_3view_v17_2_oodguard_raw"] = pred
    out["rankscore_3view_v17_2_oodguard"] = _rank(pred)
    return out
