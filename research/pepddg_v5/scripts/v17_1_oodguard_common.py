#!/usr/bin/env python
"""Common utilities for PepDDG v17.1 OOD guard (strict clean-3)."""

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
class V171Config:
    name: str = "v17_1_oodguard"
    col_v17: str = "rankscore_3view_v17_mainboost"
    col_v16: str = "rankscore_3view_v16"
    col_v13: str = "rankscore_3view_v13_v13_cons"
    # Safe anchor mixture selected from Pareto search under main>=0.68.
    safe_w_v16: float = 0.397128935426064
    safe_w_v13: float = 0.33782555817991144
    safe_w_base: float = 0.2650455063940247
    gate_w0: float = 0.7505778698235728
    gate_w_charge_buried: float = -3.8789803402045453
    gate_w_vol_buried: float = 1.2772915378739305
    gate_w_q_risk: float = -2.022865296149624
    gate_w_q_disp: float = 1.9321835205840183
    gate_w_shift: float = 1.1967328762523521
    gate_temp: float = 0.19588352917630047


def cfg_to_dict(cfg: V171Config) -> dict:
    return asdict(cfg)


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).values


def _safe_numeric(df: pd.DataFrame, col: str, *, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    fill = float(x.median()) if x.notna().any() else float(default)
    return x.fillna(fill).values.astype(float)


def get_v171_used_columns() -> list[str]:
    return sorted(
        set(
            [
                "rankscore_3view_v17_mainboost",
                "rankscore_3view_v16",
                "rankscore_3view_v13_v13_cons",
                "rankscore_3view_base",
                "delta_charge",
                "delta_volume",
                "abs_charge",
                "burial_proxy_v10",
                "q_risk",
                "q_view_dispersion",
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
    used = get_v171_used_columns() if used_columns is None else sorted(set(used_columns))
    used_viol = violates_denylist(used)
    input_viol = violates_denylist(input_columns or [])
    return {
        "used_columns": used,
        "denylist_patterns": DENYLIST_PATTERNS,
        "used_columns_violations": used_viol,
        "input_columns_with_banned_patterns": input_viol,
        "pass": len(used_viol) == 0,
    }


def apply_v171_variant(df: pd.DataFrame, cfg: V171Config) -> pd.DataFrame:
    out = df.copy()
    for col in [cfg.col_v17, cfg.col_v16, cfg.col_v13]:
        if col not in out.columns:
            raise ValueError(f"Missing required score column for v17.1: {col}")

    v17 = _rank(_safe_numeric(out, cfg.col_v17))
    v16 = _rank(_safe_numeric(out, cfg.col_v16))
    v13 = _rank(_safe_numeric(out, cfg.col_v13))

    abs_charge = _safe_numeric(out, "abs_charge", default=np.nan)
    if np.isnan(abs_charge).all():
        abs_charge = np.abs(_safe_numeric(out, "delta_charge"))
    abs_volume = np.abs(_safe_numeric(out, "delta_volume"))
    burial = np.clip(_safe_numeric(out, "burial_proxy_v10", default=0.5), 0.0, 1.0)
    q_risk = _safe_numeric(out, "q_risk", default=0.5)
    q_disp = _safe_numeric(out, "q_view_dispersion", default=0.5)
    shift = np.abs(v17 - v16)

    z_charge_buried = _rank(abs_charge * burial)
    z_vol_buried = _rank(abs_volume * burial)
    z_q_risk = _rank(q_risk)
    z_q_disp = _rank(q_disp)
    z_shift = _rank(shift)

    gate_logit = (
        cfg.gate_w0
        + cfg.gate_w_charge_buried * z_charge_buried
        + cfg.gate_w_vol_buried * z_vol_buried
        + cfg.gate_w_q_risk * z_q_risk
        + cfg.gate_w_q_disp * z_q_disp
        + cfg.gate_w_shift * z_shift
    )
    gate_temp = max(float(cfg.gate_temp), 1e-6)
    alpha = 1.0 / (1.0 + np.exp(-gate_logit / gate_temp))

    base = _rank(_safe_numeric(out, "rankscore_3view_base"))
    wsum = max(float(cfg.safe_w_v16 + cfg.safe_w_v13 + cfg.safe_w_base), 1e-12)
    safe = (
        (cfg.safe_w_v16 / wsum) * v16
        + (cfg.safe_w_v13 / wsum) * v13
        + (cfg.safe_w_base / wsum) * base
    )
    pred = v17 + alpha * (safe - v17)

    out["v171_z_charge_buried"] = z_charge_buried
    out["v171_z_vol_buried"] = z_vol_buried
    out["v171_z_q_risk"] = z_q_risk
    out["v171_z_q_disp"] = z_q_disp
    out["v171_z_shift"] = z_shift
    out["v171_gate_alpha"] = alpha
    out["rankscore_3view_v17_1_oodguard_raw"] = pred
    out["rankscore_3view_v17_1_oodguard"] = _rank(pred)
    return out
