#!/usr/bin/env python
"""Common utilities for PepDDG v17 main-rho boost (strict clean-3)."""

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
class V17Config:
    name: str = "v17_mainboost"
    core_col: str = "rankscore_3view_v16"
    w_core: float = 1.268710425994338
    w_absdiff_charge_dual: float = -0.19530748535016507
    w_dual_x_mpnn: float = -0.4007241782905294
    w_absdiff_bind_qcharge: float = -0.1845918407008933
    w_absdiff_mech_phys: float = 0.2394070761715098
    w_vol_buried: float = 0.062117041275082996
    w_charge_buried: float = 0.18349727292705487


def cfg_to_dict(cfg: V17Config) -> dict:
    return asdict(cfg)


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).values


def _safe_numeric(df: pd.DataFrame, col: str, *, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    fill = float(x.median()) if x.notna().any() else float(default)
    return x.fillna(fill).values.astype(float)


def _resolve_first(df: pd.DataFrame, choices: list[str], *, default: float = 0.0) -> np.ndarray:
    for c in choices:
        if c in df.columns:
            return _safe_numeric(df, c, default=default)
    return np.full(len(df), float(default), dtype=float)


def get_v17_used_columns() -> list[str]:
    return sorted(
        set(
            [
                "rankscore_3view_v16",
                "abs_charge",
                "delta_charge",
                "ddg_dual_blend_iface",
                "ddg_paired_dual_blend_iface",
                "ddg_bind_proxy",
                "ddg_paired_bind_proxy",
                "q_charge",
                "mpnn_ddg_bind",
                "mech_signal",
                "phys_view_score",
                "delta_volume",
                "burial_proxy_v10",
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
    used = get_v17_used_columns() if used_columns is None else sorted(set(used_columns))
    used_viol = violates_denylist(used)
    input_viol = violates_denylist(input_columns or [])
    return {
        "used_columns": used,
        "denylist_patterns": DENYLIST_PATTERNS,
        "used_columns_violations": used_viol,
        "input_columns_with_banned_patterns": input_viol,
        "pass": len(used_viol) == 0,
    }


def apply_v17_variant(df: pd.DataFrame, cfg: V17Config) -> pd.DataFrame:
    out = df.copy()
    if cfg.core_col not in out.columns:
        raise ValueError(f"Missing core column for v17: {cfg.core_col}")

    core = _rank(_safe_numeric(out, cfg.core_col))
    abs_charge = _safe_numeric(out, "abs_charge", default=np.nan)
    if np.isnan(abs_charge).all():
        abs_charge = np.abs(_safe_numeric(out, "delta_charge"))

    dual = _resolve_first(out, ["ddg_dual_blend_iface", "ddg_paired_dual_blend_iface"])
    bind = _resolve_first(out, ["ddg_bind_proxy", "ddg_paired_bind_proxy"])
    q_charge = _safe_numeric(out, "q_charge")
    mpnn_bind = _safe_numeric(out, "mpnn_ddg_bind")
    mech_signal = _safe_numeric(out, "mech_signal")
    phys_signal = _safe_numeric(out, "phys_view_score")
    burial = np.clip(_safe_numeric(out, "burial_proxy_v10", default=0.5), 0.0, 1.0)
    dvol = np.abs(_safe_numeric(out, "delta_volume"))
    dchg = np.abs(_safe_numeric(out, "delta_charge"))

    t_absdiff_charge_dual = _rank(np.abs(abs_charge - dual))
    t_dual_x_mpnn = _rank(dual * mpnn_bind)
    t_absdiff_bind_qcharge = _rank(np.abs(bind - q_charge))
    t_absdiff_mech_phys = _rank(np.abs(mech_signal - phys_signal))
    t_vol_buried = _rank(dvol * burial)
    t_charge_buried = _rank(dchg * burial)

    raw = (
        cfg.w_core * core
        + cfg.w_absdiff_charge_dual * t_absdiff_charge_dual
        + cfg.w_dual_x_mpnn * t_dual_x_mpnn
        + cfg.w_absdiff_bind_qcharge * t_absdiff_bind_qcharge
        + cfg.w_absdiff_mech_phys * t_absdiff_mech_phys
        + cfg.w_vol_buried * t_vol_buried
        + cfg.w_charge_buried * t_charge_buried
    )
    out["v17_term_absdiff_charge_dual"] = t_absdiff_charge_dual
    out["v17_term_dual_x_mpnn"] = t_dual_x_mpnn
    out["v17_term_absdiff_bind_qcharge"] = t_absdiff_bind_qcharge
    out["v17_term_absdiff_mech_phys"] = t_absdiff_mech_phys
    out["v17_term_vol_buried"] = t_vol_buried
    out["v17_term_charge_buried"] = t_charge_buried
    out["rankscore_3view_v17_mainboost_raw"] = raw
    out["rankscore_3view_v17_mainboost"] = _rank(raw)
    return out

