#!/usr/bin/env python
"""Common utilities for PepDDG v13 elegant deterministic fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rank(method="average", pct=True).values


def safe_spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho, _ = spearmanr(y, yhat)
    return float(rho) if not np.isnan(rho) else np.nan


@dataclass(frozen=True)
class V13ElegantConfig:
    """Deterministic v13 config: trust-index + continuous tri-expert fusion."""

    name: str

    # Trust positive terms.
    w_disp: float
    w_risk: float
    w_local: float
    w_burial: float

    # Trust penalties (including two new physical primitives).
    u_exposed_pack: float
    u_disp_pack: float
    u_charge_buried: float
    u_large_volume: float

    bias: float

    # Continuous triage thresholds.
    tau_main: float
    tau_safe: float

    # Additional trust down-shift on high-risk combinations.
    combo_boost: float

    # Endpoint normalization for cross-variant comparability.
    endpoint_rank_normalize: bool = True


def cfg_to_dict(cfg: V13ElegantConfig) -> dict:
    return asdict(cfg)


RULE_CONFIGS_V13: dict[str, V13ElegantConfig] = {
    # Conservatively favors transfer robustness.
    "v13_cons": V13ElegantConfig(
        name="v13_cons",
        w_disp=0.35,
        w_risk=0.20,
        w_local=0.08,
        w_burial=0.08,
        u_exposed_pack=0.12,
        u_disp_pack=0.10,
        u_charge_buried=0.38,
        u_large_volume=0.22,
        bias=0.00,
        tau_main=0.72,
        tau_safe=0.56,
        combo_boost=0.95,
        endpoint_rank_normalize=True,
    ),
    # Balanced near-gate variant.
    "v13_bal": V13ElegantConfig(
        name="v13_bal",
        w_disp=0.35,
        w_risk=0.20,
        w_local=0.10,
        w_burial=0.05,
        u_exposed_pack=0.10,
        u_disp_pack=0.10,
        u_charge_buried=0.35,
        u_large_volume=0.20,
        bias=0.00,
        tau_main=0.70,
        tau_safe=0.50,
        combo_boost=0.80,
        endpoint_rank_normalize=True,
    ),
    # In-domain-leaning, less conservative penalties.
    "v13_agg": V13ElegantConfig(
        name="v13_agg",
        w_disp=0.32,
        w_risk=0.20,
        w_local=0.12,
        w_burial=0.04,
        u_exposed_pack=0.06,
        u_disp_pack=0.06,
        u_charge_buried=0.30,
        u_large_volume=0.15,
        bias=0.00,
        tau_main=0.66,
        tau_safe=0.50,
        combo_boost=0.50,
        endpoint_rank_normalize=True,
    ),
}


def add_v13_primitives(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable physical primitives for v13 trust scoring."""
    out = df.copy()

    required = [
        "delta_charge",
        "delta_volume",
        "burial_proxy_v10",
        "q_view_dispersion",
        "q_charge",
        "q_pack",
        "local_apply_flag_target065",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for v13 primitives: {missing}")

    out["abs_charge"] = out["delta_charge"].abs().astype(float)
    out["abs_volume"] = out["delta_volume"].abs().astype(float)
    out["q_risk"] = np.maximum(out["q_charge"].astype(float).values, out["q_pack"].astype(float).values)

    out["phi_exposed_pack"] = out["q_pack"].astype(float).values * (
        1.0 - out["burial_proxy_v10"].astype(float).values
    )
    out["phi_disp_pack"] = out["q_view_dispersion"].astype(float).values * out["q_pack"].astype(float).values

    out["burial_midness"] = 1.0 - np.abs(out["burial_proxy_v10"].astype(float).values - 0.5) / 0.5
    out["burial_midness"] = out["burial_midness"].clip(0.0, 1.0)

    out["risk_charge_buried"] = (
        (out["abs_charge"].values >= 1.0) & (out["burial_proxy_v10"].astype(float).values >= 0.60)
    ).astype(float)
    out["risk_large_volume"] = (out["abs_volume"].values >= 85.0).astype(float)
    out["risk_combo"] = ((out["risk_charge_buried"] > 0.0) | (out["risk_large_volume"] > 0.0)).astype(float)

    return out


def compute_v13_trust(df: pd.DataFrame, cfg: V13ElegantConfig) -> pd.DataFrame:
    """Compute trust score in [0, 1] from monotone physical terms."""
    out = df.copy()

    p_disp = 1.0 - out["q_view_dispersion"].astype(float).values
    p_risk = 1.0 - out["q_risk"].astype(float).values
    p_local = out["local_apply_flag_target065"].astype(float).values
    p_burial = out["burial_midness"].astype(float).values

    n_exposed = out["phi_exposed_pack"].astype(float).values
    n_disp_pack = out["phi_disp_pack"].astype(float).values
    n_charge = out["risk_charge_buried"].astype(float).values
    n_volume = out["risk_large_volume"].astype(float).values

    raw = (
        cfg.w_disp * p_disp
        + cfg.w_risk * p_risk
        + cfg.w_local * p_local
        + cfg.w_burial * p_burial
        - cfg.u_exposed_pack * n_exposed
        - cfg.u_disp_pack * n_disp_pack
        - cfg.u_charge_buried * n_charge
        - cfg.u_large_volume * n_volume
        + cfg.bias
    )

    raw_max = cfg.w_disp + cfg.w_risk + cfg.w_local + cfg.w_burial + max(0.0, cfg.bias)
    raw_min = -(cfg.u_exposed_pack + cfg.u_disp_pack + cfg.u_charge_buried + cfg.u_large_volume) + min(0.0, cfg.bias)

    trust = (raw - raw_min) / (raw_max - raw_min + 1e-12)
    trust = np.clip(trust, 0.0, 1.0)

    out["trust_raw_v13"] = raw
    out["trust_v13"] = trust
    return out


def apply_v13_elegant_variant(df: pd.DataFrame, cfg: V13ElegantConfig) -> pd.DataFrame:
    """Apply continuous tri-expert deterministic fusion for one v13 config."""
    out = compute_v13_trust(df, cfg)

    endpoints = {
        "main": out["rankscore_3view_p3_v12_p123_mainmax"].astype(float).values,
        "trade": out["rankscore_3view_p3_v12_p123_tradeoff"].astype(float).values,
        "safe": out["rankscore_3view_p3_v12_p123_safefallback"].astype(float).values,
    }
    if cfg.endpoint_rank_normalize:
        for k in list(endpoints):
            endpoints[k] = _percentile_rank(endpoints[k])

    trust_adj = np.clip(out["trust_v13"].values - cfg.combo_boost * out["risk_combo"].values, 0.0, 1.0)

    # Piecewise-linear continuous expert weights.
    w_main = np.clip((trust_adj - cfg.tau_main) / (1.0 - cfg.tau_main + 1e-12), 0.0, 1.0)
    w_safe = np.clip((cfg.tau_safe - trust_adj) / (cfg.tau_safe + 1e-12), 0.0, 1.0)
    w_trade = np.clip(1.0 - w_main - w_safe, 0.0, 1.0)

    w_sum = w_main + w_trade + w_safe + 1e-12
    w_main = w_main / w_sum
    w_trade = w_trade / w_sum
    w_safe = w_safe / w_sum

    pred = w_main * endpoints["main"] + w_trade * endpoints["trade"] + w_safe * endpoints["safe"]

    out["trust_adj_v13"] = trust_adj
    out["weight_main_v13"] = w_main
    out["weight_trade_v13"] = w_trade
    out["weight_safe_v13"] = w_safe

    out["endpoint_main_v13"] = endpoints["main"]
    out["endpoint_trade_v13"] = endpoints["trade"]
    out["endpoint_safe_v13"] = endpoints["safe"]

    out["rankscore_3view_v13"] = pred
    return out


__all__ = [
    "RULE_CONFIGS_V13",
    "V13ElegantConfig",
    "add_v13_primitives",
    "apply_v13_elegant_variant",
    "cfg_to_dict",
    "compute_v13_trust",
    "safe_spearman",
]
