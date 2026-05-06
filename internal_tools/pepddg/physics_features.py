"""Physics feature extraction for PepDDG upstream pipeline.

Does NOT run OpenMM — only parses pre-computed results.
Two ingestion paths: dict-based (for programmatic use) and CSV-based.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def extract_physics_from_openmm_results(
    wt_result: dict, mut_result: dict
) -> dict[str, float]:
    """Extract physics features from WT and mutant OpenMM energy dicts.

    Computes delta energies (mutant - WT) for interface cross-interaction
    and binding proxy.

    Args:
        wt_result: Wild-type energy dict with keys:
            e_cross_interface_total_screened_heavy_kcal_mol, dg_bind_kcal_mol
        mut_result: Mutant energy dict with same keys.

    Returns:
        Dict with ddg_xint_iface and ddg_bind_proxy.
    """
    ddg_xint = (
        mut_result["e_cross_interface_total_screened_heavy_kcal_mol"]
        - wt_result["e_cross_interface_total_screened_heavy_kcal_mol"]
    )
    ddg_bind = (
        mut_result["dg_bind_kcal_mol"]
        - wt_result["dg_bind_kcal_mol"]
    )
    return {
        "ddg_xint_iface": float(ddg_xint),
        "ddg_bind_proxy": float(ddg_bind),
    }


def extract_physics_from_csv(results_csv: str | Path) -> pd.DataFrame:
    """Parse benchmark results.csv into a physics feature DataFrame.

    If paired columns (ddg_paired_xint_iface, ddg_paired_bind_proxy) exist,
    they are used preferentially and renamed to the standard column names.

    Args:
        results_csv: Path to CSV with physics results.

    Returns:
        DataFrame with columns: target, mutation, ddg_xint_iface, ddg_bind_proxy.
    """
    df = pd.read_csv(results_csv)
    out = df[["target", "mutation"]].copy()

    # Prefer paired columns if available
    if "ddg_paired_xint_iface" in df.columns:
        out["ddg_xint_iface"] = df["ddg_paired_xint_iface"]
    else:
        out["ddg_xint_iface"] = df["ddg_xint_iface"]

    if "ddg_paired_bind_proxy" in df.columns:
        out["ddg_bind_proxy"] = df["ddg_paired_bind_proxy"]
    else:
        out["ddg_bind_proxy"] = df["ddg_bind_proxy"]

    return out


def compute_physics_rankscore(xint: np.ndarray, bind: np.ndarray) -> np.ndarray:
    """Compute physics rankscore = rank(rank(xint) + rank(bind)), normalized to [0,1].

    Uses ordinal ranking for determinism.

    Args:
        xint: Array of ddg_xint_iface values.
        bind: Array of ddg_bind_proxy values.

    Returns:
        Normalized rank scores in [0, 1].
    """
    xint = np.asarray(xint, dtype=float)
    bind = np.asarray(bind, dtype=float)
    n = len(xint)
    if n <= 1:
        return np.zeros(n, dtype=float)

    r1 = rankdata(xint, method="ordinal")
    r2 = rankdata(bind, method="ordinal")
    combined = r1 + r2
    r_final = rankdata(combined, method="ordinal")
    return (r_final - 1.0) / float(n - 1)
