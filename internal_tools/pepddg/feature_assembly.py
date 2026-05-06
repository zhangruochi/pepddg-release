"""Feature assembly for PepDDG upstream pipeline.

Merges physics, structural, and MPNN feature channels into a single
DataFrame ready for scoring.py consumption.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .mpnn_features import compute_mpnn_rankscore
from .physics_features import compute_physics_rankscore
from .structural_features import compute_struct_composite


def assemble_features(
    physics_df: pd.DataFrame,
    structural_df: pd.DataFrame,
    mpnn_df: pd.DataFrame,
    join_on: list[str] | None = None,
    compute_rankscores: bool = True,
) -> pd.DataFrame:
    """Merge all three channels and optionally compute derived rankscores.

    Args:
        physics_df: DataFrame with ddg_xint_iface, ddg_bind_proxy.
        structural_df: DataFrame with n_iface_contacts_8a, n_neighbors_10a.
        mpnn_df: DataFrame with mpnn_neg_llr_complex, mpnn_ddg_bind.
        join_on: Columns to join on (default: ["target", "mutation"]).
        compute_rankscores: If True, compute rankscore_phys, struct_composite,
            rankscore_mpnn columns.

    Returns:
        Merged DataFrame ready for scoring.py.
    """
    if join_on is None:
        join_on = ["target", "mutation"]

    merged = physics_df.merge(structural_df, on=join_on, how="inner")
    merged = merged.merge(mpnn_df, on=join_on, how="inner")

    if compute_rankscores and len(merged) > 0:
        # Physics rankscore
        merged["rankscore_phys"] = compute_physics_rankscore(
            merged["ddg_xint_iface"].values,
            merged["ddg_bind_proxy"].values,
        )
        # Structural composite
        merged["struct_composite"] = compute_struct_composite(
            merged["n_iface_contacts_8a"].fillna(0).values,
            merged["n_neighbors_10a"].fillna(0).values,
        )
        # MPNN rankscore
        merged["rankscore_mpnn"] = compute_mpnn_rankscore(
            merged["mpnn_neg_llr_complex"].values,
            merged["mpnn_ddg_bind"].values,
        )

    return merged


# Required columns for scoring.py
_REQUIRED_COLUMNS = ["mpnn_neg_llr_complex", "mpnn_ddg_bind"]

# Columns that enable anchor derivation (at least one group needed)
_ANCHOR_GROUPS = [
    # Pre-computed anchor
    ["rankscore_3view_v17_mainboost"],
    ["rankscore_3view_v17_2_oodguard"],
    ["rankscore_3view_base"],
    # Derived anchor components (all three needed)
    ["rankscore_phys", "struct_composite", "rankscore_mpnn"],
    ["ddg_xint_iface", "ddg_bind_proxy", "struct_composite"],
]


def validate_scoring_input(df: pd.DataFrame) -> dict[str, Any]:
    """Check that a DataFrame has the required columns for scoring.py.

    Args:
        df: Assembled feature DataFrame.

    Returns:
        Dict with keys: valid (bool), missing (list), warnings (list).
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    warnings: list[str] = []

    # Check anchor availability
    has_anchor = False
    for group in _ANCHOR_GROUPS:
        if all(c in df.columns for c in group):
            has_anchor = True
            break

    if not has_anchor and not missing:
        warnings.append(
            "No anchor columns found. scoring.py will attempt derived clean-3 fallback "
            "but may fail if physics/structural columns are also missing."
        )

    # Check optional enrichment columns
    optional_checks = {
        "struct_composite": "Structural composite not present; will need struct column for anchor.",
        "rankscore_phys": "Physics rankscore not present; will derive from raw columns if available.",
    }
    for col, msg in optional_checks.items():
        if col not in df.columns:
            warnings.append(msg)

    return {
        "valid": len(missing) == 0,
        "missing": missing,
        "warnings": warnings,
    }
