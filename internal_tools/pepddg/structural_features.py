"""Structural feature computation for PepDDG upstream pipeline.

Computes per-mutation structural features:
- n_iface_contacts_8a: heavy-atom pairs between mutation residue and partner chains
- n_neighbors_10a: CA atoms within cutoff of mutation site
- struct_composite: zscore(contacts) + zscore(neighbors)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import zscore as scipy_zscore

from .structure_io import AtomRecord, get_residue_coord, parse_structure


@dataclass(frozen=True)
class MutationSite:
    chain_id: str
    resnum: int
    wt_aa: str
    mut_aa: str
    label: str


def compute_interface_contacts(
    atoms: list[AtomRecord],
    mutation: MutationSite,
    partner_chains: list[str],
    cutoff: float = 8.0,
) -> int:
    """Count heavy-atom pairs between mutation residue and partner chains within cutoff.

    For each heavy atom of the mutation residue, counts partner-chain heavy atoms
    within the distance cutoff, then sums all counts.

    Args:
        atoms: Parsed atom records.
        mutation: Mutation site specification.
        partner_chains: Chain IDs of the interaction partner.
        cutoff: Distance cutoff in Angstroms (default 8.0).

    Returns:
        Total number of heavy-atom contact pairs.
    """
    # Collect partner chain heavy atoms
    partner_coords = []
    for a in atoms:
        if a.chain in partner_chains:
            partner_coords.append(a.coord)

    if not partner_coords:
        return 0

    partner_tree = cKDTree(np.array(partner_coords))

    # Collect mutation residue heavy atoms
    mut_coords = []
    for a in atoms:
        if a.chain == mutation.chain_id and a.resnum == mutation.resnum:
            mut_coords.append(a.coord)

    if not mut_coords:
        return 0

    # Count contacts
    n_contacts = 0
    for coord in mut_coords:
        n_contacts += len(partner_tree.query_ball_point(coord, r=cutoff))

    return n_contacts


def compute_neighbor_count(
    atoms: list[AtomRecord],
    mutation: MutationSite,
    cutoff: float = 10.0,
) -> int:
    """Count CA atoms (any chain) within cutoff of mutation site CB/CA.

    Excludes the mutation residue's own CA to match reference implementations.

    Args:
        atoms: Parsed atom records.
        mutation: Mutation site specification.
        cutoff: Distance cutoff in Angstroms (default 10.0).

    Returns:
        Number of CA atoms within cutoff (excluding self).
    """
    site_coord = get_residue_coord(
        atoms, mutation.chain_id, mutation.resnum, prefer="CB", fallback="CA"
    )
    if site_coord is None:
        return 0

    # Collect all CA atoms excluding the mutation residue itself
    ca_coords = []
    for a in atoms:
        if a.atom_name == "CA":
            if a.chain == mutation.chain_id and a.resnum == mutation.resnum:
                continue
            ca_coords.append(a.coord)

    if not ca_coords:
        return 0

    ca_tree = cKDTree(np.array(ca_coords))
    return len(ca_tree.query_ball_point(site_coord, r=cutoff))


def compute_structural_features_batch(
    structure_path: str | Path,
    mutations: list[MutationSite],
    mutation_chain: str,
    partner_chains: list[str],
    interface_cutoff: float = 8.0,
    neighbor_cutoff: float = 10.0,
) -> pd.DataFrame:
    """Compute structural features for a batch of mutations.

    Args:
        structure_path: Path to PDB or CIF file.
        mutations: List of MutationSite objects.
        mutation_chain: Chain containing the mutations.
        partner_chains: Partner chain IDs.
        interface_cutoff: Cutoff for interface contacts (default 8.0).
        neighbor_cutoff: Cutoff for neighbor counting (default 10.0).

    Returns:
        DataFrame with columns: label, n_iface_contacts_8a, n_neighbors_10a.
    """
    atoms = parse_structure(structure_path)
    results = []
    for mut in mutations:
        contacts = compute_interface_contacts(atoms, mut, partner_chains, interface_cutoff)
        neighbors = compute_neighbor_count(atoms, mut, neighbor_cutoff)
        results.append({
            "label": mut.label,
            "n_iface_contacts_8a": contacts,
            "n_neighbors_10a": neighbors,
        })
    return pd.DataFrame(results)


def compute_struct_composite(contacts: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Compute struct_composite = zscore(contacts) + zscore(neighbors).

    Handles edge cases: single value or uniform arrays produce 0.

    Args:
        contacts: Array of interface contact counts.
        neighbors: Array of neighbor counts.

    Returns:
        Array of composite structural scores.
    """
    contacts = np.asarray(contacts, dtype=float)
    neighbors = np.asarray(neighbors, dtype=float)

    def _safe_zscore(x: np.ndarray) -> np.ndarray:
        if len(x) <= 1 or np.std(x) == 0:
            return np.zeros_like(x)
        return scipy_zscore(x)

    return _safe_zscore(contacts) + _safe_zscore(neighbors)
