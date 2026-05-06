#!/usr/bin/env python3
"""
PepDDG v6: Geometric Features v2.

Enriched geometric channel with 5 new features on top of the v1 baseline
(n_iface_contacts_8a, n_neighbors_10a):

  1. weighted_contacts    -- sum(1/r^2) for heavy-atom pairs within cutoff
  2. n_hbonds             -- donor-acceptor pairs across the interface
  3. n_salt_bridges       -- charged-pair contacts across the interface
  4. delta_sasa_total     -- burial of the mutation residue upon complexation
  5. delta_sasa_polar     -- polar component of burial
  6. delta_sasa_nonpolar  -- non-polar component of burial
  7. graph_centrality     -- betweenness centrality in the CA contact graph

compute_all_geometric_features_v2() returns all 9 features (7 new + 2 v1).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

# Atoms considered as hydrogen-bond donors
HBOND_DONOR_ATOMS = frozenset({
    "N", "NZ", "NH1", "NH2", "NE", "NE2", "ND1", "ND2",
    "OG", "OG1", "OH", "NE1",
})

# Atoms considered as hydrogen-bond acceptors
HBOND_ACCEPTOR_ATOMS = frozenset({
    "O", "OD1", "OD2", "OE1", "OE2",
    "OG", "OG1", "OH",
    "ND1", "NE2", "SD",
})

# Positively charged residues and their charged atoms
POSITIVE_RESIDUES = {"ARG", "LYS", "HIS"}
POSITIVE_CHARGED_ATOMS = frozenset({"NZ", "NH1", "NH2", "NE", "ND1", "NE2"})

# Negatively charged residues and their charged atoms
NEGATIVE_RESIDUES = {"ASP", "GLU"}
NEGATIVE_CHARGED_ATOMS = frozenset({"OD1", "OD2", "OE1", "OE2"})

# Polar elements for SASA classification
POLAR_ELEMENTS = frozenset({"N", "O", "S"})


def _parse_element(line: str, atom_name: str) -> str:
    """Extract element from a PDB ATOM line.

    Tries columns 76-78 first, falls back to first non-digit character
    of the atom name.
    """
    if len(line) >= 78:
        elem = line[76:78].strip().upper()
        if elem:
            return elem
    # Fallback: first alphabetic char of atom_name
    for ch in atom_name:
        if ch.isalpha():
            return ch.upper()
    return ""


def parse_pdb_atoms(
    pdb_path: str,
) -> Dict[str, List[Tuple[str, int, str, float, float, float, str]]]:
    """Parse ATOM records from a PDB file, skipping hydrogens.

    Returns:
        dict: chain_id -> list of (atom_name, resnum, resname, x, y, z, element)
    """
    chains: Dict[str, List[Tuple[str, int, str, float, float, float, str]]] = {}
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            if not chain:
                chain = "A"
            try:
                resnum = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            element = _parse_element(line, atom_name)
            # Skip hydrogens
            if element == "H" or atom_name.startswith("H"):
                continue

            chains.setdefault(chain, []).append(
                (atom_name, resnum, resname, x, y, z, element)
            )
    return chains


# ---------------------------------------------------------------------------
# Feature 1: Weighted contacts (sum of 1/r^2)
# ---------------------------------------------------------------------------

def compute_weighted_contacts(
    pdb_path: str,
    mutation_chain: str,
    mutation_resnum: int,
    partner_chains: List[str],
    cutoff: float = 8.0,
) -> float:
    """Sum of 1/r^2 for all heavy-atom pairs between the mutation residue
    and atoms on partner chains within *cutoff* angstroms.

    Distances are clamped to >= 1.0 A to avoid singularities.
    """
    chains_data = parse_pdb_atoms(pdb_path)

    # Mutation residue atoms
    mut_coords = []
    if mutation_chain in chains_data:
        for atom_name, resnum, resname, x, y, z, elem in chains_data[mutation_chain]:
            if resnum == mutation_resnum:
                mut_coords.append(np.array([x, y, z]))

    if not mut_coords:
        return 0.0

    # Partner atoms
    partner_coords = []
    for pc in partner_chains:
        if pc in chains_data:
            for atom_name, resnum, resname, x, y, z, elem in chains_data[pc]:
                partner_coords.append(np.array([x, y, z]))

    if not partner_coords:
        return 0.0

    mut_arr = np.array(mut_coords)     # (M, 3)
    par_arr = np.array(partner_coords) # (P, 3)

    # Compute all pairwise distances
    # diff[i, j] = mut_arr[i] - par_arr[j]
    diff = mut_arr[:, np.newaxis, :] - par_arr[np.newaxis, :, :]  # (M, P, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (M, P)

    # Mask: within cutoff
    mask = dists <= cutoff
    dists_in_cutoff = dists[mask]

    # Clamp to >= 1.0
    dists_in_cutoff = np.maximum(dists_in_cutoff, 1.0)

    return float(np.sum(1.0 / dists_in_cutoff ** 2))


# ---------------------------------------------------------------------------
# Feature 2: Interface hydrogen bonds
# ---------------------------------------------------------------------------

def count_interface_hbonds(
    pdb_path: str,
    mutation_chain: str,
    mutation_resnum: int,
    partner_chains: List[str],
    dist_cutoff: float = 3.5,
) -> int:
    """Count donor-acceptor pairs between the mutation residue and partner
    chains where the distance is below *dist_cutoff*.

    An H-bond is counted when a donor atom on one side is within dist_cutoff
    of an acceptor atom on the other side (bidirectional: mutation can be
    donor or acceptor).
    """
    chains_data = parse_pdb_atoms(pdb_path)

    # Collect mutation residue donor/acceptor atoms with coords
    mut_donors: List[np.ndarray] = []
    mut_acceptors: List[np.ndarray] = []
    if mutation_chain in chains_data:
        for atom_name, resnum, resname, x, y, z, elem in chains_data[mutation_chain]:
            if resnum == mutation_resnum:
                coord = np.array([x, y, z])
                if atom_name in HBOND_DONOR_ATOMS:
                    mut_donors.append(coord)
                if atom_name in HBOND_ACCEPTOR_ATOMS:
                    mut_acceptors.append(coord)

    # Collect partner donor/acceptor atoms with coords
    par_donors: List[np.ndarray] = []
    par_acceptors: List[np.ndarray] = []
    for pc in partner_chains:
        if pc in chains_data:
            for atom_name, resnum, resname, x, y, z, elem in chains_data[pc]:
                coord = np.array([x, y, z])
                if atom_name in HBOND_DONOR_ATOMS:
                    par_donors.append(coord)
                if atom_name in HBOND_ACCEPTOR_ATOMS:
                    par_acceptors.append(coord)

    count = 0

    # mutation donors -> partner acceptors
    for d in mut_donors:
        for a in par_acceptors:
            if np.linalg.norm(d - a) < dist_cutoff:
                count += 1

    # partner donors -> mutation acceptors
    for d in par_donors:
        for a in mut_acceptors:
            if np.linalg.norm(d - a) < dist_cutoff:
                count += 1

    return count


# ---------------------------------------------------------------------------
# Feature 3: Salt bridges
# ---------------------------------------------------------------------------

def count_salt_bridges(
    pdb_path: str,
    mutation_chain: str,
    mutation_resnum: int,
    partner_chains: List[str],
    cutoff: float = 4.0,
) -> int:
    """Count salt bridge contacts between the mutation residue and partner
    chains.

    A salt bridge is a positively-charged atom within *cutoff* of a
    negatively-charged atom. The mutation residue can be either the positive
    or negative partner.
    """
    chains_data = parse_pdb_atoms(pdb_path)

    # Mutation residue charged atoms
    mut_positive: List[np.ndarray] = []
    mut_negative: List[np.ndarray] = []
    if mutation_chain in chains_data:
        for atom_name, resnum, resname, x, y, z, elem in chains_data[mutation_chain]:
            if resnum == mutation_resnum:
                coord = np.array([x, y, z])
                if resname in POSITIVE_RESIDUES and atom_name in POSITIVE_CHARGED_ATOMS:
                    mut_positive.append(coord)
                if resname in NEGATIVE_RESIDUES and atom_name in NEGATIVE_CHARGED_ATOMS:
                    mut_negative.append(coord)

    # Partner charged atoms
    par_positive: List[np.ndarray] = []
    par_negative: List[np.ndarray] = []
    for pc in partner_chains:
        if pc in chains_data:
            for atom_name, resnum, resname, x, y, z, elem in chains_data[pc]:
                coord = np.array([x, y, z])
                if resname in POSITIVE_RESIDUES and atom_name in POSITIVE_CHARGED_ATOMS:
                    par_positive.append(coord)
                if resname in NEGATIVE_RESIDUES and atom_name in NEGATIVE_CHARGED_ATOMS:
                    par_negative.append(coord)

    count = 0

    # mutation positive <-> partner negative
    for p in mut_positive:
        for n in par_negative:
            if np.linalg.norm(p - n) < cutoff:
                count += 1

    # partner positive <-> mutation negative
    for p in par_positive:
        for n in mut_negative:
            if np.linalg.norm(p - n) < cutoff:
                count += 1

    return count


# ---------------------------------------------------------------------------
# Feature 4: Delta SASA (burial upon complexation)
# ---------------------------------------------------------------------------

def _build_freesasa_structure(
    atoms: List[Tuple[str, int, str, float, float, float, str]],
    chain_id: str,
) -> Any:
    """Build a freesasa Structure from parsed atom records."""
    import freesasa

    struct = freesasa.Structure()
    for atom_name, resnum, resname, x, y, z, elem in atoms:
        # Pad atom name to 4 chars (PDB convention) to help classifier
        padded = f" {atom_name:<3s}" if len(atom_name) < 4 else atom_name
        struct.addAtom(padded, resname, str(resnum), chain_id, x, y, z)
    return struct


def compute_delta_sasa(
    pdb_path: str,
    mutation_chain: str,
    mutation_resnum: int,
    partner_chains: List[str],
) -> Dict[str, float]:
    """Compute the change in SASA of the mutation residue upon complexation.

    delta_sasa = SASA_monomer - SASA_complex  (positive = more buried in complex)

    Returns dict with keys:
        delta_sasa_total, delta_sasa_polar, delta_sasa_nonpolar
    """
    import freesasa

    chains_data = parse_pdb_atoms(pdb_path)

    # Build the full complex structure (all chains)
    all_chains = set()
    all_chains.add(mutation_chain)
    all_chains.update(partner_chains)

    complex_struct = freesasa.Structure()
    n_complex_atoms = 0
    for ch in sorted(all_chains):
        if ch in chains_data:
            for atom_name, resnum, resname, x, y, z, elem in chains_data[ch]:
                padded = f" {atom_name:<3s}" if len(atom_name) < 4 else atom_name
                complex_struct.addAtom(
                    padded, resname, str(resnum), ch, x, y, z
                )
                n_complex_atoms += 1

    # Guard against empty structures (FreeSASA segfaults on empty input)
    if n_complex_atoms == 0:
        return {
            "delta_sasa_total": 0.0,
            "delta_sasa_polar": 0.0,
            "delta_sasa_nonpolar": 0.0,
        }

    complex_result = freesasa.calc(complex_struct)
    complex_areas = complex_result.residueAreas()

    # Get complex SASA for the mutation residue
    resnum_str = str(mutation_resnum)
    complex_total = 0.0
    complex_polar = 0.0
    complex_apolar = 0.0
    if mutation_chain in complex_areas and resnum_str in complex_areas[mutation_chain]:
        ra = complex_areas[mutation_chain][resnum_str]
        complex_total = ra.total
        complex_polar = ra.polar
        complex_apolar = ra.apolar

    # Build isolated chain structure (monomer)
    mono_struct = freesasa.Structure()
    n_mono_atoms = 0
    if mutation_chain in chains_data:
        for atom_name, resnum, resname, x, y, z, elem in chains_data[mutation_chain]:
            padded = f" {atom_name:<3s}" if len(atom_name) < 4 else atom_name
            mono_struct.addAtom(
                padded, resname, str(resnum), mutation_chain, x, y, z
            )
            n_mono_atoms += 1

    if n_mono_atoms == 0:
        return {
            "delta_sasa_total": 0.0,
            "delta_sasa_polar": 0.0,
            "delta_sasa_nonpolar": 0.0,
        }

    mono_result = freesasa.calc(mono_struct)
    mono_areas = mono_result.residueAreas()

    mono_total = 0.0
    mono_polar = 0.0
    mono_apolar = 0.0
    if mutation_chain in mono_areas and resnum_str in mono_areas[mutation_chain]:
        ra = mono_areas[mutation_chain][resnum_str]
        mono_total = ra.total
        mono_polar = ra.polar
        mono_apolar = ra.apolar

    return {
        "delta_sasa_total": mono_total - complex_total,
        "delta_sasa_polar": mono_polar - complex_polar,
        "delta_sasa_nonpolar": mono_apolar - complex_apolar,
    }


# ---------------------------------------------------------------------------
# Feature 5: Graph centrality (betweenness)
# ---------------------------------------------------------------------------

def compute_graph_centrality(
    pdb_path: str,
    mutation_chain: str,
    mutation_resnum: int,
    cutoff: float = 8.0,
) -> float:
    """Betweenness centrality of the mutation residue in a CA contact graph.

    Nodes are residues (identified by chain + resnum). Edges connect residues
    whose CA atoms are within *cutoff* angstroms.

    Returns a float in [0, 1]. Returns 0.0 if the residue has no CA atom
    or the graph has fewer than 3 nodes.
    """
    import networkx as nx

    chains_data = parse_pdb_atoms(pdb_path)

    # Collect all CA atoms: node_id -> coord
    ca_nodes: Dict[Tuple[str, int], np.ndarray] = {}
    for chain_id, atoms in chains_data.items():
        for atom_name, resnum, resname, x, y, z, elem in atoms:
            if atom_name == "CA":
                ca_nodes[(chain_id, resnum)] = np.array([x, y, z])

    target_node = (mutation_chain, mutation_resnum)
    if target_node not in ca_nodes:
        return 0.0

    if len(ca_nodes) < 3:
        return 0.0

    # Build adjacency graph
    node_list = list(ca_nodes.keys())
    coords = np.array([ca_nodes[n] for n in node_list])

    G = nx.Graph()
    G.add_nodes_from(node_list)

    # Compute pairwise CA-CA distances
    n = len(node_list)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= cutoff:
                G.add_edge(node_list[i], node_list[j])

    # Betweenness centrality
    bc = nx.betweenness_centrality(G, normalized=True)

    return float(bc.get(target_node, 0.0))


# ---------------------------------------------------------------------------
# Combined feature computation
# ---------------------------------------------------------------------------

def compute_all_geometric_features_v2(
    pdb_path: str,
    mutation_chain: str,
    mutation_resnum: int,
    receptor_chains: str,
    peptide_chains: str,
) -> Dict[str, float]:
    """Compute all geometric features (v2 + backward-compatible v1).

    Args:
        pdb_path: Path to PDB file.
        mutation_chain: Chain containing the mutation.
        mutation_resnum: Residue number of the mutation.
        receptor_chains: String of receptor chain letters (e.g. "AB").
        peptide_chains: String of peptide chain letters (e.g. "C").

    Returns:
        Dict with 9 keys:
            - weighted_contacts (v2)
            - n_hbonds (v2)
            - n_salt_bridges (v2)
            - delta_sasa_total (v2)
            - delta_sasa_polar (v2)
            - delta_sasa_nonpolar (v2)
            - graph_centrality (v2)
            - n_iface_contacts_8a (v1 backward-compat)
            - n_neighbors_10a (v1 backward-compat)
    """
    from scipy.spatial import cKDTree

    # Determine partner chains
    if mutation_chain in peptide_chains:
        partner_chain_list = list(receptor_chains)
    elif mutation_chain in receptor_chains:
        partner_chain_list = list(peptide_chains)
    else:
        partner_chain_list = list(
            set(list(receptor_chains) + list(peptide_chains)) - {mutation_chain}
        )

    # v2 features
    wc = compute_weighted_contacts(
        pdb_path, mutation_chain, mutation_resnum, partner_chain_list, cutoff=8.0
    )
    hb = count_interface_hbonds(
        pdb_path, mutation_chain, mutation_resnum, partner_chain_list, dist_cutoff=3.5
    )
    sb = count_salt_bridges(
        pdb_path, mutation_chain, mutation_resnum, partner_chain_list, cutoff=4.0
    )
    sasa = compute_delta_sasa(
        pdb_path, mutation_chain, mutation_resnum, partner_chain_list
    )
    gc = compute_graph_centrality(
        pdb_path, mutation_chain, mutation_resnum, cutoff=8.0
    )

    # v1 backward-compatible features using cKDTree
    chains_data = parse_pdb_atoms(pdb_path)

    # Mutation residue atoms
    mut_coords = []
    mut_ca = None
    if mutation_chain in chains_data:
        for atom_name, resnum, resname, x, y, z, elem in chains_data[mutation_chain]:
            if resnum == mutation_resnum:
                mut_coords.append([x, y, z])
                if atom_name == "CA":
                    mut_ca = np.array([x, y, z])

    # v1: n_iface_contacts_8a (count of heavy-atom pairs within 8A)
    partner_coords = []
    for pc in partner_chain_list:
        if pc in chains_data:
            for atom_name, resnum, resname, x, y, z, elem in chains_data[pc]:
                partner_coords.append([x, y, z])

    n_iface_contacts = 0
    if mut_coords and partner_coords:
        mut_arr = np.array(mut_coords)
        par_arr = np.array(partner_coords)
        tree = cKDTree(par_arr)
        for pt in mut_arr:
            n_iface_contacts += len(tree.query_ball_point(pt, r=8.0))

    # v1: n_neighbors_10a (CA atoms within 10A, excluding self)
    n_neighbors = 0
    if mut_ca is not None:
        all_ca_coords = []
        for chain_id, atoms in chains_data.items():
            for atom_name, resnum, resname, x, y, z, elem in atoms:
                if atom_name == "CA":
                    if chain_id == mutation_chain and resnum == mutation_resnum:
                        continue
                    all_ca_coords.append([x, y, z])
        if all_ca_coords:
            ca_arr = np.array(all_ca_coords)
            ca_tree = cKDTree(ca_arr)
            n_neighbors = len(ca_tree.query_ball_point(mut_ca, r=10.0))

    return {
        "weighted_contacts": wc,
        "n_hbonds": hb,
        "n_salt_bridges": sb,
        "delta_sasa_total": sasa["delta_sasa_total"],
        "delta_sasa_polar": sasa["delta_sasa_polar"],
        "delta_sasa_nonpolar": sasa["delta_sasa_nonpolar"],
        "graph_centrality": gc,
        "n_iface_contacts_8a": float(n_iface_contacts),
        "n_neighbors_10a": float(n_neighbors),
    }


# ---------------------------------------------------------------------------
# Batch CLI
# ---------------------------------------------------------------------------

AA3_TO_AA1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def build_residue_chain_map(pdb_path: str) -> Dict[Tuple[int, str], str]:
    """Build (resnum, aa1) -> chain_id from a PDB file using CA atoms."""
    res_chain: Dict[Tuple[int, str], str] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21].strip() or "A"
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            aa1 = AA3_TO_AA1.get(resname, "X")
            res_chain[(resnum, aa1)] = chain
    return res_chain


def build_sequential_map(
    pdb_path: str,
) -> Dict[str, List[Tuple[int, str]]]:
    """Build chain_id -> sorted list of (crystal_resnum, aa1) from CA atoms.

    This enables mapping from 1-based sequential position (as used in
    predicted-structure cohort numbering) to the actual crystal PDB resnum.
    """
    chain_residues: Dict[str, List[Tuple[int, str]]] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21].strip() or "A"
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            aa1 = AA3_TO_AA1.get(resname, "X")
            chain_residues.setdefault(chain, []).append((resnum, aa1))
    # Sort by crystal resnum within each chain
    for ch in chain_residues:
        chain_residues[ch] = sorted(chain_residues[ch])
    return chain_residues


def resolve_crystal_position(
    target: str,
    cohort_chain: str,
    cohort_resnum: int,
    wt_aa1: str,
    chain_map: Dict[Tuple[int, str], str],
    seq_map: Dict[str, List[Tuple[int, str]]],
    receptor_chains: str,
    peptide_chains: str,
) -> Tuple[str, int]:
    """Resolve a cohort mutation position to crystal (chain_id, resnum).

    The cohort uses predicted-structure conventions:
      - chain A = receptor (1-based numbering)
      - chain B = peptide (1-based numbering)
    Crystal PDBs use their own chain IDs and numbering (often non-1-based).

    Strategy:
    1. Determine expected crystal chain(s) from cohort chain identity
    2. Try direct match: (cohort_resnum, wt_aa1) on expected chains
    3. Try sequential mapping: cohort_resnum as 1-based index into
       sorted crystal chain residues, with aa1 verification
    4. Fall back to broader matching across all chains

    Returns (crystal_chain, crystal_resnum).
    """
    # Determine expected crystal chains from cohort chain
    if cohort_chain.upper() == "B":
        expected_chains = list(peptide_chains)
    elif cohort_chain.upper() == "A":
        expected_chains = list(receptor_chains)
    else:
        expected_chains = list(
            set(list(receptor_chains) + list(peptide_chains))
        )

    # Strategy 1: direct match (resnum, wt_aa1) on expected chains
    if (cohort_resnum, wt_aa1) in chain_map:
        matched_chain = chain_map[(cohort_resnum, wt_aa1)]
        if matched_chain in expected_chains:
            return matched_chain, cohort_resnum

    # Strategy 2: per-chain sequential mapping (with aa verification)
    for crystal_ch in expected_chains:
        if crystal_ch not in seq_map:
            continue
        residues = seq_map[crystal_ch]
        seq_idx = cohort_resnum - 1  # 1-based to 0-based
        if 0 <= seq_idx < len(residues):
            crystal_resnum, crystal_aa = residues[seq_idx]
            if crystal_aa == wt_aa1:
                return crystal_ch, crystal_resnum

    # Strategy 2b: concatenated multi-chain sequential mapping
    # For multi-chain groups (e.g., receptor=HL), the cohort may use
    # concatenated 1-based numbering across all chains in the group.
    if len(expected_chains) > 1:
        combined: List[Tuple[str, int, str]] = []
        for crystal_ch in expected_chains:
            if crystal_ch in seq_map:
                for crn, caa in seq_map[crystal_ch]:
                    combined.append((crystal_ch, crn, caa))
        seq_idx = cohort_resnum - 1
        if 0 <= seq_idx < len(combined):
            crystal_ch, crystal_resnum, crystal_aa = combined[seq_idx]
            if crystal_aa == wt_aa1:
                return crystal_ch, crystal_resnum

    # Strategy 3: direct match on any chain (cross-chain fallback)
    if (cohort_resnum, wt_aa1) in chain_map:
        return chain_map[(cohort_resnum, wt_aa1)], cohort_resnum

    # Strategy 4: sequential mapping, relaxed aa match (per-chain)
    for crystal_ch in expected_chains:
        if crystal_ch not in seq_map:
            continue
        residues = seq_map[crystal_ch]
        seq_idx = cohort_resnum - 1
        if 0 <= seq_idx < len(residues):
            crystal_resnum, _ = residues[seq_idx]
            return crystal_ch, crystal_resnum

    # Strategy 4b: concatenated multi-chain, relaxed aa match
    if len(expected_chains) > 1:
        combined_relax: List[Tuple[str, int]] = []
        for crystal_ch in expected_chains:
            if crystal_ch in seq_map:
                for crn, _ in seq_map[crystal_ch]:
                    combined_relax.append((crystal_ch, crn))
        seq_idx = cohort_resnum - 1
        if 0 <= seq_idx < len(combined_relax):
            return combined_relax[seq_idx]

    # Strategy 5: resnum-only match on expected chains
    for (rn, aa), ch in chain_map.items():
        if rn == cohort_resnum and ch in expected_chains:
            return ch, cohort_resnum

    # Fallback: return cohort values (will likely produce zero features)
    return cohort_chain, cohort_resnum


def main() -> None:
    """Batch-compute geometric features v2 for all mutations in the cohort."""
    import argparse
    import time

    import pandas as pd

    # Resolve repo root (scripts/ is one level below pepddg_v5/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent  # research/../../../

    parser = argparse.ArgumentParser(
        description="Compute geometric features v2 for all cohort mutations."
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=str(repo_root / "research" / "pepddg_v5" / "results"
                     / "tier2_3view" / "cohort_3view.csv"),
        help="Path to cohort CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "research" / "pepddg_v5" / "results"
                     / "v6" / "geometric_features_v2.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(repo_root / "research" / "shared_data"
                     / "skempi_peptide_subset" / "target_manifest.csv"),
        help="Path to target manifest CSV",
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        default=str(repo_root / "research" / "shared_data"
                     / "skempi_peptide_subset" / "structures"),
        help="Directory containing PDB files",
    )
    args = parser.parse_args()

    cohort_path = Path(args.cohort)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    pdb_dir = Path(args.pdb_dir)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    cohort = pd.read_csv(cohort_path)
    manifest = pd.read_csv(manifest_path)
    print(f"Cohort: {len(cohort)} mutations, {cohort['target'].nunique()} targets")

    # Build target -> (receptor_chains, peptide_chains) from manifest
    target_info: Dict[str, Dict[str, str]] = {}
    for _, row in manifest.iterrows():
        pdb_code = row["pdb_code"]
        if pdb_code not in target_info:
            target_info[pdb_code] = {
                "receptor_chains": str(row["receptor_chains"]),
                "peptide_chains": str(row["peptide_chains"]),
            }

    # ------------------------------------------------------------------
    # 2. Build chain maps and sequential maps for each target PDB
    # ------------------------------------------------------------------
    chain_maps: Dict[str, Dict[Tuple[int, str], str]] = {}
    seq_maps: Dict[str, Dict[str, List[Tuple[int, str]]]] = {}
    for target in cohort["target"].unique():
        if target not in target_info:
            print(f"  WARNING: target {target} not in manifest, skipping")
            continue
        pdb_path = pdb_dir / f"{target}.pdb"
        if pdb_path.exists():
            chain_maps[target] = build_residue_chain_map(str(pdb_path))
            seq_maps[target] = build_sequential_map(str(pdb_path))

    # ------------------------------------------------------------------
    # 3. Compute features for each mutation
    # ------------------------------------------------------------------
    results = []
    t0 = time.time()
    n_total = len(cohort)
    n_success = 0
    n_skip = 0
    n_direct = 0
    n_sequential = 0

    for idx, row in cohort.iterrows():
        target = row["target"]
        mutation_id = row["mutation_id"]
        wt_aa1 = str(row["wt_aa1"]).strip()
        cohort_chain = str(row["chain_id"]).strip()
        resnum = int(float(row["resnum"]))

        if target not in target_info:
            print(f"  SKIP {mutation_id}: target {target} not in manifest")
            n_skip += 1
            continue

        info = target_info[target]
        pdb_path = pdb_dir / f"{target}.pdb"
        if not pdb_path.exists():
            print(f"  SKIP {mutation_id}: PDB {pdb_path} not found")
            n_skip += 1
            continue

        receptor_chains = info["receptor_chains"]
        peptide_chains = info["peptide_chains"]

        # Resolve crystal chain and resnum
        cmap = chain_maps.get(target, {})
        smap = seq_maps.get(target, {})

        # Track whether direct match or sequential mapping was used
        direct_match = (resnum, wt_aa1) in cmap
        if not direct_match:
            resnum_match = any(rn == resnum for (rn, _) in cmap)
        else:
            resnum_match = True

        crystal_chain, crystal_resnum = resolve_crystal_position(
            target=target,
            cohort_chain=cohort_chain,
            cohort_resnum=resnum,
            wt_aa1=wt_aa1,
            chain_map=cmap,
            seq_map=smap,
            receptor_chains=receptor_chains,
            peptide_chains=peptide_chains,
        )

        if direct_match or resnum_match:
            n_direct += 1
        else:
            n_sequential += 1

        try:
            feats = compute_all_geometric_features_v2(
                pdb_path=str(pdb_path),
                mutation_chain=crystal_chain,
                mutation_resnum=crystal_resnum,
                receptor_chains=receptor_chains,
                peptide_chains=peptide_chains,
            )
            feats["mutation_id"] = mutation_id
            results.append(feats)
            n_success += 1
        except Exception as e:
            print(f"  ERROR {mutation_id}: {e}")
            n_skip += 1

        # Progress reporting every 50 mutations
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx + 1}/{n_total}] {elapsed:.1f}s elapsed, "
                  f"~{eta:.0f}s remaining ({n_success} OK, {n_skip} skip)")

    elapsed = time.time() - t0
    print(f"\nDone: {n_success}/{n_total} mutations in {elapsed:.1f}s "
          f"({n_skip} skipped)")
    print(f"Chain resolution: {n_direct} direct, {n_sequential} sequential")

    # ------------------------------------------------------------------
    # 4. Save output
    # ------------------------------------------------------------------
    if not results:
        print("ERROR: No results to save!")
        return

    df = pd.DataFrame(results)
    # Canonical column order: mutation_id first, then sorted feature columns
    feature_cols = [c for c in df.columns if c != "mutation_id"]
    col_order = ["mutation_id"] + sorted(feature_cols)
    df = df[col_order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {output_path}")

    # Quick quality check
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("\nWARNING: NaN values found:")
        for col, cnt in nan_counts.items():
            if cnt > 0:
                print(f"  {col}: {cnt} NaN")
    else:
        print("Quality check: No NaN values found")

    # Print feature ranges
    print("\nFeature ranges:")
    for col in sorted(feature_cols):
        vals = df[col]
        print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, "
              f"mean={vals.mean():.4f}, std={vals.std():.4f}")


if __name__ == "__main__":
    main()
