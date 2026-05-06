#!/usr/bin/env python3
"""
PepDDG v5: Geometric channel hyperparameter sensitivity analysis.

Evaluates how the structural composite score (and downstream 3-view rank-sum)
varies with two key geometric parameters:
  1. contact_cutoff (default 8A) - defines inter-chain contacts
  2. neighbor_cutoff (default 10A) - defines CA neighborhood

For each combination:
  - Recomputes n_iface_contacts and n_neighbors from PDB structures
  - Recomputes struct_composite = zscore(contacts) + zscore(neighbors)
  - Evaluates pooled Spearman rho of struct_composite alone vs ddg_exp
  - Evaluates 3-view rank-sum rho (physics + structural + MPNN)

Usage:
    conda activate research
    python research/pepddg_v5/scripts/sensitivity_geometric.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, zscore


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent  # research/pepddg_v5
REPO_ROOT = BASE_DIR.parent.parent  # repo root

COHORT_CSV = BASE_DIR / "results" / "tier2_3view" / "cohort_3view.csv"
MANIFEST_CSV = REPO_ROOT / "research" / "shared_data" / "skempi_peptide_subset" / "target_manifest.csv"
PDB_DIR = REPO_ROOT / "research" / "shared_data" / "skempi_peptide_subset" / "structures"
OUT_DIR = BASE_DIR / "results" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mutation token regex: e.g. "LB38G" -> wt=L, chain=B, resnum=38, mut=G
_MUTATION_RE = re.compile(r"^([A-Z])([A-Za-z])(\d+)([A-Z])$")

# Cutoff grids
CONTACT_CUTOFFS = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
NEIGHBOR_CUTOFFS = [8.0, 9.0, 10.0, 11.0, 12.0]


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------
def parse_pdb_atoms(pdb_path: str) -> Dict[str, List[Tuple[str, int, str, float, float, float]]]:
    """Parse ATOM records from PDB, returning atoms grouped by chain.

    Returns dict: chain -> list of (atom_name, resnum, resname, x, y, z)
    """
    chains: Dict[str, List[Tuple[str, int, str, float, float, float]]] = {}
    with open(pdb_path, "r") as f:
        for line in f:
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

            # Skip hydrogen atoms and virtual sites
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if atom_name.startswith("H") or element == "H":
                continue

            chains.setdefault(chain, []).append((atom_name, resnum, resname, x, y, z))

    return chains


def compute_geometric_features(
    pdb_path: str,
    mutation_chain: str,
    mutation_resnum: int,
    receptor_chains: str,
    peptide_chains: str,
    contact_cutoff: float = 8.0,
    neighbor_cutoff: float = 10.0,
) -> Dict[str, float]:
    """Compute geometric features for a mutation position.

    Returns:
        n_iface_contacts: number of heavy-atom pairs between mutation residue
                          and partner chain(s) within contact_cutoff
        n_neighbors: number of CA atoms within neighbor_cutoff of mutation
                     residue's CA (across all chains, excluding self)
    """
    chains_data = parse_pdb_atoms(pdb_path)

    # Determine which chains are "partner" chains
    if mutation_chain in peptide_chains:
        partner_chains = list(receptor_chains)
    elif mutation_chain in receptor_chains:
        partner_chains = list(peptide_chains)
    else:
        # Fallback: partner = everything except mutation chain
        partner_chains = [c for c in chains_data if c != mutation_chain]

    # Get atoms for the mutation residue
    mut_atoms = []
    mut_ca = None
    if mutation_chain in chains_data:
        for atom_name, resnum, resname, x, y, z in chains_data[mutation_chain]:
            if resnum == mutation_resnum:
                mut_atoms.append((x, y, z))
                if atom_name == "CA":
                    mut_ca = np.array([x, y, z])

    if not mut_atoms:
        return {"n_iface_contacts": np.nan, "n_neighbors": np.nan}

    mut_coords = np.array(mut_atoms)

    # --- Interface contacts ---
    partner_coords = []
    for pc in partner_chains:
        if pc in chains_data:
            for atom_name, resnum, resname, x, y, z in chains_data[pc]:
                partner_coords.append((x, y, z))

    if partner_coords:
        partner_arr = np.array(partner_coords)
        tree = cKDTree(partner_arr)
        n_contacts = 0
        for pt in mut_coords:
            n_contacts += len(tree.query_ball_point(pt, r=contact_cutoff))
    else:
        n_contacts = 0

    # --- CA neighbors ---
    if mut_ca is not None:
        all_ca_coords = []
        for chain_id, atoms in chains_data.items():
            for atom_name, resnum, resname, x, y, z in atoms:
                if atom_name == "CA":
                    # Exclude the mutation residue itself
                    if chain_id == mutation_chain and resnum == mutation_resnum:
                        continue
                    all_ca_coords.append((x, y, z))

        if all_ca_coords:
            ca_arr = np.array(all_ca_coords)
            ca_tree = cKDTree(ca_arr)
            n_neighbors = len(ca_tree.query_ball_point(mut_ca, r=neighbor_cutoff))
        else:
            n_neighbors = 0
    else:
        n_neighbors = np.nan

    return {
        "n_iface_contacts": float(n_contacts),
        "n_neighbors": float(n_neighbors),
    }


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Rank data with average tie-breaking."""
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j < len(sorted_a) and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = np.mean(ranks[order[i:j]])
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def target_stratified_bootstrap_rho(
    y: np.ndarray, yhat: np.ndarray, targets: np.ndarray,
    n_bootstrap: int = 5000, seed: int = 42,
) -> Tuple[float, float, float]:
    """Target-stratified bootstrap for Spearman rho CI."""
    rng = np.random.RandomState(seed)
    unique_targets = np.unique(targets)
    n_targets = len(unique_targets)

    target_indices = {}
    for t in unique_targets:
        target_indices[t] = np.where(targets == t)[0]

    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_targets, size=n_targets, replace=True)
        indices = np.concatenate([target_indices[t] for t in sampled])
        if len(indices) < 5:
            continue
        rho, _ = spearmanr(y[indices], yhat[indices])
        if not np.isnan(rho):
            bootstrap_rhos.append(rho)

    bootstrap_rhos = np.array(bootstrap_rhos)
    point_rho, _ = spearmanr(y, yhat)
    ci_lo = np.percentile(bootstrap_rhos, 2.5)
    ci_hi = np.percentile(bootstrap_rhos, 97.5)
    return float(point_rho), float(ci_lo), float(ci_hi)


def main():
    print("=" * 70)
    print("PepDDG v5: Geometric Channel Sensitivity Analysis")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load cohort data
    # ------------------------------------------------------------------
    cohort = pd.read_csv(COHORT_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    print(f"Cohort: {len(cohort)} mutations, {cohort['target'].nunique()} targets")

    # Build target -> (pdb_code, receptor_chains, peptide_chains) map
    # The cohort uses pdb_code (e.g. "1ACB") as the target identifier,
    # while the manifest uses pdb_entry (e.g. "1ACB_E_I").
    # Map by pdb_code. If multiple entries share a pdb_code, take the first.
    target_info = {}
    for _, row in manifest.iterrows():
        pdb_code = row["pdb_code"]
        if pdb_code not in target_info:
            target_info[pdb_code] = {
                "pdb_code": pdb_code,
                "receptor_chains": str(row["receptor_chains"]),
                "peptide_chains": str(row["peptide_chains"]),
            }

    # ------------------------------------------------------------------
    # 2. Parse mutation positions and map to crystal chain IDs
    # ------------------------------------------------------------------
    # The cohort mutation tokens use predicted-structure chains (A=receptor,
    # B=peptide), but PDB files use crystal chain IDs. We need to find
    # which crystal chain contains each mutation residue.
    #
    # Strategy: For each target, build a map of (resnum, wt_aa1) -> crystal_chain
    # by scanning the PDB file.

    # First, build residue -> chain lookup from PDB files
    AA3_TO_AA1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    def build_residue_chain_map(pdb_path: str) -> Dict[Tuple[int, str], str]:
        """Build (resnum, aa1) -> chain_id from a PDB file.
        Uses CA atoms to avoid ambiguity."""
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

    # Build chain maps for all targets
    chain_maps: Dict[str, Dict[Tuple[int, str], str]] = {}
    for target in cohort["target"].unique():
        if target not in target_info:
            continue
        pdb_code = target_info[target]["pdb_code"]
        pdb_path = PDB_DIR / f"{pdb_code}.pdb"
        if pdb_path.exists():
            chain_maps[target] = build_residue_chain_map(str(pdb_path))

    # Parse positions and resolve crystal chain
    positions = []
    resolved = 0
    unresolved = 0
    for _, row in cohort.iterrows():
        target = row["target"]
        mut_token = row["mut"]

        # Parse mutation token
        m = _MUTATION_RE.match(str(mut_token).strip())
        if m:
            wt_aa1 = m.group(1)
            cohort_chain = m.group(2).upper()
            resnum = int(m.group(3))
        else:
            wt_aa1 = str(row.get("wt_aa1", "X")).strip()
            cohort_chain = str(row.get("chain_id", "")).strip()
            resnum = int(row.get("resnum", 0))

        # Resolve to crystal chain
        crystal_chain = cohort_chain  # fallback
        if target in chain_maps:
            cmap = chain_maps[target]
            # Try exact match (resnum, wt_aa1)
            if (resnum, wt_aa1) in cmap:
                crystal_chain = cmap[(resnum, wt_aa1)]
                resolved += 1
            else:
                # Try just resnum (any aa)
                candidates = [ch for (rn, aa), ch in cmap.items() if rn == resnum]
                if candidates:
                    crystal_chain = candidates[0]
                    resolved += 1
                else:
                    unresolved += 1
        else:
            unresolved += 1

        positions.append({
            "target": target,
            "mut": mut_token,
            "chain_id": crystal_chain,
            "resnum": resnum,
        })

    positions_df = pd.DataFrame(positions)
    print(f"Chain resolution: {resolved} resolved, {unresolved} unresolved")

    # Unique positions for computation
    unique_positions = positions_df[["target", "chain_id", "resnum"]].drop_duplicates()
    print(f"Unique mutation positions: {len(unique_positions)}")

    # ------------------------------------------------------------------
    # 3. Compute geometric features at all cutoff combinations
    # ------------------------------------------------------------------
    print(f"\nComputing geometric features across "
          f"{len(CONTACT_CUTOFFS)} x {len(NEIGHBOR_CUTOFFS)} = "
          f"{len(CONTACT_CUTOFFS) * len(NEIGHBOR_CUTOFFS)} cutoff combinations...")

    # First, precompute features for all unique positions at all cutoffs
    # Structure: features[(target, chain, resnum)][(contact_cutoff, neighbor_cutoff)] = (n_contacts, n_neighbors)
    all_features: Dict[
        Tuple[str, str, int],
        Dict[Tuple[float, float], Tuple[float, float]]
    ] = {}

    # Group by target for efficient PDB loading
    targets_in_cohort = unique_positions["target"].unique()
    t0 = time.time()

    for tidx, target in enumerate(sorted(targets_in_cohort)):
        if target not in target_info:
            print(f"  WARNING: target {target} not in manifest, skipping")
            continue

        info = target_info[target]
        pdb_code = info["pdb_code"]
        receptor_chains = info["receptor_chains"]
        peptide_chains = info["peptide_chains"]

        pdb_path = PDB_DIR / f"{pdb_code}.pdb"
        if not pdb_path.exists():
            print(f"  WARNING: PDB {pdb_path} not found, skipping")
            continue

        # Get positions for this target
        target_positions = unique_positions[unique_positions["target"] == target]

        # Preparse PDB once
        chains_data = parse_pdb_atoms(str(pdb_path))

        for _, pos_row in target_positions.iterrows():
            chain_id = pos_row["chain_id"]
            resnum = pos_row["resnum"]
            key = (target, chain_id, resnum)

            all_features[key] = {}

            for cc in CONTACT_CUTOFFS:
                for nc in NEIGHBOR_CUTOFFS:
                    feats = compute_geometric_features(
                        str(pdb_path), chain_id, resnum,
                        receptor_chains, peptide_chains,
                        contact_cutoff=cc,
                        neighbor_cutoff=nc,
                    )
                    all_features[key][(cc, nc)] = (
                        feats["n_iface_contacts"],
                        feats["n_neighbors"],
                    )

        if (tidx + 1) % 10 == 0 or tidx == len(targets_in_cohort) - 1:
            elapsed = time.time() - t0
            print(f"  Processed {tidx + 1}/{len(targets_in_cohort)} targets ({elapsed:.1f}s)")

    print(f"  Feature computation done ({time.time() - t0:.1f}s total)")

    # ------------------------------------------------------------------
    # 4. For each cutoff combination, compute struct_composite and rho
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Evaluating cutoff combinations...")
    print("=" * 70)

    results = []

    for cc in CONTACT_CUTOFFS:
        for nc in NEIGHBOR_CUTOFFS:
            # Build feature columns for this combination
            contacts_vals = []
            neighbors_vals = []
            valid_mask = []

            for idx, row in cohort.iterrows():
                # Use the resolved crystal chain from positions_df
                pos_row = positions_df.iloc[idx]
                target = pos_row["target"]
                chain_id = pos_row["chain_id"]
                resnum = pos_row["resnum"]

                key = (target, chain_id, resnum)

                if key in all_features and (cc, nc) in all_features[key]:
                    n_c, n_n = all_features[key][(cc, nc)]
                    contacts_vals.append(n_c)
                    neighbors_vals.append(n_n)
                    valid_mask.append(True)
                else:
                    contacts_vals.append(np.nan)
                    neighbors_vals.append(np.nan)
                    valid_mask.append(False)

            contacts_arr = np.array(contacts_vals, dtype=float)
            neighbors_arr = np.array(neighbors_vals, dtype=float)
            valid = np.array(valid_mask)

            n_valid = valid.sum()
            if n_valid < 10:
                print(f"  cc={cc:.0f}A, nc={nc:.0f}A: too few valid ({n_valid}), skipping")
                continue

            # Compute struct_composite = zscore(contacts) + zscore(neighbors)
            # Only on valid rows
            working = cohort.copy()
            working["n_contacts_new"] = contacts_arr
            working["n_neighbors_new"] = neighbors_arr

            # Drop rows with NaN in features
            sub = working.dropna(subset=[
                "n_contacts_new", "n_neighbors_new",
                "ddg_exp", "ddg_paired_xint_iface", "ddg_paired_bind_proxy",
                "mpnn_neg_llr_complex", "mpnn_ddg_bind",
            ]).copy()

            if len(sub) < 10:
                continue

            # Compute struct_composite
            c1 = sub["n_contacts_new"].values
            c2 = sub["n_neighbors_new"].values

            # zscore with safety
            c1_z = zscore(c1) if np.std(c1) > 0 else np.zeros_like(c1)
            c2_z = zscore(c2) if np.std(c2) > 0 else np.zeros_like(c2)
            struct_composite_new = c1_z + c2_z

            # --- Structural composite alone ---
            rho_struct, p_struct = spearmanr(sub["ddg_exp"].values, struct_composite_new)

            # --- Physics view score ---
            phys_view = (
                _rankdata(sub["ddg_paired_xint_iface"].values)
                + _rankdata(sub["ddg_paired_bind_proxy"].values)
            )

            # --- MPNN view score ---
            mpnn_view = (
                _rankdata(sub["mpnn_neg_llr_complex"].values)
                + _rankdata(sub["mpnn_ddg_bind"].values)
            )

            # --- 3-view rank-sum ---
            rankscore_phys = _rankdata(phys_view)
            rankscore_struct = _rankdata(struct_composite_new)
            rankscore_mpnn = _rankdata(mpnn_view)
            rankscore_3view = rankscore_phys + rankscore_struct + rankscore_mpnn

            rho_3view, p_3view = spearmanr(sub["ddg_exp"].values, rankscore_3view)

            # --- Bootstrap CI for 3-view ---
            rho_3view_ci, ci_lo, ci_hi = target_stratified_bootstrap_rho(
                sub["ddg_exp"].values,
                rankscore_3view,
                sub["target"].values,
                n_bootstrap=2000,
            )

            # --- Per-target mean rho for structural ---
            per_target_rhos_struct = []
            for t in sub["target"].unique():
                t_data = sub[sub["target"] == t]
                if len(t_data) < 3:
                    continue
                t_idx = t_data.index
                # Get struct_composite values for this target
                t_struct = struct_composite_new[sub.index.get_indexer(t_idx)]
                r, _ = spearmanr(t_data["ddg_exp"].values, t_struct)
                if not np.isnan(r):
                    per_target_rhos_struct.append(r)

            mean_per_target_struct = np.mean(per_target_rhos_struct) if per_target_rhos_struct else np.nan

            is_default = (abs(cc - 8.0) < 0.01 and abs(nc - 10.0) < 0.01)
            marker = " <-- DEFAULT" if is_default else ""

            print(f"  cc={cc:5.1f}A, nc={nc:5.1f}A | "
                  f"struct rho={rho_struct:.4f} | "
                  f"3-view rho={rho_3view:.4f} [{ci_lo:.3f}, {ci_hi:.3f}] | "
                  f"N={len(sub)}{marker}")

            results.append({
                "contact_cutoff_A": cc,
                "neighbor_cutoff_A": nc,
                "n_mutations": len(sub),
                "n_targets": sub["target"].nunique(),
                "rho_struct_composite": rho_struct,
                "p_struct_composite": p_struct,
                "mean_per_target_rho_struct": mean_per_target_struct,
                "rho_3view": rho_3view,
                "p_3view": p_3view,
                "rho_3view_bootstrap": rho_3view_ci,
                "ci_lo_3view": ci_lo,
                "ci_hi_3view": ci_hi,
                "is_default": is_default,
            })

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    out_csv = OUT_DIR / "geometric_sensitivity.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(results_df)} configurations to {out_csv}")

    if len(results_df) == 0:
        print("\nERROR: No valid results. Check target name matching.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Summary analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Best configurations by structural composite rho
    print("\n--- Top 5 by structural composite rho ---")
    top_struct = results_df.nlargest(5, "rho_struct_composite")
    for _, row in top_struct.iterrows():
        default_marker = " [DEFAULT]" if row["is_default"] else ""
        print(f"  cc={row['contact_cutoff_A']:.0f}A, nc={row['neighbor_cutoff_A']:.0f}A: "
              f"rho_struct={row['rho_struct_composite']:.4f}{default_marker}")

    # Best configurations by 3-view rank-sum rho
    print("\n--- Top 5 by 3-view rank-sum rho ---")
    top_3view = results_df.nlargest(5, "rho_3view")
    for _, row in top_3view.iterrows():
        default_marker = " [DEFAULT]" if row["is_default"] else ""
        print(f"  cc={row['contact_cutoff_A']:.0f}A, nc={row['neighbor_cutoff_A']:.0f}A: "
              f"rho_3view={row['rho_3view']:.4f} [{row['ci_lo_3view']:.3f}, {row['ci_hi_3view']:.3f}]{default_marker}")

    # Default configuration
    default_row = results_df[results_df["is_default"]]
    if len(default_row) > 0:
        dr = default_row.iloc[0]
        print(f"\n--- Default configuration (cc=8A, nc=10A) ---")
        print(f"  rho_struct:  {dr['rho_struct_composite']:.4f}")
        print(f"  rho_3view:   {dr['rho_3view']:.4f} [{dr['ci_lo_3view']:.3f}, {dr['ci_hi_3view']:.3f}]")
        print(f"  N mutations: {dr['n_mutations']:.0f}")

    # Range analysis
    print(f"\n--- Range of rho_struct_composite ---")
    print(f"  Min: {results_df['rho_struct_composite'].min():.4f}")
    print(f"  Max: {results_df['rho_struct_composite'].max():.4f}")
    print(f"  Range: {results_df['rho_struct_composite'].max() - results_df['rho_struct_composite'].min():.4f}")

    print(f"\n--- Range of rho_3view ---")
    print(f"  Min: {results_df['rho_3view'].min():.4f}")
    print(f"  Max: {results_df['rho_3view'].max():.4f}")
    print(f"  Range: {results_df['rho_3view'].max() - results_df['rho_3view'].min():.4f}")

    # Heatmap-style table for 3-view rho
    print("\n--- 3-view rho heatmap (contact_cutoff x neighbor_cutoff) ---")
    pivot = results_df.pivot(
        index="contact_cutoff_A",
        columns="neighbor_cutoff_A",
        values="rho_3view",
    )
    print(pivot.to_string(float_format="%.4f"))

    # Sensitivity to contact_cutoff (averaged over neighbor_cutoff)
    print("\n--- Marginal sensitivity to contact_cutoff (avg over neighbor_cutoff) ---")
    for cc in sorted(results_df["contact_cutoff_A"].unique()):
        sub = results_df[results_df["contact_cutoff_A"] == cc]
        mean_rho = sub["rho_3view"].mean()
        std_rho = sub["rho_3view"].std()
        print(f"  cc={cc:5.1f}A: mean_rho_3view={mean_rho:.4f} +/- {std_rho:.4f}")

    # Sensitivity to neighbor_cutoff (averaged over contact_cutoff)
    print("\n--- Marginal sensitivity to neighbor_cutoff (avg over contact_cutoff) ---")
    for nc in sorted(results_df["neighbor_cutoff_A"].unique()):
        sub = results_df[results_df["neighbor_cutoff_A"] == nc]
        mean_rho = sub["rho_3view"].mean()
        std_rho = sub["rho_3view"].std()
        print(f"  nc={nc:5.1f}A: mean_rho_3view={mean_rho:.4f} +/- {std_rho:.4f}")


if __name__ == "__main__":
    main()
