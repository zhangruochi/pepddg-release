"""Token mapping from SKEMPI mutations to Boltz-2 PAE matrix indices.

Maps SKEMPI mutation identifiers (target, chain, resid) to Boltz-2 token
indices by reading the Boltz-2 manifest.json and matching chain lengths
against the SKEMPI peptide_length.

Key insight: SKEMPI chain letters do NOT always match Boltz-2 chain letters
(e.g., 3EQY: SKEMPI "chain B" = 12-residue peptide, but Boltz-2 "chain B"
is the 84-residue homodimer partner; the peptide is Boltz-2 "chain C").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class BoltzChainInfo:
    """Chain information extracted from a Boltz-2 manifest."""
    chain_name: str
    chain_id: int
    num_residues: int
    entity_id: int
    token_start: int  # inclusive
    token_end: int    # exclusive


@dataclass
class TokenMapping:
    """Complete token mapping for a target prediction."""
    target: str
    chains: list[BoltzChainInfo]
    peptide_chain: BoltzChainInfo
    receptor_chains: list[BoltzChainInfo]
    total_tokens: int

    @property
    def receptor_token_indices(self) -> np.ndarray:
        """All token indices belonging to receptor chain(s)."""
        indices = []
        for c in self.receptor_chains:
            indices.extend(range(c.token_start, c.token_end))
        return np.array(indices, dtype=int)

    def mutation_token_index(self, resid: int) -> int:
        """Map a 1-indexed residue number to a Boltz-2 token index.

        Args:
            resid: 1-indexed residue number within the peptide chain
                   (as it appears in SKEMPI mutation notation).

        Returns:
            Boltz-2 token index (0-indexed into the PAE matrix).

        Raises:
            ValueError: If resid is out of range for the peptide chain.
        """
        if resid < 1 or resid > self.peptide_chain.num_residues:
            raise ValueError(
                f"Residue {resid} out of range for peptide chain "
                f"(1-{self.peptide_chain.num_residues})"
            )
        return self.peptide_chain.token_start + (resid - 1)


def parse_manifest(manifest_path: str | Path) -> list[BoltzChainInfo]:
    """Parse a Boltz-2 manifest.json to extract chain information with token offsets.

    Args:
        manifest_path: Path to the manifest.json file from Boltz-2 processed/ dir.

    Returns:
        List of BoltzChainInfo ordered by chain_id (= token order).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    record = manifest["records"][0]
    chains_raw = sorted(record["chains"], key=lambda c: c["chain_id"])

    chains = []
    offset = 0
    for c in chains_raw:
        n = c["num_residues"]
        chains.append(BoltzChainInfo(
            chain_name=c["chain_name"],
            chain_id=c["chain_id"],
            num_residues=n,
            entity_id=c["entity_id"],
            token_start=offset,
            token_end=offset + n,
        ))
        offset += n

    return chains


def build_token_mapping(
    manifest_path: str | Path,
    peptide_length: int,
    target: str = "",
) -> TokenMapping:
    """Build a complete token mapping for a target by matching peptide length.

    Identifies the peptide chain as the chain whose num_residues matches
    peptide_length. All other chains are treated as receptor.

    Args:
        manifest_path: Path to the Boltz-2 manifest.json.
        peptide_length: Expected peptide chain length from SKEMPI data.
        target: Target identifier (for error messages).

    Returns:
        TokenMapping with peptide and receptor chains identified.

    Raises:
        ValueError: If no chain or multiple chains match peptide_length.
    """
    chains = parse_manifest(manifest_path)

    # Find the peptide chain by matching length
    matches = [c for c in chains if c.num_residues == peptide_length]

    if len(matches) == 0:
        chain_sizes = [(c.chain_name, c.num_residues) for c in chains]
        raise ValueError(
            f"No chain with {peptide_length} residues found for {target}. "
            f"Available: {chain_sizes}"
        )
    if len(matches) > 1:
        # Multiple chains with same length — pick the last one
        # (in our YAML convention, peptide is the last chain)
        matches = [matches[-1]]

    peptide_chain = matches[0]
    receptor_chains = [c for c in chains if c.chain_id != peptide_chain.chain_id]

    total = sum(c.num_residues for c in chains)

    return TokenMapping(
        target=target,
        chains=chains,
        peptide_chain=peptide_chain,
        receptor_chains=receptor_chains,
        total_tokens=total,
    )


def load_best_pae(
    predictions_dir: str | Path,
    prediction_name: str,
    n_models: int = 3,
) -> tuple[np.ndarray, int]:
    """Load the PAE matrix from the best model (highest ipTM).

    Args:
        predictions_dir: Path to the predictions/<name>/ directory.
        prediction_name: Name of the prediction (e.g., "3EQY_WT").
        n_models: Number of diffusion samples.

    Returns:
        Tuple of (PAE matrix [N_tokens, N_tokens], best model index).
    """
    predictions_dir = Path(predictions_dir)

    best_iptm = -1.0
    best_model = 0
    for m in range(n_models):
        conf_path = predictions_dir / f"confidence_{prediction_name}_model_{m}.json"
        with open(conf_path) as f:
            conf = json.load(f)
        iptm = conf["iptm"]
        if iptm > best_iptm:
            best_iptm = iptm
            best_model = m

    pae_path = predictions_dir / f"pae_{prediction_name}_model_{best_model}.npz"
    pae = np.load(pae_path)["pae"]

    return pae, best_model


def extract_pae_features(
    pae: np.ndarray,
    mapping: TokenMapping,
    resid: int,
) -> dict[str, float]:
    """Extract PAE-derived features for a mutation at a given residue.

    Args:
        pae: PAE matrix [N_tokens, N_tokens].
        mapping: Token mapping for this target.
        resid: 1-indexed residue number of the mutation in the peptide.

    Returns:
        Dictionary of feature names to values.
    """
    mut_tok = mapping.mutation_token_index(resid)
    rec_idx = mapping.receptor_token_indices

    pae_to_rec = pae[mut_tok, rec_idx]

    # Primary: mean of 5 lowest PAE values to receptor
    top5 = np.sort(pae_to_rec)[:5]
    pae_mut_to_rec_top5 = float(np.mean(top5))

    # Secondary: mean PAE to all receptor tokens
    pae_mut_to_rec_mean = float(np.mean(pae_to_rec))

    # Secondary: std of PAE to receptor (interaction specificity)
    pae_mut_row_std = float(np.std(pae_to_rec))

    return {
        "pae_mut_to_rec_top5": pae_mut_to_rec_top5,
        "pae_mut_to_rec_mean": pae_mut_to_rec_mean,
        "pae_mut_row_std": pae_mut_row_std,
    }
