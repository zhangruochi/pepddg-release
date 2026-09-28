#!/usr/bin/env python
"""
PepDDG v5 Phase 1: ProteinMPNN scoring with StaB-ddG thermodynamic decomposition.

Computes the structure-conditioned log-likelihood ratio (LLR) for each mutation
in three contexts: the full complex, the isolated mutated chain, and the
binding-specific component:

    LLR_complex = log P(mut | complex_backbone) - log P(wt | complex_backbone)
    LLR_chain   = log P(mut | isolated_chain)   - log P(wt | isolated_chain)
    DDG_bind    = -LLR_complex + LLR_chain

This decomposition isolates the binding contribution from the overall
structural fitness, following the StaB framework (Buss et al., 2024).

Usage:
    conda run -n research python research/pepddg_v5/scripts/mpnn_scoring.py \
        --cohort research/pepddg_v5/results/phase0/cohort_locked.csv \
        --pdb-dir /data1/nfs/results/pepddg_v5_scoring/predicted \
        --output-csv research/pepddg_v5/results/phase1/mpnn_features.csv \
        --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ProteinMPNN alphabet: 20 standard amino acids + X
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
ALPHABET_DICT = {aa: i for i, aa in enumerate(ALPHABET)}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H", "HSE": "H", "HSD": "H",
    "CYX": "C", "CSS": "C",
}


def apply_temperature_scaling(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to logits and return log-probabilities.

    log_probs = log_softmax(logits / temperature)

    Args:
        logits: Raw logit values (any shape, operates on last axis).
        temperature: Positive float. T<1 sharpens, T>1 smooths. T=1 is standard log-softmax.

    Returns:
        Log-probabilities (same shape as input).

    Raises:
        ValueError: If temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    scaled = logits / temperature
    # Numerical stability: subtract max along last axis
    scaled_max = scaled.max(axis=-1, keepdims=True) if scaled.ndim > 1 else scaled.max()
    scaled = scaled - scaled_max
    log_probs = scaled - np.log(np.sum(np.exp(scaled), axis=-1, keepdims=True) if scaled.ndim > 1 else np.sum(np.exp(scaled)))
    return log_probs


def extract_chain_to_pdb(pdb_path: Path, chain_id: str, out_path: Path) -> None:
    """Extract a single chain from a PDB file, writing only ATOM records."""
    with open(pdb_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") and len(line) > 21 and line[21] == chain_id:
                fout.write(line)
        fout.write("END\n")


def get_chain_sequences_from_pdb(pdb_path: Path) -> Dict[str, Tuple[str, Dict[int, int]]]:
    """
    Parse CA atoms to extract per-chain sequences and residue-number mapping.

    Returns: {chain_id: (sequence, {pdb_resnum: 0-based_index_in_chain})}
    """
    chains: Dict[str, List[Tuple[int, str]]] = {}
    seen = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21]
            resname = line[17:20].strip()
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            aa = THREE_TO_ONE.get(resname)
            if aa is None:
                continue
            key = (chain, resnum)
            if key not in seen:
                seen.add(key)
                chains.setdefault(chain, []).append((resnum, aa))

    result = {}
    for chain, residues in chains.items():
        seq = "".join(aa for _, aa in residues)
        resnum_map = {rn: i for i, (rn, _) in enumerate(residues)}
        result[chain] = (seq, resnum_map)
    return result


def load_mpnn_model(weights_path: str, device: str):
    """Load ProteinMPNN model with standard v_48_020 architecture."""
    import torch
    from .protein_mpnn_utils import ProteinMPNN

    hidden_dim = 128
    num_layers = 3

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")

    num_edges = checkpoint.get("num_edges", 48)

    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,  # no noise for scoring
        k_neighbors=num_edges,
    )

    dev = torch.device(device)
    model.to(dev)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    return model, dev


def score_positions(
    model,
    device,
    pdb_path: Path,
    chain_id: str,
    positions: List[int],
    chain_sequences: Dict[str, Tuple[str, Dict[int, int]]],
) -> Dict[int, np.ndarray]:
    """
    Score specific residue positions using ProteinMPNN conditional_probs.

    Args:
        model: Loaded ProteinMPNN model.
        device: torch device.
        pdb_path: Path to PDB file.
        chain_id: Chain containing the mutation site.
        positions: List of PDB residue numbers to score.
        chain_sequences: Pre-parsed chain sequences and mappings.

    Returns: {pdb_resnum: log_prob_vector (shape 21,)} for each scored position.
    """
    import torch
    from .protein_mpnn_utils import parse_PDB, tied_featurize

    pdb_dict_list = parse_PDB(str(pdb_path))
    if not pdb_dict_list:
        return {}

    pdb_dict = pdb_dict_list[0]

    # Determine chains present in the parsed dict
    parsed_chains = sorted(
        [k.split("_")[-1] for k in pdb_dict if k.startswith("seq_chain_")]
    )
    if not parsed_chains:
        return {}

    # Build chain_dict: all chains are "masked" (designable) so conditional_probs
    # will iterate over positions with chain_M == 1.
    # We want ALL chains as masked so parse_PDB ordering = featurized ordering.
    chain_dict = {pdb_dict["name"]: (parsed_chains, [])}

    # Compute cumulative offset for each chain in the concatenated sequence
    chain_offsets = {}
    offset = 0
    for ch in parsed_chains:
        chain_key = f"seq_chain_{ch}"
        if chain_key in pdb_dict:
            chain_offsets[ch] = offset
            offset += len(pdb_dict[chain_key])

    # Map PDB residue numbers to sequence indices
    if chain_id not in chain_sequences:
        return {}
    _, resnum_map = chain_sequences[chain_id]
    if chain_id not in chain_offsets:
        return {}
    chain_offset = chain_offsets[chain_id]

    # Build the set of sequence-level indices we need to score
    target_seq_indices = []
    resnum_to_seqidx = {}
    for rn in positions:
        if rn in resnum_map:
            seq_idx = chain_offset + resnum_map[rn]
            target_seq_indices.append(seq_idx)
            resnum_to_seqidx[rn] = seq_idx

    if not target_seq_indices:
        return {}

    # Featurize: batch of 1
    batch = [pdb_dict]
    (
        X, S, mask, lengths, chain_M, chain_encoding_all,
        *_rest
    ) = tied_featurize(batch, device, chain_dict)

    # Override chain_M: set to 1 ONLY at target positions for efficiency.
    # conditional_probs() loops over positions where chain_M[0,:]==1.
    chain_M_custom = torch.zeros_like(chain_M)
    for si in target_seq_indices:
        chain_M_custom[0, si] = 1.0

    # Extract residue_idx and chain_encoding_all from featurize output
    # _rest = [visible_list_list, masked_list_list, masked_chain_length_list_list,
    #          chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask,
    #          tied_pos_list_of_lists_list, pssm_coef, pssm_bias,
    #          pssm_log_odds_all, bias_by_res_all, tied_beta]
    # Full unpack from tied_featurize:
    # X, S, mask, lengths, chain_M, chain_encoding_all,
    # letter_list_list, visible_list_list, masked_list_list,
    # masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
    # residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
    # pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta
    (
        letter_list_list, visible_list_list, masked_list_list,
        masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
        residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
        pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta
    ) = _rest

    randn = torch.randn(chain_M_custom.shape, device=device)

    with torch.no_grad():
        log_cond_probs = model.conditional_probs(
            X, S, mask, chain_M_custom, residue_idx, chain_encoding_all, randn
        )
        # shape: (1, seq_len, 21)

    # Extract per-position log probs
    result = {}
    log_p_np = log_cond_probs[0].cpu().numpy()  # (seq_len, 21)
    for rn, si in resnum_to_seqidx.items():
        result[rn] = log_p_np[si]  # shape (21,)

    return result


def verify_aa(
    log_probs_vec: np.ndarray,
    expected_aa: str,
    pdb_seq_aa: str,
    target: str,
    chain: str,
    resnum: int,
) -> bool:
    """Verify the amino acid identity matches between cohort and PDB."""
    if expected_aa != pdb_seq_aa:
        print(
            f"  WARNING: {target} {chain}:{resnum} cohort says {expected_aa} "
            f"but PDB has {pdb_seq_aa} — skipping"
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="ProteinMPNN scoring with StaB-ddG decomposition"
    )
    parser.add_argument("--cohort", required=True, help="Path to cohort_locked.csv")
    parser.add_argument("--pdb-dir", required=True, help="NFS predicted PDB directory")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to model weights (default: auto-detect v_48_020.pt)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax scaling (default: 1.0). T<1 sharpens, T>1 smooths.",
    )
    args = parser.parse_args()

    # Add ProteinMPNN to path
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]  # research/pepddg_v5/scripts -> repo root
    mpnn_dir = repo_root / "third-party" / "ProteinMPNN"
    sys.path.insert(0, str(mpnn_dir))

    weights_path = args.weights
    if weights_path is None:
        weights_path = str(mpnn_dir / "vanilla_model_weights" / "v_48_020.pt")

    print(f"Loading ProteinMPNN from {mpnn_dir}")
    print(f"Weights: {weights_path}")
    print(f"Device: {args.device}")
    if args.temperature != 1.0:
        print(f"Temperature scaling: {args.temperature}")

    model, device = load_mpnn_model(weights_path, args.device)
    print("Model loaded successfully.")

    # Load cohort
    df = pd.read_csv(args.cohort)
    print(f"Loaded {len(df)} mutations from {df['target'].nunique()} targets")

    pdb_dir = Path(args.pdb_dir)

    # Build target -> directory mapping
    target_dir_map: Dict[str, Path] = {}
    for d in pdb_dir.iterdir():
        if d.is_dir():
            code = d.name.split("_")[0]
            target_dir_map[code] = d

    # Output arrays
    n = len(df)
    llr_complex = np.full(n, np.nan)
    llr_chain = np.full(n, np.nan)
    ddg_bind = np.full(n, np.nan)
    neg_llr_complex = np.full(n, np.nan)
    wt_logp_complex = np.full(n, np.nan)
    wt_logp_chain = np.full(n, np.nan)

    t0 = time.time()

    for ti, target in enumerate(sorted(df["target"].unique())):
        target_subdir = target_dir_map.get(target)
        if target_subdir is None:
            print(f"  [{ti+1}] {target}: no directory found, skipping")
            continue

        wt_pdb = target_subdir / "variants" / "WT.pdb"
        if not wt_pdb.exists():
            print(f"  [{ti+1}] {target}: no WT.pdb, skipping")
            continue

        # Parse chain info from PDB
        chain_seqs = get_chain_sequences_from_pdb(wt_pdb)
        if not chain_seqs:
            print(f"  [{ti+1}] {target}: failed to parse chains, skipping")
            continue

        # Get all mutations for this target
        target_mask = df["target"] == target
        target_df = df[target_mask]
        target_indices = np.where(target_mask)[0]

        # Group mutations by chain_id
        chain_groups: Dict[str, List[Tuple[int, int, str, str]]] = {}
        for row_idx, (_, row) in zip(target_indices, target_df.iterrows()):
            ch = str(row["chain_id"])
            rn = int(row["resnum"])
            wt_aa = str(row["wt_aa1"])
            mut_aa = str(row["mut_aa1"])
            chain_groups.setdefault(ch, []).append((row_idx, rn, wt_aa, mut_aa))

        # --- Score complex ---
        # Collect all unique (chain, resnum) pairs to score
        complex_positions_by_chain: Dict[str, List[int]] = {}
        for ch, mutations in chain_groups.items():
            resnums = sorted(set(rn for _, rn, _, _ in mutations))
            complex_positions_by_chain[ch] = resnums

        # Score all needed positions in the complex in one pass per chain
        complex_log_probs: Dict[str, Dict[int, np.ndarray]] = {}
        for ch, resnums in complex_positions_by_chain.items():
            complex_log_probs[ch] = score_positions(
                model, device, wt_pdb, ch, resnums, chain_seqs
            )

        # --- Score isolated chains ---
        chain_log_probs: Dict[str, Dict[int, np.ndarray]] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for ch, resnums in complex_positions_by_chain.items():
                chain_pdb = Path(tmpdir) / f"chain_{ch}.pdb"
                extract_chain_to_pdb(wt_pdb, ch, chain_pdb)

                # Parse the isolated chain PDB
                chain_seqs_isolated = get_chain_sequences_from_pdb(chain_pdb)
                if ch not in chain_seqs_isolated:
                    print(
                        f"  [{ti+1}] {target}: chain {ch} extraction failed, skipping"
                    )
                    continue

                chain_log_probs[ch] = score_positions(
                    model, device, chain_pdb, ch, resnums, chain_seqs_isolated
                )

        # --- Compute features for each mutation ---
        n_scored = 0
        for ch, mutations in chain_groups.items():
            if ch not in chain_seqs:
                continue
            ch_seq, ch_resnum_map = chain_seqs[ch]

            for row_idx, rn, wt_aa, mut_aa in mutations:
                # Verify AA identity
                if rn not in ch_resnum_map:
                    continue
                pdb_aa = ch_seq[ch_resnum_map[rn]]
                if not verify_aa(None, wt_aa, pdb_aa, target, ch, rn):
                    continue

                wt_tok = ALPHABET_DICT.get(wt_aa)
                mut_tok = ALPHABET_DICT.get(mut_aa)
                if wt_tok is None or mut_tok is None:
                    continue

                # Complex log probs
                if ch in complex_log_probs and rn in complex_log_probs[ch]:
                    lp_complex = complex_log_probs[ch][rn]
                    # Apply temperature scaling (re-normalizes log-probs)
                    lp_complex = apply_temperature_scaling(
                        lp_complex, temperature=args.temperature
                    )
                    lp_wt_c = lp_complex[wt_tok]
                    lp_mut_c = lp_complex[mut_tok]
                    llr_c = lp_mut_c - lp_wt_c
                    llr_complex[row_idx] = llr_c
                    neg_llr_complex[row_idx] = -llr_c
                    wt_logp_complex[row_idx] = lp_wt_c
                else:
                    continue

                # Isolated chain log probs
                if ch in chain_log_probs and rn in chain_log_probs[ch]:
                    lp_chain = chain_log_probs[ch][rn]
                    # Apply temperature scaling (re-normalizes log-probs)
                    lp_chain = apply_temperature_scaling(
                        lp_chain, temperature=args.temperature
                    )
                    lp_wt_ch = lp_chain[wt_tok]
                    lp_mut_ch = lp_chain[mut_tok]
                    llr_ch = lp_mut_ch - lp_wt_ch
                    llr_chain[row_idx] = llr_ch
                    wt_logp_chain[row_idx] = lp_wt_ch

                    # DDG_bind = -LLR_complex + LLR_chain
                    ddg_bind[row_idx] = -llr_c + llr_ch
                    n_scored += 1

        elapsed = time.time() - t0
        print(
            f"  [{ti+1}/{df['target'].nunique()}] {target}: "
            f"{n_scored}/{len(target_df)} scored "
            f"({elapsed:.1f}s elapsed)"
        )

    # Build output DataFrame
    result_df = df[["target", "mut", "mutation_id", "ddg_exp"]].copy()
    result_df["mpnn_llr_complex"] = llr_complex
    result_df["mpnn_llr_chain"] = llr_chain
    result_df["mpnn_ddg_bind"] = ddg_bind
    result_df["mpnn_neg_llr_complex"] = neg_llr_complex
    result_df["mpnn_wt_logp_complex"] = wt_logp_complex
    result_df["mpnn_wt_logp_chain"] = wt_logp_chain

    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    n_scored_total = (~np.isnan(ddg_bind)).sum()
    print(
        f"\nScored {n_scored_total}/{len(df)} mutations "
        f"({n_scored_total / len(df) * 100:.1f}%)"
    )

    # Quick standalone evaluation
    from scipy.stats import spearmanr

    valid = result_df.dropna(subset=["ddg_exp", "mpnn_neg_llr_complex"])
    if len(valid) >= 3:
        rho_neg, pval_neg = spearmanr(valid["ddg_exp"], valid["mpnn_neg_llr_complex"])
        print(f"\nProteinMPNN standalone evaluation:")
        print(
            f"  neg_llr_complex vs ddg_exp: rho={rho_neg:.4f} "
            f"(p={pval_neg:.2e}, N={len(valid)})"
        )

    valid_bind = result_df.dropna(subset=["ddg_exp", "mpnn_ddg_bind"])
    if len(valid_bind) >= 3:
        rho_bind, pval_bind = spearmanr(
            valid_bind["ddg_exp"], valid_bind["mpnn_ddg_bind"]
        )
        print(
            f"  ddg_bind vs ddg_exp:        rho={rho_bind:.4f} "
            f"(p={pval_bind:.2e}, N={len(valid_bind)})"
        )

    valid_llr = result_df.dropna(subset=["ddg_exp", "mpnn_llr_chain"])
    if len(valid_llr) >= 3:
        rho_chain, pval_chain = spearmanr(
            valid_llr["ddg_exp"], -valid_llr["mpnn_llr_chain"]
        )
        print(
            f"  neg_llr_chain vs ddg_exp:   rho={rho_chain:.4f} "
            f"(p={pval_chain:.2e}, N={len(valid_llr)})"
        )

    # Gate P1a check
    print(f"\n--- Gate P1a: mpnn_neg_llr_complex rho > 0.239 (beat ESM-IF1) ---")
    if len(valid) >= 3:
        gate_pass = rho_neg > 0.239
        print(f"  rho = {rho_neg:.4f} -> {'PASS' if gate_pass else 'FAIL'}")
    else:
        print("  Insufficient data for gate check")

    elapsed_total = time.time() - t0
    print(f"\nTotal time: {elapsed_total:.1f}s")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
