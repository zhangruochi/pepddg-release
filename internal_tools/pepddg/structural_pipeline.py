"""Generate all three PepDDG channels from a validated linear complex."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .api import ScoreResult, score_features
from .mpnn_features import compute_mpnn_ddg_features
from .structure_contract import ComplexSpec, UnsupportedChemistry, validate_complex
from .structural_features import MutationSite, compute_structural_features_batch


_BIND_KEY = "dg_bind_kcal_mol_restarts"
_XINT_KEY = "e_cross_interface_total_screened_kcal_mol_restarts"


@dataclass(frozen=True)
class StructuralResult:
    features: pd.DataFrame
    scores: ScoreResult
    provenance: dict[str, Any]


def paired_restart_ddg(wild_type: Sequence[float], mutant: Sequence[float]) -> float:
    """Historical paired-index median, failing closed below three finite pairs."""
    if len(wild_type) != len(mutant):
        raise ValueError("WT and mutant restart arrays must have equal length")
    pairs = []
    for wt, mut in zip(wild_type, mutant):
        if wt is not None and mut is not None and math.isfinite(float(wt)) and math.isfinite(float(mut)):
            pairs.append(float(mut) - float(wt))
    if len(pairs) < 3:
        raise ValueError("at least three common finite restart pairs are required")
    return float(np.median(pairs))


def _selected_pdb(spec: ComplexSpec, destination: Path) -> Path:
    import gemmi

    selected = set(spec.receptor_chains) | {spec.peptide_chain}
    if any(len(chain) != 1 for chain in selected):
        raise UnsupportedChemistry("PDB backend requires one-character chain IDs")
    source = Path(spec.structure_path)
    if source.suffix.lower() == ".pdb":
        records = [
            line for line in source.read_text().splitlines(keepends=True)
            if line.startswith("ATOM  ") and len(line) >= 22 and line[21] in selected
        ]
        destination.write_text("".join(records) + "TER\nEND\n")
    else:
        structure = gemmi.read_structure(str(source))
        for chain in list(structure[0]):
            if chain.name not in selected:
                structure[0].remove_chain(chain.name)
        structure.write_pdb(str(destination))
        records = [
            line for line in destination.read_text().splitlines(keepends=True)
            if line.startswith("ATOM  ") and len(line) >= 22 and line[21] in selected
        ]
        destination.write_text("".join(records) + "TER\nEND\n")
    if not destination.is_file() or not any(
        line.startswith("ATOM  ") for line in destination.read_text().splitlines()
    ):
        raise ValueError("selected receptor-peptide PDB has no atoms")
    return destination


def _build_variant_pdb(base: Path, variant: Any, output: Path) -> str:
    from ._backend.variant_builder import build_variant_pdb

    return build_variant_pdb(str(base), variant, str(output))


def _score_openmm(pdb_path: str, receptor_chain: str, ligand_chain: str, **kwargs: Any) -> dict[str, Any]:
    from ._backend.implicit_relax_score import compute_binding_energy_implicit

    return compute_binding_energy_implicit(pdb_path, receptor_chain, ligand_chain, **kwargs)


def _score_mpnn(wt_pdb: Path, spec: ComplexSpec, seed: int, checkpoint: Path) -> dict[str, tuple[float, float]]:
    import torch
    from ._backend.mpnn_scoring_legacy import (
        ALPHABET_DICT,
        extract_chain_to_pdb,
        get_chain_sequences_from_pdb,
        load_mpnn_model,
        score_positions,
    )

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model, device = load_mpnn_model(str(checkpoint), "cpu")
    positions = sorted({mutation.resnum for mutation in spec.mutations})
    peptide = spec.peptide_chain
    sequences = get_chain_sequences_from_pdb(wt_pdb)
    if peptide not in sequences:
        raise ValueError("ProteinMPNN WT peptide chain is missing")
    torch.manual_seed(seed)
    complex_probs = score_positions(model, device, wt_pdb, peptide, positions, sequences)
    with tempfile.TemporaryDirectory(prefix="pepddg-mpnn-") as directory:
        isolated = Path(directory) / "peptide.pdb"
        extract_chain_to_pdb(wt_pdb, peptide, isolated)
        isolated_sequences = get_chain_sequences_from_pdb(isolated)
        chain_probs = score_positions(model, device, isolated, peptide, positions, isolated_sequences)
    out: dict[str, tuple[float, float]] = {}
    sequence, numbering = sequences[peptide]
    for mutation in spec.mutations:
        if mutation.resnum not in numbering or sequence[numbering[mutation.resnum]] != mutation.wt:
            raise ValueError(f"ProteinMPNN WT identity mismatch: {mutation.label}")
        if mutation.resnum not in complex_probs or mutation.resnum not in chain_probs:
            raise ValueError(f"ProteinMPNN missing mutation position: {mutation.label}")
        complex_logp = complex_probs[mutation.resnum]
        chain_logp = chain_probs[mutation.resnum]
        features = compute_mpnn_ddg_features(
            float(complex_logp[ALPHABET_DICT[mutation.wt]]),
            float(complex_logp[ALPHABET_DICT[mutation.mut]]),
            float(chain_logp[ALPHABET_DICT[mutation.wt]]),
            float(chain_logp[ALPHABET_DICT[mutation.mut]]),
        )
        out[mutation.label] = (features["mpnn_neg_llr_complex"], features["mpnn_ddg_bind"])
    return out


def run_structural_cohort(
    spec: ComplexSpec,
    *,
    target: str,
    parent_id: str,
    output_dir: str | Path,
    n_restarts: int = 7,
    seed: int = 20260302,
    platform: str = "CPU",
    cpu_threads: int = 2,
) -> StructuralResult:
    """Score a complete one-parent linear substitution cohort from coordinates.

    Each WT/mutant energy uses the historical two-stage OBC2 protocol. Outputs
    publish only after all features and cohort scores pass validation.
    """
    validation = validate_complex(spec)
    if len(spec.receptor_chains) != 1:
        raise UnsupportedChemistry("select one receptor chain for this published protocol")
    if any(mutation.icode for mutation in spec.mutations):
        raise UnsupportedChemistry("insertion codes are not supported by the mutation builder")
    if not target.strip() or not parent_id.strip():
        raise ValueError("target and parent_id must be nonempty")
    if n_restarts < 3 or cpu_threads < 1 or cpu_threads > 4:
        raise ValueError("n_restarts must be >=3 and cpu_threads must be 1..4")
    if platform.upper() not in {"CPU", "CUDA"}:
        raise ValueError("platform must be CPU or CUDA")

    from ._backend.variant_builder import VariantSpec

    weights = Path(__file__).parent / "_backend/model_weights/v_48_020.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"ProteinMPNN v_48_020 checkpoint is absent: {weights}")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("output directory must be absent or empty")
    exclusions = [f"{mutation.chain}:{mutation.resnum}" for mutation in spec.mutations]
    exclusions = list(dict.fromkeys(exclusions))
    kwargs = dict(
        platform=platform.upper(), cpu_threads=cpu_threads, n_restarts=n_restarts,
        jitter_nm=0.005, aggregate="median", min_protocol="twostage",
        stage1_iters=200, stage2_iters=500, restraint_k1=1000.0,
        restraint_k2=100.0, cross_interaction_interface_cutoff_a=8.0,
        coulomb_screening_distance_nm=0.10, coulomb_screening_dielectric=80.0,
        restraint_exclusion_distance_a=10.0, random_seed=seed,
    )
    with tempfile.TemporaryDirectory(prefix="pepddg-structural-") as scratch:
        work = Path(scratch)
        base = _selected_pdb(spec, work / "selected.pdb")
        wt_pdb = Path(_build_variant_pdb(base, None, work / "wt"))
        wt = _score_openmm(
            str(wt_pdb), spec.receptor_chains[0], spec.peptide_chain,
            **kwargs, restraint_exclusion_residues=exclusions,
        )
        mpnn = _score_mpnn(wt_pdb, spec, seed, weights)
        sites = [MutationSite(m.chain, m.resnum, m.wt, m.mut, m.label) for m in spec.mutations]
        geometry = compute_structural_features_batch(
            base, sites, spec.peptide_chain, list(spec.receptor_chains)
        ).set_index("label")
        rows = []
        for mutation in spec.mutations:
            variant = VariantSpec(mutation.wt, mutation.chain, mutation.resnum, mutation.mut)
            mutant_pdb = _build_variant_pdb(base, variant, work / mutation.label)
            mutant = _score_openmm(
                mutant_pdb, spec.receptor_chains[0], spec.peptide_chain,
                **kwargs, restraint_exclusion_residues=[f"{mutation.chain}:{mutation.resnum}"],
            )
            for key in (_BIND_KEY, _XINT_KEY):
                if key not in wt or key not in mutant:
                    raise ValueError(f"OpenMM missing restart channel {key}: {mutation.label}")
                if len(wt[key]) != n_restarts or len(mutant[key]) != n_restarts:
                    raise ValueError(f"OpenMM incomplete restart channel {key}: {mutation.label}")
            rows.append({
                "target": target.strip(), "parent_id": parent_id.strip(), "mutation": mutation.label,
                "ddg_bind_proxy": paired_restart_ddg(wt[_BIND_KEY], mutant[_BIND_KEY]),
                "ddg_xint_iface": paired_restart_ddg(wt[_XINT_KEY], mutant[_XINT_KEY]),
                "n_iface_contacts_8a": int(geometry.loc[mutation.label, "n_iface_contacts_8a"]),
                "n_neighbors_10a": int(geometry.loc[mutation.label, "n_neighbors_10a"]),
                "mpnn_neg_llr_complex": mpnn[mutation.label][0],
                "mpnn_ddg_bind": mpnn[mutation.label][1],
            })
    features = pd.DataFrame(rows)
    scores = score_features(features)
    provenance = {
        "status": "ok", "scope": "linear_single_receptor_cohort",
        "structure_sha256": validation.structure_sha256,
        "excluded_water_atoms": validation.excluded_water_atoms,
        "checkpoint_sha256": sha256(weights.read_bytes()).hexdigest(),
        "n_restarts": n_restarts, "seed": seed, "platform": platform.upper(),
        "mpnn_rng": "torch manual seed before complex; isolated chain consumes continued stream",
        "physics_channels": {"bind": _BIND_KEY, "interface": _XINT_KEY},
        "physics_protocol": kwargs, "mutation_ids": list(validation.mutation_ids),
        "historical_reproduction_status": "unverified",
    }
    destination.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination / "features.csv", index=False)
    scores.table.to_csv(destination / "scores.csv", index=False)
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return StructuralResult(features, scores, provenance)
