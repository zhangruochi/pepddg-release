"""Generate all three PepDDG channels from a validated linear complex."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .api import FEATURE_COLUMNS, IDENTITY_COLUMNS, ScoreResult, score_features
from .mpnn_features import compute_mpnn_ddg_features
from .structure_contract import ComplexSpec, UnsupportedChemistry, validate_complex
from .structural_features import MutationSite, compute_structural_features_batch


_BIND_KEY = "dg_bind_kcal_mol_restarts"
_XINT_KEY = "e_cross_interface_total_screened_kcal_mol_restarts"
_CHECKPOINT_SCHEMA = "pepddg-structural-checkpoint/v1"


@dataclass(frozen=True)
class StructuralResult:
    features: pd.DataFrame
    scores: ScoreResult
    provenance: dict[str, Any]


def _digest_json(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _normalized_checkpoint(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalized_checkpoint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_normalized_checkpoint(item) for item in value]
    if isinstance(value, np.generic):
        return _normalized_checkpoint(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_checkpoint(path: Path, identity: str, payload: Any) -> None:
    payload = _normalized_checkpoint(payload)
    record = {"identity_sha256": identity, "payload": payload, "payload_sha256": _digest_json(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_checkpoint(path: Path, identity: str) -> Any:
    try:
        record = json.loads(path.read_text())
        if record["identity_sha256"] != identity:
            raise ValueError("checkpoint identity mismatch")
        if record["payload_sha256"] != _digest_json(record["payload"]):
            raise ValueError("checkpoint payload hash mismatch")
        return record["payload"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint: {path}") from exc


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
    wt_restraint_scope: str = "per_mutation",
    wt_union_sites: Sequence[str] | None = None,
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
    if wt_restraint_scope not in {"per_mutation", "target_union"}:
        raise ValueError("wt_restraint_scope must be per_mutation or target_union")
    if wt_restraint_scope != "target_union" and wt_union_sites is not None:
        raise ValueError("wt_union_sites requires target_union scope")

    requested_sites = tuple(dict.fromkeys(
        f"{mutation.chain}:{mutation.resnum}" for mutation in spec.mutations
    ))
    if wt_union_sites is None:
        union_exclusion = requested_sites
    else:
        if isinstance(wt_union_sites, (str, bytes)):
            raise ValueError("wt_union_sites must be a sequence of chain:resnum tokens")
        allowed_numbers = {number for number, _, _ in validation.peptide_residues}
        explicit_sites = []
        for site in wt_union_sites:
            if not isinstance(site, str):
                raise ValueError("wt_union_sites must contain chain:resnum strings")
            chain, separator, number = site.partition(":")
            if (separator != ":" or chain != spec.peptide_chain
                    or not number.isdecimal() or int(number) not in allowed_numbers):
                raise ValueError(f"invalid WT union site: {site!r}")
            explicit_sites.append(site)
        union_exclusion = tuple(dict.fromkeys(explicit_sites))
        if not set(requested_sites).issubset(union_exclusion):
            raise ValueError("wt_union_sites omits a scored mutation site")

    from ._backend.variant_builder import VariantSpec

    weights = Path(__file__).parent / "_backend/model_weights/v_48_020.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"ProteinMPNN v_48_020 checkpoint is absent: {weights}")
    destination = Path(output_dir)
    kwargs = dict(
        platform=platform.upper(), cpu_threads=cpu_threads, n_restarts=n_restarts,
        jitter_nm=0.005, aggregate="median", min_protocol="twostage",
        stage1_iters=200, stage2_iters=500, restraint_k1=1000.0,
        restraint_k2=100.0, cross_interaction_interface_cutoff_a=8.0,
        coulomb_screening_distance_nm=0.10, coulomb_screening_dielectric=80.0,
        restraint_exclusion_distance_a=10.0, random_seed=seed,
        wt_restraint_scope=wt_restraint_scope,
        wt_union_sites=union_exclusion if wt_restraint_scope == "target_union" else None,
    )
    physics_kwargs = {
        key: value for key, value in kwargs.items()
        if key not in {"wt_restraint_scope", "wt_union_sites"}
    }
    source_files = [
        Path(__file__), Path(__file__).parent / "api.py",
        Path(__file__).parent / "config.py",
        Path(__file__).parent / "scoring.py",
        Path(__file__).parent / "structure_contract.py",
        Path(__file__).parent / "structure_io.py",
        Path(__file__).parent / "structural_features.py",
        Path(__file__).parent / "mpnn_features.py",
        Path(__file__).parent / "_backend/implicit_relax_score.py",
        Path(__file__).parent / "_backend/variant_builder.py",
        Path(__file__).parent / "_backend/mpnn_scoring_legacy.py",
        Path(__file__).parent / "_backend/protein_mpnn_utils.py",
    ]
    identity = _digest_json({
        "schema": _CHECKPOINT_SCHEMA, "structure_sha256": validation.structure_sha256,
        "target": target.strip(), "parent_id": parent_id.strip(),
        "peptide_chain": spec.peptide_chain, "receptor_chains": spec.receptor_chains,
        "mutations": [mutation.__dict__ for mutation in spec.mutations],
        "closure": spec.closure_kind, "protocol": kwargs,
        "checkpoint_sha256": sha256(weights.read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(Path(__file__).parent)): sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
    })
    checkpoint_dir = destination / ".pepddg-work"
    manifest = checkpoint_dir / "manifest.json"
    publishing = checkpoint_dir / "publishing.json"
    if destination.exists():
        entries = {path.name for path in destination.iterdir()}
        if checkpoint_dir.exists() and not manifest.is_file():
            raise ValueError("checkpoint manifest is missing")
        if manifest.is_file():
            saved = _load_checkpoint(manifest, identity)
            if saved != {"schema": _CHECKPOINT_SCHEMA}:
                raise ValueError("checkpoint manifest schema mismatch")
        final_entries = entries - {checkpoint_dir.name}
        if final_entries:
            if not publishing.is_file():
                raise FileExistsError("output directory already contains final or unrelated files")
            publication = _load_checkpoint(publishing, identity)
            if publication.get("schema") != "pepddg-publication/v1" or set(publication.get("files", {})) != {
                "features.csv", "scores.csv", "provenance.json"
            } or not final_entries.issubset(publication["files"]):
                raise ValueError("invalid publication checkpoint")
            for name in final_entries:
                if sha256((destination / name).read_bytes()).hexdigest() != publication["files"][name]:
                    raise ValueError(f"published file hash mismatch: {name}")
    checkpoint_dir.joinpath("mutations").mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        _save_checkpoint(manifest, identity, {"schema": _CHECKPOINT_SCHEMA})
    with tempfile.TemporaryDirectory(prefix="pepddg-structural-") as scratch:
        work = Path(scratch)
        base = _selected_pdb(spec, work / "selected.pdb")
        wt_pdb = Path(_build_variant_pdb(base, None, work / "wt"))
        mpnn_checkpoint = checkpoint_dir / "mpnn.json"
        if mpnn_checkpoint.exists():
            mpnn = _load_checkpoint(mpnn_checkpoint, identity)
        else:
            mpnn = _score_mpnn(wt_pdb, spec, seed, weights)
            _save_checkpoint(mpnn_checkpoint, identity, mpnn)
        sites = [MutationSite(m.chain, m.resnum, m.wt, m.mut, m.label) for m in spec.mutations]
        geometry = compute_structural_features_batch(
            base, sites, spec.peptide_chain, list(spec.receptor_chains)
        ).set_index("label")
        rows = []
        shared_wt = None
        shared_wt_checkpoint = checkpoint_dir / "wt.target_union.json"
        if wt_restraint_scope == "target_union":
            if shared_wt_checkpoint.exists():
                shared_wt = _load_checkpoint(shared_wt_checkpoint, identity)
                if not isinstance(shared_wt, dict) or any(
                    key not in shared_wt or not isinstance(shared_wt[key], list)
                    or len(shared_wt[key]) != n_restarts
                    for key in (_BIND_KEY, _XINT_KEY)
                ):
                    raise ValueError("target-union WT checkpoint has incomplete restart channels")
                for key in (_BIND_KEY, _XINT_KEY):
                    paired_restart_ddg(shared_wt[key], shared_wt[key])
            elif any(
                (checkpoint_dir / "mutations" / f"{index:04d}.json").exists()
                for index in range(len(spec.mutations))
            ):
                raise ValueError("target-union WT checkpoint is missing for completed mutation work")
        for index, mutation in enumerate(spec.mutations):
            row_checkpoint = checkpoint_dir / "mutations" / f"{index:04d}.json"
            if row_checkpoint.exists():
                row = _load_checkpoint(row_checkpoint, identity)
                if row.get("mutation") != mutation.label:
                    raise ValueError("checkpoint mutation identity mismatch")
                rows.append(row)
                continue
            exclusion = [f"{mutation.chain}:{mutation.resnum}"]
            if wt_restraint_scope == "target_union":
                if shared_wt is None:
                    shared_wt = _score_openmm(
                        str(wt_pdb), spec.receptor_chains[0], spec.peptide_chain,
                        **physics_kwargs, restraint_exclusion_residues=union_exclusion,
                    )
                    _save_checkpoint(shared_wt_checkpoint, identity, shared_wt)
                wt = shared_wt
            else:
                wt_checkpoint = checkpoint_dir / "mutations" / f"{index:04d}.wt.json"
                if wt_checkpoint.exists():
                    wt = _load_checkpoint(wt_checkpoint, identity)
                else:
                    wt = _score_openmm(
                        str(wt_pdb), spec.receptor_chains[0], spec.peptide_chain,
                        **physics_kwargs, restraint_exclusion_residues=exclusion,
                    )
                    _save_checkpoint(wt_checkpoint, identity, wt)
            variant = VariantSpec(mutation.wt, mutation.chain, mutation.resnum, mutation.mut)
            mutant_pdb = _build_variant_pdb(base, variant, work / mutation.label)
            mutant = _score_openmm(
                mutant_pdb, spec.receptor_chains[0], spec.peptide_chain,
                **physics_kwargs, restraint_exclusion_residues=exclusion,
            )
            for key in (_BIND_KEY, _XINT_KEY):
                if key not in wt or key not in mutant:
                    raise ValueError(f"OpenMM missing restart channel {key}: {mutation.label}")
                if len(wt[key]) != n_restarts or len(mutant[key]) != n_restarts:
                    raise ValueError(f"OpenMM incomplete restart channel {key}: {mutation.label}")
            row = {
                "target": target.strip(), "parent_id": parent_id.strip(), "mutation": mutation.label,
                "ddg_bind_proxy": paired_restart_ddg(wt[_BIND_KEY], mutant[_BIND_KEY]),
                "ddg_xint_iface": paired_restart_ddg(wt[_XINT_KEY], mutant[_XINT_KEY]),
                "n_iface_contacts_8a": int(geometry.loc[mutation.label, "n_iface_contacts_8a"]),
                "n_neighbors_10a": int(geometry.loc[mutation.label, "n_neighbors_10a"]),
                "mpnn_neg_llr_complex": mpnn[mutation.label][0],
                "mpnn_ddg_bind": mpnn[mutation.label][1],
            }
            _save_checkpoint(row_checkpoint, identity, row)
            rows.append(row)
    features = pd.DataFrame(rows, columns=list(IDENTITY_COLUMNS + FEATURE_COLUMNS))
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
        "wt_restraint_scope": wt_restraint_scope,
        "wt_union_sites": list(union_exclusion) if wt_restraint_scope == "target_union" else None,
        "historical_reproduction_status": "unverified",
        "resume_identity_sha256": identity,
    }
    staging = checkpoint_dir / "publish"
    staging.mkdir(parents=True, exist_ok=True)
    features.to_csv(staging / "features.csv", index=False)
    scores.table.to_csv(staging / "scores.csv", index=False)
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    file_hashes = {name: sha256((staging / name).read_bytes()).hexdigest()
                   for name in ("features.csv", "scores.csv", "provenance.json")}
    publication = {"schema": "pepddg-publication/v1", "files": file_hashes}
    if publishing.exists():
        if _load_checkpoint(publishing, identity) != publication:
            raise ValueError("publication checkpoint content mismatch")
    else:
        _save_checkpoint(publishing, identity, publication)
    for name in file_hashes:
        os.replace(staging / name, destination / name)
    publishing.unlink()
    return StructuralResult(features, scores, provenance)
