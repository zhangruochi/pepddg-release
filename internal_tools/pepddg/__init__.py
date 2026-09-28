"""PepDDG internal tool package.

Upstream feature pipeline modules:
- structure_io: PDB/CIF parsing
- structural_features: Contact/neighbor counting
- physics_features: OpenMM energy extraction
- mpnn_features: ProteinMPNN log-likelihood features
- feature_assembly: Channel merging and validation
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    # Config + pipeline
    "predict": "prediction",
    "PepDDGConfig": "config",
    "V19SanitizedWeights": "config",
    "load_config": "config",
    "run_pepddg": "pipeline",
    # Structure I/O
    "AtomRecord": "structure_io",
    "parse_structure": "structure_io",
    "get_residue_coord": "structure_io",
    "get_chain_sequence": "structure_io",
    # Structural features
    "MutationSite": "structural_features",
    "compute_interface_contacts": "structural_features",
    "compute_neighbor_count": "structural_features",
    "compute_structural_features_batch": "structural_features",
    "compute_struct_composite": "structural_features",
    # Physics features
    "extract_physics_from_openmm_results": "physics_features",
    "extract_physics_from_csv": "physics_features",
    "compute_physics_rankscore": "physics_features",
    # MPNN features
    "compute_mpnn_ddg_features": "mpnn_features",
    "extract_mpnn_from_csv": "mpnn_features",
    "compute_mpnn_rankscore": "mpnn_features",
    # Assembly
    "assemble_features": "feature_assembly",
    "validate_scoring_input": "feature_assembly",
    "ScoreResult": "api",
    "score_features": "api",
    "score_feature_csv": "api",
    "diagnose_runtime": "diagnostics",
    "MutationSpec": "structure_contract",
    "ComplexSpec": "structure_contract",
    "ComplexValidation": "structure_contract",
    "UnsupportedChemistry": "structure_contract",
    "read_mutations_csv": "structure_contract",
    "validate_complex": "structure_contract",
    "StructuralResult": "structural_pipeline",
    "paired_restart_ddg": "structural_pipeline",
    "run_structural_cohort": "structural_pipeline",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "predict",
    # Config + pipeline
    "PepDDGConfig",
    "V19SanitizedWeights",
    "load_config",
    "run_pepddg",
    # Structure I/O
    "AtomRecord",
    "parse_structure",
    "get_residue_coord",
    "get_chain_sequence",
    # Structural features
    "MutationSite",
    "compute_interface_contacts",
    "compute_neighbor_count",
    "compute_structural_features_batch",
    "compute_struct_composite",
    # Physics features
    "extract_physics_from_openmm_results",
    "extract_physics_from_csv",
    "compute_physics_rankscore",
    # MPNN features
    "compute_mpnn_ddg_features",
    "extract_mpnn_from_csv",
    "compute_mpnn_rankscore",
    # Assembly
    "assemble_features",
    "validate_scoring_input",
    "ScoreResult",
    "score_features",
    "score_feature_csv",
    "diagnose_runtime",
    "MutationSpec",
    "ComplexSpec",
    "ComplexValidation",
    "UnsupportedChemistry",
    "read_mutations_csv",
    "StructuralResult",
    "paired_restart_ddg",
    "run_structural_cohort",
    "validate_complex",
]
