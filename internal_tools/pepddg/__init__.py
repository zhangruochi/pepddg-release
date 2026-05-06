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
    "PepDDGConfig": "internal_tools.pepddg.config",
    "V19SanitizedWeights": "internal_tools.pepddg.config",
    "load_config": "internal_tools.pepddg.config",
    "run_pepddg": "internal_tools.pepddg.pipeline",
    # Structure I/O
    "AtomRecord": "internal_tools.pepddg.structure_io",
    "parse_structure": "internal_tools.pepddg.structure_io",
    "get_residue_coord": "internal_tools.pepddg.structure_io",
    "get_chain_sequence": "internal_tools.pepddg.structure_io",
    # Structural features
    "MutationSite": "internal_tools.pepddg.structural_features",
    "compute_interface_contacts": "internal_tools.pepddg.structural_features",
    "compute_neighbor_count": "internal_tools.pepddg.structural_features",
    "compute_structural_features_batch": "internal_tools.pepddg.structural_features",
    "compute_struct_composite": "internal_tools.pepddg.structural_features",
    # Physics features
    "extract_physics_from_openmm_results": "internal_tools.pepddg.physics_features",
    "extract_physics_from_csv": "internal_tools.pepddg.physics_features",
    "compute_physics_rankscore": "internal_tools.pepddg.physics_features",
    # MPNN features
    "compute_mpnn_ddg_features": "internal_tools.pepddg.mpnn_features",
    "extract_mpnn_from_csv": "internal_tools.pepddg.mpnn_features",
    "compute_mpnn_rankscore": "internal_tools.pepddg.mpnn_features",
    # Assembly
    "assemble_features": "internal_tools.pepddg.feature_assembly",
    "validate_scoring_input": "internal_tools.pepddg.feature_assembly",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
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
]
