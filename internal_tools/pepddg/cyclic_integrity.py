"""Fail-closed disulfide checks on actual OpenMM structures and systems."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def verify_disulfide_integrity(
    topology: Any,
    positions: Any,
    *,
    peptide_chain: str,
    expected_pairs: tuple[tuple[int, int], ...],
    system: Any | None = None,
    stage: str,
) -> dict[str, float]:
    """Check exact S–S topology, geometry, stereochemistry and force-field bonds."""
    from openmm import HarmonicBondForce, unit

    if not expected_pairs:
        raise ValueError(f"{stage}: no declared disulfide bonds")
    atoms = list(topology.atoms())
    residues = {
        int(residue.id): residue
        for chain in topology.chains() if chain.id == peptide_chain
        for residue in chain.residues()
    }
    if not residues:
        raise ValueError(f"{stage}: peptide chain is missing")
    expected = {tuple(sorted(pair)) for pair in expected_pairs}
    if len(expected) != len(expected_pairs):
        raise ValueError(f"{stage}: duplicate declared disulfide bonds")
    sg_atoms: dict[int, Any] = {}
    for number in {number for pair in expected for number in pair}:
        residue = residues.get(number)
        if residue is None or residue.name not in {"CYS", "CYX"}:
            raise ValueError(f"{stage}: disulfide cysteine {number} is missing")
        by_name = {atom.name: atom for atom in residue.atoms()}
        if "SG" not in by_name or "HG" in by_name:
            raise ValueError(f"{stage}: disulfide cysteine {number} has missing SG or thiol HG")
        sg_atoms[number] = by_name["SG"]

    topology_pairs = {
        tuple(sorted((int(first.residue.id), int(second.residue.id))))
        for first, second in topology.bonds()
        if first.name == second.name == "SG"
        and first.residue.chain.id == second.residue.chain.id == peptide_chain
    }
    if topology_pairs != expected:
        raise ValueError(f"{stage}: disulfide topology mismatch: {sorted(topology_pairs)}")

    coordinates = np.asarray(positions.value_in_unit(unit.angstrom), dtype=float)
    if coordinates.shape != (len(atoms), 3) or not np.isfinite(coordinates).all():
        raise ValueError(f"{stage}: invalid OpenMM coordinates")
    distances = {}
    for first, second in sorted(expected):
        distance = float(np.linalg.norm(
            coordinates[sg_atoms[first].index] - coordinates[sg_atoms[second].index]
        ))
        if not 1.7 <= distance <= 2.5:
            raise ValueError(f"{stage}: disulfide geometry outside 1.7–2.5 A: {first}-{second}")
        distances[f"{first}-{second}"] = distance

    for number, residue in residues.items():
        if residue.name == "GLY":
            continue
        by_name = {atom.name: atom for atom in residue.atoms()}
        if not {"N", "CA", "C", "CB"}.issubset(by_name):
            raise ValueError(f"{stage}: peptide stereochemistry cannot be checked at {number}")
        ca = coordinates[by_name["CA"].index]
        handedness = float(np.dot(
            np.cross(coordinates[by_name["N"].index] - ca,
                     coordinates[by_name["C"].index] - ca),
            coordinates[by_name["CB"].index] - ca,
        ))
        if not math.isfinite(handedness) or handedness <= 0.5:
            raise ValueError(f"{stage}: peptide stereochemistry changed at {number}")

    if system is not None:
        if system.getNumParticles() != len(atoms):
            raise ValueError(f"{stage}: OpenMM system/structure atom count mismatch")
        bonded_indices = {
            tuple(sorted((int(force.getBondParameters(index)[0]),
                          int(force.getBondParameters(index)[1]))))
            for force in system.getForces() if isinstance(force, HarmonicBondForce)
            for index in range(force.getNumBonds())
        }
        for first, second in expected:
            indices = tuple(sorted((sg_atoms[first].index, sg_atoms[second].index)))
            if indices not in bonded_indices:
                raise ValueError(f"{stage}: force-field disulfide bond missing: {first}-{second}")
    return distances
