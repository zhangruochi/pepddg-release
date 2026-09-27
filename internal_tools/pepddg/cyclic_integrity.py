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

    expected_indices = {
        tuple(sorted((sg_atoms[first].index, sg_atoms[second].index)))
        for first, second in expected
    }
    topology_indices = set()
    for first, second in topology.bonds():
        if (first.name == "SG" and first.residue.chain.id == peptide_chain
                and second.residue.chain.id != peptide_chain) or (
            second.name == "SG" and second.residue.chain.id == peptide_chain
            and first.residue.chain.id != peptide_chain
        ):
            raise ValueError(f"{stage}: disulfide topology mismatch: cross-chain peptide sulfur bond")
        if first.name == second.name == "SG" and (
            first.residue.chain.id == peptide_chain or second.residue.chain.id == peptide_chain
        ):
            topology_indices.add(tuple(sorted((first.index, second.index))))
    if topology_indices != expected_indices:
        raise ValueError(f"{stage}: disulfide topology mismatch")

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
        sulfur_indices = {atom.index for atom in atoms if atom.name == "SG"}
        peptide_sulfur_indices = {
            atom.index for atom in atoms
            if atom.name == "SG" and atom.residue.chain.id == peptide_chain
        }
        bonded_indices = set()
        for force in system.getForces():
            if not isinstance(force, HarmonicBondForce):
                continue
            for index in range(force.getNumBonds()):
                first, second, length, stiffness = force.getBondParameters(index)
                indices = tuple(sorted((int(first), int(second))))
                if not (set(indices) <= sulfur_indices and set(indices) & peptide_sulfur_indices):
                    continue
                if indices not in expected_indices or indices in bonded_indices:
                    raise ValueError(f"{stage}: force-field disulfide bond mismatch")
                bond_length = float(length.value_in_unit(unit.nanometer))
                bond_stiffness = float(stiffness.value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer**2
                ))
                if (not math.isfinite(bond_length) or not 0.17 <= bond_length <= 0.25
                        or not math.isfinite(bond_stiffness) or bond_stiffness <= 0):
                    raise ValueError(f"{stage}: force-field disulfide bond parameters invalid")
                bonded_indices.add(indices)
        if bonded_indices != expected_indices:
            raise ValueError(f"{stage}: force-field disulfide bond missing")
    return distances
