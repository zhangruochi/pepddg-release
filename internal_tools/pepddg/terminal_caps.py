"""Exact acetyl and primary-amide terminal graphs for canonical peptides."""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
from threading import RLock
from typing import Any

import numpy as np

CAP_HEAVY_ATOMS = {"ACE": {"CH3": "C", "C": "C", "O": "O"}, "NH2": {"N": "N"}}
_PREPARATION_LOCK = RLock()


@contextmanager
def scoped_preparation_seed(seed: int):
    """Keep native hydrogen-placement RNG deterministic and restore caller state."""
    with _PREPARATION_LOCK:
        state = random.getstate()
        try:
            random.seed(int(seed))
            yield
        finally:
            random.setstate(state)


def validate_terminal_caps(chain: Any, amino_acids: set[str] | frozenset[str]) -> tuple[tuple[str, int], ...]:
    """Validate existing cap atoms and chemical attachment without rebuilding them."""
    from .structure_contract import UnsupportedChemistry
    caps = [r for r in chain if r.name in CAP_HEAVY_ATOMS]
    if not caps:
        return ()
    peptide = sorted([r for r in chain if r.name in amino_acids], key=lambda r: r.seqid.num)
    if not peptide:
        raise UnsupportedChemistry("terminal cap has no canonical peptide")
    names = [r.name for r in caps]
    if len(names) != len(set(names)):
        raise UnsupportedChemistry("duplicate terminal cap")
    for cap in caps:
        if cap.seqid.icode not in {" ","\x00"}:
            raise UnsupportedChemistry("terminal cap insertion code is unsupported")
        atoms = list(cap)
        if any(a.altloc not in {" ","\x00"} for a in atoms):
            raise UnsupportedChemistry("alternate terminal cap atoms require preparation")
        heavy = [a for a in atoms if a.element.name != "H"]
        observed = {a.name: a.element.name for a in heavy}
        if len(observed) != len(heavy) or observed != CAP_HEAVY_ATOMS[cap.name]:
            raise UnsupportedChemistry("terminal cap heavy-atom graph is unsupported")
        allowed_h = {"H1","H2","H3"} if cap.name == "ACE" else {"HN1","HN2"}
        if any(a.element.name == "H" and a.name not in allowed_h for a in atoms):
            raise UnsupportedChemistry("terminal cap hydrogen identity is unsupported")
        neighbor = peptide[0] if cap.name == "ACE" else peptide[-1]
        number = neighbor.seqid.num + (-1 if cap.name == "ACE" else 1)
        if cap.seqid.num != number:
            raise UnsupportedChemistry("terminal cap numbering must flank the peptide")
        ca = next(a for a in cap if a.name == ("C" if cap.name == "ACE" else "N"))
        na = next(a for a in neighbor if a.name == ("N" if cap.name == "ACE" else "C"))
        distance = math.sqrt((ca.pos.x-na.pos.x)**2 + (ca.pos.y-na.pos.y)**2 + (ca.pos.z-na.pos.z)**2)
        if not math.isfinite(distance) or not 1.1 <= distance <= 1.6:
            raise UnsupportedChemistry("terminal cap amide connection lacks covalent geometry")
    return tuple(sorted(((r.name,r.seqid.num) for r in caps), key=lambda item:item[1]))


def preserve_caps_in_fixer(fixer: Any, *, keep_water: bool = False) -> None:
    """Register only the exact primary amide on this PDBFixer instance."""
    from openmm import Vec3, unit
    from openmm.app import Topology, element
    from .structure_contract import _ONE_LETTER
    allowed = set(_ONE_LETTER) | set(CAP_HEAVY_ATOMS)
    if keep_water:
        allowed |= {"HOH","WAT"}
    unwanted = [r for r in fixer.topology.residues() if r.name not in allowed]
    if unwanted:
        # No caller may use cap preservation to silently lose other chemistry.
        raise ValueError("terminal-cap preparation contains unsupported heterogens")
    topology = Topology()
    residue = topology.addResidue("NH2",topology.addChain())
    topology.addAtom("N",element.nitrogen,residue)
    fixer.registerTemplate(topology,[Vec3(0,0,0)]*unit.nanometer)


def cap_residue_templates(topology: Any) -> dict[Any, str]:
    """Select Amber's primary-amide template while retaining input NH2 identity."""
    return {r:"NHE" for r in topology.residues() if r.name == "NH2"}


def verify_terminal_cap_integrity(topology: Any, positions: Any, *, peptide_chain: str,
                                  expected_caps: tuple[tuple[str,int], ...],
                                  system: Any | None = None, stage: str) -> dict[str,Any]:
    """Check exact cap topology and nonzero force-field amide bond terms."""
    from openmm import HarmonicBondForce, unit
    from .structure_contract import _ONE_LETTER
    residues = [r for c in topology.chains() if c.id == peptide_chain for r in c.residues()]
    caps = [r for r in residues if r.name in CAP_HEAVY_ATOMS]
    actual = tuple(sorted(((r.name,int(r.id)) for r in caps),key=lambda item:item[1]))
    if actual != expected_caps:
        raise ValueError(f"{stage}: terminal cap identity changed")
    peptide = sorted([r for r in residues if r.name in _ONE_LETTER],key=lambda r:int(r.id))
    if caps and not peptide:
        raise ValueError(f"{stage}: cap peptide missing")
    coordinates = np.asarray(positions.value_in_unit(unit.angstrom),dtype=float)
    atoms = list(topology.atoms())
    if coordinates.shape != (len(atoms),3) or not np.isfinite(coordinates).all():
        raise ValueError(f"{stage}: invalid cap coordinates")
    bonds = {frozenset((a.index,b.index)) for a,b in topology.bonds()}
    required_heavy = set()
    report = {}
    for cap in caps:
        by_name = {a.name:a for a in cap.atoms()}
        names = {"C","O","CH3","H1","H2","H3"} if cap.name == "ACE" else {"N","HN1","HN2"}
        if set(by_name) != names or len(by_name) != len(list(cap.atoms())):
            raise ValueError(f"{stage}: terminal cap atom identity changed")
        for name,element in CAP_HEAVY_ATOMS[cap.name].items():
            if by_name[name].element.symbol != element:
                raise ValueError(f"{stage}: terminal cap element changed")
        for name in names-set(CAP_HEAVY_ATOMS[cap.name]):
            if by_name[name].element.symbol != "H":
                raise ValueError(f"{stage}: terminal cap hydrogen changed")
        neighbor = peptide[0] if cap.name == "ACE" else peptide[-1]
        a = by_name["C" if cap.name == "ACE" else "N"]
        b = next(atom for atom in neighbor.atoms() if atom.name == ("N" if cap.name == "ACE" else "C"))
        edges = {frozenset((a.index,b.index))}
        internal = [("C","O"),("C","CH3"),("CH3","H1"),("CH3","H2"),("CH3","H3")] if cap.name == "ACE" else [("N","HN1"),("N","HN2")]
        edges |= {frozenset((by_name[x].index,by_name[y].index)) for x,y in internal}
        cap_indices = {atom.index for atom in cap.atoms()}
        if {edge for edge in bonds if edge & cap_indices} != edges:
            raise ValueError(f"{stage}: terminal cap covalent bonds changed")
        distance = float(np.linalg.norm(coordinates[a.index]-coordinates[b.index]))
        if not 1.1 <= distance <= 1.6:
            raise ValueError(f"{stage}: terminal cap amide geometry changed")
        report[f"{cap.name}:{cap.id}"] = {
            "amide_distance_angstrom": distance,
            "atom_names": sorted(by_name),
            "topology_bonds": [sorted(edge) for edge in sorted(edges,key=lambda edge:tuple(sorted(edge)))],
            "force_field_bonds": [],
        }
        required_heavy |= {edge for edge in edges if all(atoms[i].element.symbol != "H" for i in edge)}
    if system is not None:
        if system.getNumParticles() != len(atoms):
            raise ValueError(f"{stage}: terminal cap system atom count changed")
        found = set()
        for force in system.getForces():
            if isinstance(force,HarmonicBondForce):
                for i in range(force.getNumBonds()):
                    a,b,length,k = force.getBondParameters(i)
                    edge = frozenset((int(a),int(b)))
                    if edge in required_heavy:
                        stiffness = float(k.value_in_unit(unit.kilojoule_per_mole/unit.nanometer**2))
                        equilibrium = float(length.value_in_unit(unit.angstrom))
                        if edge in found or not math.isfinite(stiffness) or stiffness <= 0 or not 1 <= equilibrium <= 1.7:
                            raise ValueError(f"{stage}: terminal cap force-field bond invalid")
                        found.add(edge)
                        for cap in caps:
                            if edge & {atom.index for atom in cap.atoms()}:
                                report[f"{cap.name}:{cap.id}"]["force_field_bonds"].append({
                                    "atom_indices": sorted(edge),"equilibrium_angstrom": equilibrium,
                                    "stiffness_kj_mol_nm2": stiffness,
                                })
        if found != required_heavy:
            raise ValueError(f"{stage}: terminal cap force-field bond missing")
    return report
