"""Convenient WT-structure entrypoint over the validated structural protocol."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
import re
import tempfile

import pandas as pd

from .structure_contract import ComplexSpec, MutationSpec, UnsupportedChemistry, validate_complex
from .structural_pipeline import run_structural_cohort

_MUTATION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])(-?\d+)([ACDEFGHIKLMNPQRSTVWY])")


def _mutation_specs(mutations: str | Sequence[str], chain: str) -> tuple[MutationSpec, ...]:
    tokens = [mutations] if isinstance(mutations, str) else mutations
    if not isinstance(tokens, Sequence) or isinstance(tokens, (bytes, bytearray)):
        raise TypeError("mutations must be a string or an ordered sequence of strings")
    if not tokens:
        raise ValueError("provide at least one mutation, such as 'T11A'")
    specs = []
    seen = set()
    for token in tokens:
        if not isinstance(token, str) or (match := _MUTATION.fullmatch(token.strip().upper())) is None:
            raise ValueError(f"invalid mutation {token!r}; use one substitution per item, such as 'T11A'")
        wt, position, mutant = match.groups()
        number = int(position)
        label = f"{wt}{number}{mutant}"
        if wt == mutant:
            raise ValueError(f"wild-type and mutant must differ: {label}")
        if label in seen:
            raise ValueError(f"duplicate mutation: {label}")
        seen.add(label)
        specs.append(MutationSpec(label, chain, number, "", wt, mutant))
    return tuple(specs)


def predict(
    wt_structure: str | Path,
    mutations: str | Sequence[str],
    *,
    peptide_chain: str,
    receptor_chain: str,
    output_dir: str | Path | None = None,
    closure: str = "auto",
    target: str | None = None,
    parent_id: str = "WT",
    platform: str = "CPU",
    n_restarts: int = 7,
    seed: int = 20260302,
    cpu_threads: int = 2,
) -> pd.DataFrame:
    """Predict one or several independent single substitutions from a WT complex.

    Mutation numbers are PDB/mmCIF residue numbers, not sequence offsets.
    ``predict('wt.pdb', ['T11A', 'Y7F'], peptide_chain='I', receptor_chain='E')``
    computes all channels and ranks the submitted cohort. One mutation returns
    raw channels with unavailable relative ranks, rather than a trivial zero.
    Scores are not calibrated binding energies or measured affinities.

    Linear or supported disulfide topology is validated automatically. Other
    chemistry is refused by the existing structural checks. Use ``closure`` to
    require an explicit topology. A list means separate single mutants, never
    a combined multi-site mutant.

    Output files/checkpoints are retained. If omitted, ``output_dir`` is a new
    temporary directory, reported in ``frame.attrs['output_dir']`` and error
    notes. Supply a persistent directory to enable an identical-input retry.
    ``frame.attrs['provenance']`` contains the structural run provenance.
    """
    for name, value in (("peptide_chain", peptide_chain), ("receptor_chain", receptor_chain)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty chain ID")
    if closure not in {"auto", "linear", "disulfide"}:
        raise UnsupportedChemistry(f"unsupported peptide closure: {closure}")
    peptide_chain, receptor_chain = peptide_chain.strip(), receptor_chain.strip()
    spec = ComplexSpec(wt_structure, peptide_chain, (receptor_chain,),
                       _mutation_specs(mutations, peptide_chain),
                       "linear" if closure == "auto" else closure)
    try:
        validate_complex(spec)
    except UnsupportedChemistry as exc:
        # Retry only the validator's specific supported SG-SG detection signal.
        if closure != "auto" or str(exc) != "possible peptide disulfide closure detected":
            raise
        spec = replace(spec, closure_kind="disulfide")
        validate_complex(spec)
    destination = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="pepddg-predict-"))
    try:
        result = run_structural_cohort(
            spec, target=Path(wt_structure).stem if target is None else target,
            parent_id=parent_id, output_dir=destination, platform=platform,
            n_restarts=n_restarts, seed=seed, cpu_threads=cpu_threads,
        )
    except Exception as exc:
        exc.add_note(f"PepDDG output/checkpoint location: {destination.resolve()}. Retry with the same inputs and output_dir.")
        raise
    table = result.scores.table.copy()
    table.attrs = {
        **result.scores.summary,
        "output_dir": str(destination.resolve()),
        "provenance": result.provenance,
    }
    return table
