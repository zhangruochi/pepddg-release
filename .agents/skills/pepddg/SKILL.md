---
name: pepddg
description: Use when a researcher wants PepDDG predictions for one or several independent peptide substitutions in a bound WT complex, or wants to reproduce the supplied PepDDG SKEMPI examples.
---

# PepDDG mutation prediction

Use the installed `pepddg` CLI/Python package. In a PepDDG clone, read
`README.agents.md`; in another project, locate the installed package and the
user's complex rather than assuming this skill directory is a source checkout.
Full guide: https://github.com/zhangruochi/pepddg-release/blob/main/README.agents.md

## Inputs and preparation

Require a WT peptide–receptor PDB/mmCIF complex, explicit peptide/receptor chain
IDs and mutation tokens such as `T2A`. Numbers are structure residue IDs. A list
means independent single mutants, not combined multi-site mutants. Inspect WT
identity; let the strict validator reject mismatches or unsupported chemistry.
Unannotated unsupported covalent closures may evade detection; inspect the
chemistry and bond records rather than treating successful parsing as support.
A peptide sequence or isolated peptide cannot substitute for the bound complex.

In a source clone, create/activate the `pepddg` environment from environment.yaml.
Run `pepddg doctor --json`; its dependency check is not structural execution proof.
For a quick installation check, score examples/raw_features.csv with `score-features`.
Use the standard protocol for predictions; CPU inference can be slow. Choose CUDA
only with a compatible installed platform. Never reduce restarts silently.

## Execute

```python
import pepddg

results = pepddg.predict(
    "inputs/wt_complex.pdb", ["T2A", "K3A"],
    peptide_chain="I", receptor_chain="E",
    output_dir="results/pepddg",
)
print(results)
```

Replace the example structure/chains/mutations with the user's verified inputs.
Pass a string for one mutation. Keep the same output directory and exact inputs
when resuming interrupted work. Preserve partial outputs and error notes.

## Interpret and deliver

Return the actual model table, output paths and provenance. Lower cohort-relative
scores favor mutations within the same parent–target cohort; do not compare ranks
across cohorts. One candidate returns raw channels with relative ranks unavailable
(`NaN`, `results.attrs['rank_available']` is false). Do not invent a rank or an
absolute ΔΔG, affinity or experimental improvement.

Supported linear/disulfide inputs and exact ACE/NH2 caps are documented in
https://github.com/zhangruochi/pepddg-release/blob/main/docs/STRUCTURES.md
Head-to-tail rings, linkers and noncanonical residues require additional support;
do not linearize a ring or remove chemistry to make an input pass.

For the supplied real structural reproduction, run
`bash examples/skempi_cyclic/run.sh results/skempi-cyclic` from the clone.
`python -m pepddg.frozen_benchmark --repo-root . --output /tmp/pepddg-frozen-replay`
is only a frozen-feature replay. Report their distinct scope and any incompleteness.
Never treat installation checks or unavailable predictions as a reproduction PASS.

Preserve license/data attribution. Commercial use, including company-internal
R&D, requires separate written authorization under LICENSE_SCOPE.md.
