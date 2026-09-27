# Structural workflow

`pepddg score-structures` accepts one coordinate complex, one peptide chain,
one receptor chain and a CSV of single-residue substitutions on that same
peptide parent. It generates three raw channels and then ranks the entire
submitted cohort. It does not infer binding affinity in physical units.

## Input

Use an existing `.pdb`, `.cif` or `.mmcif` file with one coordinate model.
Specify exact chain IDs as they appear in that file. The peptide and receptor
must have standard amino acids and complete N/CA/C/O backbone atoms. Crystal
waters are removed for the implicit-solvent calculation and their atom count
is recorded; other selected-chain heterogens are rejected. The tool
checks the wild-type residue at every numbered position before constructing a
mutant. Numbering, insertion code and chain mapping are molecular identity,
not optional labels.

Mutation CSV schema:

```csv
mutation,chain,resnum,icode,wt,mut
TI11A,I,11,,T,A
```

Each row is one substitution. `mutation` is a unique output ID; `chain`,
`resnum`, `icode`, `wt` and `mut` identify the actual edit. `wt` and `mut`
are one-letter standard amino-acid symbols. An empty `icode` is required for
the present PDBFixer mutation builder. The CLI requires exactly one receptor
chain; choose the intended chain pair explicitly when the source structure
contains more chains. Other chains are excluded from scoring. This changes the
physical system, so report the selected pair with any result.

The validated structural path currently supports **linear peptides only**.
Head-to-tail, side-chain, linker and disulfide cyclization, covalent
connections, modified amino acids, ligands and alternate atom locations need
a separately validated topology workflow. Explicit or suspected closures are
rejected where detectable; absence of a rejection is not proof that an
unannotated chemistry is supported. For cyclic peptide optimization, use the
feature-table interface only if all channels were produced by a separately
validated cyclic protocol, and retain that protocol's provenance.

## Commands

```bash
conda env create -f environment.yaml
conda activate pepddg
pepddg score-structures \
  --structure /path/to/complex.pdb \
  --peptide-chain I --receptor-chain F \
  --mutations /path/to/mutations.csv \
  --target my_target --parent-id parent_001 \
  --output /path/to/new-results
```

`--output` must be absent or empty. For a research workflow the default
`--n-restarts 7` keeps the historical paired-restart setting; smaller values
are for plumbing checks and are not paper-equivalent. `--seed` fixes the
OpenMM and ProteinMPNN random generators. `--platform CPU` is the default;
`--platform CUDA` needs a correctly installed OpenMM CUDA plugin. Set
`--cpu-threads` from 1 to 4 based on actual allocation. On CPU, seven
restarts for a full cohort may be expensive.

Python API:

```python
from pepddg import ComplexSpec, MutationSpec, run_structural_cohort

spec = ComplexSpec(
    structure_path="/path/to/complex.pdb",
    peptide_chain="I",
    receptor_chains=("F",),
    mutations=(MutationSpec("TI11A", "I", 11, "", "T", "A"),),
)
result = run_structural_cohort(
    spec, target="my_target", parent_id="parent_001",
    output_dir="/path/to/new-results",
)
print(result.scores.table)
```

## Computation and outputs

The public structural workflow builds WT and mutant PDBs with PDBFixer;
OpenMM scores each with an Amber14/OBC2 implicit-solvent model, two-stage
restrained minimization (200/500 iterations) and seven restarts by default.
The interface and binding-proxy ΔΔG features are the median of mutant minus
WT values **at matching restart indices**, requiring at least three finite
pairs. The interface channel uses the screened non-heavy interface energy
array. Geometry counts peptide-site to receptor heavy-atom contacts within
8 Å and nearby CA atoms within 10 Å of the site's CB/CA. ProteinMPNN
v_48_020 scores WT and mutant amino-acid log probabilities on the WT complex
and isolated peptide backbones, using the bundled checkpoint and fixed seed.
The final PepDDG-ZS score is a cohort-relative rank fusion.

Successful output directory contents:

| File | Meaning |
|---|---|
| `features.csv` | Unranked complete raw-channel table with target, parent and mutation identities |
| `scores.csv` | Same cohort after frozen PepDDG rank fusion |
| `provenance.json` | Input/checkpoint hashes, excluded water count, selected protocol, seed, restart count and channel keys |

No score files are written until every requested mutation has validated
channels. Failed jobs can leave transient work in a system temp directory;
they never supply partial ranks. Keep the mutation list and the full input
structure with the result. A numerical result is a model prediction, not a
measurement. For the status of paper-level SKEMPI reproduction, see
[BENCHMARK.md](BENCHMARK.md).
