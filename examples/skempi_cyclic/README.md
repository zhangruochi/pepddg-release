# Official SKEMPI examples

Activate the repository's `pepddg` environment, then run from the repository root:

```bash
python examples/skempi_cyclic/run.py --target 1SMF --output /tmp/pepddg-1SMF
```

The command downloads the official CSV and cleaned PDB archive, verifies the pinned hashes and selected original members, verifies the bundled prepared inputs, then calls the installed structural CLI. Add `--platform CUDA` when your OpenMM CUDA environment is working. `--verify-only` checks acquisition/input identity without model computation. The default seven paired restarts can take substantial time on CPU. Repeat the identical command to resume completed mutation checkpoints.

| Target | Mutations | Actual peptide chemistry |
|---|---:|---|
| 1SMF | 5 | Cys1–Cys9 disulfide, peptide chain I |
| 5XCO | 9 | Cys5–Cys15 disulfide, peptide chain B |
| 3EQS | 10 | Linear PMI, peptide chain B |
| 3EQY | 11 | Linear PMI, peptide chain C |

The paper's CYCLIC annotation includes all35; 3EQS/3EQY are linear controls. The actual disulfide subgroup has14 observations. Never add a terminal bond to these structures. Disulfide support is a preview until the separate fresh-computation acceptance report passes. Head-to-tail, linker and noncanonical chemistry remain unsupported by this example.

Prepared coordinates are derived from the official cleaned PDBs. Where alternate conformers exist, preparation selects a complete whole-residue conformer by highest mean occupancy, with lexical tie-breaking; the manifest records selected/discarded labels and atom identities and both source/prepared hashes. The included inputs are frozen to this exact snapshot. Changed upstream data or edited coordinates fail verification, rather than silently defining a different benchmark.

Each target is one complete ranking cohort; lower final score is more favorable within that cohort. A target's score is not an absolute binding energy and target-relative scores cannot be pooled to claim the paper's global correlation. No experimental labels or archived scores enter this example. The engineering acceptance report separately compares fresh keyed results against historical within-target orders and reports all failures. Full332/33 and original predicted-coordinate reproduction are not asserted.

## Data attribution

SKEMPI 2.0: Jankauskaite et al., Bioinformatics35,462–469(2019), https://doi.org/10.1093/bioinformatics/bty635. Official source: https://life.bsc.es/pid/skempi2/database/index. These SKEMPI-derived examples and prepared coordinates are **CC BY4.0**, copyright2018Barcelona Supercomputing Center (BSC), https://creativecommons.org/licenses/by/4.0/. This repository modified the subset selection, chain/numbering documentation and alternate-conformer preparation; no BSC endorsement is claimed. PepDDG's commercial-use authorization requirement does not restrict independent rights in these source data.
