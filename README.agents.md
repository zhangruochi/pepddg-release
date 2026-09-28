# PepDDG for coding agents

Use PepDDG from Claude Code or Codex to rank single-residue peptide mutations
from a WT peptide–receptor complex. The agent runs the real tool, preserves the
outputs and explains the evidence; it does not generate numeric predictions itself.

## Agent entrypoints

| Runtime | Repository instructions | Reusable skill |
|---|---|---|
| Codex | [AGENTS.md](AGENTS.md) | [.agents/skills/pepddg/SKILL.md](.agents/skills/pepddg/SKILL.md) |
| Claude Code | [CLAUDE.md](CLAUDE.md) | [.claude/skills/pepddg/SKILL.md](.claude/skills/pepddg/SKILL.md) |

The two instruction files and the two skill files are identical. Open the cloned
repository in your agent to use its repository-scoped guidance. The skills contain
no private platform dependencies, accounts or credentials.

To use the skill in another project, run the appropriate command from this clone,
replacing `/path/to/my-project` with your project directory:

```bash
# Codex
mkdir -p /path/to/my-project/.agents/skills
cp -R .agents/skills/pepddg /path/to/my-project/.agents/skills/

# Claude Code
mkdir -p /path/to/my-project/.claude/skills
cp -R .claude/skills/pepddg /path/to/my-project/.claude/skills/
```

This copies guidance; it does not install PepDDG or its scientific dependencies.
Discovery conventions follow the [Codex skill documentation](https://developers.openai.com/codex/skills)
and [Claude Code skill documentation](https://code.claude.com/docs/en/skills).

## Install and check the tool

From the repository root:

```bash
conda env create -f environment.yaml
conda activate pepddg
pepddg doctor --json
pepddg score-features --input examples/raw_features.csv --output /tmp/pepddg-agent-check.csv
```

If the environment already exists, activate it instead of recreating it. `doctor`
checks dependency availability; it does not certify a successful structural run.
The feature check above is lightweight and does not create structural features.

## A useful request to your agent

> Use PepDDG on `inputs/wt_complex.pdb`. The peptide is chain I and the receptor
> is chain E. Predict T2A, K3A and S4A as separate single mutants. Keep results in
> `results/pepddg`. Inspect WT identity and supported chemistry first, then run
> the standard protocol. Return the raw channels, the mutation ranking, output
> paths and any limitations. Do not claim an absolute affinity or invent scores.

Supply the actual chain IDs and structure residue numbers. A sequence alone,
an isolated peptide structure, or an unspecified receptor is not sufficient.

## Run predictions

```python
import pepddg

results = pepddg.predict(
    "inputs/wt_complex.pdb",
    ["T2A", "K3A", "S4A"],
    peptide_chain="I",
    receptor_chain="E",
    output_dir="results/pepddg",
)
print(results)
print(results.attrs["output_dir"])
```

Use `mutations="T2A"` for one substitution. A list means independent single
mutants; combined multi-site mutants are not supported by this interface.
Numbers are PDB/mmCIF residue IDs, not positions in a renumbered sequence.

The API recognizes supported linear/disulfide topology and validates WT identities.
Known unsupported chemistry is refused, but unannotated covalent closures may
evade detection. Inspect the supplied chemistry and bond records; a successful
parse alone does not establish topology support. Default structural inference uses seven paired
WT/mutant restarts on CPU and may take substantial time. A supported CUDA setup
can use `platform="CUDA"`; the supplied conda environment is the CPU environment.

## Read and preserve results

- `scores.csv`: raw channel columns and cohort-relative scores. Lower scores
  favor a mutation within the same parent–target cohort.
- `features.csv`: inspectable generated features.
- `provenance.json` and `.pepddg-work/`: input/protocol identity and checkpoints.
- DataFrame attributes: output directory, provenance and `rank_available`.

One mutation has raw channels but unavailable relative ranks (`NaN`). Do not
interpret a missing rank as a failed prediction or replace it with a claimed gain.
Do not compare ranks across different cohorts or targets. Scores are not calibrated
binding ΔΔG, measured affinities or evidence of wet-lab improvement.

If interrupted, retry the same inputs, settings and `output_dir`. Preserve partial
outputs; changing the structure, mutation list, protocol or source version may
invalidate the checkpoint. Report incomplete work as incomplete.

## Reproduction and development

```bash
# Real structure-to-score inference: four targets, 35 observations; can be slow.
bash examples/skempi_cyclic/run.sh results/skempi-cyclic

# Lightweight replay of existing frozen features; no new structure prediction.
python -m pepddg.frozen_benchmark --repo-root . --output /tmp/pepddg-frozen-replay

# Package regression tests after implementation changes.
python -m pip install -e .
pytest -q unit_tests/pepddg
```

See [STRUCTURES.md](docs/STRUCTURES.md) for chemistry and input constraints,
[BENCHMARK.md](docs/BENCHMARK.md) for acceptance scopes, and
[CONTRIBUTING.md](CONTRIBUTING.md) for development guidance. Use a new branch
for experiments and keep generated results out of Git. Preserve versioned reference
results; a successful frozen replay does not prove fresh structural reproduction.

Permitted noncommercial research use is available under the stated license.
Commercial use, including company-internal R&D, requires separate written
permission; preserve historical MIT and third-party/data rights. See
[LICENSE_SCOPE.md](LICENSE_SCOPE.md).
