# PepDDG agent instructions

Read [README.agents.md](README.agents.md) for setup, prediction, output handling
and reproduction commands. Claude Code and Codex share these instructions.

- Use `pepddg.predict(wt_structure, mutations, peptide_chain=..., receptor_chain=..., output_dir=...)`
  for WT-complex inputs. Obtain explicit chain IDs and structure residue numbers;
  never guess them or manufacture prediction values. Check the molecular chemistry
  explicitly: unannotated unsupported closures may evade automatic detection.
- Use an isolated PepDDG environment and inspect supported chemistry before a long
  run. Keep the standard seven-restart protocol unless the requested experiment
  explicitly specifies another setting. Dependency checks are not prediction proof.
- A mutation list represents independent single substitutions. Preserve the complete
  comparison cohort. Single-candidate relative ranks are unavailable; score values
  are not calibrated binding energies, measured affinities or proven optimization.
- Preserve outputs, provenance and partial checkpoints. Retry identical inputs in
  the same output directory; report unfinished computations as unfinished.
- Read docs/STRUCTURES.md for chemistry limits and docs/BENCHMARK.md before making
  paper-consistency claims. Keep frozen-feature replay separate from fresh inference.
- For code changes, use a feature branch, run affected package tests, preserve
  reference evidence, and keep generated results and credentials out of Git.
- Preserve LICENSE_SCOPE.md, historical MIT notices and third-party/data attribution.
  Commercial use, including company-internal R&D, requires written authorization.

Reusable skills: `.agents/skills/pepddg/SKILL.md` for Codex and
`.claude/skills/pepddg/SKILL.md` for Claude Code. Keep each pair byte-identical.
