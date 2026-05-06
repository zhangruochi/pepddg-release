# PepDDG v5: Comprehensive Score Mode Evaluation

## Executive Summary

Systematically evaluated **48 scoring modes** (17 base modes x {aggregate, paired-restart} + 14 blend modes) for peptide DDG prediction using implicit-solvent OpenMM minimization on the expanded SKEMPI peptide subset (**40 targets, 509 mutations**, predicted structures; 39 targets, 489 mutations, crystal structures).

**Key Findings:**
1. **dual_blend_iface** remains the best fixed mode (predicted rho=0.449, p=0.028 vs baseline)
2. **Heavy-atom modes are catastrophically bad** (negative rho, -0.31 worst) — hydrogen atoms carry crucial scoring information
3. **Local mutation-neighborhood terms are useless** (rho < 0.1)
4. **Soft-interface switching** is decent (0.36) but does not beat hard cutoff (0.44)
5. **Paired-restart DDG** provides marginal improvement (+0.004 max)
6. **Adaptive mode selection fundamentally cannot help** — the oracle (cheating) upper bound (0.444) is lower than the best fixed mode (0.449)

## Dataset

| Property | Predicted | Crystal |
|----------|-----------|---------|
| Targets | 40 | 39 |
| Mutations | 509 | 489 |
| Saturation targets excluded | 4 (1CHO, 1R0R, 3SGB, 1PPF) |
| Failed targets | 4CPA (UNK residue), 4J2L (minimization failure) |
| Structure source | Boltz-2 | PDB crystal |
| Scoring | OpenMM implicit (7 restarts, median aggregation) |

## Phase A1: Mode Evaluation

### Predicted Structures (Primary Endpoint)

| Rank | Mode | Pooled rho | 95% CI | Sign Acc |
|------|------|-----------|--------|----------|
| 1 | paired_dual_blend_iface | 0.453 | [0.351, 0.530] | 0.796 |
| 2 | dual_blend_iface | 0.449 | [0.350, 0.527] | 0.798 |
| 3 | xint_iface | 0.442 | [0.353, 0.501] | 0.695 |
| 4 | paired_xint_iface | 0.438 | [0.349, 0.499] | 0.687 |
| 5 | paired_hybrid_iface | 0.433 | [0.330, 0.513] | 0.810 |
| 6 | hybrid_iface | 0.429 | [0.330, 0.510] | 0.810 |
| 7 | paired_hybrid_screened | 0.418 | [0.316, 0.500] | 0.800 |
| 8 | hybrid_screened | 0.417 | [0.317, 0.500] | 0.810 |
| 9 | xint_screened | 0.409 | [0.327, 0.468] | 0.699 |
| 10 | xint_soft_iface_screened | 0.364 | [0.252, 0.453] | 0.647 |

### Crystal Structures

| Rank | Mode | Pooled rho | 95% CI |
|------|------|-----------|--------|
| 1 | paired_xint_iface | 0.392 | [0.295, 0.491] |
| 2 | xint_iface | 0.391 | [0.295, 0.491] |
| 3 | paired_xint_screened | 0.389 | [0.284, 0.500] |
| 4 | xint_screened | 0.386 | [0.279, 0.497] |
| 5 | paired_dual_blend_iface | 0.382 | [0.261, 0.510] |

### Failed Mode Categories

| Category | Best rho (predicted) | Conclusion |
|----------|---------------------|------------|
| Heavy-atom (H-stripped) | -0.139 to -0.310 | **Catastrophically bad**. H-bonds critical for scoring. |
| Local (mutation nbr) | -0.033 to 0.092 | **Useless**. 10A neighborhood too restrictive. |
| Soft-interface (cosine switch) | 0.364 | Decent but consistently worse than hard cutoff. |
| Paired-restart DDG | +0.004 max delta | **Marginal**. Not worth the implementation complexity. |

### Significance Tests

- **paired_dual_blend_iface vs hybrid_screened** (predicted): p=0.006, Bonferroni=0.028 (**SIGNIFICANT**)
- **paired_xint_iface vs hybrid_screened** (crystal): p=0.198, Bonferroni=0.989 (NOT significant)

## Phase A0.3: v5 vs v3 Non-Regression

v5 scoring uses `restraint_exclusion_residues` (freeing mutation sites in WT minimization), which changes the energy landscape compared to v3.

| Mode | v3 rho | v5 rho | Delta |
|------|--------|--------|-------|
| xint_iface | 0.419 | 0.442 | **+0.022** |
| xint_screened | 0.390 | 0.409 | **+0.019** |
| bind_proxy | 0.404 | 0.283 | **-0.121** |
| dual_blend_iface | 0.510 | 0.449 | -0.061 |
| hybrid_screened | 0.493 | 0.417 | -0.076 |

**Root cause**: Exclusion residues improve cross-interaction terms (+0.02) but severely degrade bind_proxy (-0.12). Blend modes are hurt because they weight bind_proxy at 40%.

**Grid search on v5 data**: Optimal blend = 0.85 * iface + 0.15 * bind = 0.467 (in-sample). The v3-era 60/40 weights are suboptimal for v5.

## Phase B: Adaptive Mode Selection (LOTO)

| Method | Pooled rho | CI 95% | vs Fixed |
|--------|-----------|--------|----------|
| **dual_blend_iface (fixed)** | **0.449** | **[0.350, 0.527]** | baseline |
| Oracle (cheating upper bound) | 0.444 | [0.328, 0.535] | -0.005 |
| Soft Ridge blending (B3b) | 0.442 | [0.343, 0.521] | -0.007 |
| Stacking meta-learner (B3c) | 0.438 | [0.339, 0.509] | -0.011 |
| Length-stratified (B2) | 0.422 | [0.324, 0.491] | -0.027 |
| Hard KNN classifier (B3a) | 0.306 | [0.218, 0.397] | -0.143 |

**The oracle is BELOW the best fixed mode.** This is a fundamental negative result: per-target mode selection cannot improve pooled Spearman rho because individual per-target rho estimates are too noisy (3-48 mutations per target). Mode selection introduces variance that outweighs any potential gain from target-specific optimization.

## Per-Target Mode Preference

Despite the failure of adaptive selection, mode preference is highly heterogeneous:

| Mode winning most targets | Count (of 40) |
|--------------------------|---------------|
| xint_total | 5 |
| xint_soft_iface | 5 |
| xint_local_heavy | 4 |
| bind_proxy | 3 |
| dual_blend_iface | 2 |
| xint_iface | 2 |

No single mode wins more than 12.5% of targets. But pooled evaluation favors consistent mediocrity over noisy per-target optimization.

## Conclusions and Recommendations

1. **Use dual_blend_iface** (0.24 * xint_iface + 0.76 * (0.6 * xint_iface + 0.4 * bind_proxy)) as the production scoring mode
2. **Do NOT strip hydrogen atoms** when computing cross-interaction energy
3. **Do NOT restrict scoring to mutation neighborhood** (local terms)
4. **Paired-restart DDG is not worth the complexity** — aggregate DDG is nearly as good
5. **Adaptive mode selection is not recommended** — it introduces noise without benefit
6. **Exclusion residues trade-off**: improves iface/screened but degrades bind_proxy. Consider scoring without exclusion residues if bind_proxy is needed for blend modes.

## Reproducibility

- Scoring script: `research/pepddg_v5/scripts/score_all_mutations_v5.py`
- Evaluation script: `research/pepddg_v5/scripts/evaluate_all_modes.py`
- Adaptive selection: `research/pepddg_v5/scripts/adaptive_mode_selection.py`
- Cluster jobs (internal infra): `pepddg-v5-scoring-crystal`, `pepddg-v5-scoring-predicted`
- Reproduction note: per-restart energy sidecars are needed only for the
  `sensitivity_energetic.py` ablation; the canonical pooled energetic columns
  (`ddg_paired_xint_iface`, `ddg_paired_bind_proxy`) are already shipped in
  `results/v19_sanitized_baseline/main_eval_v19.csv`.
- Bootstrap seed: 42, n_boot: 5000
