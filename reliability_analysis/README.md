# Reliability Analysis — ICC Reproduction Package

This folder contains the underlying evaluation scores, sample ID lists, and a
self-contained reproduction script for **every inter-rater reliability (IRR)
value printed in the paper** (main reliability paragraph, Table 3, and
Table 5).

## Quick start

```bash
python reproduce_icc.py
```

Requires only `pandas` and `numpy`. The script recomputes all Fleiss' kappa
and ICC values from the deposited scores and compares them against the values
printed in the paper. It exits with code 0 and prints
`ALL VALUES REPRODUCED SUCCESSFULLY.` when every value matches within
rounding (±0.001).

## Provenance: which file produces which numbers

| Paper location | Data file | Sample | Method |
|---|---|---|---|
| Reliability and Robustness Testing paragraph (Novelty 0.283, Usefulness/Feasibility 0.545, Flexibility 0.448, Elaboration 0.386, Cultural Appropriateness/Sensitivity 0.736) | `data/evaluation_scores_70.csv` (IDs: `data/solution_ids_70.txt`) | 70 stratified solutions × 5 judges = 350 evaluations | ICC(2,k): two-way random effects, **absolute agreement**, average of *k* = 5 raters |
| Table 3 — first prompt optimization (prompt v1) | `data/evaluation_scores_stratified_o1.0.csv` (IDs: `data/solution_ids_30_prompt_v1.txt`) | 30 stratified solutions × 5 judges = 150 evaluations | Fleiss' κ; ICC computed with the single-measure **consistency** formula ICC(C,1) |
| Table 5 — second prompt optimization (prompt v2) | `data/evaluation_scores_3.0.csv` (IDs: `data/solution_ids_30_prompt_v2.txt`) | 30 stratified solutions × 5 judges = 150 evaluations | Fleiss' κ; ICC computed with the single-measure **consistency** formula ICC(C,1) |

**Note on ICC variants.** The main reliability paragraph's values correspond to
ICC(2,k) with absolute agreement (average measures), while the values in
Tables 3 and 5 were computed from the same two-way ANOVA
(ratings ~ solution + judge) with the single-measure consistency formula
ICC(C,1) = (MS_solution − MS_error) / (MS_solution + (k−1)·MS_error).
Both variants lead to the same conclusions: single-judge reliability is poor to
moderate, which motivates the ensemble (mean-of-5-judges) scoring used for the
main analyses.

## File descriptions

- `reproduce_icc.py` — reproduction script (see above).
- `data/evaluation_scores_70.csv` — the 350 evaluations behind the main
  reliability paragraph: original evaluation prompt, 70 solutions, 5 judges,
  with per-criterion scores (1–5) and textual justifications.
- `data/evaluation_scores_stratified_o1.0.csv` — the 150 evaluations behind
  Table 3 (first optimized prompt, v1).
- `data/evaluation_scores_3.0.csv` — the 150 evaluations behind Table 5
  (second optimized prompt, v2).
- `data/solution_ids_70.txt`, `data/solution_ids_30_prompt_v1.txt`,
  `data/solution_ids_30_prompt_v2.txt` — the `Solution_ID`s of the stratified
  samples, one per line. Each ID resolves to the full solution text in
  `all_experiment_solutions.json` at the repository root.

The full-dataset scores (all 6,375 solutions × 5 judges) are in
`cleaned_evaluation_scores.csv` at the repository root; the ensemble scores
used for the main analyses are in `ensemble_scores_FULL_DATA_6375.csv`.
