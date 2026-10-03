#!/usr/bin/env python3
"""
reproduce_icc.py — Reproduce every inter-rater reliability value printed in:
"Measuring and Explaining Creativity in Large Language Models: A Mixed-Methods
Investigation into the Effect of Cultural Persona Prompts on Business Solution
Generation" (Hariri, 2025).

What it reproduces
------------------
1. Main reliability paragraph (Section: Reliability and Robustness Testing):
   ICC values computed on a stratified sample of 70 solutions x 5 LLM judges
   (350 evaluations), using ICC(2,k): two-way random effects, absolute
   agreement, average of k raters (Shrout & Fleiss, 1979; Koo & Li, 2016).
   Source data: data/evaluation_scores_70.csv
   IDs:         data/solution_ids_70.txt

2. Table 3 (first prompt optimization, prompt v1):
   Fleiss' kappa and ICC on a stratified sample of 30 solutions x 5 judges
   (150 evaluations). NOTE: these values were computed with the single-measure
   consistency formula ICC(C,1) = (MS_item - MS_error) / (MS_item + (k-1)*MS_error)
   from a two-way ANOVA (ratings ~ item + rater), although the table is
   captioned "ICC". Source data: data/evaluation_scores_stratified_o1.0.csv
   IDs:         data/solution_ids_30_prompt_v1.txt

3. Table 5 (second prompt optimization, prompt v2):
   Same method as Table 3. Source data: data/evaluation_scores_3.0.csv
   IDs:         data/solution_ids_30_prompt_v2.txt

Dependencies: python >= 3.9, pandas, numpy. No other packages required.

Usage:  python reproduce_icc.py
Exit code 0 if every printed value reproduces within rounding (±0.001), else 1.
"""
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CRITERIA = [
    "Novelty",
    "Usefulness/Feasibility",
    "Flexibility",
    "Elaboration",
    "Cultural_Appropriateness_Sensitivity",
]
SOLUTION_KEYS = ["Solution_ID", "Generating_LLM_ID", "Business_Problem",
                 "Culture", "Generating_Iteration_Number"]

# Values as printed in the paper.
PAPER_MAIN_ICC = {"Novelty": 0.283, "Usefulness/Feasibility": 0.545,
                  "Flexibility": 0.448, "Elaboration": 0.386,
                  "Cultural_Appropriateness_Sensitivity": 0.736}
PAPER_TABLE3 = {  # (Fleiss' kappa, ICC)
    "Novelty": (0.006, 0.288), "Usefulness/Feasibility": (0.021, 0.122),
    "Flexibility": (0.009, 0.344), "Elaboration": (-0.001, 0.169),
    "Cultural_Appropriateness_Sensitivity": (0.012, 0.375)}
PAPER_TABLE5 = {
    "Novelty": (0.000396, 0.152), "Usefulness/Feasibility": (-0.004, 0.053),
    "Flexibility": (0.101, 0.291), "Elaboration": (-0.120, 0.028),
    "Cultural_Appropriateness_Sensitivity": (0.151, 0.395)}


def fleiss_kappa_np(table):
    """Fleiss' kappa for an N x k count matrix (equal raters per subject)."""
    table = np.asarray(table, dtype=float)
    n_subj, _ = table.shape
    n = table.sum(axis=1)[0]
    p_i = (np.sum(table ** 2, axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = table.sum(axis=0) / (n_subj * n)
    p_e = np.sum(p_j ** 2)
    return (p_bar - p_e) / (1 - p_e)


def two_way_anova_ms(item_ids, rater_ids, y):
    """Mean squares from a two-way ANOVA, ratings ~ item + rater (Type-II SS),
    computed with plain least squares (no statsmodels dependency)."""
    items = pd.unique(item_ids)
    raters = pd.unique(rater_ids)
    k, n = len(raters), len(items)
    y = np.asarray(y, dtype=float)
    x_item = np.column_stack([(item_ids == it).astype(float) for it in items])
    x_rater = np.column_stack([(rater_ids == r).astype(float) for r in raters[1:]])
    one = np.ones((len(y), 1))

    def sse(x):
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        return float(resid @ resid), len(y) - np.linalg.matrix_rank(x)

    sse_full, df_err = sse(np.hstack([one, x_item[:, 1:], x_rater]))
    sse_rater_only, _ = sse(np.hstack([one, x_rater]))
    sse_item_only, _ = sse(np.hstack([one, x_item[:, 1:]]))
    ms_item = (sse_rater_only - sse_full) / (n - 1)
    ms_rater = (sse_item_only - sse_full) / (k - 1)
    ms_err = sse_full / df_err
    return ms_item, ms_rater, ms_err, n, k


def long_format(df, criterion):
    """Pivot one criterion to wide (solution x judge), then melt to long."""
    col = f'{criterion.replace("/", "_")}_Score'
    cdf = df[SOLUTION_KEYS + ["Evaluator_LLM_ID", col]].dropna(subset=[col])
    wide = cdf.pivot_table(index=SOLUTION_KEYS, columns="Evaluator_LLM_ID",
                           values=col, aggfunc="first")
    wide = wide.dropna(how="all")
    tmp = wide.copy()
    tmp["item_id"] = np.arange(len(tmp))
    long = tmp.melt(id_vars=["item_id"], value_vars=list(wide.columns),
                    var_name="rater", value_name="ratings")
    return long.dropna(subset=["ratings"]), wide


def icc_c1(ms_item, ms_err, k):
    """Single-measure consistency: (MS_item - MS_err) / (MS_item + (k-1)*MS_err).
    This is the formula used for Tables 3 and 5."""
    return max(0.0, (ms_item - ms_err) / (ms_item + (k - 1) * ms_err))


def icc_2k_absolute(ms_item, ms_rater, ms_err, n):
    """ICC(2,k) absolute agreement, average measures:
    (MS_item - MS_err) / (MS_item + (MS_rater - MS_err)/n).
    This is the formula matching the main 70-solution reliability paragraph."""
    return max(0.0, (ms_item - ms_err) / (ms_item + (ms_rater - ms_err) / n))


def kappa_from_wide(wide):
    kd = wide.astype(float).round().astype("Int64").dropna()
    kd = kd[(kd >= 1) & (kd <= 5)].dropna()
    if kd.empty:
        return np.nan
    m = np.zeros((len(kd), 5))
    for i, row in enumerate(kd.values):
        for s in row:
            m[i, int(s) - 1] += 1
    return fleiss_kappa_np(m)


def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, name), dtype=str)
    for c in df.columns:
        if c.endswith("_Score"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    failures = 0

    def check(label, got, want, tol=0.001):
        nonlocal failures
        ok = abs(got - want) <= tol + 1e-12
        if not ok:
            failures += 1
        print(f"  {label:<45} computed={got:>9.6f}   paper={want:>8.3f}   "
              f"{'OK' if ok else 'MISMATCH'}")

    # --- 1. Main 70-solution paragraph: ICC(2,k) absolute agreement ---------
    print("\n[1] Main reliability paragraph (70 solutions x 5 judges, "
          "evaluation_scores_70.csv), ICC(2,k) absolute agreement")
    df70 = load("evaluation_scores_70.csv")
    assert df70["Solution_ID"].nunique() == 70 and len(df70) == 350
    for crit in CRITERIA:
        long, _ = long_format(df70, crit)
        ms_i, ms_r, ms_e, n, k = two_way_anova_ms(
            long["item_id"].values, long["rater"].values, long["ratings"].values)
        check(crit, icc_2k_absolute(ms_i, ms_r, ms_e, n), PAPER_MAIN_ICC[crit])

    # --- 2 & 3. Tables 3 and 5: Fleiss' kappa + ICC(C,1) --------------------
    for table, fname, paper in [
            ("Table 3 (prompt v1)", "evaluation_scores_stratified_o1.0.csv", PAPER_TABLE3),
            ("Table 5 (prompt v2)", "evaluation_scores_3.0.csv", PAPER_TABLE5)]:
        print(f"\n[{'2' if '3' in table else '3'}] {table} "
              f"(30 solutions x 5 judges, {fname}), "
              "Fleiss' kappa and ICC(C,1) single-measure consistency")
        df = load(fname)
        assert df["Solution_ID"].nunique() == 30 and len(df) == 150
        for crit in CRITERIA:
            long, wide = long_format(df, crit)
            ms_i, ms_r, ms_e, n, k = two_way_anova_ms(
                long["item_id"].values, long["rater"].values, long["ratings"].values)
            kappa = kappa_from_wide(wide)
            check(f"{crit} — Fleiss' kappa", kappa, paper[crit][0])
            check(f"{crit} — ICC", icc_c1(ms_i, ms_e, k), paper[crit][1])

    print("\n" + ("ALL VALUES REPRODUCED SUCCESSFULLY."
                  if failures == 0 else f"{failures} VALUE(S) DID NOT REPRODUCE."))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
