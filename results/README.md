# Result files

Output of the reported run (`PROFILE = "full"`, master seed `20261029`, 120 conditions, 864 mutants), as written by the notebook. Every number in the paper can be recomputed from these files. Property identifiers: CI, OVP, NFO, RNG, StatRep, RP, FP, DR, SE, TE, PFS, CS, MV, DT, IV, CB, NN.

## Inputs and reference behaviour

| File | Rows | Content |
|---|---|---|
| `conditions.csv` | 120 | One row per randomized condition: `cid`, data `source`, missingness `mechanism`, target and effective rate (`rate_target`, `rate_eff`), size (`n`, `p`), `tie_frac`, `scale_spread`, and the stressor flags `flag_coarse`, `flag_rescaled`, `flag_dup_rows`, `flag_const_col`, `flag_empty_col`, `flag_empty_row`. |
| `reference_gate.csv` | 10440 | Reference validation: each applicable property of each unmutated subject (`sid`) on each condition (`cid`), with the deviation `dev`, the tolerance `tol`, any error `err`, and `violated`. A mutant is credited with a kill only where `violated` is false here. |
| `natural_behaviour_findings.csv` | 15 | Summary of the reference gate: (subject, property) pairs that the unmutated library violates, with the number of violating conditions out of 120, tie statistics and the maximum deviation. |

## Mutants and classification

| File | Rows | Content |
|---|---|---|
| `mutant_classification.csv` | 864 | One row per mutant: location (`target`, `func`, `lineno`), `operator`, `category`, original and replacement code, coverage, number of eligible checks (`n_elig`), `killed_prop`, `killed_crash`, and the final `class`: `NO_COVERAGE`, `EQUIV_CANDIDATE`, `CRASH_ALL`, `KILLED_PROP`, `KILLED_CRASH`, `SURVIVED` (also `STILLBORN`, `TIMEOUT` and `NOT_SAMPLED`, empty in this run), plus the heuristic `subtle_family`. |
| `kill_matrix.csv` | 248 | Analysed mutants (rows) by property (columns): true if the property kills the mutant in at least one condition in which the reference satisfies it. |
| `equivalent_candidates_for_manual_audit.csv` | 162 | Mutants with no observable change on this workload, exported for manual audit. `manual_verdict` is empty. |
| `survivors_for_manual_audit.csv` | 34 | Analysed mutants that no property killed (`mag_max` is the largest output deviation they cause), exported for manual audit. `manual_verdict` and `manual_note` are empty. |

## Research questions

| File | Content |
|---|---|
| `rq1_overall.csv`, `rq1_by_target.csv`, `rq1_by_category.csv`, `rq1_by_target_category.csv`, `rq1_by_family.csv` | RQ1: mutation score (killed by property over analysed mutants) with 95% bootstrap interval, overall and by component, operator category and fault family. |
| `rq2_property_contributions.csv` | RQ2: kills, unique kills and kills lost if the property is removed, per property. |
| `rq2_minimal_sets.csv` | RQ2: exact minimal property set that preserves all kills, overall and by component. |
| `rq2_suite_comparison.csv` | RQ2: score of the baseline suite of seven properties, the full suite of 17 and the ten added properties alone. |
| `rq3_detection_probability.csv` | RQ3: for each (mutant, subject) pair, the number of activated conditions, the detection probability `p_detect` and its regime (always, never or condition-dependent). |
| `rq3_condition_factors.csv` | RQ3: mean detection probability by missingness mechanism, rate, size, ties, scale, stressor and data source. |
| `rq3_condition_sampling_reliability.csv` | RQ3: condition-bootstrap and split-half stability of the score. |
| `rq4_score_by_magnitude.csv` | RQ4: score by size of the output deviation caused by the mutant. |
| `rq4_tolerance_sensitivity.csv` | RQ4: score when every tolerance is multiplied by 0.01, 1 and 100. |

## Raw outcomes and metadata

| File | Content |
|---|---|
| `mutation_property_rows.csv.gz` | One row per eligible check, i.e. per (mutant, subject, condition, property) where the unmutated implementation satisfies the property: columns `mutant_id`, `sid`, `cid`, `pid`, `ref_dev` (deviation of the unmutated implementation), `mut_dev` (deviation of the mutant) and `mut_err` (error raised by the mutant, if any). A violation is `mut_dev` above the tolerance or a non-empty `mut_err`. |
| `mutation_activation_rows.csv.gz` | One row per (mutant, subject, condition): columns `mutant_id`, `sid`, `cid`, `phase`, `differs` (whether the mutant changed the output), `base_mag` and `any_mag` (size of the change), `base_exc` and `n_exc` (exceptions raised). |
| `experiment_metadata.json` | Configuration, seed, tolerances, property names and package versions. |
