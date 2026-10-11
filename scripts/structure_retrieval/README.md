# Frozen T1/T2/T3 five-fold retrieval evaluation

Paper-facing workflow. The counts below are enforced by the code, not newly
measured from a bundled run. The final `structure_retrieval_outputs/` archive is
absent from this source copy.

## Required inputs

Paths are relative to the repository root; associated source manifests are needed.

| Input | Location |
|---|---|
| Ground truth | `data/processed/retrieval_ground_truth.jsonl` |
| Legacy split provenance | `data/db_export/split.csv` |
| Candidate export | `structure_retrieval_outputs/t2_candidate_corpus_v1/text_candidate_patents_common.csv` |
| Original structures | `structure_retrieval_outputs/t2_surechembl/surechembl_structures.csv` |
| Prepared structures | `structure_retrieval_outputs/t2_molformer_input_v1/molformer_eligible_structures.csv` |
| NV/Qwen full rankings/per-query results | `structure_retrieval_outputs/t3_rrf_v1/inputs/t1_text_full/` |
| Morgan full rankings/per-query results | `structure_retrieval_outputs/t3_rrf_v1/inputs/t2_morgan_full/` |
| Arctic baseline | `structure_retrieval_outputs/t123_5fold_v1/inputs/t1_arctic_whole_claims_full/` |
| T2 common-input baseline results | `structure_retrieval_outputs/t2_morgan_common_input_v1/`, `t2_molformer_mean_v1/`, `t2_representative_tanimoto_k16_v1/` |
| Assignment/manifest | `structure_retrieval_outputs/t123_5fold_v1/t123_5fold_assignment.csv` and its manifest |

M1–M4 require both full `*top2217.jsonl` ranking files, not only Top-100 results.
Vector stores and `data/epo_txt/` are needed to regenerate the text inputs.
No public archive URL is provided. The freeze script enforces exact SHA-256
requirements. Do not change expected hashes or cohorts to accept a different run.

## Population and folds

| Property | Enforced value |
|---|---|
| Export / structure-bearing candidates | 2,376 / 2,218 publications |
| Ranked candidates | 2,217 after self-exclusion |
| Citation / Conflict-X | 1,126 / 47 queries |
| Groups | 908 shared-gold connected components |
| Split | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260924)` |
| Stratification | Conflict-X eligibility |

| Fold | Citation | Conflict-X | Groups |
|---|---:|---:|---:|
| 0 | 228 | 11 | 179 |
| 1 | 224 | 9 | 179 |
| 2 | 224 | 9 | 181 |
| 3 | 226 | 9 | 185 |
| 4 | 224 | 9 | 184 |

Groups link eligible queries sharing gold publications, not family IDs. Query
nodes and gold-publication nodes use separate namespaces: a publication ID's
appearance as both a query and gold does not itself link those roles.
This is complete-gold closed-pool evaluation, not a
temporal deployment test. Each query has one held-out result. OOF summaries
average query-level metrics, not the five fold means.

## Methods

| Method | Behaviour |
|---|---|
| Morgan mean cosine | Mean binary radius-2, 2,048-bit Morgan fingerprints, cosine |
| MoLFormer mean pooling | Mean of molecular embeddings per patent, cosine, common prepared input |
| Representative max Tanimoto | Farthest-first representatives, up to 16 per patent, maximum cross-patent Tanimoto |
| M1 | Equal RRF: sum of `1/(60+rank)` |
| M2 | Per-query z-score weighted text/Morgan fusion over all candidates |
| M3 | Weighted reranking within text Top-100/Top-500; tail retains text order |
| M4 | Pairwise logistic fusion on score/rank features |

MoLFormer defaults: `ibm-research/MoLFormer-XL-both-10pct`, revision
`361063d0ad524ef77cf39b08469f6be770dc550f`, preparation token ceiling 202.
The evaluator uses `pooler_output` and checks attention-mask-mean equivalence at
runtime. Original `canonical_smiles` and prepared `model_smiles` counts are not
interchangeable.

M2/M3 alpha grid: 0.0–1.0, step 0.1, selected on Citation MRR@100 of the other
four folds. Within 1e-12 of the best value, prefer closest to 0.5, then larger alpha.

M4 features: text/structure z-scores, two `1/log2(rank+1)` features, and z-score
product. C grid: 0.01, 0.1, 1.0, 10.0. Select C on an inner shared-gold-disjoint
development partition targeting 20% of outer training queries, then refit on
all four training folds. Ties within 1e-12 prefer smaller C. Negatives exclude
known Citation gold: up to 50 unseen negatives from each modality's ranking,
then up to 50 random remaining negatives without replacement. Symmetric
positive-minus-negative feature differences form training pairs. Training uses
`StandardScaler`, logistic `liblinear`, no intercept, and 1,000 maximum iterations.

## Execution order

Restore the required inputs first. Run from the root in the structure/fusion
environment. Preserve frozen output directories: many scripts refuse overwrite.
Some accept `--output-dir`, but downstream scripts use fixed default paths.

```bash
python scripts/structure_retrieval/freeze_t123_5fold_assignment.py
python scripts/structure_retrieval/evaluate_t123_5fold_standalone.py
python scripts/structure_retrieval/evaluate_t123_5fold_m1_rrf.py
python scripts/structure_retrieval/evaluate_t123_5fold_m2_weighted.py
python scripts/structure_retrieval/evaluate_t123_5fold_m3_reranking.py
python scripts/structure_retrieval/evaluate_t123_5fold_m4_learned.py
python scripts/structure_retrieval/compare_t123_5fold_oof.py
```

Standalone evaluation aggregates fixed baseline results; it does not recompute
embeddings. T3 combines NV-document or Qwen-whole-claims with Morgan.
Additional T1 settings use `evaluate_t123_extra_t1_setting.py` separately and do
not silently expand the prespecified comparison family.
That script defaults to `data/qdrant_text_units/`; its `--qdrant-path` option can
point to another existing text-unit store without changing the frozen cohort or
ranking definition. A store must be restored/supplied first; it is not bundled.

## Validation

Final independent recalculation/hash-check scripts are retained:

```bash
python scripts/structure_retrieval/validate_t123_5fold_standalone.py --analysis-dir structure_retrieval_outputs/t123_5fold_v1/standalone
python scripts/structure_retrieval/validate_t123_5fold_m1_rrf.py --analysis-dir structure_retrieval_outputs/t123_5fold_v1/m1_equal_rrf
python scripts/structure_retrieval/validate_t123_5fold_m2_weighted.py --analysis-dir structure_retrieval_outputs/t123_5fold_v1/m2_weighted
python scripts/structure_retrieval/validate_t123_5fold_m3_reranking.py --analysis-dir structure_retrieval_outputs/t123_5fold_v1/m3_text_first_reranking
python scripts/structure_retrieval/validate_t123_5fold_m4_learned.py --analysis-dir structure_retrieval_outputs/t123_5fold_v1/m4_pairwise_logistic
python scripts/structure_retrieval/validate_t123_5fold_oof_comparisons.py --analysis-dir structure_retrieval_outputs/t123_5fold_v1/comparisons
```

Metrics: R@1/5/10/50/100, MRR@100, MAP@100, binary nDCG@10. Group-paired
comparison defaults: 10,000 iterations, base seed 20260926. Primary Citation-MRR
family: ten T3-versus-T1 tests; secondary: six T1-versus-T2 tests. Use manifests
and Holm-adjusted tests before making a significance claim. Final validation has
not been rerun on this source-only copy.

## Preparation and helper dependencies

Preparation: `extract_surechembl_structures.py`, `export_text_candidate_corpus.py`,
`prepare_molformer_input.py`, `evaluate_t1_matched_subset.py`, and the three
`evaluate_t2_*.py` evaluators. Data provenance/counting tools:
`audit_dataset_filtering_counts.py`, `audit_dataset_upstream_provenance.py`,
`audit_patent_structure_statistics.py`.

The two `audit_t123_5fold_*` modules and legacy `evaluate_t3_*.py` files are retained
because the final workflow imports their grouping/scoring/training helpers.
Their legacy standalone entry points are not the final five-fold workflow.
