# Legacy ranking evaluator

`evaluator.py` defines `ChemicalRAGEvaluator` for legacy scored records. It
supports pooled legacy AP and query-grouped evaluation. Pooled AP is not standard
query-averaged MAP. Query-grouped output separates single-candidate queries.

Legacy result files, partial worker outputs, and cached summaries are excluded
from this submission copy. They are not the frozen T123 OOF bundle.
`run_chem_rag.py` can recreate `benchmark/results/`.

Paper-facing metrics/comparisons are in `scripts/structure_retrieval/`; the
primary metric is Citation MRR@100. Do not merge results from different candidate
pools or gold definitions.
