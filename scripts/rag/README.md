# Downstream citation judgment and RAG

This package integrates a separately supplied downstream implementation without
replacing the frozen [T1–T3 workflow](../structure_retrieval/README.md).
The supplied folder contained source files only: no model responses, ranking
bundles, text databases, result tables, or historical environment lockfile.
The implementation below is source-backed; downstream scores, timings, sample
counts, and compatibility with a particular server have not been verified.

## Entry points

Run from the repository root with `python -m scripts.rag.<module>`.
Package-relative imports prevent collisions with the root `config/` and `data/`.

| Module | Behavior |
|---|---|
| `run_judgment` | Gives a query and examiner-cited document directly to the LLM; conditions `base`, `think`, `nodefs`, `3class` |
| `run_rag` | Judges retrieved query/candidate pairs; `citation` or `conflict_x`; default k=50 and 16 workers |
| `build_bm25` | Builds BM25Okapi rankings from title plus numbered individual claims; excludes the query itself; overwrites the selected BM25 output |
| `evaluate` | Evaluates stored rankings or judgment/RAG JSONL results; does not call the LLM |
| `analyze_3class` | Breaks down available judgment records by category; does not generate predictions |
| `serve_vllm.sh` | Optional inference-server launcher; not an evaluation or verified deployment recipe |

### Conditions and inputs

| Condition | Ranking selection or construction |
|---|---|
| `T1_nv` | T1 JSONL with `source=nv_document` |
| `T1_qwen` | T1 JSONL with `source=qwen_whole_claims` |
| `T2_morgan` | Morgan JSONL, without a source-key filter |
| `T3_m4` | M4 JSONL with `text_source=nv_document`; supplied filename indicates Top-100 |
| `BM25` | Automatically enabled only when the configured BM25 ranking file exists |
| `oracle` | Task-specific gold in the structure pool plus random distractors, shuffled and truncated to k; may omit gold when more than k are available |
| `closedbook` | No candidate documents; asks for publication identifiers from model memory |

`--retriever all` includes active retrievers and both reference conditions.
`experiment` includes active retrievers only. This implementation does not
add Arctic or M1–M3 downstream conditions. Oracle and closed-book are reference
conditions, not guaranteed upper/lower performance bounds or a proven
contamination test.

## Required artifacts and paths

`CHEMRAG_SERVER` defaults to this repository root; it is an input-artifact root,
not an SSH host. Scientific inputs are not generated or moved by configuration.

| Setting | Default relative to the artifact root / required schema |
|---|---|
| `CHEMRAG_GT` | Sibling `../epo_dataset/in_corpus_candidate_pairs.csv`; `query_patent_id,cited_patent_id,category` |
| `CHEMRAG_FOLDS` | `structure_retrieval_outputs/t123_5fold_v1/t123_5fold_assignment.csv`; `query_id,outer_fold,citation_eligible,conflict_x_eligible` |
| `CHEMRAG_STRUCT` | `structure_retrieval_outputs/t2_molformer_input_v1/molformer_eligible_structures.csv`; `patent_id` |
| `CHEMRAG_RANK_T1` | `structure_retrieval_outputs/t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_rankings_top2217.jsonl` |
| `CHEMRAG_RANK_MORGAN` | `structure_retrieval_outputs/t3_rrf_v1/inputs/t2_morgan_full/t2_structure_rankings_top2217.jsonl` |
| `CHEMRAG_RANK_M4` | `structure_retrieval_outputs/t123_5fold_v1/m4_pairwise_logistic/t123_m4_learned_rankings_top100.jsonl` |
| `CHEMRAG_TEXT_SQLITE` | `data/qdrant_text_units/collection/patent_text_units_qwen3_0_6b/storage.sqlite` |
| `CHEMRAG_GROUNDING` | `data/db_export/grounding_gold.csv`; auxiliary loader, not scored by the current evaluator |
| `CHEMRAG_OUT` | `structure_retrieval_outputs/rag/results/` |
| `CHEMRAG_CACHE` | `structure_retrieval_outputs/rag/cache/` |
| `CHEMRAG_BM25` | `structure_retrieval_outputs/rag/bm25_rankings_top2217.jsonl` |

Ranking records require `query_patent_id` and a `ranking` list containing
`patent_id` and one-based `rank`. The fold loader interprets eligibility as
the exact string `1`. The CSV ground truth is a separate input from the
existing processed JSONL; do not silently substitute it or assume equivalence.
Repeated query/cited CSV pairs overwrite earlier categories in this loader.
Conflict-X uses an exact `X` token after splitting the category on commas,
without stripping whitespace or changing case.

The SQLite store is opened read-only. Its `points` table must have `id` and
`point` columns, with pickled objects holding payload fields `patent_id`,
`unit_type`, `text`, `title`, and `claim_no` for individual claims.
Only load trusted databases/caches: pickle deserialization can execute code.
Store layout and text coverage must be checked on the intended server.
If an existing store has another directory name, set `CHEMRAG_TEXT_SQLITE`
to its actual SQLite file; the integration does not rename a server database.
`CHEMRAG_QDRANT` can alternatively override the parent of `qdrant_text_units/`.

## Prompts and execution settings

Prompts are preserved from the supplied `prompts.py`. RAG uses X/CITED/NOT_CITED;
judgment conditions use X/NOT_X or X/Y/A. Inputs are patent ID, document-unit
text, and individual claims, limited to 9,000 characters by default through
`CHEMRAG_MAX_DOC_CHARS`; publication dates and IPC are not separately injected.
The prompt explicitly instructs the model to ignore publication dates.

The client defaults to `LLM_BACKEND=vllm`, `LLM_MODEL=Qwen3-8B`,
`LLM_BASE_URL=http://localhost:8000/v1`, temperature 0, output limit 2,000 tokens,
and timeout 600 seconds. Thinking mode raises the output limit to at least
8,000 tokens. An Ollama path is also implemented. These are configured settings,
not evidence of historical model revisions or successful server execution.
The client dependencies are in `requirements.txt`; inference serving needs a
separately prepared compatible environment.

After restoring and checking the required inputs:

```bash
python -m pip install -r scripts/rag/requirements.txt

# Stored-rank evaluation; writes ranking caches, but makes no LLM calls.
python -m scripts.rag.evaluate --retrieval

# Optional lexical baseline; overwrites CHEMRAG_BM25.
python -m scripts.rag.build_bm25

# Planning makes no LLM calls, but may create ranking caches.
python -m scripts.rag.run_rag --plan --retriever all --task citation --k 50

# Run only after checking endpoint/GPU availability and using fresh run paths.
python -m scripts.rag.run_judgment --cond all --workers 16
python -m scripts.rag.evaluate --judgment
python -m scripts.rag.run_rag --retriever T1_nv --task citation --k 50 --workers 16
python -m scripts.rag.evaluate 'structure_retrieval_outputs/rag/results/rag_*.jsonl'
python -m scripts.rag.analyze_3class
```

These are execution instructions, not operations performed during integration.
If `CHEMRAG_OUT` is changed, also change the evaluator's explicit file glob.
For an optional planning-time estimate, supply your own measured
`--calls-per-second`; no unverified hardware-throughput constant is assumed.
The launcher defaults to GPU 0, port 8000, BF16, context 32,768, 32 sequences,
and GPU-memory-utilization setting 0.90. These settings are not a guarantee of
memory fit or performance; do not launch on a shared GPU without authorization.

## Metric definitions and preserved limitations

- Retrieval `R@k` is the per-query fraction of gold documents retrieved,
  averaged across the evaluated queries. `MRR` is not truncated at 100;
  missing gold ranks count as zero reciprocal rank. M4's truncated input limits
  its available ranks. Do not equate this column with frozen T1–T3 `MRR@100`.
- Retrieval reports a query-level mean and sample SD of fold means, not the
  unweighted mean of five fold means. Judgment and RAG evaluation pool records
  and do not report per-fold SD.
- RAG's column named `retrieval R@k` is a query hit rate (at least one gold
  candidate), unlike retrieval-only `R@k`. `final found` is the fraction of
  processed query records with at least one correctly positive judgment.
  Candidate recall counts judged, successfully parsed pairs, not missing gold
  documents outside retrieval.
- `--sample-neg N` keeps task-specific retrieved gold and samples non-gold.
  Precision/F1 use inverse-sampling-weighted false positives. The displayed FP
  rate is a pooled unweighted sample proportion, not guaranteed to estimate
  the full-pool rate when query sampling fractions differ.
- Retrieval/RAG queries use eligibility flags but are further restricted to
  queries present in the selected rankings. Judgment pairs only require fold
  membership, not `citation_eligible`. Check the actual denominators before
  reporting results. The fold CSV is reused; this package does not recreate or
  independently validate its groups, stratification, or leakage properties.
- RAG loads every active retriever ranking even when selecting one condition.
  Top-candidate caches store only 200 entries; use k no greater than 200 and
  no greater than the depth available in each input (100 for the named M4 file).
- Ranking-cache validity checks only the ranking-file modification time;
  judgment-cache keys contain only query/candidate IDs. They do not identify
  model, prompt, thinking mode, input hash, or text version. Use fresh output
  and cache directories for any configuration/input change.
- Existing output IDs are skipped on resume, including error records. Error
  and parse-failure counts must be inspected. The RAG evaluator's inherited
  error filter checks only the first 60 characters of a JSONL line; it is not
  a complete result-schema validator. Do not publish unchecked mixed runs.
- With no judgment result files, the judgment evaluator can still print a
  majority baseline; that is not evidence of a completed LLM experiment.

Integration changed namespacing, neutral configurable paths, documentation,
console descriptions, planning-time estimates, and one formatting expression
for pre-Python-3.12 syntax compatibility. Scientific prompts, candidate
selection, sampling, label parsing, and metric formulas were not redesigned.
No downstream experiment or numeric reproduction is claimed by this merge.
Generated logs may contain machine paths in errors; review them before release.
