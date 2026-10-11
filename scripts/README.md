# Scripts

Run from the repository root. Start with the
[paper workflow](structure_retrieval/README.md).

| Directory | Scope |
|---|---|
| `structure_retrieval/` | Retrieval, frozen evaluation/fusion, comparison, final validation |
| [rag/](rag/README.md) | Downstream cited-document judgment, RAG, BM25, and result evaluation |
| `db_export/` | Retrieval ground-truth construction and legacy CSV preparation/validation |
| `qdrant/` | Text-unit embedding construction and separate patent-level baseline tooling |
| `evaluation/` | Earlier text-unit evaluator and legacy corpus/Qdrant evaluation |
| `ocsr/` | Optional cached OCSR execution |

Exploratory ADMET/MMPA property analysis, server inventory, OPS pilot probes,
token-length diagnostics, and pre-five-fold post-hoc comparisons/validators are
excluded. Final evaluation helpers, data-provenance/counting audits, final
independent validators, and baseline tests are retained.

## Text-unit embeddings

`qdrant/build_epo_txt_vector_db.py` accepts `--model qwen|nvembed|arctic`,
data/store paths, collection, and batch settings. It builds document, whole-claims,
and individual-claim units.

| Model option | Identifier |
|---|---|
| `qwen` | `Qwen/Qwen3-Embedding-0.6B` |
| `nvembed` | `nvidia/NV-Embed-v2` |
| `arctic` | `Snowflake/snowflake-arctic-embed-l-v2.0` |

It imports PyTorch, sentence-transformers, NumPy, and qdrant-client but does not
pin every historical model/dependency revision. A rebuild is not guaranteed to
equal archived embeddings. `--recreate` deletes the selected collection.

The builder and earlier text evaluator default to the neutral text-unit store
`data/qdrant_text_units/`. The separate patent-level baseline uses
`data/qdrant_patent_baseline/`. No vector stores are bundled or migrated here.
For the builder, set `--qdrant-path` explicitly to use another existing store.

`db_export/build_retrieval_ground_truth.py` reads `data/epo_txt/epo_txt_EP*.json`
and `data/**/epo_srpt_label_*.json`. Executing it overwrites processed pair/gold
files; do not run against a frozen input bundle.
`evaluation/evaluate_text_retrieval.py` is the earlier evaluator; final matched
T1 inputs are produced by `structure_retrieval/evaluate_t1_matched_subset.py`.

Both ground-truth construction and the earlier evaluator run their workflow at
module execution. Do not import them as libraries or run them against frozen
inputs without preparing a separate working copy.

For legacy baseline/export/OCSR details, see `qdrant_engine/README.md`,
`data/db_export/README.md`, and `vision_ocsr/README.md` at the repository root.
These auxiliary workflows do not replace the frozen paper archive.

## Downstream RAG

The integrated RAG code is isolated in `scripts/rag/` and uses package-relative
imports. Run its entry points from the repository root as
`python -m scripts.rag.<module>`, not as standalone files. It consumes existing
rankings and a read-only text-unit SQLite store; it does not rebuild T1–T3.
Required inputs, output/cache paths, and metric differences are documented in
[rag/README.md](rag/README.md). Result logs were not included with the source.
