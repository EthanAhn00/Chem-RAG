# Patent-level Qdrant baseline

Separate legacy baseline, not the frozen T123 ranking bundle or long-context
text-unit embedding builder.

| Module | Role |
|---|---|
| `config.py` | Defaults, optional JSON settings, environment and explicit overrides |
| `client.py` | Embedded/server clients, collection/alias utilities |
| `schemas.py` | Vector/payload schemas and deterministic point IDs |
| `filters.py` | Date/family/self-exclusion filters |
| `patent_retriever.py` | Sparse, dense, hybrid retrieval |
| `structure_repository.py` | Structure lookup and exact RDKit Tanimoto |
| `hybrid_fusion.py` | RRF/DBSF and structure reranking |
| `provenance.py` | Checksums and runtime metadata |
| `snapshot.py` | Snapshot export/restore and integrity checks |

`embeddings/` provides BM25, static vectors, hashing, and optional transformer
backends. The default dense model is `glove-wiki-gigaword-300`, not NV-Embed-v2.
Use `requirements.txt` in a separate legacy environment.

Set `CHEMRAG_QDRANT_URL` for a server or `CHEMRAG_QDRANT_LOCAL_PATH` for an
embedded store. Supply API keys through the named environment variable.
The neutral default embedded path is `data/qdrant_patent_baseline/`, separate
from the text-unit builder/evaluator path `data/qdrant_text_units/`.
No existing database is renamed or migrated by the source refactor.
Do not recreate existing research collections for a fresh-install example.

CLIs in `scripts/qdrant/` create/index/search/verify collections and export/restore
snapshots. `scripts/evaluation/evaluate_qdrant_retrieval.py` evaluates the legacy
baseline. `tests/run_tests.py` reports unit, integration, and full-index checks.
Its `--skip-full-index` flag does not skip integration tests, which require an
accessible environment and create temporary collections.

Changing these baseline settings does not reproduce or redefine the archived
rankings used by five-fold fusion.
