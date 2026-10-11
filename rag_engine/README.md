# Legacy retrieval and ranking

Used by `run_chem_rag.py`. This directory does not implement an LLM answer
generator or the final T3 five-fold fusion workflow.

| Module | Behaviour |
|---|---|
| `chemical_tokenizer.py` | Chemical-aware tokenization |
| `morgan_fingerprint_matcher.py` | Binary Morgan fingerprints (radius 2, 2,048 bits), maximum Tanimoto across available molecule pairs |
| `hybrid_retriever.py` | Legacy text and structure scoring |
| `multimodal_ranker.py` | Weighted text/structure/vision combination, normalized according to available signals |

Legacy default weights are 0.5/0.3/0.2 in `config/settings.py`, not the tuned
M1–M4 settings. `scripts/evaluation/evaluate_full_corpus_retrieval.py` is a separate
legacy evaluator. See `scripts/structure_retrieval/README.md` for the frozen
experiment; different candidate pools and ground-truth tracks are not interchangeable.
