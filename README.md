# Chem-RAG: text and chemical-structure patent retrieval

This repository implements patent-data preparation, text/structure retrieval, and
text–structure fusion. The paper-facing experiment is the frozen T1/T2/T3
five-fold evaluation in [scripts/structure_retrieval](scripts/structure_retrieval/README.md).
Downstream citation judgment and retrieval-augmented judgment are implemented
separately in [scripts/rag](scripts/rag/README.md).

**Release status:** the frozen experiment bundle, text-unit vector databases, and
`data/epo_txt/` are not included in this source copy. This is not a self-contained
reproduction of the reported results. Do not substitute legacy data or change
frozen input hashes to make a different experiment appear equivalent.

## Experiment scope

| Branch | Implementation |
|---|---|
| T1 | NV-Embed-v2 document, Qwen3 whole-claims, Arctic whole-claims; additional fixed input settings have a separate evaluator |
| T2 | Mean binary Morgan fingerprint cosine; mean molecular MoLFormer embedding cosine; maximum Morgan Tanimoto over up to 16 representatives |
| T3 | Equal RRF, weighted score fusion, text-first Top-100/Top-500 reranking, pairwise logistic fusion |

The frozen code enforces 1,126 Citation queries, a 47-query Conflict-X subset,
908 shared-gold connected groups, and 2,217 ranked candidates after self-exclusion
from 2,218 structure-bearing candidates. These are input requirements, not counts
newly measured from an included result bundle. The larger candidate export contains
2,376 publications. This is a complete-gold closed-pool evaluation, not a
time-filtered or family-deduplicated benchmark. The folds concern evaluation and
fusion fitting, not embedding-model fine-tuning.

## Repository map

| Directory | Purpose |
|---|---|
| [scripts/structure_retrieval](scripts/structure_retrieval/README.md) | Retrieval, frozen folds, fusion, comparison, reproducibility checks |
| [scripts/rag](scripts/rag/README.md) | Downstream judgment, RAG, BM25, and result-log evaluation |
| [scripts](scripts/README.md) | Preparation and auxiliary/legacy CLIs |
| [data/db_export](data/db_export/README.md) | Legacy relational CSV export, not the final ranking bundle |
| [data_parser](data_parser/README.md) | Search-report, patent XML, and full-text parsing |
| `embeddings/` | Encoders for the separate patent-level Qdrant baseline |
| [qdrant_engine](qdrant_engine/README.md) | Patent-level baseline and snapshot tooling |
| [rag_engine](rag_engine/README.md) | Legacy token/structure retrieval and ranking |
| [vision_ocsr](vision_ocsr/README.md) | Optional OPS/OCSR support, not the final T2 structure source |
| [benchmark](benchmark/README.md) | Legacy ranking evaluator |
| [config](config/README.md) | Legacy configuration |
| `tests/` | Legacy baseline unit and Qdrant integration tests |

`rag_engine/` remains a legacy retrieval package; the downstream LLM evaluator
is in `scripts/rag/`. Its result logs and required text database were not supplied
with the integrated source. Retrieval results are not end-to-end generator
results, and source integration does not validate downstream experimental scores.

## Environment

Use separate environments for the legacy baseline and structure/fusion workflow:
`requirements.txt` and `requirements-structure-retrieval.txt` specify different
RDKit versions. A single installation of both is not a reproduction recipe.

```bash
python -m venv .venv_structure
source .venv_structure/bin/activate
python -m pip install -r requirements-structure-retrieval.txt
```

The structure requirements list direct dependencies, not a complete historical
lockfile. Install PyTorch separately for the intended CPU/CUDA platform.
Text-embedding construction also imports `sentence_transformers` and requires a
separately resolved compatible environment. Runtime manifests are the authority
for the environment of a particular archived run.

## Reproduction and data

Follow the [paper workflow](scripts/structure_retrieval/README.md) from the
repository root after restoring its required inputs. Ground truth and legacy
CSVs are present but insufficient alone. An archive omitting full
`*top2217.jsonl` text/Morgan rankings cannot reproduce M1–M4.

External SureChEMBL/PatCID dumps, downloaded images, internal diagnostics, legacy
result caches, and inherited Git history are excluded from this submission copy.
Obtain external assets under their own terms. See the repository `LICENSE`;
external data/model weights are separate assets. The contributed RAG folder
did not contain a separate license or redistribution statement. Redistribution
permission for included research data must be checked before publication. No
data licence or public data-download location is asserted here.

## Anonymous release and local paths

Run commands from your own checkout, for example:

```bash
cd /putyourdirectory/Project/Chem-RAG
```

Repository paths are derived from the checkout or supplied through CLI arguments
and environment variables. External parser inputs use `CHEMRAG_EXTERNAL_DATA_ROOT`,
`CHEMRAG_EPO_PARSED_BASE`, and `CHEMRAG_EPO_XML_DIR`; see the
[parser documentation](data_parser/README.md). Do not replace real scientific
inputs with placeholder data or weaken frozen hash checks for anonymization.

Before publishing additional files, inspect manifests, logs, notebook outputs,
document/image metadata, and archive metadata for private paths, credentials,
hostnames, author names, and contact details. Generated artifacts can record paths
from the machine on which they were created; source anonymization does not
automatically sanitize future outputs. Keep private backups outside the release.

The original Git history is not supplied. Use an anonymous hosting account and
anonymous commit author/committer identity when creating the submission history;
also review remotes and download/share links. Publish tracked, reviewed files,
not an unreviewed workstation folder. When making an archive, exclude operating-
system extended attributes and user/group ownership metadata. A `.gitignore`
rule does not remove files from an archive or from existing Git history.

## Source language and compatibility

Source comments, docstrings, command-line help, and diagnostic messages are in
English. The legacy `effect_decomposition` JSON keys in
`scripts/evaluation/evaluate_qdrant_retrieval.py` retain their original spelling
for compatibility with saved reports and readers; their test references are
preserved as well. They are schema identifiers, not user-facing descriptions.
Do not translate experimental prompts, source-data values, parsing patterns,
or existing result files as a documentation cleanup.
