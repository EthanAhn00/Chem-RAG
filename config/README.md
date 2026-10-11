# Legacy configuration

`settings.py` defines project-relative data/output paths, external EPO input paths,
IPC prefixes, legacy score weights, and OPS/OCSR settings. Importing it creates
configured data, sample, benchmark-output, and OCSR-cache directories.

| Setting | Code default / meaning |
|---|---|
| `ID_PATENT_MAP_PATH` | `data/reference/id_patent_map.csv` |
| `TRAIN_TEST_SPLIT_PATH` | Optional legacy `data/reference/train_test_split.json` |
| IPC prefixes | `C`, `A61K`, `A61P` |
| Text / Tanimoto / vision weights | 0.5 / 0.3 / 0.2 |
| Stream sample cap | 2,000 |
| `ENABLE_OCSR_FALLBACK` | Enabled only by environment value `1` |
| OPS credentials | `EPO_OPS_CONSUMER_KEY`, `EPO_OPS_CONSUMER_SECRET` |
| Maximum OPS drawing pages | 3 per patent |

Set `CHEMRAG_EXTERNAL_DATA_ROOT`, or the more specific `CHEMRAG_EPO_JSON_DIR`
and `CHEMRAG_EPO_XML_DIR`, for external sources. Those inputs are not bundled.
The legacy split filename does not establish a verified family-safe split.

The paper-facing evaluators use their own CLI inputs/constants/manifests; legacy
score weights here do not change M1–M4. Qdrant settings are defined separately in
`qdrant_engine/config.py`. Credentials are never supplied by this repository.
