# Optional OPS and OCSR support

Legacy full-text acquisition and exploratory optical chemical-structure
recognition. These outputs are not the final T2/T3 structure input.

| Module | Role |
|---|---|
| `epo_ops_fetcher.py` | OPS authentication, bibliographic/full-text retrieval, image downloads |
| `ocsr_pipeline.py` | Cached fallback, preferring available PatCID `Clean` SMILES |
| `ocsr_smiles_converter.py` | Optional DECIMER wrapper; reports unavailable/failed recognition |
| `ocsr_cache_lookup.py` | Standard-library cache lookup for the main legacy pipeline |
| `patcid_bbox_cropper.py` | PatCID bounding-box cropper |
| `chem_vision_cropper.py` | Optional contour-based cropper exported by the package |

OPS requires environment credentials. Images and responses are stored in local
caches, not supplied here. DECIMER/image-processing dependencies are optional
and are not fully specified by the structure/fusion requirements.

`OCSRPipeline.ENABLE_BBOX_CROP_FALLBACK=False` disables the bbox fallback. Do not
describe it as a verified source of exact structure regions or frozen SureChEMBL
retrieval results.
