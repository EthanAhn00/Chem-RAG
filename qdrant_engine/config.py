"""Centralized Qdrant configuration. Priority: environment > config/qdrant_settings.json > defaults. API keys are environment-only; never write secrets into files or code."""
import os
import json
import dataclasses
from dataclasses import dataclass, field, asdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SETTINGS_PATH = os.path.join(PROJECT_DIR, 'config', 'qdrant_settings.json')

# Fixed namespace for UUID5 point IDs. Never change it:
# changing it changes existing point IDs and breaks idempotent upserts.
ID_NAMESPACE = 'f0a1c3d2-5b6e-4f70-9a12-chemrag000001'.replace('chemrag', 'c8e9d1')


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, '') else default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ''):
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, '') else default


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else default


@dataclass
class QdrantSettings:
    # --- Connection ---
    # An empty URL selects local embedded mode (local_path). For a server,
    # set e.g. CHEMRAG_QDRANT_URL=http://localhost:6333.
    url: str = ''
    api_key_env: str = 'CHEMRAG_QDRANT_API_KEY'   # Store the environment-variable name, not the secret value.
    # Patent-level GloVe/BM25 baseline store, separate from the text-unit store.
    local_path: str = os.path.join(PROJECT_DIR, 'data', 'qdrant_patent_baseline')
    prefer_grpc: bool = False
    timeout: int = 120

    # --- Collections ---
    patents_collection: str = 'chemrag_patents_v1'
    structures_collection: str = 'chemrag_structures_v1'
    patents_alias: str = 'chemrag_patents'
    structures_alias: str = 'chemrag_structures'

    # --- Vector names ---
    # Separate input-field effects (claims vs title+claims) from algorithm effects
    # by storing dense/sparse vectors separately for each field.
    dense_vector_name: str = 'title_claims_dense'      # Dense vector for default search.
    sparse_vector_name: str = 'title_claims_bm25'      # Sparse vector for default search.
    fingerprint_vector_name: str = 'morgan_ecfp4_2048'

    # --- Embeddings ---
    dense_backend: str = 'static-vectors'   # static-vectors | hashing | fastembed | sentence-transformers
    dense_model: str = 'glove-wiki-gigaword-300'
    static_vectors_path: str = ''           # If empty, use data/embeddings_cache/{dense_model}.npz.
    dense_dim: int = 384                    # Relevant only to hashing; model backends determine their own dimension.
    embedding_version: str = 'v1'
    max_text_chars: int = 8000              # Maximum embedding-input length to protect model token limits.

    # --- BM25 ---
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # --- Indexing ---
    batch_size: int = 128

    # --- Retrieval ---
    search_limit: int = 1000
    prefetch_limit: int = 2000
    fusion: str = 'rrf'                     # rrf | dbsf
    structure_rerank_top_k: int = 100       # Number of top candidates for RDKit exact Tanimoto.
    structure_rerank_weight: float = 0.3    # Matches multimodal_ranker's default Tanimoto weight.
    text_weight: float = 0.5                # Matches multimodal_ranker's default text weight.

    # --- Chemical fingerprints ---
    morgan_radius: int = 2                  # ECFP4
    morgan_nbits: int = 2048

    def api_key(self):
        return os.environ.get(self.api_key_env) or None

    def to_public_dict(self):
        """Settings snapshot without secrets, for result provenance."""
        d = asdict(self)
        d['api_key_configured'] = bool(self.api_key())
        d.pop('api_key_env', None)
        return d


_FIELD_TYPES = {f.name: f.type for f in dataclasses.fields(QdrantSettings)}

_ENV_MAP = {
    'url': 'CHEMRAG_QDRANT_URL',
    'local_path': 'CHEMRAG_QDRANT_LOCAL_PATH',
    'prefer_grpc': 'CHEMRAG_QDRANT_PREFER_GRPC',
    'timeout': 'CHEMRAG_QDRANT_TIMEOUT',
    'patents_collection': 'CHEMRAG_PATENTS_COLLECTION',
    'structures_collection': 'CHEMRAG_STRUCTURES_COLLECTION',
    'patents_alias': 'CHEMRAG_PATENTS_ALIAS',
    'structures_alias': 'CHEMRAG_STRUCTURES_ALIAS',
    'dense_backend': 'CHEMRAG_DENSE_BACKEND',
    'dense_model': 'CHEMRAG_DENSE_MODEL',
    'static_vectors_path': 'CHEMRAG_STATIC_VECTORS_PATH',
    'dense_dim': 'CHEMRAG_DENSE_DIM',
    'embedding_version': 'CHEMRAG_EMBEDDING_VERSION',
    'bm25_k1': 'CHEMRAG_BM25_K1',
    'bm25_b': 'CHEMRAG_BM25_B',
    'batch_size': 'CHEMRAG_BATCH_SIZE',
    'search_limit': 'CHEMRAG_SEARCH_LIMIT',
    'prefetch_limit': 'CHEMRAG_PREFETCH_LIMIT',
    'fusion': 'CHEMRAG_FUSION',
    'structure_rerank_top_k': 'CHEMRAG_STRUCTURE_RERANK_TOP_K',
    'structure_rerank_weight': 'CHEMRAG_STRUCTURE_RERANK_WEIGHT',
    'text_weight': 'CHEMRAG_TEXT_WEIGHT',
}


def load_settings(settings_path=None, overrides=None):
    """Merge defaults -> JSON configuration -> environment -> explicit overrides."""
    settings = QdrantSettings()

    path = settings_path or _env('CHEMRAG_QDRANT_SETTINGS', DEFAULT_SETTINGS_PATH)
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for k, v in (json.load(f) or {}).items():
                if hasattr(settings, k):
                    setattr(settings, k, v)

    for attr, env_name in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw in (None, ''):
            continue
        t = _FIELD_TYPES[attr]
        if t is bool or t == 'bool':
            setattr(settings, attr, raw.strip().lower() in ('1', 'true', 'yes', 'on'))
        elif t is int or t == 'int':
            setattr(settings, attr, int(raw))
        elif t is float or t == 'float':
            setattr(settings, attr, float(raw))
        else:
            setattr(settings, attr, raw)

    for k, v in (overrides or {}).items():
        if v is not None and hasattr(settings, k):
            setattr(settings, k, v)

    return settings
