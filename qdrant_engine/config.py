"""Qdrant 파이프라인 설정 - 하드코딩 금지 지점을 한곳에 모은 모듈.

우선순위: 환경변수 > config/qdrant_settings.json > 아래 기본값.
비밀값(API key)은 환경변수로만 받고 파일/코드에 쓰지 않는다.
"""
import os
import json
import dataclasses
from dataclasses import dataclass, field, asdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SETTINGS_PATH = os.path.join(PROJECT_DIR, 'config', 'qdrant_settings.json')

# 포인트 ID(UUID5) 생성용 고정 네임스페이스. 절대 바꾸지 말 것 -
# 바꾸면 기존에 적재된 포인트와 ID가 달라져 upsert 멱등성이 깨진다.
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
    # --- 접속 ---
    # url이 비어 있으면 로컬 임베디드 모드(local_path)를 쓴다. 서버를 쓰려면
    # CHEMRAG_QDRANT_URL=http://localhost:6333 처럼 지정.
    url: str = ''
    api_key_env: str = 'CHEMRAG_QDRANT_API_KEY'   # 값이 아니라 '환경변수 이름'만 저장
    # Patent-level GloVe/BM25 baseline store, separate from the text-unit store.
    local_path: str = os.path.join(PROJECT_DIR, 'data', 'qdrant_patent_baseline')
    prefer_grpc: bool = False
    timeout: int = 120

    # --- 컬렉션 ---
    patents_collection: str = 'chemrag_patents_v1'
    structures_collection: str = 'chemrag_structures_v1'
    patents_alias: str = 'chemrag_patents'
    structures_alias: str = 'chemrag_structures'

    # --- 벡터 이름 ---
    # 입력 필드(claims only vs title+claims) 효과를 알고리즘 효과와 분리해 평가하기 위해
    # 필드별로 dense/sparse 벡터를 따로 저장한다.
    dense_vector_name: str = 'title_claims_dense'      # 기본 검색에 쓰는 dense 벡터
    sparse_vector_name: str = 'title_claims_bm25'      # 기본 검색에 쓰는 sparse 벡터
    fingerprint_vector_name: str = 'morgan_ecfp4_2048'

    # --- 임베딩 ---
    dense_backend: str = 'static-vectors'   # static-vectors | hashing | fastembed | sentence-transformers
    dense_model: str = 'glove-wiki-gigaword-300'
    static_vectors_path: str = ''           # 비우면 data/embeddings_cache/{dense_model}.npz
    dense_dim: int = 384                    # hashing 백엔드에서만 의미 있음(모델 백엔드는 모델이 결정)
    embedding_version: str = 'v1'
    max_text_chars: int = 8000              # 임베딩 입력 텍스트 상한(모델 토큰 한계 보호)

    # --- BM25 ---
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # --- 색인 ---
    batch_size: int = 128

    # --- 검색 ---
    search_limit: int = 1000
    prefetch_limit: int = 2000
    fusion: str = 'rrf'                     # rrf | dbsf
    structure_rerank_top_k: int = 100       # RDKit exact Tanimoto를 계산할 상위 후보 수
    structure_rerank_weight: float = 0.3    # multimodal_ranker 기본 tanimoto 가중치와 동일
    text_weight: float = 0.5                # multimodal_ranker 기본 text 가중치와 동일

    # --- 화학구조 지문 ---
    morgan_radius: int = 2                  # ECFP4
    morgan_nbits: int = 2048

    def api_key(self):
        return os.environ.get(self.api_key_env) or None

    def to_public_dict(self):
        """비밀값을 제외한 설정 스냅샷(결과 파일에 기록용)."""
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
    """기본값 -> JSON 설정파일 -> 환경변수 -> 명시적 overrides 순으로 병합."""
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
