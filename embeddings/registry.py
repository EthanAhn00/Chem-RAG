"""설정 문자열 -> 인코더 인스턴스 해석.

dense 백엔드는 요청한 것이 설치돼 있지 않으면 예외를 던진다(조용히 다른 모델로
바꿔치기하면 결과 해석이 틀어지므로). 명시적으로 fallback을 허용할 때만
allow_fallback=True를 쓴다.
"""
import os

from .dense_encoder import (
    HashingDenseEncoder, FastEmbedDenseEncoder, SentenceTransformerDenseEncoder,
    StaticVectorDenseEncoder)
from .sparse_encoder import BM25SparseEncoder

DENSE_BACKENDS = ('static-vectors', 'hashing', 'fastembed', 'sentence-transformers')


def build_dense_encoder(settings, allow_fallback=False, idf_state_path=None):
    backend = (settings.dense_backend or 'hashing').strip().lower()
    if backend not in DENSE_BACKENDS:
        raise ValueError(f'알 수 없는 dense 백엔드: {backend} (가능: {DENSE_BACKENDS})')

    try:
        if backend == 'fastembed':
            return FastEmbedDenseEncoder(settings.dense_model, max_chars=settings.max_text_chars)
        if backend == 'sentence-transformers':
            return SentenceTransformerDenseEncoder(settings.dense_model, max_chars=settings.max_text_chars)
        if backend == 'static-vectors':
            from config.settings import DATA_DIR
            npz = settings.static_vectors_path or os.path.join(
                DATA_DIR, 'embeddings_cache', f'{settings.dense_model}.npz')
            idf = idf_state_path or os.path.join(
                DATA_DIR, 'db_export', 'derived', 'bm25_state.json')
            return StaticVectorDenseEncoder(npz, idf_state_path=idf, max_chars=settings.max_text_chars)
        return HashingDenseEncoder(dim=settings.dense_dim)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                f"dense 백엔드 '{backend}'({settings.dense_model}) 초기화 실패: {type(exc).__name__}: {exc}\n"
                f"CHEMRAG_DENSE_BACKEND=hashing 으로 두면 외부 모델 없이 동작합니다."
            ) from exc
        print(f'[registry] {backend} 사용 불가({type(exc).__name__}) - hashing 백엔드로 대체합니다')
        return HashingDenseEncoder(dim=settings.dense_dim)


def build_sparse_encoder(settings):
    return BM25SparseEncoder(k1=settings.bm25_k1, b=settings.bm25_b)
