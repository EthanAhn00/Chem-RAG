"""Resolve configuration strings to encoder instances. Raise when the requested dense backend is unavailable rather than silently substituting a model and invalidating interpretation. Set allow_fallback=True only when fallback is explicitly permitted."""
import os

from .dense_encoder import (
    HashingDenseEncoder, FastEmbedDenseEncoder, SentenceTransformerDenseEncoder,
    StaticVectorDenseEncoder)
from .sparse_encoder import BM25SparseEncoder

DENSE_BACKENDS = ('static-vectors', 'hashing', 'fastembed', 'sentence-transformers')


def build_dense_encoder(settings, allow_fallback=False, idf_state_path=None):
    backend = (settings.dense_backend or 'hashing').strip().lower()
    if backend not in DENSE_BACKENDS:
        raise ValueError(f'Unknown dense backend: {backend} (available: {DENSE_BACKENDS})')

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
                f"dense backend '{backend}'({settings.dense_model}) initialization failed: {type(exc).__name__}: {exc}\n"
                f"Set CHEMRAG_DENSE_BACKEND=hashing to run without external models."
            ) from exc
        print(f'[registry] {backend} unavailable ({type(exc).__name__}) - falling back to the hashing backend')
        return HashingDenseEncoder(dim=settings.dense_dim)


def build_sparse_encoder(settings):
    return BM25SparseEncoder(k1=settings.bm25_k1, b=settings.bm25_b)
