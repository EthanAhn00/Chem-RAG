"""Dense encoder backends.

static-vectors: IDF-weighted mean pooling of pretrained GloVe word vectors from GitHub releases. Requires only numpy and supports varied Python environments. These are pretrained distributional vectors, not transformer sentence embeddings; distinguish them when reporting.
hashing: deterministic lexical projection without external models/network access. Not semantic embeddings; similar in role to BM25. A default/fallback for restricted environments.
fastembed: ONNX sentence embeddings, e.g. BAAI/bge-small-en-v1.5.
sentence-transformers: torch sentence embeddings.
Always record the backend in results. Never interpret hashing-based dense scores as semantic-embedding performance.
"""
import hashlib
import json
import math
import os
import re

from .base import DenseEncoder

_TOKEN_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9\-\(\)\[\]\,\=\+\.]{1,}')


def tokenize(text):
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def _stable_hash(token, salt=b''):
    """Deterministic hash unaffected by PYTHONHASHSEED."""
    return int.from_bytes(hashlib.blake2b(token.encode('utf-8'), digest_size=8, key=salt).digest(), 'big')


class HashingDenseEncoder(DenseEncoder):
    """Signed hashing with log-tf weighting and L2 normalization. Identical text yields identical vectors across machines/processes. Requires no model download and works offline or with restricted Python versions."""

    name = 'hashing'

    def __init__(self, dim=384, model_id='hashing-signed-logtf-v1'):
        self._dim = int(dim)
        self.model_id = model_id

    @property
    def dim(self):
        return self._dim

    def encode(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * self._dim
            tokens = tokenize(text)
            if tokens:
                counts = {}
                for t in tokens:
                    counts[t] = counts.get(t, 0) + 1
                for token, tf in counts.items():
                    h = _stable_hash(token)
                    idx = h % self._dim
                    sign = 1.0 if ((h >> 61) & 1) else -1.0
                    vec[idx] += sign * (1.0 + math.log(tf))
                norm = math.sqrt(sum(v * v for v in vec))
                if norm > 0:
                    vec = [v / norm for v in vec]
            out.append(vec)
        return out


class FastEmbedDenseEncoder(DenseEncoder):
    """Sentence embeddings using fastembed (ONNX)."""

    name = 'fastembed'

    def __init__(self, model_id='BAAI/bge-small-en-v1.5', max_chars=8000):
        from fastembed import TextEmbedding
        self.model_id = model_id
        self.max_chars = max_chars
        self._model = TextEmbedding(model_name=model_id)
        self._dim = len(list(self._model.embed(['dimension probe']))[0])

    @property
    def dim(self):
        return self._dim

    def encode(self, texts):
        clipped = [(t or '')[: self.max_chars] for t in texts]
        vecs = []
        for v in self._model.embed(clipped):
            v = list(map(float, v))
            norm = math.sqrt(sum(x * x for x in v))
            vecs.append([x / norm for x in v] if norm > 0 else v)
        return vecs


class SentenceTransformerDenseEncoder(DenseEncoder):
    """Sentence embeddings using sentence-transformers (torch)."""

    name = 'sentence-transformers'

    def __init__(self, model_id='BAAI/bge-small-en-v1.5', max_chars=8000):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self.max_chars = max_chars
        self._model = SentenceTransformer(model_id)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self):
        return self._dim

    def encode(self, texts):
        clipped = [(t or '')[: self.max_chars] for t in texts]
        vecs = self._model.encode(clipped, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]


class StaticVectorDenseEncoder(DenseEncoder):
    """IDF-weighted mean pooling of pretrained static GloVe word vectors.
    - Read the word/vector .npz produced by scripts/qdrant_fetch_dense_model.py.
    - Reuse ChemicalTokenizer to match BM25 lexical processing.
    - Weight by BM25 IDF x (1 + log tf); use uniform weights without IDF state and log that choice.
    - L2-normalize output for cosine distance.
    - Track out-of-vocabulary token ratios for quality diagnostics.
    """

    name = 'static-vectors'

    def __init__(self, npz_path, idf_state_path=None, max_chars=None, pooling='idf-mean'):
        import numpy as np
        self._np = np
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f'{npz_path} is missing. Run scripts/qdrant_fetch_dense_model.py first.')
        data = np.load(npz_path, allow_pickle=True)
        words = [str(w) for w in data['words']]
        self._matrix = np.asarray(data['vectors'], dtype=np.float32)
        self._index = {w: i for i, w in enumerate(words)}
        self._dim = int(self._matrix.shape[1])
        self.npz_path = npz_path
        self.pooling = pooling
        self.max_chars = max_chars

        base = os.path.basename(npz_path)
        self.model_id = f"{base[:-4] if base.endswith('.npz') else base}+{pooling}"

        self._idf = {}
        self._idf_source = None
        if idf_state_path and os.path.exists(idf_state_path):
            with open(idf_state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self._idf = {k: float(v) for k, v in state.get('idf', {}).items()}
            self._idf_source = idf_state_path

        from rag_engine.chemical_tokenizer import ChemicalTokenizer
        self._tokenizer = ChemicalTokenizer()

        self.oov_tokens = 0
        self.total_tokens = 0
        self.empty_vectors = 0

    @property
    def dim(self):
        return self._dim

    def signature_parts(self):
        h = hashlib.sha256()
        with open(self.npz_path, 'rb') as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
        # Different IDF weights produce different vectors even with the same vector file; include them in the signature.
        # Field-specific BM25 states require different claims / title_claims signatures.
        idf_sig = 'none'
        if self._idf:
            ih = hashlib.sha256()
            for k in sorted(self._idf):
                ih.update(f'{k}:{self._idf[k]:.6f}'.encode('utf-8'))
            idf_sig = ih.hexdigest()[:16]
        return [h.hexdigest()[:16], self.pooling, f'idf:{idf_sig}',
                f'src:{os.path.basename(self._idf_source or "")}']

    def encode(self, texts, token_lists=None):
        """Use supplied token_lists without retokenizing, avoiding duplicate tokenization during indexing."""
        np = self._np
        out = []
        for i, text in enumerate(texts):
            if token_lists is not None:
                tokens = token_lists[i]
            else:
                if self.max_chars:
                    text = (text or '')[: self.max_chars]
                tokens = self._tokenizer.tokenize(text or '')
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.total_tokens += len(tokens)

            rows, weights = [], []
            for term, tf in counts.items():
                idx = self._index.get(term)
                if idx is None:
                    self.oov_tokens += tf
                    continue
                w = (1.0 + math.log(tf))
                if self._idf:
                    w *= self._idf.get(term, 0.0)
                if w > 0:
                    rows.append(idx)
                    weights.append(w)
            if not rows:
                self.empty_vectors += 1
                out.append([0.0] * self._dim)
                continue
            vec = (self._matrix[rows] * np.asarray(weights, dtype=np.float32)[:, None]).sum(axis=0)
            norm = float(np.linalg.norm(vec))
            out.append((vec / norm).tolist() if norm > 0 else [0.0] * self._dim)
        return out

    def describe(self):
        d = super().describe()
        d.update({
            'kind': 'pretrained static word vectors (GloVe) + IDF-weighted mean pooling',
            'is_transformer_sentence_encoder': False,
            'vocab_size': len(self._index),
            'idf_state': self._idf_source,
            'oov_token_rate': round(self.oov_tokens / self.total_tokens, 4) if self.total_tokens else None,
            'empty_vectors': self.empty_vectors,
        })
        return d
