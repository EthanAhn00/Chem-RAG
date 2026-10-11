"""Dense 인코더 백엔드.

네 가지를 제공한다.

  static-vectors        GitHub 릴리스로 배포되는 사전학습 GloVe 단어 벡터를 IDF 가중 평균
                        풀링해 문장 벡터를 만든다. numpy만 있으면 동작하고 어떤 파이썬
                        버전에서도 쓸 수 있다. **실제 사전학습 분포 의미 벡터**지만
                        transformer 문장 인코더는 아니다(보고 시 구분 필요).

  hashing               - 외부 모델/네트워크 없이 동작하는 결정론적 해싱 인코더.
                          의미(semantic) 임베딩이 아니라 어휘 신호를 고정 차원에
                          투영한 것이므로 BM25와 비슷한 성격이다. 어떤 환경에서도
                          파이프라인이 돌아가게 하는 기본값 겸 폴백.
  fastembed             - ONNX 기반 실제 문장 임베딩 모델(BAAI/bge-small-en-v1.5 등).
  sentence-transformers - torch 기반 문장 임베딩 모델.

주의: 결과 파일에는 항상 어떤 백엔드로 만든 수치인지 기록한다. hashing 백엔드로
낸 dense 점수를 '의미 기반 임베딩 성능'이라고 해석하면 안 된다.
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
    """PYTHONHASHSEED에 영향받지 않는 결정론적 해시."""
    return int.from_bytes(hashlib.blake2b(token.encode('utf-8'), digest_size=8, key=salt).digest(), 'big')


class HashingDenseEncoder(DenseEncoder):
    """부호 해싱(signed hashing) + log-tf 가중 + L2 정규화.

    - 같은 텍스트는 어떤 머신/프로세스에서든 같은 벡터를 만든다.
    - 모델 다운로드가 필요 없어 오프라인/제한된 파이썬 버전에서도 동작한다.
    """

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
    """fastembed(ONNX) 기반 실제 문장 임베딩."""

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
    """sentence-transformers(torch) 기반 문장 임베딩."""

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
    """사전학습 정적 단어 벡터(GloVe)의 IDF 가중 평균 풀링.

    - scripts/qdrant_fetch_dense_model.py 가 만든 .npz(단어 + 벡터)를 읽는다.
    - 토크나이저는 BM25와 동일한 ChemicalTokenizer를 써서 어휘 처리 기준을 맞춘다.
    - 가중치: IDF(BM25 학습 상태) x (1 + log tf). IDF 상태가 없으면 균등 가중(로그에 명시).
    - 출력은 L2 정규화되어 cosine 거리와 맞는다.
    - OOV(사전에 없는 토큰) 비율을 추적해 품질 판단 근거로 남긴다.
    """

    name = 'static-vectors'

    def __init__(self, npz_path, idf_state_path=None, max_chars=None, pooling='idf-mean'):
        import numpy as np
        self._np = np
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f'{npz_path} 가 없습니다. 먼저 scripts/qdrant_fetch_dense_model.py 를 실행하세요.')
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
        # IDF 가중치가 다르면 같은 벡터 파일이라도 결과 벡터가 달라지므로 서명에 포함한다
        # (필드별 BM25 상태를 쓰기 때문에 claims / title_claims 서명이 달라야 한다)
        idf_sig = 'none'
        if self._idf:
            ih = hashlib.sha256()
            for k in sorted(self._idf):
                ih.update(f'{k}:{self._idf[k]:.6f}'.encode('utf-8'))
            idf_sig = ih.hexdigest()[:16]
        return [h.hexdigest()[:16], self.pooling, f'idf:{idf_sig}',
                f'src:{os.path.basename(self._idf_source or "")}']

    def encode(self, texts, token_lists=None):
        """token_lists를 주면 재토큰화하지 않는다(색인 시 토큰화 중복 제거용)."""
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
