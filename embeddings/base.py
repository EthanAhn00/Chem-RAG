"""임베딩 인코더 공통 인터페이스."""
import abc
import hashlib


def text_hash(text):
    """임베딩 입력 텍스트의 SHA-256(앞 16자리). 재색인 필요 여부 판단용."""
    return hashlib.sha256((text or '').encode('utf-8')).hexdigest()[:16]


class DenseEncoder(abc.ABC):
    """고정 차원 dense 벡터 인코더."""

    name = 'abstract'
    model_id = 'abstract'

    @property
    @abc.abstractmethod
    def dim(self):
        ...

    @abc.abstractmethod
    def encode(self, texts):
        """list[str] -> list[list[float]] (L2 정규화된 벡터)."""

    def encode_one(self, text):
        return self.encode([text])[0]

    def describe(self):
        return {'backend': self.name, 'model': self.model_id, 'dim': self.dim}

    def signature(self):
        """적재 시 인코더와 검색 시 인코더가 같은지 확인하기 위한 지문.

        백엔드/모델/차원 + 백엔드별 추가 상태(예: 벡터 파일 해시)를 합쳐 만든다.
        payload의 embedding_signature와 비교하면 다른 인코더로 검색하는 사고를 막을 수 있다.
        """
        cached = getattr(self, '_signature_cache', None)
        if cached:
            return cached
        import hashlib as _h
        parts = [self.name, str(self.model_id), str(self.dim)] + list(self.signature_parts())
        sig = _h.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:16]
        self._signature_cache = sig       # 파일 해시 등 비싼 계산은 한 번만
        return sig

    def signature_parts(self):
        return []


class SparseEncoder(abc.ABC):
    """sparse(어휘) 벡터 인코더."""

    name = 'abstract'

    @abc.abstractmethod
    def encode_documents(self, texts):
        """list[str] -> list[(indices, values)]"""

    @abc.abstractmethod
    def encode_query(self, text):
        """str -> (indices, values)"""

    def describe(self):
        return {'backend': self.name}
