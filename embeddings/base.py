"""Shared embedding-encoder interfaces."""
import abc
import hashlib


def text_hash(text):
    """First 16 hexadecimal characters of the embedding input's SHA-256; used to detect reindexing needs."""
    return hashlib.sha256((text or '').encode('utf-8')).hexdigest()[:16]


class DenseEncoder(abc.ABC):
    """Fixed-dimensional dense vector encoder."""

    name = 'abstract'
    model_id = 'abstract'

    @property
    @abc.abstractmethod
    def dim(self):
        ...

    @abc.abstractmethod
    def encode(self, texts):
        """list[str] -> list[list[float]], with L2-normalized vectors."""

    def encode_one(self, text):
        return self.encode([text])[0]

    def describe(self):
        return {'backend': self.name, 'model': self.model_id, 'dim': self.dim}

    def signature(self):
        """Fingerprint for verifying that indexing and search use the same encoder. Combines backend/model/dimension with backend-specific state, such as vector-file hashes. Compare with payload embedding_signature to prevent searching with an incompatible encoder."""
        cached = getattr(self, '_signature_cache', None)
        if cached:
            return cached
        import hashlib as _h
        parts = [self.name, str(self.model_id), str(self.dim)] + list(self.signature_parts())
        sig = _h.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:16]
        self._signature_cache = sig       # Cache expensive calculations such as file hashes.
        return sig

    def signature_parts(self):
        return []


class SparseEncoder(abc.ABC):
    """Sparse lexical vector encoder."""

    name = 'abstract'

    @abc.abstractmethod
    def encode_documents(self, texts):
        """list[str] -> list[(indices, values)]"""

    @abc.abstractmethod
    def encode_query(self, text):
        """str -> (indices, values)"""

    def describe(self):
        return {'backend': self.name}
