"""BM25 sparse encoder.
Compute BM25 on the client rather than using Qdrant's IDF modifier, keeping local/server statistics consistent and the scoring formula explicit and reproducible.
Factorization for BM25 by dot product:
document value = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
query value = idf(term), using Lucene-style smoothed IDF
score = sum_over_matched_terms(doc_value * query_value) = BM25.
Reuse rag_engine.ChemicalTokenizer for lexical consistency with the Jaccard baseline.
"""
import json
import math
import os

from .base import SparseEncoder


def _default_tokenizer():
    from rag_engine.chemical_tokenizer import ChemicalTokenizer
    return ChemicalTokenizer()


class BM25SparseEncoder(SparseEncoder):
    name = 'bm25'

    def __init__(self, k1=1.5, b=0.75, tokenizer=None):
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or _default_tokenizer()
        self.vocab = {}          # term -> index
        self.idf = {}            # term -> idf
        self.avgdl = 0.0
        self.num_docs = 0
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, texts, token_lists=None):
        """Fit document frequencies and average length on the corpus. Supplied token_lists avoid duplicate tokenization during bulk indexing."""
        df = {}
        total_len = 0
        n = 0
        for i, text in enumerate(texts):
            tokens = token_lists[i] if token_lists is not None else self.tokenizer.tokenize(text or '')
            total_len += len(tokens)
            n += 1
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        self.num_docs = n
        self.avgdl = (total_len / n) if n else 0.0
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        # Lucene-style smoothed IDF is always positive.
        self.idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.fitted = True
        return self

    # ------------------------------------------------------------- encoding
    def encode_documents(self, texts, token_lists=None):
        if not self.fitted:
            raise RuntimeError('Call BM25SparseEncoder.fit() first')
        out = []
        for i, text in enumerate(texts):
            tokens = token_lists[i] if token_lists is not None else self.tokenizer.tokenize(text or '')
            dl = len(tokens)
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            indices, values = [], []
            denom_len = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
            for term, tf in counts.items():
                idx = self.vocab.get(term)
                if idx is None:
                    continue
                w = tf * (self.k1 + 1.0) / (tf + denom_len)
                if w > 0:
                    indices.append(int(idx))
                    values.append(float(w))
            out.append((indices, values))
        return out

    def encode_query(self, text):
        if not self.fitted:
            raise RuntimeError('Call BM25SparseEncoder.fit() first')
        tokens = self.tokenizer.tokenize(text or '')
        seen = {}
        for t in tokens:
            idx = self.vocab.get(t)
            if idx is None:
                continue
            seen[int(idx)] = float(self.idf.get(t, 0.0))
        if not seen:
            return [], []
        indices = sorted(seen)
        return indices, [seen[i] for i in indices]

    # ------------------------------------------------------------ Save/restore
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'k1': self.k1, 'b': self.b, 'avgdl': self.avgdl, 'num_docs': self.num_docs,
                'vocab': self.vocab, 'idf': self.idf,
            }, f)
        return path

    @classmethod
    def load(cls, path, tokenizer=None):
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        enc = cls(k1=state['k1'], b=state['b'], tokenizer=tokenizer)
        enc.vocab = {k: int(v) for k, v in state['vocab'].items()}
        enc.idf = {k: float(v) for k, v in state['idf'].items()}
        enc.avgdl = float(state['avgdl'])
        enc.num_docs = int(state['num_docs'])
        enc.fitted = True
        return enc

    def describe(self):
        return {'backend': self.name, 'k1': self.k1, 'b': self.b,
                'vocab_size': len(self.vocab), 'num_docs': self.num_docs,
                'avgdl': round(self.avgdl, 3)}
