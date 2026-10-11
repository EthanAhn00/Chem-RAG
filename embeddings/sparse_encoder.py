"""BM25 sparse 인코더.

Qdrant의 IDF modifier에 의존하지 않고 BM25를 클라이언트에서 직접 계산한다.
이유: (1) 로컬/서버 모드에서 통계가 달라지지 않게 하고, (2) 점수 산출식을
코드에 명시해 재현 가능하게 하기 위해서다.

분해 방식(내적이 곧 BM25 점수가 되도록):
    문서 벡터 값 = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    질의 벡터 값 = idf(term)   (Lucene식 smoothed idf)
    score = sum_over_matched_terms(doc_value * query_value) = BM25

토크나이저는 프로젝트 기존 rag_engine.ChemicalTokenizer를 그대로 쓴다
(기존 Jaccard baseline과 어휘 처리 기준을 맞추기 위함).
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
        """코퍼스 전체로 문서빈도/평균 길이를 학습한다.

        token_lists를 주면 재토큰화하지 않는다(대량 색인 시 토큰화 중복 제거용).
        """
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
        # Lucene식 smoothed IDF - 항상 양수
        self.idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.fitted = True
        return self

    # ------------------------------------------------------------- encoding
    def encode_documents(self, texts, token_lists=None):
        if not self.fitted:
            raise RuntimeError('BM25SparseEncoder.fit()을 먼저 호출해야 합니다')
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
            raise RuntimeError('BM25SparseEncoder.fit()을 먼저 호출해야 합니다')
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

    # ------------------------------------------------------------ 저장/복원
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
