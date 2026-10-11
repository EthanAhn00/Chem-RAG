import math
from collections import Counter
from .chemical_tokenizer import ChemicalTokenizer

class HybridChemicalRetriever:
    """타깃 특허 청구항과 선행기술 청구항 간의 텍스트 유사도를 계산하는 검색기

    학습된 임베딩 모델(BERT 등)이 아닌, 화학 용어 토큰 단위의 Jaccard/overlap
    집합 연산으로 점수를 매기는 어휘 기반(lexical) 유사도. B1(BM25)과 B2(임베딩)
    사이에 위치하는 baseline 성격의 텍스트 축 구현
    """

    def __init__(self):
        self.tokenizer = ChemicalTokenizer()

    def compute_text_similarity(self, query_text, doc_text):
        # Jaccard 유사도(|교집합|/|합집합|)와 길이 정규화 overlap 유사도
        # (|교집합|/sqrt(길이곱), 코사인 유사도와 유사한 성격)를 0.5:0.5로 평균
        # 두 지표를 함께 쓰는 이유: 어느 한쪽만 쓰면 텍스트 길이 편향이 생기기 쉬움
        q_tokens = self.tokenizer.tokenize(query_text)
        d_tokens = self.tokenizer.tokenize(doc_text)

        if not q_tokens or not d_tokens:
            return 0.0

        q_counts = Counter(q_tokens)
        d_counts = Counter(d_tokens)

        intersection = sum((q_counts & d_counts).values())
        union = sum((q_counts | d_counts).values())

        jaccard = intersection / float(union) if union > 0 else 0.0
        overlap_ratio = intersection / math.sqrt(len(q_tokens) * len(d_tokens))

        return round(0.5 * jaccard + 0.5 * overlap_ratio, 4)
