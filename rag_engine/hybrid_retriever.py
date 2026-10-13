import math
from collections import Counter
from .chemical_tokenizer import ChemicalTokenizer

class HybridChemicalRetriever:
    """Lexical similarity between target and prior-art patent claims.

    Uses Jaccard/overlap set operations on chemical-term tokens, not learned embeddings such as BERT. This is a baseline text component between B1 (BM25) and B2 (embeddings).
    """

    def __init__(self):
        self.tokenizer = ChemicalTokenizer()

    def compute_text_similarity(self, query_text, doc_text):
        # Average Jaccard (intersection/union) and length-normalized overlap
        # (intersection/sqrt(length product), similar to cosine) at 0.5:0.5.
        # Combining both measures reduces the length bias of either alone.
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
