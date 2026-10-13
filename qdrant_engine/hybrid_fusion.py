"""Hybrid fusion and structure reranking.
1) Fuse text with Qdrant Query API RRF/DBSF, selected by configuration.
2) Compute RDKit exact Tanimoto for top returned candidates and combine with text scores.
Never interpret sparse fingerprint dot products as Tanimoto: recompute actual similarity with RDKit. Do not assign zero structure scores to patents without structures; renormalize available weights as in MultimodalChemicalRanker.
"""
from qdrant_client import models

FUSION_MAP = {
    'rrf': models.Fusion.RRF,
    'dbsf': models.Fusion.DBSF,
}


def fusion_query(name):
    key = (name or 'rrf').strip().lower()
    if key not in FUSION_MAP:
        raise ValueError(f'Unsupported fusion: {name} (available: {sorted(FUSION_MAP)})')
    return models.FusionQuery(fusion=FUSION_MAP[key])


def minmax_normalize(scores):
    """Normalize scores to [0,1]. RRF has no absolute scale and cannot be added directly to Tanimoto (0~1). Normalize relatively within the reranking window; ranks outside it remain unaffected."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def blend_text_and_structure(text_score, tanimoto_score, text_weight, structure_weight):
    """Renormalized fusion matching rag_engine/multimodal_ranker.py. With tanimoto_score=None (missing structure), use text alone rather than forcing a zero structure score."""
    if tanimoto_score is None:
        return float(text_score), ['text']
    total = text_weight + structure_weight
    if total <= 0:
        return float(text_score), ['text']
    blended = (text_weight * text_score + structure_weight * tanimoto_score) / total
    return float(blended), ['text', 'structure']


def rerank_with_structure(hits, tanimoto_fn, text_weight, structure_weight, top_k):
    """Rerank only the top_k candidates using structure.
    hits: descending-score [{'patent_id':..., 'score':..., ...}].
    tanimoto_fn: patent_id -> float | None, with RDKit exact Tanimoto or None for missing structure.
    Return the full reranked list, retaining the original order outside the window.
    High-scoring candidates without structures retain their text score; structure does not necessarily overturn the top text result.
    Scores inside the window are normalized to [0,1] and fused; outside scores retain the original retrieval scale. These scales are not comparable: final order is the reranked window first, then the original remainder.
    """
    window = hits[:top_k]
    rest = hits[top_k:]
    if not window:
        return list(hits), {'reranked': 0, 'with_structure': 0}

    norm = minmax_normalize([h['score'] for h in window])
    with_structure = 0
    rescored = []
    for h, text_norm in zip(window, norm):
        tan = tanimoto_fn(h['patent_id'])
        if tan is not None:
            with_structure += 1
        blended, used = blend_text_and_structure(text_norm, tan, text_weight, structure_weight)
        item = dict(h)
        item['text_score_normalized'] = round(text_norm, 6)
        item['tanimoto_score'] = tan
        item['components_used'] = used
        item['score'] = blended
        rescored.append(item)

    rescored.sort(key=lambda x: x['score'], reverse=True)
    return rescored + rest, {'reranked': len(window), 'with_structure': with_structure}
