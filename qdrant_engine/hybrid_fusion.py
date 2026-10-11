"""하이브리드 융합 및 구조 재정렬(rerank) 유틸.

1) 텍스트 축 융합: Qdrant Query API의 RRF/DBSF를 사용(설정으로 교체 가능).
2) 구조 축 재정렬: Qdrant가 돌려준 상위 후보에 한해 RDKit exact Tanimoto를
   계산해 텍스트 점수와 결합한다.

중요한 정책 두 가지
  - Qdrant의 sparse 지문 내적 점수를 Tanimoto로 간주하지 않는다. 실제 Tanimoto는
    항상 RDKit으로 다시 계산한다.
  - 구조 정보가 없는 특허에 구조 점수 0을 강제하지 않는다. rag_engine의
    MultimodalChemicalRanker와 동일하게 '사용 가능한 신호의 가중치만 재정규화'한다.
"""
from qdrant_client import models

FUSION_MAP = {
    'rrf': models.Fusion.RRF,
    'dbsf': models.Fusion.DBSF,
}


def fusion_query(name):
    key = (name or 'rrf').strip().lower()
    if key not in FUSION_MAP:
        raise ValueError(f'지원하지 않는 fusion: {name} (가능: {sorted(FUSION_MAP)})')
    return models.FusionQuery(fusion=FUSION_MAP[key])


def minmax_normalize(scores):
    """서로 다른 스케일의 점수를 [0,1]로 맞춘다.

    RRF 점수는 절대적 의미가 없어 Tanimoto(0~1)와 직접 더할 수 없으므로,
    재정렬 대상 윈도우 안에서 상대 정규화한다(윈도우 밖 순위에는 영향 없음).
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def blend_text_and_structure(text_score, tanimoto_score, text_weight, structure_weight):
    """rag_engine/multimodal_ranker.py와 동일한 재정규화 결합.

    tanimoto_score가 None이면(=구조 데이터 없음) 텍스트 점수만으로 결정한다.
    0점을 강제로 매기지 않는다.
    """
    if tanimoto_score is None:
        return float(text_score), ['text']
    total = text_weight + structure_weight
    if total <= 0:
        return float(text_score), ['text']
    blended = (text_weight * text_score + structure_weight * tanimoto_score) / total
    return float(blended), ['text', 'structure']


def rerank_with_structure(hits, tanimoto_fn, text_weight, structure_weight, top_k):
    """상위 top_k 후보만 구조 점수로 재정렬한다.

    hits: [{'patent_id':..., 'score':..., ...}] (점수 내림차순)
    tanimoto_fn: patent_id -> float | None (RDKit exact Tanimoto, 구조 없으면 None)
    반환: 재정렬된 전체 리스트(윈도우 밖은 원래 순서 유지)

    주의 1) 구조 점수가 텍스트 최상위를 무조건 뒤집지는 않는다. 구조가 없는 후보에
            0점을 강제하지 않는 정책 때문에, 텍스트 점수가 높고 구조가 없는 후보는
            점수가 그대로 유지된다(rag_engine의 late fusion 정책과 동일).
    주의 2) 반환 리스트에서 윈도우(top_k) 안의 score는 [0,1] 정규화 후 결합한 값이고,
            윈도우 밖은 원래 검색 점수 그대로다. 두 구간의 score는 서로 비교 가능한
            척도가 아니며, 최종 순서는 '윈도우 우선 + 나머지 원순서'로 결정된다.
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
