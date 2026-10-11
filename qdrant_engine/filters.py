"""TARGET->PRIOR 검색용 공통 필터 빌더.

이 프로젝트의 선행기술 검색은 항상 아래 3개 조건을 만족해야 한다:

    candidate.pub_date  <= query.app_date      (미래 문헌 배제)
    candidate.family_id != query.family_id     (동일 패밀리 자기인용 배제)
    candidate.patent_id != query.patent_id     (자기 자신 배제)

필터 없는 검색 경로가 실수로 쓰이는 것을 막기 위해, 검색 함수들은 이 모듈이 만든
PriorArtFilter를 반드시 인자로 받도록 설계되어 있다(무필터를 원하면 명시적으로
allow_unfiltered=True를 넘겨야 한다).

*** 중요(Qdrant 동작 주의) ***
prefetch를 쓰는 하이브리드 질의에서는 최상위 query_filter만 지정하면 prefetch 단계에
필터가 적용되지 않아 걸러져야 할 후보가 결과에 남는다(로컬 모드에서 실측 확인).
따라서 이 모듈의 attach_filter()로 모든 prefetch 분기에 동일 필터를 주입한다.
"""
from dataclasses import dataclass
from qdrant_client import models


def _to_int_date(value):
    """'20200131' -> 20200131. 비어있거나 숫자가 아니면 None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or not s.isdigit():
        return None
    return int(s)


@dataclass(frozen=True)
class PriorArtFilter:
    """검색에 쓰는 필터와 그 근거를 함께 담는다(디버깅/검증용)."""
    query_patent_id: str
    query_app_date: str
    query_family_id: str
    qdrant_filter: models.Filter

    def describe(self):
        return {
            'query_patent_id': self.query_patent_id,
            'query_app_date': self.query_app_date,
            'query_family_id': self.query_family_id,
            'rules': [
                'candidate.patent_id != query.patent_id',
                'candidate.pub_date <= query.app_date' if _to_int_date(self.query_app_date) else 'time filter SKIPPED (query app_date 없음)',
                'candidate.family_id != query.family_id' if self.query_family_id else 'family filter SKIPPED (query family_id 없음)',
            ],
        }


def build_prior_art_filter(query_patent_id, query_app_date, query_family_id, extra_must=None):
    """TARGET->PRIOR 검색 표준 필터.

    baseline(scripts/evaluate_full_corpus_retrieval.py)과 의미를 맞춘다:
      - app_date가 없으면 시간 필터를 걸지 않는다
      - family_id가 비어 있으면 패밀리 필터를 걸지 않는다
      - 후보의 pub_date가 비어 있으면 시간 조건으로 배제하지 않는다
    """
    must = list(extra_must or [])
    must_not = [models.FieldCondition(key='patent_id', match=models.MatchValue(value=query_patent_id))]

    app_int = _to_int_date(query_app_date)
    if app_int is not None:
        # pub_date가 없는 후보까지 배제되지 않도록 (범위 OR 비어있음) 형태로 구성
        must.append(models.Filter(should=[
            models.FieldCondition(key='pub_date_int', range=models.Range(lte=app_int)),
            models.IsEmptyCondition(is_empty=models.PayloadField(key='pub_date_int')),
        ]))

    if query_family_id and str(query_family_id).strip():
        must_not.append(models.FieldCondition(
            key='family_id', match=models.MatchValue(value=str(query_family_id))))

    return PriorArtFilter(
        query_patent_id=query_patent_id,
        query_app_date=str(query_app_date or ''),
        query_family_id=str(query_family_id or ''),
        qdrant_filter=models.Filter(must=must or None, must_not=must_not),
    )


def build_filter_from_payload(payload):
    """특허 payload(dict)에서 바로 필터를 만든다."""
    return build_prior_art_filter(
        payload.get('patent_id'), payload.get('app_date'), payload.get('family_id'))


def attach_filter(prefetches, prior_art_filter):
    """모든 prefetch 분기에 동일 필터를 주입한다.

    최상위 query_filter만으로는 prefetch가 필터링되지 않으므로 이 함수를 반드시
    거쳐야 한다. (retriever가 이 함수를 통해서만 prefetch를 만든다)
    """
    if prior_art_filter is None:
        return prefetches
    out = []
    for p in prefetches:
        out.append(models.Prefetch(
            query=p.query, using=p.using, limit=p.limit,
            filter=prior_art_filter.qdrant_filter,
            prefetch=attach_filter(p.prefetch, prior_art_filter) if p.prefetch else None,
        ))
    return out
