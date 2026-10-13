"""Shared TARGET->PRIOR filter builder.
Prior-art retrieval requires:
candidate.pub_date <= query.app_date (exclude future documents)
candidate.family_id != query.family_id (exclude same-family self-citations)
candidate.patent_id != query.patent_id (exclude self).
Retrievers require PriorArtFilter to prevent accidental unfiltered search; explicitly set allow_unfiltered=True to bypass it.
Qdrant caveat: a top-level query_filter does not filter hybrid prefetch branches, as observed in local mode. attach_filter() therefore injects the same filter into every prefetch branch.
"""
from dataclasses import dataclass
from qdrant_client import models


def _to_int_date(value):
    """'20200131' -> 20200131. Return None for empty or nonnumeric input."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or not s.isdigit():
        return None
    return int(s)


@dataclass(frozen=True)
class PriorArtFilter:
    """Keep the search filter and its rationale together for debugging/verification."""
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
                'candidate.pub_date <= query.app_date' if _to_int_date(self.query_app_date) else 'time filter SKIPPED (query app_date missing)',
                'candidate.family_id != query.family_id' if self.query_family_id else 'family filter SKIPPED (query family_id missing)',
            ],
        }


def build_prior_art_filter(query_patent_id, query_app_date, query_family_id, extra_must=None):
    """Standard TARGET->PRIOR filter matching scripts/evaluate_full_corpus_retrieval.py: skip time filtering when app_date is missing, skip family filtering when family_id is empty, and retain candidates with missing pub_date under the time condition."""
    must = list(extra_must or [])
    must_not = [models.FieldCondition(key='patent_id', match=models.MatchValue(value=query_patent_id))]

    app_int = _to_int_date(query_app_date)
    if app_int is not None:
        # Use range OR missing to retain candidates without pub_date.
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
    """Build a filter directly from a patent payload dictionary."""
    return build_prior_art_filter(
        payload.get('patent_id'), payload.get('app_date'), payload.get('family_id'))


def attach_filter(prefetches, prior_art_filter):
    """Inject the same filter into every prefetch branch. The top-level query_filter is insufficient; the retriever builds prefetch only through this function."""
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
