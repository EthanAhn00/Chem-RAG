"""Patent retrieval with BM25/dense/hybrid.
1) Require prior_art_filter to avoid accidentally omitting time/family/self-exclusion; bypass only with explicit allow_unfiltered=True.
2) Select field-specific vectors using the claims / title+claims argument, separating input-field effects from algorithm effects.
3) Inject identical filters into every prefetch branch using filters.attach_filter(); the top-level query_filter is insufficient.
"""
import time

from qdrant_client import models

from .filters import attach_filter
from .hybrid_fusion import fusion_query
from .schemas import TEXT_FIELDS

SEARCH_MODES = ('bm25', 'dense', 'hybrid')

# Default payload fields returned with search hits.
# Fetching large fields such as claims_text slows bulk retrieval;
# request only needed fields, extending payload_fields when required.
DEFAULT_PAYLOAD_FIELDS = ['patent_id', 'family_id', 'pub_date', 'app_date', 'role', 'split',
                          'has_structure']


class PatentRetriever:
    def __init__(self, client, settings, dense_encoders=None, sparse_encoders=None,
                 collection=None, payload_fields=None, default_field='title_claims'):
        """dense_encoders / sparse_encoders are {field: encoder} dictionaries. For backward compatibility, a single encoder is registered only under default_field."""
        self.client = client
        self.settings = settings
        self.collection = collection or settings.patents_collection
        self.payload_fields = list(payload_fields) if payload_fields else list(DEFAULT_PAYLOAD_FIELDS)
        self.default_field = default_field
        self.dense_encoders = self._as_map(dense_encoders, default_field)
        self.sparse_encoders = self._as_map(sparse_encoders, default_field)

    @staticmethod
    def _as_map(enc, default_field):
        if enc is None:
            return {}
        if isinstance(enc, dict):
            return dict(enc)
        return {default_field: enc}

    # ------------------------------------------------------------- Internal utilities
    def _field(self, field):
        field = field or self.default_field
        if field not in TEXT_FIELDS:
            raise ValueError(f'Unknown text field: {field} (available: {sorted(TEXT_FIELDS)})')
        return field

    def _require_filter(self, prior_art_filter, allow_unfiltered):
        if prior_art_filter is None and not allow_unfiltered:
            raise ValueError(
                'Missing prior_art_filter. TARGET->PRIOR retrieval requires time/family/self-exclusion '
                'filters. To intentionally perform unfiltered retrieval, '
                'explicitly set allow_unfiltered=True.')
        return prior_art_filter.qdrant_filter if prior_art_filter is not None else None

    @staticmethod
    def _to_hits(points):
        out = []
        for rank, p in enumerate(points):
            payload = p.payload or {}
            out.append({
                'patent_id': payload.get('patent_id'),
                'score': float(p.score) if p.score is not None else 0.0,
                'rank': rank,
                'point_id': str(p.id),
                'payload': payload,
            })
        return out

    def _dense_query(self, text, field):
        enc = self.dense_encoders.get(field)
        if enc is None:
            raise RuntimeError(f'{field} has no configured dense encoder')
        return enc.encode_one(text)

    def _sparse_query(self, text, field):
        enc = self.sparse_encoders.get(field)
        if enc is None:
            raise RuntimeError(f'{field} has no configured sparse (BM25) encoder')
        indices, values = enc.encode_query(text)
        return models.SparseVector(indices=indices, values=values)

    def encoder_signature(self, field=None):
        field = self._field(field)
        enc = self.dense_encoders.get(field)
        return enc.signature() if enc else None

    # --------------------------------------------------------------- Retrieval
    def search_bm25(self, text, prior_art_filter, limit=None, field=None, allow_unfiltered=False):
        field = self._field(field)
        qfilter = self._require_filter(prior_art_filter, allow_unfiltered)
        sparse = self._sparse_query(text, field)
        if not sparse.indices:
            return [], 0.0
        t0 = time.perf_counter()
        res = self.client.query_points(
            self.collection, query=sparse, using=TEXT_FIELDS[field]['sparse'],
            query_filter=qfilter, limit=limit or self.settings.search_limit,
            with_payload=self.payload_fields)
        return self._to_hits(res.points), (time.perf_counter() - t0) * 1000.0

    def search_dense(self, text, prior_art_filter, limit=None, field=None, allow_unfiltered=False):
        field = self._field(field)
        qfilter = self._require_filter(prior_art_filter, allow_unfiltered)
        vec = self._dense_query(text, field)
        t0 = time.perf_counter()
        res = self.client.query_points(
            self.collection, query=vec, using=TEXT_FIELDS[field]['dense'],
            query_filter=qfilter, limit=limit or self.settings.search_limit,
            with_payload=self.payload_fields)
        return self._to_hits(res.points), (time.perf_counter() - t0) * 1000.0

    def search_hybrid(self, text, prior_art_filter, limit=None, prefetch_limit=None,
                      fusion=None, field=None, allow_unfiltered=False):
        """Fuse dense + BM25 for the same input field via Qdrant Query API RRF/DBSF."""
        field = self._field(field)
        qfilter = self._require_filter(prior_art_filter, allow_unfiltered)
        sparse = self._sparse_query(text, field)
        vec = self._dense_query(text, field)
        plimit = prefetch_limit or self.settings.prefetch_limit

        prefetches = [models.Prefetch(query=vec, using=TEXT_FIELDS[field]['dense'], limit=plimit)]
        if sparse.indices:
            prefetches.append(models.Prefetch(query=sparse, using=TEXT_FIELDS[field]['sparse'],
                                              limit=plimit))
        prefetches = attach_filter(prefetches, prior_art_filter)

        t0 = time.perf_counter()
        res = self.client.query_points(
            self.collection, prefetch=prefetches,
            query=fusion_query(fusion or self.settings.fusion),
            query_filter=qfilter, limit=limit or self.settings.search_limit,
            with_payload=self.payload_fields)
        return self._to_hits(res.points), (time.perf_counter() - t0) * 1000.0

    def search(self, mode, text, prior_art_filter, **kwargs):
        if mode == 'bm25':
            return self.search_bm25(text, prior_art_filter, **kwargs)
        if mode == 'dense':
            return self.search_dense(text, prior_art_filter, **kwargs)
        if mode == 'hybrid':
            return self.search_hybrid(text, prior_art_filter, **kwargs)
        raise ValueError(f'Unknown retrieval mode: {mode} (available: {SEARCH_MODES})')

    def get_patent_payload(self, patent_id, fields=True):
        """Fetch patent payloads; defaults to all fields (fields=True)."""
        res = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key='patent_id', match=models.MatchValue(value=patent_id))]),
            limit=1, with_payload=fields, with_vectors=False)
        points = res[0]
        return points[0].payload if points else None
