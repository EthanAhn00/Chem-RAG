"""특허 컬렉션 검색기 - BM25 / dense / hybrid.

설계 원칙
  1) 시간·패밀리·자기제외 필터가 빠진 검색이 실수로 실행되지 않도록, 모든 검색
     메서드는 prior_art_filter를 필수 인자로 받는다(무필터는 allow_unfiltered=True 명시).
  2) 입력 텍스트 필드(claims / title+claims)를 인자로 받아 그 필드 전용 벡터를 쓴다.
     알고리즘 효과와 입력 필드 효과를 분리해 평가하기 위함이다.
  3) 하이브리드는 최상위 query_filter만으로는 prefetch가 필터링되지 않으므로
     filters.attach_filter()로 모든 prefetch 분기에 같은 필터를 주입한다.
"""
import time

from qdrant_client import models

from .filters import attach_filter
from .hybrid_fusion import fusion_query
from .schemas import TEXT_FIELDS

SEARCH_MODES = ('bm25', 'dense', 'hybrid')

# 검색 결과에서 기본으로 가져올 payload 필드.
# claims_text 같은 큰 필드까지 전부 끌어오면 대량 검색이 매우 느려지므로
# 필요한 필드만 선택적으로 받는다(필요하면 payload_fields로 확장).
DEFAULT_PAYLOAD_FIELDS = ['patent_id', 'family_id', 'pub_date', 'app_date', 'role', 'split',
                          'has_structure']


class PatentRetriever:
    def __init__(self, client, settings, dense_encoders=None, sparse_encoders=None,
                 collection=None, payload_fields=None, default_field='title_claims'):
        """dense_encoders / sparse_encoders 는 {field: encoder} 딕셔너리.

        하위 호환을 위해 인코더 객체 하나만 주면 default_field 에만 등록한다.
        """
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

    # ------------------------------------------------------------- 내부 유틸
    def _field(self, field):
        field = field or self.default_field
        if field not in TEXT_FIELDS:
            raise ValueError(f'알 수 없는 텍스트 필드: {field} (가능: {sorted(TEXT_FIELDS)})')
        return field

    def _require_filter(self, prior_art_filter, allow_unfiltered):
        if prior_art_filter is None and not allow_unfiltered:
            raise ValueError(
                'prior_art_filter가 없습니다. TARGET->PRIOR 검색은 시간/패밀리/자기제외 '
                '필터가 반드시 필요합니다. 의도적으로 무필터 검색을 하려면 '
                'allow_unfiltered=True를 명시하세요.')
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
            raise RuntimeError(f'{field} 필드용 dense 인코더가 설정되지 않았습니다')
        return enc.encode_one(text)

    def _sparse_query(self, text, field):
        enc = self.sparse_encoders.get(field)
        if enc is None:
            raise RuntimeError(f'{field} 필드용 sparse(BM25) 인코더가 설정되지 않았습니다')
        indices, values = enc.encode_query(text)
        return models.SparseVector(indices=indices, values=values)

    def encoder_signature(self, field=None):
        field = self._field(field)
        enc = self.dense_encoders.get(field)
        return enc.signature() if enc else None

    # --------------------------------------------------------------- 검색들
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
        """같은 입력 필드의 dense + BM25 를 Qdrant Query API의 RRF/DBSF로 융합."""
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
        raise ValueError(f'알 수 없는 검색 모드: {mode} (가능: {SEARCH_MODES})')

    def get_patent_payload(self, patent_id, fields=True):
        """특허 payload 조회. 기본은 전체 payload(fields=True)."""
        res = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key='patent_id', match=models.MatchValue(value=patent_id))]),
            limit=1, with_payload=fields, with_vectors=False)
        points = res[0]
        return points[0].payload if points else None
