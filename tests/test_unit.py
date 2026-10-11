"""단위 테스트 - Qdrant DB 없이 실행 가능.

    python3 tests/test_unit.py          (또는 python3 -m pytest tests/test_unit.py)
"""
import os
import sys
import json
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import models

from qdrant_engine.config import load_settings, ID_NAMESPACE
from qdrant_engine.schemas import (
    patent_point_id, structure_point_id, build_field_text, TEXT_FIELDS,
    patent_vectors_config, patent_sparse_config, TRUE, FALSE, UNKNOWN)
from qdrant_engine.filters import build_prior_art_filter, attach_filter, _to_int_date
from qdrant_engine.hybrid_fusion import (
    blend_text_and_structure, minmax_normalize, rerank_with_structure, fusion_query)
from qdrant_engine.structure_repository import build_fingerprint, max_tanimoto, morgan_bits
from embeddings.dense_encoder import HashingDenseEncoder
from embeddings.sparse_encoder import BM25SparseEncoder
from scripts.evaluation.evaluate_qdrant_retrieval import (
    evaluate_ranking, metric_key_for_cutoff, build_system_summary, build_effects,
    resolve_recompute_cutoff)
from qdrant_engine.snapshot import validate_manifest_structure


def test_point_ids_are_deterministic_uuid5():
    a, b = patent_point_id('EP3660125A1'), patent_point_id('EP3660125A1')
    assert a == b and uuid.UUID(a).version == 5
    assert a == str(uuid.uuid5(uuid.UUID(ID_NAMESPACE), 'patent:EP3660125A1'))
    assert patent_point_id('EP1') != patent_point_id('EP2')
    s = structure_point_id('EP1', 'MOL_1')
    assert s == structure_point_id('EP1', 'MOL_1') != structure_point_id('EP1', 'MOL_2')
    for bad in ('', None):
        try:
            patent_point_id(bad)
            raise AssertionError('빈 patent_id는 거부되어야 한다')
        except ValueError:
            pass


def test_text_fields_and_vector_config():
    assert build_field_text('T', 'C', 'claims') == ('C', False)
    assert build_field_text('T', 'C', 'title_claims') == ('T\nC', False)
    # claims가 없으면 버리지 않고 title만 쓰며 missing 표시
    assert build_field_text('T', '', 'title_claims') == ('T', True)
    assert build_field_text('T', '', 'claims') == ('', True)
    try:
        build_field_text('T', 'C', 'nope')
        raise AssertionError('알 수 없는 필드는 거부되어야 한다')
    except ValueError:
        pass
    assert set(patent_vectors_config(300)) == {'claims_dense', 'title_claims_dense'}
    assert set(patent_sparse_config()) == {'claims_bm25', 'title_claims_bm25'}


def test_filter_rules_and_policies():
    f = build_prior_art_filter('EP_Q', '20200101', 'FAM1')
    keys = [c.key for c in f.qdrant_filter.must_not if hasattr(c, 'key')]
    assert 'patent_id' in keys and 'family_id' in keys and f.qdrant_filter.must
    # app_date / family_id 가 없으면 해당 조건을 걸지 않는다
    f2 = build_prior_art_filter('EP_Q', '', '')
    assert not f2.qdrant_filter.must
    assert [c.key for c in f2.qdrant_filter.must_not if hasattr(c, 'key')] == ['patent_id']
    f3 = build_prior_art_filter('EP_Q', '20200101', '')
    assert f3.qdrant_filter.must and len(f3.qdrant_filter.must_not) == 1
    assert _to_int_date('20200131') == 20200131
    assert _to_int_date('') is None and _to_int_date(None) is None and _to_int_date('x') is None


def test_attach_filter_injects_into_every_prefetch_branch():
    """최상위 query_filter만으로는 prefetch가 필터링되지 않는다(실측 확인된 Qdrant 동작)."""
    f = build_prior_art_filter('EP_Q', '20200101', 'FAM1')
    prefetches = [models.Prefetch(query=[0.1, 0.2], using='title_claims_dense', limit=10),
                  models.Prefetch(query=models.SparseVector(indices=[1], values=[1.0]),
                                  using='title_claims_bm25', limit=10)]
    out = attach_filter(prefetches, f)
    assert len(out) == 2 and all(p.filter == f.qdrant_filter for p in out)


def test_fusion_query_options():
    assert fusion_query('rrf').fusion == models.Fusion.RRF
    assert fusion_query('dbsf').fusion == models.Fusion.DBSF
    try:
        fusion_query('nope')
        raise AssertionError('알 수 없는 fusion은 거부되어야 한다')
    except ValueError:
        pass


def test_encoders_deterministic():
    enc = HashingDenseEncoder(dim=128)
    assert enc.encode_one('polymer composition') == enc.encode_one('polymer composition')
    assert abs(sum(x * x for x in enc.encode_one('polymer')) - 1.0) < 1e-9
    assert enc.encode_one('') == [0.0] * 128

    b = BM25SparseEncoder().fit(['ethylene polymer composition', 'polymer resin additive',
                                 'catalyst ziegler natta'])
    d_idx, d_val = b.encode_documents(['ethylene polymer composition'])[0]
    q_idx, q_val = b.encode_query('ethylene composition')
    score = sum(dv * qv for di, dv in zip(d_idx, d_val) for qi, qv in zip(q_idx, q_val) if di == qi)
    assert score > 0
    assert b.encode_query('zzzznonexistent') == ([], [])
    # 토큰 리스트를 직접 넘겨도 동일 결과
    toks = [b.tokenizer.tokenize('ethylene polymer composition')]
    assert b.encode_documents(['ethylene polymer composition'], token_lists=toks)[0] == (d_idx, d_val)


def test_exact_tanimoto_semantics():
    f1, f2 = build_fingerprint('CCO'), build_fingerprint('CCCO')
    assert max_tanimoto([f1], [f2]) == 0.5556 and max_tanimoto([f1], [f1]) == 1.0
    assert max_tanimoto([f1], []) is None and max_tanimoto([], []) is None   # 없음 != 0
    assert build_fingerprint('not-a-smiles!!') is None and morgan_bits('CCO')


def test_structure_weight_renormalization_and_rerank():
    assert blend_text_and_structure(0.8, None, 0.5, 0.3) == (0.8, ['text'])
    score, used = blend_text_and_structure(0.8, 0.4, 0.5, 0.3)
    assert abs(score - (0.5 * 0.8 + 0.3 * 0.4) / 0.8) < 1e-9 and used == ['text', 'structure']
    assert minmax_normalize([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]

    hits = [{'patent_id': f'P{i}', 'score': 1.0 - i * 0.01} for i in range(10)]
    out, stats = rerank_with_structure(hits, lambda pid: None, 0.5, 0.3, top_k=5)
    assert [h['patent_id'] for h in out] == [h['patent_id'] for h in hits]
    assert stats == {'reranked': 5, 'with_structure': 0}
    out2, stats2 = rerank_with_structure(hits, lambda p: 1.0 if p == 'P4' else None, 0.5, 0.3, 5)
    assert [h['patent_id'] for h in out2[:5]] == ['P0', 'P1', 'P2', 'P4', 'P3']
    assert stats2['with_structure'] == 1
    assert [h['patent_id'] for h in out2[5:]] == ['P5', 'P6', 'P7', 'P8', 'P9']


def test_average_precision_definition():
    """AP = sum(정답 위치별 precision) / 전체 정답 수. 미회수 정답은 0 기여."""
    gold = {'P2', 'P5'}
    full = evaluate_ranking([f'P{i}' for i in range(1, 11)], gold)
    assert abs(full['ap'] - 0.45) < 1e-9
    partial = evaluate_ranking(['P2'], gold)               # (1/1)/2
    assert abs(partial['ap'] - 0.5) < 1e-9 and partial['num_gold_retrieved'] == 1
    assert partial['recall_at_10'] == 0.5
    nothing = evaluate_ranking(['ZZZ'], gold)
    assert nothing['ap'] == 0.0 and nothing['best_rank'] is None
    empty = evaluate_ranking([], gold)
    assert empty['ap'] == 0.0 and empty['retrieved'] == 0


def test_structure_location_metadata_consistency():
    """파생 산출물이 있으면 3-값 논리/근거/횟수 일관성을 검사한다."""
    from config.settings import DATA_DIR
    path = os.path.join(DATA_DIR, 'db_export', 'derived', 'structure_location_metadata.jsonl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} 없음 - scripts/qdrant/qdrant_build_structure_metadata.py 실행 필요')
    seen = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            key = (rec['patent_id'], rec['mol_id'])
            assert key not in seen, f'중복 {key}'
            seen.add(key)
            assert rec['in_claim'] in (TRUE, FALSE, UNKNOWN)
            assert rec['in_figure'] in (TRUE, FALSE, UNKNOWN)
            ev = rec.get('location_evidence') or {}
            occ, ccws = rec['claim_occurrence_count'], rec['claim_count_with_structure']
            if rec['in_claim'] == TRUE:
                assert ev.get('claim'), 'in_claim=true 는 근거 필요'
            if rec['in_claim'] == UNKNOWN:
                assert occ is None, 'unknown 은 0이 아니라 null'
            if rec['in_claim'] == FALSE:
                assert occ == 0
            if occ is not None and ccws is not None:
                assert ccws <= occ, 'claim_count_with_structure 는 occurrence 보다 클 수 없다'
            if rec['in_figure'] == TRUE:
                assert ev.get('figure') and (rec.get('figure_unit_refs') or any(
                    e.get('fig_id') or e.get('unit_ref') for e in ev['figure']))
    assert len(seen) == 6928, f'구조 메타데이터 레코드 수 {len(seen)}'


class _FakePoint:
    def __init__(self, payload):
        self.payload = payload


class _FakeScrollClient:
    """scroll()만 흉내내는 최소 클라이언트 - DB 없이 서명 검사 경로를 시험한다."""

    def __init__(self, payload):
        self._payload = payload

    def scroll(self, *args, **kwargs):
        return ([_FakePoint(self._payload)] if self._payload is not None else [], None)


class _FakeDense:
    def __init__(self, sig):
        self._sig = sig

    def signature(self):
        return self._sig

    def describe(self):
        return {'backend': 'fake', 'model': 'fake', 'dim': 0}


def test_search_cli_field_option_and_state_paths():
    """검색 CLI가 필드별 상태 파일을 쓰고 기본 필드가 title_claims 인지."""
    from scripts.qdrant import qdrant_search as qs

    assert qs.DEFAULT_FIELD == 'title_claims'
    for field in TEXT_FIELDS:
        assert os.path.basename(qs.bm25_state_path(field)) == f'bm25_state_{field}.json'
    # 레거시 단일 상태 파일(bm25_state.json)은 더 이상 참조하지 않는다
    assert 'bm25_state.json' not in {os.path.basename(qs.bm25_state_path(f)) for f in TEXT_FIELDS}

    ap = qs.build_arg_parser()
    assert ap.parse_args(['--patent', 'EP1']).field == 'title_claims'
    assert ap.parse_args(['--patent', 'EP1', '--field', 'claims']).field == 'claims'
    assert ap.parse_args(['--text', 'x', '--field', 'title_claims']).field == 'title_claims'
    try:
        ap.parse_args(['--patent', 'EP1', '--field', 'nope'])
        raise AssertionError('알 수 없는 --field 는 거부되어야 한다')
    except SystemExit:
        pass


def test_search_cli_missing_bm25_state_stops_search():
    from scripts.qdrant import qdrant_search as qs

    settings = load_settings()
    try:
        qs.load_field_encoders(settings, 'claims', state_path='/nonexistent/bm25_state_claims.json')
        raise AssertionError('BM25 상태 파일이 없으면 검색을 시작하면 안 된다')
    except qs.SearchSetupError as exc:
        assert 'BM25 상태 파일이 없습니다' in str(exc), exc
    try:
        qs.load_field_encoders(settings, 'nope')
        raise AssertionError('알 수 없는 필드는 거부되어야 한다')
    except qs.SearchSetupError:
        pass


def test_search_cli_encoder_signature_guard():
    """payload의 embedding_signature_by_field 와 다르면 검색 전에 중단한다."""
    from scripts.qdrant import qdrant_search as qs

    settings = load_settings()
    payload = {'embedding_signature_by_field': {'claims': 'AAA', 'title_claims': 'BBB'}}

    ok = qs.verify_encoder_signature(_FakeScrollClient(payload), settings, 'claims',
                                     _FakeDense('AAA'))
    assert ok == {'field': 'claims', 'indexed': 'AAA', 'search': 'AAA', 'match': True}
    assert qs.verify_encoder_signature(_FakeScrollClient(payload), settings, 'title_claims',
                                       _FakeDense('BBB'))['match']

    # 1) 서명이 다르면 중단
    try:
        qs.verify_encoder_signature(_FakeScrollClient(payload), settings, 'claims',
                                    _FakeDense('ZZZ'))
        raise AssertionError('인코더 불일치인데 검색이 허용됨')
    except qs.SearchSetupError as exc:
        assert '인코더 불일치' in str(exc), exc

    # 2) 필드를 잘못 짚어도(claims 벡터에 title_claims 인코더) 중단
    try:
        qs.verify_encoder_signature(_FakeScrollClient(payload), settings, 'claims',
                                    _FakeDense('BBB'))
        raise AssertionError('필드가 어긋났는데 검색이 허용됨')
    except qs.SearchSetupError:
        pass

    # 3) 필드별 서명이 아예 없는 구버전 색인이면 중단
    try:
        qs.verify_encoder_signature(_FakeScrollClient({'embedding_signature': 'BBB'}), settings,
                                    'title_claims', _FakeDense('BBB'))
        raise AssertionError('필드별 서명이 없는 색인인데 검색이 허용됨')
    except qs.SearchSetupError:
        pass

    # 4) 빈 컬렉션이면 중단
    try:
        qs.verify_encoder_signature(_FakeScrollClient(None), settings, 'claims', _FakeDense('AAA'))
        raise AssertionError('빈 컬렉션인데 검색이 허용됨')
    except qs.SearchSetupError:
        pass


def test_map_at_cutoff_naming_and_semantics():
    """evaluate_qdrant_retrieval.py의 MAP@cutoff 이름/절단/alias 일관성 (search_cli 테스트와 별개)."""
    assert metric_key_for_cutoff(1000) == 'mean_average_precision_at_1000'
    assert metric_key_for_cutoff(1) == 'mean_average_precision_at_1'

    gold = {'P1', 'P1500'}
    ranked = [f'P{i}' for i in range(1, 1501)]
    ap_full = evaluate_ranking(ranked, gold)['ap']
    ap_1000 = evaluate_ranking(ranked[:1000], gold)['ap']
    assert ap_full > ap_1000, 'cutoff 밖 정답이 절단 시 0 기여해야 한다'
    assert abs(ap_1000 - 0.5) < 1e-9

    per_query = [{'ap': 0.5, 'recall_at_10': 1.0, 'recall_at_100': 1.0, 'recall_at_1000': 1.0,
                 'mrr_at_10': 1.0, 'ndcg_at_10_binary': 1.0, 'num_gold': 1, 'num_gold_retrieved': 1,
                 'retrieved': 10, 'latency_ms': 5.0}]
    s1000 = build_system_summary(per_query, 'bm25', 'claims', 1000)
    assert s1000['mean_average_precision_at_1000'] == s1000['mean_average_precision_map'] == 0.5

    effects = build_effects({'A_jaccard_claims': {'mean_average_precision_at_1000': 0.1},
                             'B_bm25_claims': {'mean_average_precision_at_1000': 0.4}},
                            'mean_average_precision_at_1000')
    assert effects['jaccard_to_bm25 (claims 고정)']['delta'] == round(0.4 - 0.1, 4)


def _base_v2_collection_meta(**overrides):
    meta = {
        'collection': 'chemrag_patents_v1', 'role': 'patents', 'file': 'chemrag_patents_v1.jsonl.gz',
        'points': 10, 'archive_bytes': 1234,
        'archive_sha256': 'a' * 64, 'content_sha256': 'b' * 64,
        'vectors': {'claims_dense': {'size': 300, 'distance': 'Cosine'}},
        'sparse_vectors': ['claims_bm25'], 'payload_indexes': [{'field': 'patent_id', 'schema': 'KEYWORD'}],
        'alias': 'chemrag_patents',
    }
    meta.update(overrides)
    return meta


def _v2_manifest(*collections):
    return {'manifest_version': 2, 'created_at': '2026-01-01T00:00:00+0000',
           'provenance': {}, 'collections': list(collections)}


def test_manifest_v2_strict_validation_and_v1_compat():
    """v2 manifest는 필수 필드 삭제/해시 축약/중복 이름/경로 이탈을 전부 거부해야 한다
    (필드를 지우면 무결성 검사를 조용히 우회할 수 있었던 취약점에 대한 회귀 테스트).
    v1(레거시, manifest_version 키 자체가 없던 시절)은 예전처럼 관대하게 받아야 한다."""

    # 1) 정상 v2는 문제 없이 통과
    ok_manifest = _v2_manifest(_base_v2_collection_meta())
    assert validate_manifest_structure(ok_manifest) == []

    # 2) archive_sha256 필드 자체를 삭제 -> 거부(이게 이번에 고친 원래 취약점)
    m = _base_v2_collection_meta()
    del m['archive_sha256']
    problems = validate_manifest_structure(_v2_manifest(m))
    assert problems and any('archive_sha256' in p or '필수 필드' in p for p in problems), problems

    # 3) content_sha256 필드 삭제 -> 거부
    m = _base_v2_collection_meta()
    del m['content_sha256']
    problems = validate_manifest_structure(_v2_manifest(m))
    assert problems and any('content_sha256' in p or '필수 필드' in p for p in problems), problems

    # 4) archive_bytes 필드 삭제 -> 거부
    m = _base_v2_collection_meta()
    del m['archive_bytes']
    problems = validate_manifest_structure(_v2_manifest(m))
    assert problems and any('archive_bytes' in p or '필수 필드' in p for p in problems), problems

    # 5) archive_bytes가 0 이하 -> 거부
    problems = validate_manifest_structure(_v2_manifest(_base_v2_collection_meta(archive_bytes=0)))
    assert problems and any('archive_bytes' in p for p in problems), problems

    # 6) 짧은(축약된) SHA-256 -> v2에서는 거부(v1에서는 허용되던 것)
    problems = validate_manifest_structure(
        _v2_manifest(_base_v2_collection_meta(content_sha256='b' * 32)))
    assert problems and any('content_sha256' in p for p in problems), problems

    # 7) 잘못된 hex 문자 -> 거부
    problems = validate_manifest_structure(
        _v2_manifest(_base_v2_collection_meta(archive_sha256='z' * 64)))
    assert problems and any('archive_sha256' in p for p in problems), problems

    # 8) 중복 collection 이름 -> 거부
    m1, m2 = _base_v2_collection_meta(), _base_v2_collection_meta(
        file='chemrag_structures_v1.jsonl.gz', role='structures', alias='chemrag_structures')
    problems = validate_manifest_structure(_v2_manifest(m1, m2))
    assert problems and any('중복된 collection' in p for p in problems), problems

    # 9) 중복 alias 이름 -> 거부
    m1 = _base_v2_collection_meta()
    m2 = _base_v2_collection_meta(collection='chemrag_structures_v1',
                                  file='chemrag_structures_v1.jsonl.gz', role='structures')
    # m2의 alias를 m1과 같은 값으로 맞춰 충돌시킨다
    problems = validate_manifest_structure(_v2_manifest(m1, m2))
    assert problems and any('중복된 alias' in p for p in problems), problems

    # 10) file 경로 이탈(절대경로) -> 거부
    problems = validate_manifest_structure(
        _v2_manifest(_base_v2_collection_meta(file='/etc/passwd')))
    assert problems and any('경로' in p for p in problems), problems

    # 11) file 경로 이탈(../) -> 거부
    problems = validate_manifest_structure(
        _v2_manifest(_base_v2_collection_meta(file='../../etc/passwd')))
    assert problems and any('경로' in p for p in problems), problems

    # 12) 정상 v1(레거시) 매니페스트는 그대로 통과해야 한다
    #     (manifest_version 키가 없고, bytes/짧은 content hash 같은 옛 필드명)
    legacy_v1 = {
        'collections': [{
            'collection': 'chemrag_patents_v1', 'file': 'chemrag_patents_v1.jsonl.gz',
            'points': 10, 'bytes': 1234, 'content_sha256': 'c' * 32,
        }],
    }
    assert validate_manifest_structure(legacy_v1) == []


def _fake_pq_record(cutoff=None):
    rec = {'query_id': 'Q1', 'system': 'B_bm25_claims', 'pool_size': 100, 'latency_ms': 1.0,
          'original_split_label': 'test', 'num_gold': 1, 'num_gold_retrieved': 1, 'retrieved': 10,
          'ap': 0.5, 'recall_at_10': 1.0, 'recall_at_100': 1.0, 'recall_at_1000': 1.0,
          'mrr_at_10': 1.0, 'ndcg_at_10_binary': 1.0}
    if cutoff is not None:
        rec['cutoff'] = cutoff
    return rec


def test_resolve_recompute_cutoff_pure_logic():
    """--recompute-only가 cutoff를 임의로 추정하지 않는지에 대한 순수 로직 테스트."""
    # 1) per-query 레코드에 cutoff가 기록돼 있으면(신규 실행분) 그 값을 그대로 쓴다
    pq = [_fake_pq_record(cutoff=1000), _fake_pq_record(cutoff=1000)]
    cutoff, problem = resolve_recompute_cutoff(pq, fallback_cutoff=None)
    assert problem is None and cutoff == 1000

    # 2) 레코드에 cutoff가 없으면(예전 실행분) 과거 summary의 config.search_limit을 근거로 쓴다
    pq_legacy = [_fake_pq_record(cutoff=None), _fake_pq_record(cutoff=None)]
    cutoff, problem = resolve_recompute_cutoff(pq_legacy, fallback_cutoff=1000)
    assert problem is None and cutoff == 1000

    # 3) 레코드에도 없고 과거 summary도 없으면 추정하지 않고 중단 신호를 준다
    cutoff, problem = resolve_recompute_cutoff(pq_legacy, fallback_cutoff=None)
    assert cutoff is None and problem is not None

    # 4) 같은 partial 파일 안에 서로 다른 cutoff가 섞여 있으면(데이터 손상) 중단 신호
    pq_mixed = [_fake_pq_record(cutoff=1000), _fake_pq_record(cutoff=500)]
    cutoff, problem = resolve_recompute_cutoff(pq_mixed, fallback_cutoff=None)
    assert cutoff is None and problem is not None


def test_recompute_only_rejects_cutoff_mismatch_end_to_end():
    """run_recompute_only()를 실제로 호출해 top-1000 partial로 MAP@500을 재계산하려 하면
    종료 코드 1(SystemExit)로 중단되고, 같은 cutoff를 요청하면 정상 성공하는지 확인한다.
    기존 1,106쿼리 전체 재검색 없이, 가짜 partial 파일만으로 검증한다."""
    import argparse
    import shutil
    import tempfile
    import scripts.evaluation.evaluate_qdrant_retrieval as evr

    tmp = tempfile.mkdtemp(prefix='chemrag_ut_recompute_')
    orig_results_dir = evr.RESULTS_DIR
    try:
        evr.RESULTS_DIR = tmp
        out_prefix = 'unit_test_cutoff_guard'
        system = 'B_bm25_claims'
        partial_path = os.path.join(tmp, f'{out_prefix}_{system}.partial.jsonl')
        with open(partial_path, 'w', encoding='utf-8') as f:
            for i in range(3):
                rec = _fake_pq_record(cutoff=1000)
                rec['query_id'] = f'Q{i}'
                f.write(json.dumps(rec) + '\n')

        # (a) 같은 cutoff(1000)로 재계산 요청 -> 성공해야 한다
        args_ok = argparse.Namespace(systems=system, search_limit=1000, out_prefix=out_prefix)
        rc = evr.run_recompute_only(args_ok)
        assert rc == 0
        with open(os.path.join(tmp, f'{out_prefix}_summary.json'), encoding='utf-8') as f:
            report = json.load(f)
        mkey = 'mean_average_precision_at_1000'
        assert report['summaries'][system][mkey] == report['summaries'][system]['mean_average_precision_map']

        # (b) top-1000 partial인데 MAP@500을 요청 -> 종료 코드 1(SystemExit)로 중단돼야 한다
        args_bad = argparse.Namespace(systems=system, search_limit=500, out_prefix=out_prefix)
        try:
            evr.run_recompute_only(args_bad)
            raise AssertionError('top-1000 partial로 MAP@500 재계산이 성공해버림(cutoff 오류 미탐지)')
        except SystemExit as exc:
            assert 'top-1000' in str(exc) or 'cutoff' in str(exc).lower(), exc

        # (c) cutoff 메타데이터도, 과거 summary도 없는 새 out_prefix -> 추정하지 않고 중단
        out_prefix2 = 'unit_test_cutoff_guard_no_fallback'
        partial_path2 = os.path.join(tmp, f'{out_prefix2}_{system}.partial.jsonl')
        with open(partial_path2, 'w', encoding='utf-8') as f:
            rec = _fake_pq_record(cutoff=None)   # cutoff 필드 없음 (예전 실행분 흉내)
            f.write(json.dumps(rec) + '\n')
        args_no_meta = argparse.Namespace(systems=system, search_limit=1000, out_prefix=out_prefix2)
        try:
            evr.run_recompute_only(args_no_meta)
            raise AssertionError('cutoff 확인 불가인데 중단하지 않고 성공해버림')
        except SystemExit:
            pass
    finally:
        evr.RESULTS_DIR = orig_results_dir
        shutil.rmtree(tmp, ignore_errors=True)
