"""Unit tests requiring no Qdrant DB.
Usage: python3 tests/test_unit.py or python3 -m pytest tests/test_unit.py.
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
            raise AssertionError('Empty patent_id must be rejected')
        except ValueError:
            pass


def test_text_fields_and_vector_config():
    assert build_field_text('T', 'C', 'claims') == ('C', False)
    assert build_field_text('T', 'C', 'title_claims') == ('T\nC', False)
    # Retain title-only patents with missing-claims status.
    assert build_field_text('T', '', 'title_claims') == ('T', True)
    assert build_field_text('T', '', 'claims') == ('', True)
    try:
        build_field_text('T', 'C', 'nope')
        raise AssertionError('Unknown fields must be rejected')
    except ValueError:
        pass
    assert set(patent_vectors_config(300)) == {'claims_dense', 'title_claims_dense'}
    assert set(patent_sparse_config()) == {'claims_bm25', 'title_claims_bm25'}


def test_filter_rules_and_policies():
    f = build_prior_art_filter('EP_Q', '20200101', 'FAM1')
    keys = [c.key for c in f.qdrant_filter.must_not if hasattr(c, 'key')]
    assert 'patent_id' in keys and 'family_id' in keys and f.qdrant_filter.must
    # Missing app_date / family_id skips the corresponding condition.
    f2 = build_prior_art_filter('EP_Q', '', '')
    assert not f2.qdrant_filter.must
    assert [c.key for c in f2.qdrant_filter.must_not if hasattr(c, 'key')] == ['patent_id']
    f3 = build_prior_art_filter('EP_Q', '20200101', '')
    assert f3.qdrant_filter.must and len(f3.qdrant_filter.must_not) == 1
    assert _to_int_date('20200131') == 20200131
    assert _to_int_date('') is None and _to_int_date(None) is None and _to_int_date('x') is None


def test_attach_filter_injects_into_every_prefetch_branch():
    """A top-level query_filter does not filter prefetch, as observed in Qdrant."""
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
        raise AssertionError('Unknown fusion must be rejected')
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
    # Supplying token lists directly must give the same result.
    toks = [b.tokenizer.tokenize('ethylene polymer composition')]
    assert b.encode_documents(['ethylene polymer composition'], token_lists=toks)[0] == (d_idx, d_val)


def test_exact_tanimoto_semantics():
    f1, f2 = build_fingerprint('CCO'), build_fingerprint('CCCO')
    assert max_tanimoto([f1], [f2]) == 0.5556 and max_tanimoto([f1], [f1]) == 1.0
    assert max_tanimoto([f1], []) is None and max_tanimoto([], []) is None   # Missing != 0.
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
    """AP = sum(precision at gold ranks) / all gold; unretrieved gold contributes zero."""
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
    """Check three-valued logic, evidence, and count consistency when derived artifacts exist."""
    from config.settings import DATA_DIR
    path = os.path.join(DATA_DIR, 'db_export', 'derived', 'structure_location_metadata.jsonl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing - run scripts/qdrant/qdrant_build_structure_metadata.py first')
    seen = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            key = (rec['patent_id'], rec['mol_id'])
            assert key not in seen, f'duplicates {key}'
            seen.add(key)
            assert rec['in_claim'] in (TRUE, FALSE, UNKNOWN)
            assert rec['in_figure'] in (TRUE, FALSE, UNKNOWN)
            ev = rec.get('location_evidence') or {}
            occ, ccws = rec['claim_occurrence_count'], rec['claim_count_with_structure']
            if rec['in_claim'] == TRUE:
                assert ev.get('claim'), 'in_claim=true requires evidence'
            if rec['in_claim'] == UNKNOWN:
                assert occ is None, 'unknown means null, not 0'
            if rec['in_claim'] == FALSE:
                assert occ == 0
            if occ is not None and ccws is not None:
                assert ccws <= occ, 'claim_count_with_structure cannot exceed occurrence'
            if rec['in_figure'] == TRUE:
                assert ev.get('figure') and (rec.get('figure_unit_refs') or any(
                    e.get('fig_id') or e.get('unit_ref') for e in ev['figure']))
    assert len(seen) == 6928, f'Structure metadata record count: {len(seen)}'


class _FakePoint:
    def __init__(self, payload):
        self.payload = payload


class _FakeScrollClient:
    """Minimal scroll()-only client to test signature checks without a DB."""

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
    """Search CLI uses field-specific state files and defaults to title_claims."""
    from scripts.qdrant import qdrant_search as qs

    assert qs.DEFAULT_FIELD == 'title_claims'
    for field in TEXT_FIELDS:
        assert os.path.basename(qs.bm25_state_path(field)) == f'bm25_state_{field}.json'
    # Do not reference legacy bm25_state.json.
    assert 'bm25_state.json' not in {os.path.basename(qs.bm25_state_path(f)) for f in TEXT_FIELDS}

    ap = qs.build_arg_parser()
    assert ap.parse_args(['--patent', 'EP1']).field == 'title_claims'
    assert ap.parse_args(['--patent', 'EP1', '--field', 'claims']).field == 'claims'
    assert ap.parse_args(['--text', 'x', '--field', 'title_claims']).field == 'title_claims'
    try:
        ap.parse_args(['--patent', 'EP1', '--field', 'nope'])
        raise AssertionError('Unknown --field must be rejected')
    except SystemExit:
        pass


def test_search_cli_missing_bm25_state_stops_search():
    from scripts.qdrant import qdrant_search as qs

    settings = load_settings()
    try:
        qs.load_field_encoders(settings, 'claims', state_path='/nonexistent/bm25_state_claims.json')
        raise AssertionError('Search must not start without BM25 state')
    except qs.SearchSetupError as exc:
        assert 'BM25 state file is missing' in str(exc), exc
    try:
        qs.load_field_encoders(settings, 'nope')
        raise AssertionError('Unknown fields must be rejected')
    except qs.SearchSetupError:
        pass


def test_search_cli_encoder_signature_guard():
    """Stop before search if the encoder differs from payload embedding_signature_by_field."""
    from scripts.qdrant import qdrant_search as qs

    settings = load_settings()
    payload = {'embedding_signature_by_field': {'claims': 'AAA', 'title_claims': 'BBB'}}

    ok = qs.verify_encoder_signature(_FakeScrollClient(payload), settings, 'claims',
                                     _FakeDense('AAA'))
    assert ok == {'field': 'claims', 'indexed': 'AAA', 'search': 'AAA', 'match': True}
    assert qs.verify_encoder_signature(_FakeScrollClient(payload), settings, 'title_claims',
                                       _FakeDense('BBB'))['match']

    # 1) Stop on signature mismatch.
    try:
        qs.verify_encoder_signature(_FakeScrollClient(payload), settings, 'claims',
                                    _FakeDense('ZZZ'))
        raise AssertionError('Search was allowed despite an encoder mismatch')
    except qs.SearchSetupError as exc:
        assert 'Encoder mismatch' in str(exc), exc

    # 2) Stop on field mismatch, e.g. title_claims encoder with claims vectors.
    try:
        qs.verify_encoder_signature(_FakeScrollClient(payload), settings, 'claims',
                                    _FakeDense('BBB'))
        raise AssertionError('Search was allowed despite a field mismatch')
    except qs.SearchSetupError:
        pass

    # 3) Stop for a legacy index without field-specific signatures.
    try:
        qs.verify_encoder_signature(_FakeScrollClient({'embedding_signature': 'BBB'}), settings,
                                    'title_claims', _FakeDense('BBB'))
        raise AssertionError('Search was allowed without field-specific index signatures')
    except qs.SearchSetupError:
        pass

    # 4) Stop for an empty collection.
    try:
        qs.verify_encoder_signature(_FakeScrollClient(None), settings, 'claims', _FakeDense('AAA'))
        raise AssertionError('Search was allowed on an empty collection')
    except qs.SearchSetupError:
        pass


def test_map_at_cutoff_naming_and_semantics():
    """evaluate_qdrant_retrieval.py MAP@cutoff naming/truncation/alias consistency, separate from search CLI tests."""
    assert metric_key_for_cutoff(1000) == 'mean_average_precision_at_1000'
    assert metric_key_for_cutoff(1) == 'mean_average_precision_at_1'

    gold = {'P1', 'P1500'}
    ranked = [f'P{i}' for i in range(1, 1501)]
    ap_full = evaluate_ranking(ranked, gold)['ap']
    ap_1000 = evaluate_ranking(ranked[:1000], gold)['ap']
    assert ap_full > ap_1000, 'Gold beyond the cutoff must contribute zero after truncation'
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
    """v2 rejects missing required fields, truncated hashes, duplicate names, and path escape: regression checks for deletion-based validation bypass. Legacy v1 without manifest_version remains permissive."""

    # 1) Valid v2 passes without problems.
    ok_manifest = _v2_manifest(_base_v2_collection_meta())
    assert validate_manifest_structure(ok_manifest) == []

    # 2) Remove archive_sha256 -> reject the original validation-bypass vulnerability.
    m = _base_v2_collection_meta()
    del m['archive_sha256']
    problems = validate_manifest_structure(_v2_manifest(m))
    assert problems and any('archive_sha256' in p or 'required fields' in p for p in problems), problems

    # 3) Remove content_sha256 -> reject.
    m = _base_v2_collection_meta()
    del m['content_sha256']
    problems = validate_manifest_structure(_v2_manifest(m))
    assert problems and any('content_sha256' in p or 'required fields' in p for p in problems), problems

    # 4) Remove archive_bytes -> reject.
    m = _base_v2_collection_meta()
    del m['archive_bytes']
    problems = validate_manifest_structure(_v2_manifest(m))
    assert problems and any('archive_bytes' in p or 'required fields' in p for p in problems), problems

    # 5) archive_bytes <= 0 -> reject.
    problems = validate_manifest_structure(_v2_manifest(_base_v2_collection_meta(archive_bytes=0)))
    assert problems and any('archive_bytes' in p for p in problems), problems

    # 6) Truncated SHA-256 -> reject in v2, although legacy v1 allowed it.
    problems = validate_manifest_structure(
        _v2_manifest(_base_v2_collection_meta(content_sha256='b' * 32)))
    assert problems and any('content_sha256' in p for p in problems), problems

    # 7) Invalid hexadecimal characters -> reject.
    problems = validate_manifest_structure(
        _v2_manifest(_base_v2_collection_meta(archive_sha256='z' * 64)))
    assert problems and any('archive_sha256' in p for p in problems), problems

    # 8) Duplicate collection names -> reject.
    m1, m2 = _base_v2_collection_meta(), _base_v2_collection_meta(
        file='chemrag_structures_v1.jsonl.gz', role='structures', alias='chemrag_structures')
    problems = validate_manifest_structure(_v2_manifest(m1, m2))
    assert problems and any('Duplicate collection' in p for p in problems), problems

    # 9) Duplicate alias names -> reject.
    m1 = _base_v2_collection_meta()
    m2 = _base_v2_collection_meta(collection='chemrag_structures_v1',
                                  file='chemrag_structures_v1.jsonl.gz', role='structures')
    # Set m2's alias equal to m1's to cause a collision.
    problems = validate_manifest_structure(_v2_manifest(m1, m2))
    assert problems and any('Duplicate alias' in p for p in problems), problems

    # 10) Absolute file-path escape -> reject.
    problems = validate_manifest_structure(
        _v2_manifest(_base_v2_collection_meta(file='/etc/passwd')))
    assert problems and any('path' in p for p in problems), problems

    # 11) '../' file-path escape -> reject.
    problems = validate_manifest_structure(
        _v2_manifest(_base_v2_collection_meta(file='../../etc/passwd')))
    assert problems and any('path' in p for p in problems), problems

    # 12) Valid legacy v1 manifests must still pass.
    # No manifest_version; historical bytes and short content-hash fields.
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
    """Pure cutoff-resolution checks ensure --recompute-only never guesses."""
    # 1) New per-query cutoff metadata is used directly.
    pq = [_fake_pq_record(cutoff=1000), _fake_pq_record(cutoff=1000)]
    cutoff, problem = resolve_recompute_cutoff(pq, fallback_cutoff=None)
    assert problem is None and cutoff == 1000

    # 2) Legacy records without cutoff use previous summary config.search_limit.
    pq_legacy = [_fake_pq_record(cutoff=None), _fake_pq_record(cutoff=None)]
    cutoff, problem = resolve_recompute_cutoff(pq_legacy, fallback_cutoff=1000)
    assert problem is None and cutoff == 1000

    # 3) Without either source, signal stop rather than guessing.
    cutoff, problem = resolve_recompute_cutoff(pq_legacy, fallback_cutoff=None)
    assert cutoff is None and problem is not None

    # 4) Mixed cutoffs within a partial file indicate corruption and signal stop.
    pq_mixed = [_fake_pq_record(cutoff=1000), _fake_pq_record(cutoff=500)]
    cutoff, problem = resolve_recompute_cutoff(pq_mixed, fallback_cutoff=None)
    assert cutoff is None and problem is not None


def test_recompute_only_rejects_cutoff_mismatch_end_to_end():
    """Call run_recompute_only() with synthetic partial files, without reretrieving 1,106 queries. MAP@500 from a top-1000 partial must stop with SystemExit(1); requesting the same cutoff must succeed."""
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

        # (a) Recompute at the same cutoff (1000) -> success.
        args_ok = argparse.Namespace(systems=system, search_limit=1000, out_prefix=out_prefix)
        rc = evr.run_recompute_only(args_ok)
        assert rc == 0
        with open(os.path.join(tmp, f'{out_prefix}_summary.json'), encoding='utf-8') as f:
            report = json.load(f)
        mkey = 'mean_average_precision_at_1000'
        assert report['summaries'][system][mkey] == report['summaries'][system]['mean_average_precision_map']

        # (b) Request MAP@500 from top-1000 partial -> SystemExit(1).
        args_bad = argparse.Namespace(systems=system, search_limit=500, out_prefix=out_prefix)
        try:
            evr.run_recompute_only(args_bad)
            raise AssertionError('MAP@500 recomputation from top-1000 partial incorrectly succeeded (cutoff mismatch undetected)')
        except SystemExit as exc:
            assert 'top-1000' in str(exc) or 'cutoff' in str(exc).lower(), exc

        # (c) New out_prefix without cutoff metadata or prior summary -> stop without guessing.
        out_prefix2 = 'unit_test_cutoff_guard_no_fallback'
        partial_path2 = os.path.join(tmp, f'{out_prefix2}_{system}.partial.jsonl')
        with open(partial_path2, 'w', encoding='utf-8') as f:
            rec = _fake_pq_record(cutoff=None)   # No cutoff field, simulating legacy records.
            f.write(json.dumps(rec) + '\n')
        args_no_meta = argparse.Namespace(systems=system, search_limit=1000, out_prefix=out_prefix2)
        try:
            evr.run_recompute_only(args_no_meta)
            raise AssertionError('Recomputation succeeded despite an unverifiable cutoff')
        except SystemExit:
            pass
    finally:
        evr.RESULTS_DIR = orig_results_dir
        shutil.rmtree(tmp, ignore_errors=True)
