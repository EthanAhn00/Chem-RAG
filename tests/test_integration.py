"""통합 테스트 - 전용 테스트 컬렉션을 직접 만들고 실제로 검색한다.

프로덕션 컬렉션에 의존하지 않는다(임시 저장소에 자체 데이터를 넣고 검증).
따라서 색인이 없다는 이유로 skip 되지 않는다.

    python3 tests/test_integration.py
"""
import os
import sys
import gzip
import json
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import models

from qdrant_engine.config import load_settings
from qdrant_engine.client import (get_client, ensure_patent_collection,
                                  ensure_structure_collection, check_local_storage_writable)
from qdrant_engine.schemas import patent_point_id, structure_point_id, TEXT_FIELDS
from qdrant_engine.filters import build_prior_art_filter
from qdrant_engine.patent_retriever import PatentRetriever
from embeddings.dense_encoder import HashingDenseEncoder
from embeddings.sparse_encoder import BM25SparseEncoder
from qdrant_engine.structure_repository import morgan_bits

# (patent_id, title, claims, pub_date, app_date, family_id)
FIXTURES = [
    ('Q1', 'query patent', 'ethylene polymer catalyst composition', '20200601', '20200101', 'FAM_Q'),
    ('OLD_HIT', 'old relevant', 'ethylene polymer catalyst composition for film', '20190101', '20180101', 'FAM_A'),
    ('EQUAL_DATE', 'equal date', 'polymer catalyst composition', '20200101', '20190101', 'FAM_B'),
    ('FUTURE', 'future doc', 'ethylene polymer catalyst composition', '20201231', '20200601', 'FAM_C'),
    ('SAME_FAMILY', 'same family', 'ethylene polymer catalyst composition', '20190201', '20180201', 'FAM_Q'),
    ('UNRELATED', 'tire', 'pneumatic tire bead reinforcing layer', '20190301', '20180301', 'FAM_D'),
    ('NO_DATE', 'no pub date', 'ethylene polymer composition', '', '20180401', 'FAM_E'),
]


def _build(tmpdir):
    settings = load_settings()
    settings.url = ''                       # 통합 테스트는 항상 임시 로컬 저장소 사용
    settings.local_path = tmpdir
    settings.patents_collection = 'test_patents_integration'
    settings.structures_collection = 'test_structures_integration'
    settings.dense_backend = 'hashing'      # 모델 파일 없이도 도는 결정론적 인코더
    settings.dense_dim = 64

    ok, detail = check_local_storage_writable(tmpdir)
    assert ok, f'임시 저장소 사용 불가: {detail}'

    dense = HashingDenseEncoder(dim=64)
    texts = {f: [] for f in TEXT_FIELDS}
    for pid, title, claims, *_ in FIXTURES:
        texts['claims'].append(claims)
        texts['title_claims'].append(f'{title}\n{claims}')
    sparse = {f: BM25SparseEncoder().fit(texts[f]) for f in TEXT_FIELDS}

    client = get_client(settings)
    ensure_patent_collection(client, settings, dense.dim, recreate=True)

    points = []
    for i, (pid, title, claims, pub, app, fam) in enumerate(FIXTURES):
        vector = {}
        for f, spec in TEXT_FIELDS.items():
            vector[spec['dense']] = dense.encode_one(texts[f][i])
            idx, val = sparse[f].encode_documents([texts[f][i]])[0]
            if idx:
                vector[spec['sparse']] = models.SparseVector(indices=idx, values=val)
        points.append(models.PointStruct(
            id=patent_point_id(pid), vector=vector,
            payload={'patent_id': pid, 'title': title, 'claims_text': claims,
                     'pub_date': pub, 'app_date': app,
                     'pub_date_int': int(pub) if pub else None,
                     'family_id': fam, 'role': 'corpus', 'split': 'train',
                     # 검색 CLI의 서명 검사가 실제 색인과 같은 형태의 payload를 보게 한다
                     'embedding_signature_by_field': {f: dense.signature() for f in TEXT_FIELDS}}))
    client.upsert(settings.patents_collection, points=points, wait=True)
    return settings, client, PatentRetriever(client, settings,
                                             {f: dense for f in TEXT_FIELDS}, sparse)


def test_integration_search_modes_and_filters():
    tmp = tempfile.mkdtemp(prefix='chemrag_it_')
    try:
        settings, client, retriever = _build(tmp)
        assert client.count(settings.patents_collection).count == len(FIXTURES)

        # 필터 없는 검색은 차단
        try:
            retriever.search_dense('x', None)
            raise AssertionError('무필터 검색이 차단되지 않음')
        except ValueError:
            pass

        pf = build_prior_art_filter('Q1', '20200101', 'FAM_Q')
        violations = {'self': 0, 'family': 0, 'future': 0}
        for mode in ('bm25', 'dense', 'hybrid'):
            for field in TEXT_FIELDS:
                hits, _ = retriever.search(mode, 'ethylene polymer catalyst composition', pf,
                                           limit=10, field=field)
                ids = [h['patent_id'] for h in hits]
                if 'Q1' in ids:
                    violations['self'] += 1
                if 'SAME_FAMILY' in ids:
                    violations['family'] += 1
                if 'FUTURE' in ids:
                    violations['future'] += 1
                # 경계조건: pub_date == app_date 인 문서는 포함되어야 한다
                assert 'EQUAL_DATE' in ids, f'{mode}/{field}: 동일 날짜 문서가 제외됨'
                # pub_date 가 없는 문서는 시간 조건으로 배제하지 않는다
                assert 'NO_DATE' in ids, f'{mode}/{field}: pub_date 없는 문서가 제외됨'
        assert violations == {'self': 0, 'family': 0, 'future': 0}, violations

        # 멱등성: 같은 포인트 재적재 후에도 개수 불변
        pts = list(client.scroll(settings.patents_collection, limit=10,
                                 with_payload=True, with_vectors=True)[0])
        before = client.count(settings.patents_collection).count
        client.upsert(settings.patents_collection, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in pts], wait=True)
        assert client.count(settings.patents_collection).count == before

        # 관련 문서가 무관 문서보다 위에 오는지(랭킹이 실제로 동작하는지).
        # BM25는 질의어와 겹치는 토큰이 하나도 없는 문서(UNRELATED)를 아예 반환하지 않는다.
        # 따라서 '없거나, 있더라도 더 아래'를 요구한다.
        hits, _ = retriever.search('bm25', 'ethylene polymer catalyst composition', pf,
                                   limit=10, field='claims')
        ids = [h['patent_id'] for h in hits]
        assert 'OLD_HIT' in ids, ids
        assert ('UNRELATED' not in ids) or (ids.index('OLD_HIT') < ids.index('UNRELATED')), ids

        # dense는 모든 문서에 점수를 주므로 무관 문서도 반환되지만 더 아래여야 한다
        dhits, _ = retriever.search('dense', 'ethylene polymer catalyst composition', pf,
                                    limit=10, field='claims')
        dids = [h['patent_id'] for h in dhits]
        assert 'UNRELATED' in dids and dids.index('OLD_HIT') < dids.index('UNRELATED'), dids
        client.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_integration_snapshot_roundtrip():
    """export -> 완전히 별개인 두 번째 임시 저장소로 실제 restore -> 원본과 비교 ->
    복원본에서 검색까지 실행한다(단순히 gzip을 읽기만 하는 게 아니라
    qdrant_engine.snapshot.restore_snapshot()을 실제로 호출).

    비교 범위(전부 실제로 비교한 것만 "완전 일치"라고 표현한다):
      특허 컬렉션 - 포인트 수, 전체 point ID, 전체 payload, claims_dense/title_claims_dense
        (float32 허용오차), claims_bm25/title_claims_bm25(sparse indices/values/길이/이름)
      구조 컬렉션 - 포인트 수, 전체 point ID, 전체 payload, morgan_ecfp4_2048(sparse 완전 일치)
      payload 인덱스 - 로컬 임베디드 모드 기대값(no_effect_local_mode)
      alias - 실제 재조회한 verified_target_matches
      대표 검색 - claims BM25/dense, title_claims BM25/dense, 필터 적용 검색,
        구조 fingerprint 후보 검색(approximate_candidates)
    서버가 없는 이 테스트는 로컬 모드 기대값만 검사한다. 서버 전용 검증은
    이 파일의 별도 서버 통합 테스트(SKIP 사유 명시)가 맡는다."""
    from qdrant_engine.snapshot import export_collection, write_manifest, restore_snapshot
    from qdrant_engine.client import ensure_alias
    from qdrant_engine.structure_repository import StructureRepository

    tmp1 = tempfile.mkdtemp(prefix='chemrag_it_snap_src_')
    tmp2 = tempfile.mkdtemp(prefix='chemrag_it_snap_dst_')
    out = tempfile.mkdtemp(prefix='chemrag_it_snapout_')
    try:
        settings1, client1, retriever1 = _build(tmp1)
        struct_rows = [('Q1', 'M1', 'CCO'), ('OLD_HIT', 'M2', 'CCO'), ('OLD_HIT', 'M3', 'CCOC')]
        _add_structures(client1, settings1, struct_rows)
        ensure_alias(client1, 'test_patents_alias', settings1.patents_collection)
        ensure_alias(client1, 'test_structures_alias', settings1.structures_collection)

        meta_p = export_collection(client1, settings1.patents_collection, out, role='patents',
                                   alias='test_patents_alias')
        meta_s = export_collection(client1, settings1.structures_collection, out, role='structures',
                                   alias='test_structures_alias')
        assert meta_p['points'] == len(FIXTURES) and meta_s['points'] == len(struct_rows)
        assert len(meta_p['archive_sha256']) == 64 and len(meta_p['content_sha256']) == 64
        assert meta_p['payload_indexes'], 'patents role은 payload_indexes가 채워져야 한다'
        assert meta_s['payload_indexes'], 'structures role은 payload_indexes가 채워져야 한다'
        mpath, manifest = write_manifest(out, [meta_p, meta_s], settings1, 'test')
        assert manifest['manifest_version'] == 2

        # 원본 전체 포인트(payload + 모든 벡터) - 특허/구조 둘 다
        orig_patent_points, _ = client1.scroll(settings1.patents_collection, limit=100,
                                               with_payload=True, with_vectors=True)
        orig_patent_by_id = {str(p.id): p for p in orig_patent_points}
        orig_struct_points, _ = client1.scroll(settings1.structures_collection, limit=100,
                                               with_payload=True, with_vectors=True)
        orig_struct_by_id = {str(p.id): p for p in orig_struct_points}
        assert len(orig_patent_by_id) == len(FIXTURES)
        assert len(orig_struct_by_id) == len(struct_rows)
        client1.close()

        settings2 = load_settings()
        settings2.url = ''
        settings2.local_path = tmp2
        settings2.patents_collection = settings1.patents_collection
        settings2.structures_collection = settings1.structures_collection
        client2 = get_client(settings2)

        report = restore_snapshot(out, settings2, client2, recreate=True)
        assert report['all_ok'], report
        assert report['restore_mode'] == 'local_embedded', report['restore_mode']
        for c in report['collections']:
            assert c['points_match'] and c['point_id_set_match'] and c['vector_schema_match']
            # 로컬 임베디드 모드 기대값: payload 인덱스는 no_effect_local_mode(실패 아님)
            assert not c['payload_indexes']['failed'], c['payload_indexes']
            assert c['payload_indexes_ok'], c['payload_indexes']
            # alias: 실제 재조회 결과가 대상 컬렉션과 일치해야 한다(API 호출 성공만으로 판정하지 않음)
            assert c['alias']['status'] == 'created', c['alias']
            assert c['alias']['verified_exists'] is True and c['alias']['verified_target_matches'] is True, c['alias']
            assert c['alias_ok']
            # 대표 검색이 실제로 실행 가능했는지(verify_search 기본값 True로 호출했으므로)
            assert c['representative_search_ok'], c.get('representative_search')

        claims_dense_vn = TEXT_FIELDS['claims']['dense']
        title_dense_vn = TEXT_FIELDS['title_claims']['dense']
        claims_bm25_vn = TEXT_FIELDS['claims']['sparse']
        title_bm25_vn = TEXT_FIELDS['title_claims']['sparse']

        def _assert_sparse_equal(ov, rv, label):
            assert list(ov.indices) == list(rv.indices), (label, 'indices', ov.indices, rv.indices)
            assert len(ov.values) == len(rv.values), (label, 'length')
            assert all(abs(a - b) < 1e-6 for a, b in zip(ov.values, rv.values)), \
                (label, 'values', ov.values, rv.values)

        # --- 특허 컬렉션: 전체 point ID + payload + dense(허용오차) + sparse(완전 일치) ---
        restored_patent_points, _ = client2.scroll(settings2.patents_collection, limit=100,
                                                    with_payload=True, with_vectors=True)
        assert len(restored_patent_points) == len(orig_patent_by_id)
        assert {str(p.id) for p in restored_patent_points} == set(orig_patent_by_id)
        for rp in restored_patent_points:
            op = orig_patent_by_id[str(rp.id)]
            assert rp.payload == op.payload, (rp.id, rp.payload, op.payload)
            for vn in (claims_dense_vn, title_dense_vn):
                ov, rv = list(op.vector[vn]), list(rp.vector[vn])
                assert len(ov) == len(rv), vn
                # float32 저장소를 한 번 더 거치면 소수 7째 자리 수준의 반올림 차이가 생길 수 있어
                # 정확히 같음이 아니라 float32 정밀도(1e-6) 이내 근접을 요구한다.
                assert all(abs(a - b) < 1e-6 for a, b in zip(ov, rv)), (vn, ov, rv)
            for vn in (claims_bm25_vn, title_bm25_vn):
                _assert_sparse_equal(op.vector[vn], rp.vector[vn], vn)

        # --- 구조 컬렉션: 전체 point ID + payload + fingerprint sparse(완전 일치) ---
        restored_struct_points, _ = client2.scroll(settings2.structures_collection, limit=100,
                                                    with_payload=True, with_vectors=True)
        assert len(restored_struct_points) == len(orig_struct_by_id)
        assert {str(p.id) for p in restored_struct_points} == set(orig_struct_by_id)
        fp_vn = settings2.fingerprint_vector_name
        for rp in restored_struct_points:
            op = orig_struct_by_id[str(rp.id)]
            assert rp.payload == op.payload, (rp.id, rp.payload, op.payload)
            _assert_sparse_equal(op.vector[fp_vn], rp.vector[fp_vn], fp_vn)

        # --- 복원본에서 대표 검색(claims/title_claims x BM25/dense) + 필터 적용 확인 ---
        dense = HashingDenseEncoder(dim=64)
        retriever2 = PatentRetriever(client2, settings2,
                                     {f: dense for f in TEXT_FIELDS}, retriever1.sparse_encoders)
        pf = build_prior_art_filter('Q1', '20200101', 'FAM_Q')
        for mode in ('bm25', 'dense'):
            for field in ('claims', 'title_claims'):
                hits, _ = retriever2.search(mode, 'ethylene polymer catalyst composition', pf,
                                            limit=10, field=field)
                ids = [h['patent_id'] for h in hits]
                assert 'Q1' not in ids and 'SAME_FAMILY' not in ids and 'FUTURE' not in ids, \
                    (mode, field, ids)
                assert 'OLD_HIT' in ids, (mode, field, ids)

        # --- 구조 fingerprint 후보 검색(approximate_candidates) - 복원본 대상 ---
        struct_repo2 = StructureRepository(client2, settings2)
        candidates = struct_repo2.approximate_candidates('CCO', limit=10)
        cand_ids = {(c['patent_id'], c['mol_id']) for c in candidates}
        assert ('Q1', 'M1') in cand_ids or ('OLD_HIT', 'M2') in cand_ids, candidates

        alias_report = next(c['alias'] for c in report['collections']
                            if c['collection'] == settings2.patents_collection)
        assert alias_report['status'] == 'created' and alias_report['verified_target_matches'], alias_report
        client2.close()
    finally:
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def _make_snapshot_fixture(out_dir):
    """변조 테스트용 - fixture 특허 컬렉션 하나를 export하고 (out_dir, manifest_path) 반환."""
    from qdrant_engine.snapshot import export_collection, write_manifest
    tmp = tempfile.mkdtemp(prefix='chemrag_it_tamper_src_')
    settings, client, _ = _build(tmp)
    try:
        meta = export_collection(client, settings.patents_collection, out_dir, role='patents')
        mpath, manifest = write_manifest(out_dir, [meta], settings, 'test')
    finally:
        client.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return mpath, manifest, os.path.join(out_dir, meta['file'])


def _restore_target(prepopulate=True):
    """변조 테스트의 복원 대상 저장소. prepopulate=True면 기존 데이터를 넣어두고
    반환값으로 '복원 실패 후에도 그대로인지' 확인할 수 있게 한다."""
    tmp = tempfile.mkdtemp(prefix='chemrag_it_tamper_dst_')
    settings = load_settings()
    settings.url = ''
    settings.local_path = tmp
    settings.patents_collection = 'test_patents_integration'
    client = get_client(settings)
    sentinel_payload = None
    if prepopulate:
        ensure_patent_collection(client, settings, 64, recreate=True)
        sentinel_payload = {'patent_id': 'SENTINEL', 'title': 'pre-existing', 'claims_text': 'x',
                            'pub_date': '20200101', 'app_date': '20190101', 'family_id': 'FAM_S',
                            'role': 'corpus', 'split': 'train'}
        client.upsert(settings.patents_collection, points=[models.PointStruct(
            id=patent_point_id('SENTINEL'),
            vector={TEXT_FIELDS['claims']['dense']: [0.0] * 64,
                   TEXT_FIELDS['title_claims']['dense']: [0.0] * 64},
            payload=sentinel_payload)], wait=True)
    return tmp, settings, client, sentinel_payload


def _assert_target_untouched(client, settings, sentinel_payload):
    from qdrant_engine.client import collection_exists
    assert collection_exists(client, settings.patents_collection), '기존 컬렉션이 사라짐'
    assert client.count(settings.patents_collection).count == 1
    pts, _ = client.scroll(settings.patents_collection, limit=1, with_payload=True)
    assert pts[0].payload == sentinel_payload, '기존 컬렉션 내용이 변경됨'


def test_integration_snapshot_restore_rejects_tampered_archive():
    """(1) 파일 일부 변조 - archive_sha256 불일치 -> 중단, 기존 컬렉션 보존."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    out = tempfile.mkdtemp(prefix='chemrag_it_tamper_out_')
    dst_tmp = None
    try:
        mpath, manifest, archive_path = _make_snapshot_fixture(out)
        with open(archive_path, 'r+b') as f:
            f.seek(10)
            b = f.read(1)
            f.seek(10)
            f.write(bytes([b[0] ^ 0xFF]))   # 압축 바이트 하나를 뒤집는다

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('변조된 archive인데 복원이 성공함')
        except SnapshotIntegrityError as exc:
            assert 'archive_sha256' in str(exc) or 'gzip' in str(exc) or '변조' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_same_length_content_tamper():
    """(2) 내용은 바뀌었지만 행 수는 같은 변조 - content_sha256이 archive_sha256과
    별개로 이를 잡아내는지 확인(공격자가 archive_sha256만 재계산해도 여전히 걸림)."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError, sha256_file
    out = tempfile.mkdtemp(prefix='chemrag_it_tamper_out_')
    dst_tmp = None
    try:
        mpath, manifest, archive_path = _make_snapshot_fixture(out)
        with gzip.open(archive_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
        assert len(lines) >= 1
        rec = json.loads(lines[0])
        rec['payload']['title'] = 'TAMPERED TITLE - same line count'
        lines[0] = json.dumps(rec, ensure_ascii=False) + '\n'
        with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
            f.writelines(lines)
        # 공격자가 archive_sha256/크기는 새로 계산해 manifest에 반영했다고 가정
        # (그래도 content_sha256이 예전 값이라 걸려야 한다)
        manifest['collections'][0]['archive_sha256'] = sha256_file(archive_path)
        manifest['collections'][0]['archive_bytes'] = os.path.getsize(archive_path)
        with open(mpath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('내용이 바뀌었는데(행 수 동일) 복원이 성공함')
        except SnapshotIntegrityError as exc:
            assert 'content_sha256' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_point_count_tamper():
    """(3) manifest의 포인트 수 변조 - 파일은 그대로인데 manifest.points만 조작."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    out = tempfile.mkdtemp(prefix='chemrag_it_tamper_out_')
    dst_tmp = None
    try:
        mpath, manifest, archive_path = _make_snapshot_fixture(out)
        manifest['collections'][0]['points'] += 1000
        with open(mpath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('포인트 수가 조작됐는데 복원이 성공함')
        except SnapshotIntegrityError as exc:
            assert '줄 수' in str(exc) or 'points' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_dimension_tamper():
    """(4) manifest의 vector dimension 변조 - 실제 벡터 길이와 선언이 달라지면 잡아야 한다."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    out = tempfile.mkdtemp(prefix='chemrag_it_tamper_out_')
    dst_tmp = None
    try:
        mpath, manifest, archive_path = _make_snapshot_fixture(out)
        vn = TEXT_FIELDS['claims']['dense']
        manifest['collections'][0]['vectors'][vn]['size'] = 9999
        with open(mpath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('vector dimension이 조작됐는데 복원이 성공함')
        except SnapshotIntegrityError as exc:
            assert '벡터 길이' in str(exc) or 'dimension' in str(exc) or 'content_sha256' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_missing_file():
    """(5) 파일 누락."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    out = tempfile.mkdtemp(prefix='chemrag_it_tamper_out_')
    dst_tmp = None
    try:
        mpath, manifest, archive_path = _make_snapshot_fixture(out)
        os.remove(archive_path)

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('파일이 없는데 복원이 성공함')
        except SnapshotIntegrityError as exc:
            assert '파일 없음' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_malformed_manifest():
    """manifest 구조 자체가 깨진 경우(필수 키 누락) - 파일 읽기 전에 걸려야 한다."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    out = tempfile.mkdtemp(prefix='chemrag_it_tamper_out_')
    dst_tmp = None
    try:
        mpath, manifest, archive_path = _make_snapshot_fixture(out)
        del manifest['collections'][0]['points']
        with open(mpath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('manifest 구조가 깨졌는데 복원이 성공함')
        except SnapshotIntegrityError as exc:
            assert 'manifest 구조' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_v2_required_field_deletion():
    """v2 manifest에서 archive_sha256/content_sha256/archive_bytes 중 하나를 지우면
    (예전에는 이렇게 하면 검증을 그냥 통과했던 취약점) 복원이 거부되고 기존 컬렉션은 보존돼야 한다."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    for field_to_delete in ('archive_sha256', 'content_sha256', 'archive_bytes'):
        out = tempfile.mkdtemp(prefix='chemrag_it_v2field_out_')
        dst_tmp = None
        try:
            mpath, manifest, archive_path = _make_snapshot_fixture(out)
            assert manifest['manifest_version'] == 2
            del manifest['collections'][0][field_to_delete]
            with open(mpath, 'w', encoding='utf-8') as f:
                json.dump(manifest, f)

            dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
            try:
                restore_snapshot(out, dst_settings, dst_client, recreate=True)
                raise AssertionError(f'{field_to_delete} 를 지웠는데 v2 복원이 성공함')
            except SnapshotIntegrityError as exc:
                assert field_to_delete in str(exc) or '필수 필드' in str(exc), (field_to_delete, exc)
            _assert_target_untouched(dst_client, dst_settings, sentinel)
            dst_client.close()
        finally:
            shutil.rmtree(out, ignore_errors=True)
            if dst_tmp:
                shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_path_traversal_file():
    """manifest.collections[i].file 이 스냅샷 디렉토리 밖(절대경로/../)을 가리키면
    파일을 열어보기도 전에 거부되고 기존 컬렉션은 보존돼야 한다."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    for bad_file in ('/etc/passwd', '../../etc/passwd'):
        out = tempfile.mkdtemp(prefix='chemrag_it_traversal_out_')
        dst_tmp = None
        try:
            mpath, manifest, archive_path = _make_snapshot_fixture(out)
            manifest['collections'][0]['file'] = bad_file
            with open(mpath, 'w', encoding='utf-8') as f:
                json.dump(manifest, f)

            dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
            try:
                restore_snapshot(out, dst_settings, dst_client, recreate=True)
                raise AssertionError(f'file={bad_file!r} 인데 복원이 성공함')
            except SnapshotIntegrityError as exc:
                assert '경로' in str(exc), exc
            _assert_target_untouched(dst_client, dst_settings, sentinel)
            dst_client.close()
        finally:
            shutil.rmtree(out, ignore_errors=True)
            if dst_tmp:
                shutil.rmtree(dst_tmp, ignore_errors=True)


def _add_structures(client, settings, rows):
    """(patent_id, mol_id, smiles) 목록을 테스트용 구조 컬렉션에 적재."""
    ensure_structure_collection(client, settings, recreate=True)
    points = []
    for pid, mol_id, smi in rows:
        bits = morgan_bits(smi, settings.morgan_radius, settings.morgan_nbits)
        assert bits, f'{smi} 지문 생성 실패'
        points.append(models.PointStruct(
            id=structure_point_id(pid, mol_id),
            vector={settings.fingerprint_vector_name:
                    models.SparseVector(indices=bits, values=[1.0] * len(bits))},
            payload={'patent_id': pid, 'mol_id': mol_id, 'smiles': smi}))
    client.upsert(settings.structures_collection, points=points, wait=True)


def test_integration_search_cli_field_routing_signature_and_rerank():
    """검색 CLI 경로: --field 반영 / 서명 불일치 차단 / 구조 rerank / 필터 유지."""
    from scripts.qdrant import qdrant_search as qs

    tmp = tempfile.mkdtemp(prefix='chemrag_it_cli_')
    try:
        settings, client, retriever = _build(tmp)
        pf = build_prior_art_filter('Q1', '20200101', 'FAM_Q')

        # --- 1. 적재 인코더 == 검색 인코더면 통과, 다르면 검색 전에 중단 ---
        for field in TEXT_FIELDS:
            chk = qs.verify_encoder_signature(client, settings, field, HashingDenseEncoder(dim=64))
            assert chk['match'] and chk['indexed'] == chk['search'], chk
        try:
            qs.verify_encoder_signature(client, settings, 'title_claims',
                                        HashingDenseEncoder(dim=128))
            raise AssertionError('인코더 불일치가 차단되지 않음')
        except qs.SearchSetupError as exc:
            assert '인코더 불일치' in str(exc), exc

        # --- 2. --field 가 실제로 다른 벡터를 쓰는지 (BM25) ---
        # 'relevant' 는 OLD_HIT 의 title 에만, 'film' 은 OLD_HIT 의 claims 에만 있다.
        tc_hits, _ = retriever.search('bm25', 'relevant', pf, limit=10, field='title_claims')
        assert [h['patent_id'] for h in tc_hits] == ['OLD_HIT'], tc_hits
        cl_hits, _ = retriever.search('bm25', 'relevant', pf, limit=10, field='claims')
        assert cl_hits == [], cl_hits          # claims 벡터에는 title 토큰이 없어야 한다
        film_hits, _ = retriever.search('bm25', 'film', pf, limit=10, field='claims')
        assert [h['patent_id'] for h in film_hits] == ['OLD_HIT'], film_hits

        # --- 3. bm25 / dense / hybrid 가 필드별로 동작하며 기존 필터를 유지 ---
        for mode in ('bm25', 'dense', 'hybrid'):
            for field in TEXT_FIELDS:
                hits, _ = retriever.search(mode, 'ethylene polymer catalyst composition', pf,
                                           limit=10, field=field)
                ids = [h['patent_id'] for h in hits]
                assert 'Q1' not in ids, f'{mode}/{field} 자기제외 위반: {ids}'
                assert 'SAME_FAMILY' not in ids, f'{mode}/{field} 패밀리 제외 위반: {ids}'
                assert 'FUTURE' not in ids, f'{mode}/{field} 시간 필터 위반: {ids}'
                assert 'EQUAL_DATE' in ids and 'NO_DATE' in ids, f'{mode}/{field}: {ids}'

        # --- 4. 구조 rerank 가 선택한 필드의 결과 위에서 동작 ---
        _add_structures(client, settings, [('Q1', 'M1', 'CCO'), ('OLD_HIT', 'M2', 'CCO'),
                                           ('UNRELATED', 'M3', 'c1ccccc1')])
        for field in TEXT_FIELDS:
            hits, _ = retriever.search('bm25', 'ethylene polymer catalyst composition', pf,
                                       limit=10, field=field)
            reranked, stats = qs.apply_structure_rerank(client, settings, hits, 'Q1')
            ids = [h['patent_id'] for h in reranked]
            scored = {h['patent_id']: h.get('tanimoto_score') for h in reranked}
            assert stats['reranked'] == len(hits) and stats['with_structure'] >= 1, stats
            assert scored.get('OLD_HIT') == 1.0, scored     # 같은 SMILES -> Tanimoto 1.0
            assert 'Q1' not in ids and 'SAME_FAMILY' not in ids and 'FUTURE' not in ids, ids
        client.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_integration_snapshot_restore_server_mode_full_verification():
    """서버 모드 전용 v2 스냅샷 복원 검증 - 이 항목만 실행 중인 Qdrant 서버가 있을 때
    동작한다. 이 세션/환경에는 서버 바이너리가 없고 지시에 따라 새로 받지 않았으므로
    SKIP 되며, run_tests.py 러너가 SKIP 사유를 그대로 출력한다(실패로 위장하지 않음).

    CHEMRAG_TEST_QDRANT_SERVER_URL 환경변수로 실행 중인 서버를 가리키면(예:
    http://localhost:6333) 실제로 검증한다: payload 인덱스가 서버에 실제로 걸리는지
    (verify_live_payload_schema로 재조회), alias가 실제로 존재하고 대상이 맞는지,
    restore_snapshot()의 all_ok가 서버 모드 엄격 기준으로 참인지."""
    import uuid as _uuid
    server_url = os.environ.get('CHEMRAG_TEST_QDRANT_SERVER_URL')
    if not server_url:
        raise FileNotFoundError(
            'CHEMRAG_TEST_QDRANT_SERVER_URL 환경변수가 설정되지 않음 - 이 세션/환경에는 '
            'Qdrant 서버 바이너리가 없고, 지시에 따라 새로 다운로드하지 않았습니다. '
            '실행 중인 Qdrant 서버가 있으면 예: '
            'CHEMRAG_TEST_QDRANT_SERVER_URL=http://localhost:6333 로 설정하고 다시 실행하세요.')
    from qdrant_client import QdrantClient
    try:
        probe = QdrantClient(url=server_url, timeout=5)
        probe.get_collections()
        probe.close()
    except Exception as exc:
        raise FileNotFoundError(f'{server_url} 에 연결할 수 없음 - 서버가 떠있지 않은 것으로 '
                                f'보입니다: {type(exc).__name__}: {exc}')

    from qdrant_engine.snapshot import export_collection, write_manifest, restore_snapshot

    suffix = _uuid.uuid4().hex[:8]
    tmp1 = tempfile.mkdtemp(prefix='chemrag_it_server_src_')
    out = tempfile.mkdtemp(prefix='chemrag_it_server_out_')
    settings2 = load_settings()
    settings2.url = server_url
    settings2.patents_collection = f'test_patents_server_{suffix}'
    settings2.structures_collection = f'test_structures_server_{suffix}'
    patents_alias = f'test_patents_server_alias_{suffix}'
    client2 = QdrantClient(url=server_url, timeout=30)
    try:
        settings1, client1, _ = _build(tmp1)
        _add_structures(client1, settings1, [('Q1', 'M1', 'CCO'), ('OLD_HIT', 'M2', 'CCO')])
        meta_p = export_collection(client1, settings1.patents_collection, out, role='patents',
                                   alias=patents_alias)
        meta_s = export_collection(client1, settings1.structures_collection, out, role='structures')
        # 복원 대상 컬렉션 이름으로 manifest를 바꿔친다(원본 임시 컬렉션 이름이 아니라
        # 서버에 새로 만들 이름으로 복원해야 하므로).
        meta_p['collection'] = settings2.patents_collection
        meta_s['collection'] = settings2.structures_collection
        mpath, manifest = write_manifest(out, [meta_p, meta_s], settings1, 'test-server')
        client1.close()

        report = restore_snapshot(out, settings2, client2, recreate=True)
        assert report['restore_mode'] == 'server', report['restore_mode']
        for c in report['collections']:
            assert c['points_match'] and c['point_id_set_match'] and c['vector_schema_match'], c
            assert not c['payload_indexes']['failed'], c['payload_indexes']
            assert 'live_schema_check' in c['payload_indexes'], \
                '서버 모드인데 실측 payload schema 대조가 기록되지 않음'
            assert all(v['match'] for v in c['payload_indexes']['live_schema_check'].values()), \
                c['payload_indexes']['live_schema_check']
            assert c['payload_indexes_ok'], c['payload_indexes']
        patents_c = next(c for c in report['collections']
                         if c['collection'] == settings2.patents_collection)
        assert patents_c['alias']['status'] == 'created', patents_c['alias']
        assert patents_c['alias']['verified_exists'] is True, patents_c['alias']
        assert patents_c['alias']['verified_target_matches'] is True, patents_c['alias']
        assert patents_c['alias_ok']
        assert report['all_ok'], report
    finally:
        for name in (settings2.patents_collection, settings2.structures_collection):
            try:
                client2.delete_collection(name)
            except Exception:
                pass
        client2.close()
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
