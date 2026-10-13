"""Integration tests create temporary collections and perform actual retrieval. They use fixture data in temporary storage, not production collections, and therefore do not skip merely because a production index is absent.
Usage: python3 tests/test_integration.py.
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
    settings.url = ''                       # Integration tests always use temporary local storage.
    settings.local_path = tmpdir
    settings.patents_collection = 'test_patents_integration'
    settings.structures_collection = 'test_structures_integration'
    settings.dense_backend = 'hashing'      # Deterministic encoder without model files.
    settings.dense_dim = 64

    ok, detail = check_local_storage_writable(tmpdir)
    assert ok, f'Temporary storage unavailable: {detail}'

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
                     # Give CLI signature checks the same payload shape used by actual indexing.
                     'embedding_signature_by_field': {f: dense.signature() for f in TEXT_FIELDS}}))
    client.upsert(settings.patents_collection, points=points, wait=True)
    return settings, client, PatentRetriever(client, settings,
                                             {f: dense for f in TEXT_FIELDS}, sparse)


def test_integration_search_modes_and_filters():
    tmp = tempfile.mkdtemp(prefix='chemrag_it_')
    try:
        settings, client, retriever = _build(tmp)
        assert client.count(settings.patents_collection).count == len(FIXTURES)

        # Unfiltered retrieval must be blocked.
        try:
            retriever.search_dense('x', None)
            raise AssertionError('Unfiltered retrieval was not blocked')
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
                # Equality boundary: pub_date == app_date must be included.
                assert 'EQUAL_DATE' in ids, f'{mode}/{field}: equal-date document was excluded'
                # Missing pub_date must not be excluded by time filtering.
                assert 'NO_DATE' in ids, f'{mode}/{field}: document with missing pub_date was excluded'
        assert violations == {'self': 0, 'family': 0, 'future': 0}, violations

        # Idempotence: re-upserting identical points preserves count.
        pts = list(client.scroll(settings.patents_collection, limit=10,
                                 with_payload=True, with_vectors=True)[0])
        before = client.count(settings.patents_collection).count
        client.upsert(settings.patents_collection, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in pts], wait=True)
        assert client.count(settings.patents_collection).count == before

        # Verify relevant documents outrank unrelated ones.
        # BM25 omits documents with zero query-token overlap (UNRELATED),
        # so require either absence or a lower rank.
        hits, _ = retriever.search('bm25', 'ethylene polymer catalyst composition', pf,
                                   limit=10, field='claims')
        ids = [h['patent_id'] for h in hits]
        assert 'OLD_HIT' in ids, ids
        assert ('UNRELATED' not in ids) or (ids.index('OLD_HIT') < ids.index('UNRELATED')), ids

        # Dense retrieval scores all documents, but unrelated documents must rank lower.
        dhits, _ = retriever.search('dense', 'ethylene polymer catalyst composition', pf,
                                    limit=10, field='claims')
        dids = [h['patent_id'] for h in dhits]
        assert 'UNRELATED' in dids and dids.index('OLD_HIT') < dids.index('UNRELATED'), dids
        client.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_integration_snapshot_roundtrip():
    """Export, restore into a separate temporary store, compare with the original, and execute searches on the restored DB via qdrant_engine.snapshot.restore_snapshot(), not merely gzip reading.
    Check patent counts, all point IDs/payloads, claims_dense/title_claims_dense within float32 tolerance, and exact claims_bm25/title_claims_bm25 sparse indexes/values/lengths/names. Check structure counts/IDs/payloads and exact morgan_ecfp4_2048 sparse vectors.
    Verify local index no_effect_local_mode, re-read alias verified_target_matches, and representative claims/title_claims BM25/dense/filter searches plus approximate_candidates structure retrieval. This test covers local expectations only; separate server tests explicitly report skipped prerequisites.
    """
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
        assert meta_p['payload_indexes'], 'patents role must populate payload_indexes'
        assert meta_s['payload_indexes'], 'structures role must populate payload_indexes'
        mpath, manifest = write_manifest(out, [meta_p, meta_s], settings1, 'test')
        assert manifest['manifest_version'] == 2

        # All original points, including complete payloads/vectors for patents and structures.
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
            # Embedded payload indexes are no_effect_local_mode, not failure.
            assert not c['payload_indexes']['failed'], c['payload_indexes']
            assert c['payload_indexes_ok'], c['payload_indexes']
            # Verify alias targets by re-query, not just API success.
            assert c['alias']['status'] == 'created', c['alias']
            assert c['alias']['verified_exists'] is True and c['alias']['verified_target_matches'] is True, c['alias']
            assert c['alias_ok']
            # Verify representative search execution with default verify_search=True.
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

        # --- Patents: all IDs/payloads, tolerant dense, exact sparse vectors ---
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
                # Another float32 storage round trip can cause rounding near the 7th decimal;
                # require closeness within 1e-6, not exact floating-point equality.
                assert all(abs(a - b) < 1e-6 for a, b in zip(ov, rv)), (vn, ov, rv)
            for vn in (claims_bm25_vn, title_bm25_vn):
                _assert_sparse_equal(op.vector[vn], rp.vector[vn], vn)

        # --- Structures: all IDs/payloads and exact sparse fingerprints ---
        restored_struct_points, _ = client2.scroll(settings2.structures_collection, limit=100,
                                                    with_payload=True, with_vectors=True)
        assert len(restored_struct_points) == len(orig_struct_by_id)
        assert {str(p.id) for p in restored_struct_points} == set(orig_struct_by_id)
        fp_vn = settings2.fingerprint_vector_name
        for rp in restored_struct_points:
            op = orig_struct_by_id[str(rp.id)]
            assert rp.payload == op.payload, (rp.id, rp.payload, op.payload)
            _assert_sparse_equal(op.vector[fp_vn], rp.vector[fp_vn], fp_vn)

        # --- Restored claims/title_claims x BM25/dense searches with filters ---
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

        # --- approximate_candidates fingerprint search on restored storage ---
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
    """Export one fixture patent collection for tampering tests; return (out_dir, manifest_path)."""
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
    """Restore target for tampering tests. prepopulate=True inserts existing data so it can be checked unchanged after failure."""
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
    assert collection_exists(client, settings.patents_collection), 'Existing collection disappeared'
    assert client.count(settings.patents_collection).count == 1
    pts, _ = client.scroll(settings.patents_collection, limit=1, with_payload=True)
    assert pts[0].payload == sentinel_payload, 'Existing collection contents changed'


def test_integration_snapshot_restore_rejects_tampered_archive():
    """(1) Alter archive bytes: archive_sha256 mismatch must stop restore and preserve existing collections."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    out = tempfile.mkdtemp(prefix='chemrag_it_tamper_out_')
    dst_tmp = None
    try:
        mpath, manifest, archive_path = _make_snapshot_fixture(out)
        with open(archive_path, 'r+b') as f:
            f.seek(10)
            b = f.read(1)
            f.seek(10)
            f.write(bytes([b[0] ^ 0xFF]))   # Flip one compressed byte.

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('Restore succeeded despite altered archive bytes')
        except SnapshotIntegrityError as exc:
            assert 'archive_sha256' in str(exc) or 'gzip' in str(exc) or 'altered' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_same_length_content_tamper():
    """(2) Change content while retaining row count. content_sha256 must detect it independently of archive_sha256, even if an attacker recomputes the latter."""
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
        # Assume an attacker updates archive_sha256 and size in the manifest;
        # the unchanged content_sha256 must still detect alteration.
        manifest['collections'][0]['archive_sha256'] = sha256_file(archive_path)
        manifest['collections'][0]['archive_bytes'] = os.path.getsize(archive_path)
        with open(mpath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('Restore succeeded after content changed with the same row count')
        except SnapshotIntegrityError as exc:
            assert 'content_sha256' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_point_count_tamper():
    """(3) Alter manifest.points while leaving the file unchanged."""
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
            raise AssertionError('Restore succeeded despite altered point count')
        except SnapshotIntegrityError as exc:
            assert 'line count' in str(exc) or 'points' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_dimension_tamper():
    """(4) Alter manifest vector dimension; detect disagreement with actual vector lengths."""
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
            raise AssertionError('Restore succeeded despite altered vector dimension')
        except SnapshotIntegrityError as exc:
            assert 'vector lengths' in str(exc) or 'dimension' in str(exc) or 'content_sha256' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_missing_file():
    """(5) Missing file."""
    from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
    out = tempfile.mkdtemp(prefix='chemrag_it_tamper_out_')
    dst_tmp = None
    try:
        mpath, manifest, archive_path = _make_snapshot_fixture(out)
        os.remove(archive_path)

        dst_tmp, dst_settings, dst_client, sentinel = _restore_target()
        try:
            restore_snapshot(out, dst_settings, dst_client, recreate=True)
            raise AssertionError('Restore succeeded despite a missing file')
        except SnapshotIntegrityError as exc:
            assert 'file missing' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_malformed_manifest():
    """Malformed manifest structure/missing required keys must be rejected before reading files."""
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
            raise AssertionError('Restore succeeded despite malformed manifest structure')
        except SnapshotIntegrityError as exc:
            assert 'Manifest structure' in str(exc), exc
        _assert_target_untouched(dst_client, dst_settings, sentinel)
        dst_client.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if dst_tmp:
            shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_v2_required_field_deletion():
    """Removing any v2 archive_sha256/content_sha256/archive_bytes field must reject restore and preserve existing collections, guarding against a previous validation-bypass vulnerability."""
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
                raise AssertionError(f'{field_to_delete} removed but v2 restore succeeded')
            except SnapshotIntegrityError as exc:
                assert field_to_delete in str(exc) or 'required fields' in str(exc), (field_to_delete, exc)
            _assert_target_untouched(dst_client, dst_settings, sentinel)
            dst_client.close()
        finally:
            shutil.rmtree(out, ignore_errors=True)
            if dst_tmp:
                shutil.rmtree(dst_tmp, ignore_errors=True)


def test_integration_snapshot_restore_rejects_path_traversal_file():
    """Reject manifest.collections[i].file absolute/../ paths outside the snapshot directory before opening files, preserving existing collections."""
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
                raise AssertionError(f'file={bad_file!r} but restore succeeded')
            except SnapshotIntegrityError as exc:
                assert 'path' in str(exc), exc
            _assert_target_untouched(dst_client, dst_settings, sentinel)
            dst_client.close()
        finally:
            shutil.rmtree(out, ignore_errors=True)
            if dst_tmp:
                shutil.rmtree(dst_tmp, ignore_errors=True)


def _add_structures(client, settings, rows):
    """Load fixture (patent_id, mol_id, smiles) entries into a test structure collection."""
    ensure_structure_collection(client, settings, recreate=True)
    points = []
    for pid, mol_id, smi in rows:
        bits = morgan_bits(smi, settings.morgan_radius, settings.morgan_nbits)
        assert bits, f'{smi} fingerprint generation failed'
        points.append(models.PointStruct(
            id=structure_point_id(pid, mol_id),
            vector={settings.fingerprint_vector_name:
                    models.SparseVector(indices=bits, values=[1.0] * len(bits))},
            payload={'patent_id': pid, 'mol_id': mol_id, 'smiles': smi}))
    client.upsert(settings.structures_collection, points=points, wait=True)


def test_integration_search_cli_field_routing_signature_and_rerank():
    """Search CLI: field selection, signature mismatch rejection, structure reranking, and filter preservation."""
    from scripts.qdrant import qdrant_search as qs

    tmp = tempfile.mkdtemp(prefix='chemrag_it_cli_')
    try:
        settings, client, retriever = _build(tmp)
        pf = build_prior_art_filter('Q1', '20200101', 'FAM_Q')

        # --- 1. Matching indexing/search encoders pass; mismatches stop before search ---
        for field in TEXT_FIELDS:
            chk = qs.verify_encoder_signature(client, settings, field, HashingDenseEncoder(dim=64))
            assert chk['match'] and chk['indexed'] == chk['search'], chk
        try:
            qs.verify_encoder_signature(client, settings, 'title_claims',
                                        HashingDenseEncoder(dim=128))
            raise AssertionError('Encoder mismatch was not blocked')
        except qs.SearchSetupError as exc:
            assert 'Encoder mismatch' in str(exc), exc

        # --- 2. --field actually selects different BM25 vectors ---
        # OLD_HIT has 'relevant' only in its title and 'film' only in its claims.
        tc_hits, _ = retriever.search('bm25', 'relevant', pf, limit=10, field='title_claims')
        assert [h['patent_id'] for h in tc_hits] == ['OLD_HIT'], tc_hits
        cl_hits, _ = retriever.search('bm25', 'relevant', pf, limit=10, field='claims')
        assert cl_hits == [], cl_hits          # Claims vectors must not contain title tokens.
        film_hits, _ = retriever.search('bm25', 'film', pf, limit=10, field='claims')
        assert [h['patent_id'] for h in film_hits] == ['OLD_HIT'], film_hits

        # --- 3. Field-specific BM25/dense/hybrid retain existing filters ---
        for mode in ('bm25', 'dense', 'hybrid'):
            for field in TEXT_FIELDS:
                hits, _ = retriever.search(mode, 'ethylene polymer catalyst composition', pf,
                                           limit=10, field=field)
                ids = [h['patent_id'] for h in hits]
                assert 'Q1' not in ids, f'{mode}/{field} self-exclusion violation: {ids}'
                assert 'SAME_FAMILY' not in ids, f'{mode}/{field} family-exclusion violation: {ids}'
                assert 'FUTURE' not in ids, f'{mode}/{field} time-filter violation: {ids}'
                assert 'EQUAL_DATE' in ids and 'NO_DATE' in ids, f'{mode}/{field}: {ids}'

        # --- 4. Structure reranking applies to selected-field results ---
        _add_structures(client, settings, [('Q1', 'M1', 'CCO'), ('OLD_HIT', 'M2', 'CCO'),
                                           ('UNRELATED', 'M3', 'c1ccccc1')])
        for field in TEXT_FIELDS:
            hits, _ = retriever.search('bm25', 'ethylene polymer catalyst composition', pf,
                                       limit=10, field=field)
            reranked, stats = qs.apply_structure_rerank(client, settings, hits, 'Q1')
            ids = [h['patent_id'] for h in reranked]
            scored = {h['patent_id']: h.get('tanimoto_score') for h in reranked}
            assert stats['reranked'] == len(hits) and stats['with_structure'] >= 1, stats
            assert scored.get('OLD_HIT') == 1.0, scored     # Identical SMILES -> Tanimoto 1.0.
            assert 'Q1' not in ids and 'SAME_FAMILY' not in ids and 'FUTURE' not in ids, ids
        client.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_integration_snapshot_restore_server_mode_full_verification():
    """Server-only v2 snapshot restore verification requires a running Qdrant server. This environment lacked a server binary and did not download one, so the runner reports SKIP with its reason.
    Set CHEMRAG_TEST_QDRANT_SERVER_URL, e.g. http://localhost:6333, to verify actual payload indexes through verify_live_payload_schema, alias existence/targets, and strict server-mode restore_snapshot all_ok.
    """
    import uuid as _uuid
    server_url = os.environ.get('CHEMRAG_TEST_QDRANT_SERVER_URL')
    if not server_url:
        raise FileNotFoundError(
            'CHEMRAG_TEST_QDRANT_SERVER_URL is not set; this environment has '
            'no Qdrant server binary and none was downloaded under the instructions. '
            'With a running Qdrant server, set e.g. '
            'CHEMRAG_TEST_QDRANT_SERVER_URL=http://localhost:6333 and rerun.')
    from qdrant_client import QdrantClient
    try:
        probe = QdrantClient(url=server_url, timeout=5)
        probe.get_collections()
        probe.close()
    except Exception as exc:
        raise FileNotFoundError(f'{server_url} is unreachable; the server appears '
                                f'not to be running: {type(exc).__name__}: {exc}')

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
        # Rewrite the manifest with restore-target collection names, not the original
        # temporary names, since new collections must be created on the server.
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
                'Server mode did not record live payload-schema verification'
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
