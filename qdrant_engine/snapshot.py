"""Qdrant snapshot export/restore core shared by qdrant_export_snapshot.py, qdrant_restore_snapshot.py, and tests/test_integration.py. Tests call functions directly rather than duplicating logic or wrapping CLIs in subprocesses.

Manifest v2 records manifest_version=2, created_at ('YYYY-MM-DDTHH:MM:SS+0000'), provenance, and collections with collection/file/points/archive_bytes/archive_sha256/content_sha256/vectors/sparse_vectors/payload_indexes/alias. archive_bytes is compressed size; archive_sha256 is the full hex64 digest of compressed bytes; content_sha256 hashes raw decompressed JSONL bytes including newlines. vectors maps names to size/distance; sparse_vectors lists names; payload_indexes lists declared field/schema entries; alias is str or None.

Safe restore order enforced by restore_snapshot():
1) manifest structure; 2) file existence; 3) size/hashes; 4) schema validity. Failure raises SnapshotIntegrityError without touching existing collections.
5) delete/replace old collections; 6) create collections; 7) load points; 8) create payload indexes; 9) create aliases; 10) verify final state.

all_ok requires points_match, point_id_set_match, and vector_schema_match in both modes. Local mode accepts non-failed index creation, including the known no_effect_local_mode limitation; aliases require verified_target_matches except explicit unsupported status. Server mode requires no index failures, exact live payload-schema agreement with the manifest, and verified_exists plus verified_target_matches for aliases; API success alone is insufficient. With verify_search=True (default), representative dense/sparse queries must also execute successfully.
"""
import os
import gzip
import json
import time
import hashlib

from qdrant_client import models

from .config import ID_NAMESPACE  # noqa: F401 (reference only; schemas.py owns point-ID policy)
from .schemas import PATENT_PAYLOAD_INDEXES, STRUCTURE_PAYLOAD_INDEXES
from .client import (
    collection_exists, is_local_mode, create_payload_indexes, ensure_alias, verify_alias,
)
from .provenance import runtime_provenance

MANIFEST_VERSION = 2

# Map collection names to declared payload-index schemas by collection role,
# not just settings.patents_collection / structures_collection values.
# This preserves correct manifest indexes for renamed/versioned collections.
INDEX_SPECS_BY_ROLE = {'patents': PATENT_PAYLOAD_INDEXES, 'structures': STRUCTURE_PAYLOAD_INDEXES}


class SnapshotIntegrityError(RuntimeError):
    """Pre-restore manifest/file/hash/schema verification failure. restore_snapshot() has not touched existing collections when this exception is raised; all validation precedes deletion/creation."""


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _schema_type_name(schema):
    """models.PayloadSchemaType.KEYWORD -> 'KEYWORD' for JSON/getattr round-tripping. Do not use str(schema), which returns the lowercase value 'keyword'."""
    return getattr(schema, 'name', None) or str(schema).rsplit('.', 1)[-1].upper()


def _schema_type_from_name(name):
    return getattr(models.PayloadSchemaType, name)


def vector_to_jsonable(vec):
    if not vec:
        return None
    out = {}
    for name, v in vec.items():
        if hasattr(v, 'indices'):     # SparseVector
            out[name] = {'__sparse__': True, 'indices': list(map(int, v.indices)),
                         'values': [float(x) for x in v.values]}
        else:
            out[name] = [float(x) for x in v]
    return out


def jsonable_to_vector(vec):
    if not vec:
        return {}
    out = {}
    for name, v in vec.items():
        if isinstance(v, dict) and v.get('__sparse__'):
            out[name] = models.SparseVector(indices=v['indices'], values=v['values'])
        else:
            out[name] = v
    return out


# --------------------------------------------------------------------------- export
def export_collection(client, name, out_dir, role=None, alias=None, batch=256):
    """Export one collection to {name}.jsonl.gz and return manifest metadata. role='patents'|'structures'|None selects the declared payload-index schema. With None, record an empty list rather than implying live-index coverage; callers should supply a role."""
    os.makedirs(out_dir, exist_ok=True)
    info = client.get_collection(name)
    cfg = info.config.params
    vectors_cfg = {vn: {'size': int(vp.size), 'distance': str(vp.distance).rsplit('.', 1)[-1]}
                  for vn, vp in (cfg.vectors or {}).items()}
    sparse_cfg = sorted((cfg.sparse_vectors or {}).keys())

    path = os.path.join(out_dir, f'{name}.jsonl.gz')
    tmp_path = path + '.tmp'
    count = 0
    content_hash = hashlib.sha256()
    offset = None
    with gzip.open(tmp_path, 'wt', encoding='utf-8') as f:
        while True:
            points, offset = client.scroll(name, limit=batch, offset=offset,
                                           with_payload=True, with_vectors=True)
            for p in points:
                rec = {'id': str(p.id), 'vector': vector_to_jsonable(p.vector), 'payload': p.payload}
                line = json.dumps(rec, ensure_ascii=False) + '\n'
                content_hash.update(line.encode('utf-8'))
                f.write(line)
                count += 1
            if offset is None:
                break
    os.replace(tmp_path, path)   # Replace with the final filename only after writing the complete file safely.

    payload_indexes = []
    if role and role in INDEX_SPECS_BY_ROLE:
        payload_indexes = [{'field': fn, 'schema': _schema_type_name(sc)}
                           for fn, sc in INDEX_SPECS_BY_ROLE[role]]

    return {
        'collection': name, 'role': role, 'file': os.path.basename(path),
        'points': count,
        'archive_bytes': os.path.getsize(path),
        'archive_sha256': sha256_file(path),
        'content_sha256': content_hash.hexdigest(),
        'vectors': vectors_cfg,
        'sparse_vectors': sparse_cfg,
        'payload_indexes': payload_indexes,
        'alias': alias,
    }


def write_manifest(out_dir, collections_meta, settings, command):
    manifest = {
        'manifest_version': MANIFEST_VERSION,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'provenance': runtime_provenance(settings, command),
        'collections': collections_meta,
    }
    path = os.path.join(out_dir, 'manifest.json')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path, manifest


# ------------------------------------------------------------------- Pre-restore verification
# Missing v2 fields could silently bypass integrity checks, e.g. archive_sha256;
# require all of them in v2.
# Legacy v1, before manifest_version existed, required only collection/file;
# retain permissive handling for backward compatibility.
_V2_REQUIRED_COLLECTION_KEYS = (
    'collection', 'role', 'file', 'points', 'archive_bytes',
    'archive_sha256', 'content_sha256', 'vectors', 'sparse_vectors',
    'payload_indexes', 'alias',
)


def _is_valid_hex_digest(v, exact_len=None, min_len=8, max_len=64):
    if not isinstance(v, str) or not v:
        return False
    if any(ch not in '0123456789abcdef' for ch in v.lower()):
        return False
    if exact_len is not None:
        return len(v) == exact_len
    return min_len <= len(v) <= max_len


def _is_safe_snapshot_relative_path(file_val):
    """Reject manifest.collections[i].file paths that are absolute or traverse '../' outside the snapshot directory, preventing arbitrary-file access."""
    if not isinstance(file_val, str) or not file_val:
        return False
    normalized_slashes = file_val.replace('\\', '/')
    if normalized_slashes.startswith('/'):
        return False
    first_seg = normalized_slashes.split('/', 1)[0]
    if ':' in first_seg:               # Defensively reject Windows drive letters such as C:.
        return False
    normalized = os.path.normpath(file_val)
    parts = normalized.split(os.sep)
    if '..' in parts or normalized.startswith('..') or os.path.isabs(normalized):
        return False
    return True


def validate_manifest_structure(manifest):
    """Return accumulated problems (empty means valid) rather than raising immediately. Strict v2 requires all fields, 64-character hexadecimal hashes, archive_bytes>0, unique collection/file/alias names, and paths inside the snapshot directory. This prevents bypassing validation by removing v2 fields. Legacy v1 remains permissive."""
    problems = []
    if not isinstance(manifest, dict):
        return ['manifest root is not a dictionary']
    # No manifest_version means legacy v1, not failure merely due to absence.
    # Explicit unknown version values are invalid.
    version = manifest.get('manifest_version', 1)
    if version not in (1, 2):
        problems.append(f'Unknown manifest_version: {version!r}')
    is_v2 = version == 2
    colls = manifest.get('collections')
    if not isinstance(colls, list) or not colls:
        problems.append('manifest.collections is empty or not a list')
        return problems

    seen_collections, seen_files, seen_aliases = set(), set(), set()

    for i, c in enumerate(colls):
        if not isinstance(c, dict):
            problems.append(f'collections[{i}] is not a dictionary')
            continue

        if is_v2:
            missing = [k for k in _V2_REQUIRED_COLLECTION_KEYS if k not in c]
            if missing:
                problems.append(
                    f"collections[{i}] (manifest_version=2) missing required fields: {missing} - "
                    f"All these fields are required in v2 to prevent silently bypassing integrity checks")
        else:
            for k in ('collection', 'file'):
                if not c.get(k) or not isinstance(c[k], str):
                    problems.append(f"collections[{i}].{k} missing/invalid type")

        for k in ('collection', 'file'):
            if c.get(k) is not None and not isinstance(c[k], str):
                problems.append(f"collections[{i}].{k} is not a string: {c[k]!r}")

        if 'points' in c or is_v2:
            if not isinstance(c.get('points'), int) or c.get('points', -1) < 0:
                problems.append(f"collections[{i}].points is not a valid integer: {c.get('points')!r}")

        if is_v2:
            if not isinstance(c.get('archive_bytes'), int) or c.get('archive_bytes', 0) <= 0:
                problems.append(
                    f"collections[{i}].archive_bytes is not an integer greater than 0: {c.get('archive_bytes')!r}")
            for hk in ('archive_sha256', 'content_sha256'):
                v = c.get(hk)
                if not _is_valid_hex_digest(v, exact_len=64):
                    problems.append(
                        f"collections[{i}].{hk} is not the 64-character hexadecimal digest required by v2: "
                        f"{v!r} (v2 does not allow missing/truncated hashes)")
            if not isinstance(c.get('payload_indexes'), list):
                problems.append(
                    f"collections[{i}].payload_indexes is not a list: {c.get('payload_indexes')!r}")
            if 'alias' in c and c['alias'] is not None and not isinstance(c['alias'], str):
                problems.append(f"collections[{i}].alias is neither a string nor None: {c['alias']!r}")
        else:
            for hk in ('archive_sha256', 'content_sha256'):
                v = c.get(hk)
                # Legacy v1 stored the first 32 characters of content_sha256;
                # allow hexadecimal digests of length 8~64 for compatibility.
                if v is not None and not _is_valid_hex_digest(v):
                    problems.append(f"collections[{i}].{hk} is not a valid hexadecimal digest: {v!r}")

        if not isinstance(c.get('vectors') or {}, dict):
            problems.append(f'collections[{i}].vectors is not a dictionary')
        if not isinstance(c.get('sparse_vectors') or [], list):
            problems.append(f'collections[{i}].sparse_vectors is not a list')

        file_val = c.get('file')
        if file_val is not None:
            if not _is_safe_snapshot_relative_path(file_val):
                problems.append(
                    f"collections[{i}].file may escape the snapshot directory"
                    f" (absolute path or parent traversal): {file_val!r}")
            elif file_val in seen_files:
                problems.append(f"Duplicate file name: {file_val!r} (collections[{i}])")
            else:
                seen_files.add(file_val)

        coll_name = c.get('collection')
        if coll_name:
            if coll_name in seen_collections:
                problems.append(f"Duplicate collection name: {coll_name!r} (collections[{i}])")
            else:
                seen_collections.add(coll_name)

        alias_val = c.get('alias')
        if alias_val:
            if alias_val in seen_aliases:
                problems.append(f"Duplicate alias name: {alias_val!r} (collections[{i}])")
            else:
                seen_aliases.add(alias_val)

    return problems


def verify_snapshot_files(snapshot_dir, manifest):
    """Check (1) file existence, (2) size, (3) archive_sha256, (4) content_sha256. Return problems; an empty list means OK. Assumes validate_manifest_structure has already passed."""
    problems = []
    for c in manifest['collections']:
        path = os.path.join(snapshot_dir, c['file'])
        if not os.path.exists(path):
            problems.append(f"{c['collection']}: file missing ({path})")
            continue
        actual_bytes = os.path.getsize(path)
        expected_bytes = c.get('archive_bytes', c.get('bytes'))   # 'bytes' is the legacy v1 field name.
        if expected_bytes is not None and actual_bytes != expected_bytes:
            problems.append(f"{c['collection']}: file-size mismatch (manifest {expected_bytes} "
                            f"!= actual {actual_bytes})")
            continue     # A size mismatch already indicates alteration; hashing is unnecessary.
        if c.get('archive_sha256'):
            actual_hash = sha256_file(path)
            # Compare only the recorded prefix when a legacy manifest hash is truncated.
            stored = c['archive_sha256']
            if actual_hash[:len(stored)] != stored:
                problems.append(f"{c['collection']}: archive_sha256 mismatch (file altered or "
                                f"corrupted - manifest {stored[:16]}... != "
                                f"actual {actual_hash[:16]}...)")
                continue
        if c.get('content_sha256'):
            h = hashlib.sha256()
            h_legacy = hashlib.sha256()   # Legacy v1 export_snapshot.py hashed line contents without newlines.
                                          # To avoid false content-tampering reports for that historical convention,
                                          # compute both variants and accept a match to either.
            n = 0
            dim_problems = 0
            declared_vectors = c.get('vectors') or {}
            declared_sparse = set(c.get('sparse_vectors') or [])
            try:
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        h.update(line.encode('utf-8'))
                        h_legacy.update(line.rstrip('\n').encode('utf-8'))
                        n += 1
                        # Compare actual vector dimensions/names with manifest declarations.
                        # Hashes detect content changes, but this separate check also catches
                        # alteration of manifest.vectors.size itself by using actual vector lengths.
                        # Keep this independent schema-consistency check.
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            problems.append(f"{c['collection']}: {n} line JSON parsing failed")
                            continue
                        vec = rec.get('vector') or {}
                        for vn, v in vec.items():
                            if isinstance(v, dict) and v.get('__sparse__'):
                                if vn not in declared_sparse:
                                    dim_problems += 1
                            elif vn in declared_vectors:
                                if len(v) != declared_vectors[vn]['size']:
                                    dim_problems += 1
            except OSError as exc:
                problems.append(f"{c['collection']}: gzip decompression failed - {type(exc).__name__}: {exc}")
                continue
            stored_content_hash = c['content_sha256']
            k = len(stored_content_hash)
            if h.hexdigest()[:k] != stored_content_hash and h_legacy.hexdigest()[:k] != stored_content_hash:
                problems.append(f"{c['collection']}: content_sha256 mismatch (content changed - "
                                f"gzip is valid but decompressed content differs from the manifest, "
                                f"{n} lines read)")
            if c.get('points') is not None and n != c['points']:
                problems.append(f"{c['collection']}: line count ({n}) != manifest.points({c['points']}) "
                                f"- content may have changed while preserving row count")
            if dim_problems:
                problems.append(f"{c['collection']}: actual vector lengths differ from manifest.vectors for "
                                f"points: {dim_problems} - the manifest vector dimension may have been "
                                f"altered")
    return problems


def validate_collection_schema(meta):
    """Validate vectors/sparse_vectors/payload_indexes declarations, including typos and unsupported distances."""
    problems = []
    for vn, vp in (meta.get('vectors') or {}).items():
        if not isinstance(vp.get('size'), int) or vp['size'] <= 0:
            problems.append(f"{meta['collection']}.vectors.{vn}.size is invalid: {vp.get('size')!r}")
        if not hasattr(models.Distance, str(vp.get('distance', '')).upper()):
            problems.append(f"{meta['collection']}.vectors.{vn}.distance is unknown: {vp.get('distance')!r}")
    for idx in (meta.get('payload_indexes') or []):
        try:
            _schema_type_from_name(idx['schema'])
        except (AttributeError, KeyError):
            problems.append(f"{meta['collection']}.payload_indexes schema is unknown: {idx!r}")
    return problems


def preflight_check(snapshot_dir, manifest):
    """Run steps 1~4 in order; accumulate problems in SnapshotIntegrityError. Successful return means pre-restore checks passed and collection changes may proceed."""
    problems = validate_manifest_structure(manifest)
    if problems:
        raise SnapshotIntegrityError('Manifest structure validation failed:\n  - ' + '\n  - '.join(problems))
    problems = verify_snapshot_files(snapshot_dir, manifest)
    if problems:
        raise SnapshotIntegrityError('Snapshot file verification failed (existing collections unchanged):\n  - '
                                     + '\n  - '.join(problems))
    schema_problems = []
    for c in manifest['collections']:
        schema_problems += validate_collection_schema(c)
    if schema_problems:
        raise SnapshotIntegrityError('Collection schema validation failed (existing collections unchanged):\n  - '
                                     + '\n  - '.join(schema_problems))


def verify_live_payload_schema(client, collection, index_specs):
    """Compare live server payload schema with manifest declarations. Embedded mode returns an empty schema, already classified as no_effect_local_mode by create_payload_indexes. Interpret strictly only in server mode; an empty local schema is not a failure."""
    info = client.get_collection(collection)
    live_schema = getattr(info, 'payload_schema', None) or {}
    result = {}
    for field_name, schema in index_specs:
        expected_type = getattr(schema, 'name', str(schema))
        live = live_schema.get(field_name)
        live_type = None
        if live is not None:
            dt = getattr(live, 'data_type', None)
            live_type = getattr(dt, 'name', dt)
        result[field_name] = {'declared': expected_type, 'live': live_type,
                              'match': live is not None and live_type == expected_type}
    return result


def _representative_search_check(client, collection, meta):
    """Verify that representative dense/sparse queries execute on restored collections. This checks operability, not score correctness; encoder-dependent correctness belongs to evaluate_qdrant_retrieval.py/qdrant_search.py."""
    checks = {}
    for vn, vp in (meta.get('vectors') or {}).items():
        try:
            client.query_points(collection, query=[0.0] * int(vp['size']), using=vn, limit=1)
            checks[vn] = {'ok': True}
        except Exception as exc:
            checks[vn] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    for sn in (meta.get('sparse_vectors') or []):
        try:
            client.query_points(collection, query=models.SparseVector(indices=[0], values=[1.0]),
                                using=sn, limit=1)
            checks[sn] = {'ok': True}
        except Exception as exc:
            checks[sn] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    return checks


# ------------------------------------------------------------------------- Restore
def restore_snapshot(snapshot_dir, settings, client, collections=None, recreate=True,
                     batch_size=256, verify_search=True):
    """Restore from manifest.json and return a report dictionary. Pre-validation failure raises SnapshotIntegrityError without changing existing collections."""
    mpath = os.path.join(snapshot_dir, 'manifest.json')
    if not os.path.exists(mpath):
        raise SnapshotIntegrityError(f'{mpath} is missing.')
    with open(mpath, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    target = [c for c in manifest.get('collections', [])
             if collections is None or c['collection'] in collections]
    if not target:
        raise SnapshotIntegrityError('No collections to restore are listed in the manifest.')
    manifest = dict(manifest, collections=target)

    # Steps 1~4: failure here leaves all collections unchanged.
    preflight_check(snapshot_dir, manifest)

    local_mode = is_local_mode(settings)
    report = {'snapshot_dir': snapshot_dir, 'local_mode': local_mode, 'collections': []}

    for meta in manifest['collections']:
        name = meta['collection']
        path = os.path.join(snapshot_dir, meta['file'])
        coll_report = {'collection': name, 'expected_points': meta['points']}

        # 5) Delete/replace existing collections.
        existed = collection_exists(client, name)
        if recreate and existed:
            client.delete_collection(name)
            existed = False
        coll_report['recreated'] = bool(recreate and existed is False)

        # 6) Create collections.
        if not collection_exists(client, name):
            vectors_config = {vn: models.VectorParams(size=int(v['size']),
                                                       distance=getattr(models.Distance, v['distance'].upper()))
                              for vn, v in (meta.get('vectors') or {}).items()}
            sparse_config = {sn: models.SparseVectorParams() for sn in (meta.get('sparse_vectors') or [])}
            client.create_collection(name, vectors_config=vectors_config,
                                     sparse_vectors_config=sparse_config)
            coll_report['created'] = True
        else:
            coll_report['created'] = False

        # 7) Load points.
        t0 = time.time()
        buf, total = [], 0
        restored_ids = set()
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                restored_ids.add(rec['id'])
                buf.append(models.PointStruct(id=rec['id'], vector=jsonable_to_vector(rec.get('vector')),
                                              payload=rec.get('payload')))
                if len(buf) >= batch_size:
                    client.upsert(name, points=buf, wait=True)
                    total += len(buf)
                    buf = []
        if buf:
            client.upsert(name, points=buf, wait=True)
            total += len(buf)
        coll_report['upserted'] = total
        coll_report['load_seconds'] = round(time.time() - t0, 2)

        # 8) Create payload indexes; distinguish embedded and server outcomes.
        # Local: accept unless the API call failed; no_effect_local_mode
        # is a known limitation, neither a failure nor actual index creation.
        # Server: require no failures and exact live-schema agreement with the manifest;
        # distinguish API success from verified creation.
        index_specs = [(idx['field'], _schema_type_from_name(idx['schema']))
                       for idx in (meta.get('payload_indexes') or [])]
        created, no_effect, failed = create_payload_indexes(client, name, index_specs,
                                                             local_mode=local_mode)
        coll_report['payload_indexes'] = {'created': created,
                                          'no_effect_local_mode': no_effect, 'failed': failed}
        if not local_mode and index_specs:
            live_schema_check = verify_live_payload_schema(client, name, index_specs)
            coll_report['payload_indexes']['live_schema_check'] = live_schema_check
            payload_indexes_ok = (not failed) and all(v['match'] for v in live_schema_check.values())
        else:
            payload_indexes_ok = not failed
        coll_report['payload_indexes_ok'] = payload_indexes_ok

        # 9) Create aliases only if listed in the manifest; ensure_alias re-reads them
        # and returns verified_exists / verified_target_matches.
        # Accept status='unsupported' only as a genuine embedded-backend limitation;
        # otherwise follow actual verification results in either mode.
        if meta.get('alias'):
            coll_report['alias'] = ensure_alias(client, meta['alias'], name)
            if local_mode and coll_report['alias']['status'] == 'unsupported':
                alias_ok = True
            else:
                alias_ok = (coll_report['alias'].get('verified_exists') is True
                           and coll_report['alias'].get('verified_target_matches') is True)
        else:
            coll_report['alias'] = {'status': 'not_configured'}
            alias_ok = True
        coll_report['alias_ok'] = alias_ok

        # 10) Verify final state.
        actual_count = client.count(name).count
        coll_report['actual_points'] = actual_count
        coll_report['points_match'] = actual_count == meta['points']

        live_ids = set()
        offset = None
        while True:
            pts, offset = client.scroll(name, limit=1000, offset=offset, with_payload=False,
                                        with_vectors=False)
            live_ids.update(str(p.id) for p in pts)
            if offset is None:
                break
        coll_report['point_id_set_match'] = live_ids == restored_ids
        coll_report['point_id_mismatch_count'] = len(live_ids ^ restored_ids)

        info = client.get_collection(name)
        cfg = info.config.params
        live_vectors = {vn: {'size': int(vp.size), 'distance': str(vp.distance).rsplit('.', 1)[-1]}
                        for vn, vp in (cfg.vectors or {}).items()}
        live_sparse = sorted((cfg.sparse_vectors or {}).keys())
        coll_report['vector_schema_match'] = (live_vectors == (meta.get('vectors') or {}) and
                                              live_sparse == sorted(meta.get('sparse_vectors') or []))

        sample, _ = client.scroll(name, limit=min(5, meta['points']) or 0, with_payload=True,
                                  with_vectors=False)
        coll_report['sample_payload_checked'] = len(sample)

        # Execute representative searches only when verify_search=True.
        # If disabled, treat this check as passed without affecting all_ok.
        if verify_search:
            search_checks = _representative_search_check(client, name, meta)
            coll_report['representative_search'] = search_checks
            coll_report['representative_search_ok'] = (
                all(v['ok'] for v in search_checks.values()) if search_checks else True)
        else:
            coll_report['representative_search_ok'] = True

        report['collections'].append(coll_report)

    report['restore_mode'] = 'local_embedded' if local_mode else 'server'
    ok = all(c['points_match'] and c['point_id_set_match'] and c['vector_schema_match']
             and c['payload_indexes_ok'] and c['alias_ok'] and c['representative_search_ok']
             for c in report['collections'])
    report['all_ok'] = ok
    return report
