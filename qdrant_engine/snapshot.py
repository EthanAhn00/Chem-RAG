"""Qdrant 스냅샷 export/restore 핵심 로직 - CLI(qdrant_export_snapshot.py /
qdrant_restore_snapshot.py)와 테스트(tests/test_integration.py)가 이 모듈의 함수를
그대로 호출한다(로직 중복을 피하기 위함 - CLI를 subprocess로 감싸 테스트하지 않는다).

manifest 스키마(v2)
-------------------
{
  'manifest_version': 2,
  'created_at': 'YYYY-MM-DDTHH:MM:SS+0000',
  'provenance': {...},                     # qdrant_engine.provenance.runtime_provenance()
  'collections': [
    {
      'collection': str,
      'file': 'name.jsonl.gz',
      'points': int,
      'archive_bytes': int,                # .gz 파일 자체의 크기
      'archive_sha256': 'hex64',           # .gz 파일 바이트 그대로의 sha256(전체, 잘리지 않음)
      'content_sha256': 'hex64',           # 압축 해제한 raw jsonl(파일에 실제로 쓰인 바이트, 개행 포함)의 sha256
      'vectors': {name: {'size': int, 'distance': str}},   # dense named vectors
      'sparse_vectors': [name, ...],
      'payload_indexes': [{'field': str, 'schema': str}],  # 이 컬렉션에 선언된 인덱스(스키마 정의 기준)
      'alias': str | None,
    }, ...
  ],
}

복원 순서(안전) - restore_snapshot()이 강제한다
  1) manifest 구조 검증           2) 스냅샷 파일 존재 확인
  3) 파일 크기/해시 확인           4) 컬렉션 스키마 유효성 확인
  ↑ 여기까지 전부 통과해야만 아래로 진행 - 하나라도 실패하면 기존 컬렉션을 전혀 건드리지 않고
    SnapshotIntegrityError 를 던진다.
  5) 기존 컬렉션 삭제/교체         6) 컬렉션 생성        7) 포인트 적재
  8) payload 인덱스 생성          9) alias 생성        10) 최종 상태 검증

all_ok 판정 기준 (로컬 임베디드 모드 vs 서버 모드를 다르게 본다)
  공통: points_match, point_id_set_match, vector_schema_match
  로컬 임베디드 모드: payload 인덱스는 create_payload_indexes() 호출이 실패(failed)하지
    않으면 통과 - no_effect_local_mode는 이 모드의 알려진 제약이지 실패가 아니다.
    alias는 ensure_alias()가 재조회로 검증한 verified_target_matches를 따른다
    (status='unsupported'인 경우만 모드 제약으로 인정).
  서버 모드: payload_indexes.failed가 비어 있어야 하고, verify_live_payload_schema()로
    실제 조회한 payload schema가 manifest 선언과 정확히 일치해야 한다. alias는
    verified_exists와 verified_target_matches가 모두 True여야 한다(API 호출 성공만으로는
    통과시키지 않는다).
  verify_search=True(기본)면 대표 dense/sparse 질의가 실제로 실행 가능한지도 확인해서
  all_ok에 반영한다.
"""
import os
import gzip
import json
import time
import hashlib

from qdrant_client import models

from .config import ID_NAMESPACE  # noqa: F401 (참고용 - 포인트 ID 정책은 schemas.py 소관)
from .schemas import PATENT_PAYLOAD_INDEXES, STRUCTURE_PAYLOAD_INDEXES
from .client import (
    collection_exists, is_local_mode, create_payload_indexes, ensure_alias, verify_alias,
)
from .provenance import runtime_provenance

MANIFEST_VERSION = 2

# 컬렉션 이름 -> 그 컬렉션에 선언된 payload 인덱스 스펙. settings.patents_collection /
# structures_collection 값 자체가 아니라 '어떤 스키마를 쓰는 컬렉션인지'로 매핑해야
# 컬렉션 이름이 바뀌어도(별칭/버전 접미사 등) manifest가 올바른 인덱스를 기록한다.
INDEX_SPECS_BY_ROLE = {'patents': PATENT_PAYLOAD_INDEXES, 'structures': STRUCTURE_PAYLOAD_INDEXES}


class SnapshotIntegrityError(RuntimeError):
    """manifest 구조, 파일 존재, 해시, 스키마 중 하나라도 복원 전 검증에 실패했을 때.

    이 예외가 나면 restore_snapshot()은 기존 컬렉션을 전혀 건드리지 않았음이 보장된다
    (모든 검증은 실제 삭제/생성 이전 단계에서 끝난다).
    """


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
    """models.PayloadSchemaType.KEYWORD -> 'KEYWORD' (JSON에 안전하게 저장, getattr로 왕복 가능한
    이름을 써야 한다 - str(schema)는 'keyword'(소문자 value)를 주므로 쓰면 안 됨)."""
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
    """컬렉션 하나를 {name}.jsonl.gz 로 내보내고 manifest용 메타를 반환한다.

    role: 'patents' | 'structures' | None - payload_indexes를 manifest에 기록하기 위한
    스키마 선택 키. None이면 실제 라이브 컬렉션에 걸린 인덱스 정보를 최대한 반영하지 못하므로
    빈 리스트로 기록한다(호출자가 role을 주는 것을 권장).
    """
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
    os.replace(tmp_path, path)   # 파일이 끝까지 안전하게 쓰인 뒤에만 최종 이름으로 교체

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


# ------------------------------------------------------------------- 복원 전 검증
# v2 manifest에서 이 필드들 중 하나라도 없으면 무결성 검사를 조용히 우회할 수 있으므로
# (예: archive_sha256이 없으면 그 부분 검사가 그냥 생략됨) v2에서는 전부 필수로 요구한다.
# v1(레거시, manifest_version 키 자체가 없던 시절) 스냅샷은 collection/file만 있으면 됐으므로
# 하위 호환을 위해 계속 관대하게 받는다.
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
    """manifest.collections[i].file 이 스냅샷 디렉토리 밖을 가리키는 절대경로나
    '../' 상위 디렉토리 이동이면 거부한다(경로 조작으로 임의 파일을 읽게 만드는 것을 방지)."""
    if not isinstance(file_val, str) or not file_val:
        return False
    normalized_slashes = file_val.replace('\\', '/')
    if normalized_slashes.startswith('/'):
        return False
    first_seg = normalized_slashes.split('/', 1)[0]
    if ':' in first_seg:               # Windows 드라이브 문자(C:) 등 방어적으로 차단
        return False
    normalized = os.path.normpath(file_val)
    parts = normalized.split(os.sep)
    if '..' in parts or normalized.startswith('..') or os.path.isabs(normalized):
        return False
    return True


def validate_manifest_structure(manifest):
    """problems 리스트를 반환한다(비어있으면 구조 OK). 예외를 던지지 않는다 - 호출자가
    누적된 모든 문제를 한 번에 볼 수 있게 하기 위함.

    v2는 엄격 검증(필수 필드 전부 존재, 해시는 정확히 64자 hex, archive_bytes>0,
    collection/file/alias 중복 금지, file의 경로 이탈 금지)이고, v2 필드를 지우면
    검증을 우회할 수 있었던 이전 취약점을 막는다. v1(레거시)은 예전처럼 관대하게 받는다."""
    problems = []
    if not isinstance(manifest, dict):
        return ['manifest 최상위가 dict가 아닙니다']
    # manifest_version이 아예 없는 건 v1(이전 export_snapshot.py) 스냅샷 - 필드가 없었을 뿐이니
    # 실패로 보지 않는다. 명시적으로 다른 값이 적혀 있는데 우리가 모르는 값이면 그건 문제.
    version = manifest.get('manifest_version', 1)
    if version not in (1, 2):
        problems.append(f'알 수 없는 manifest_version: {version!r}')
    is_v2 = version == 2
    colls = manifest.get('collections')
    if not isinstance(colls, list) or not colls:
        problems.append('manifest.collections 가 비어있거나 리스트가 아닙니다')
        return problems

    seen_collections, seen_files, seen_aliases = set(), set(), set()

    for i, c in enumerate(colls):
        if not isinstance(c, dict):
            problems.append(f'collections[{i}] 가 dict가 아닙니다')
            continue

        if is_v2:
            missing = [k for k in _V2_REQUIRED_COLLECTION_KEYS if k not in c]
            if missing:
                problems.append(
                    f"collections[{i}] (manifest_version=2) 필수 필드 누락: {missing} - "
                    f"v2에서는 이 필드들이 없으면 무결성 검사를 조용히 우회할 수 있어 전부 필수입니다")
        else:
            for k in ('collection', 'file'):
                if not c.get(k) or not isinstance(c[k], str):
                    problems.append(f"collections[{i}].{k} 누락/타입 오류")

        for k in ('collection', 'file'):
            if c.get(k) is not None and not isinstance(c[k], str):
                problems.append(f"collections[{i}].{k} 가 문자열이 아닙니다: {c[k]!r}")

        if 'points' in c or is_v2:
            if not isinstance(c.get('points'), int) or c.get('points', -1) < 0:
                problems.append(f"collections[{i}].points 가 올바른 정수가 아닙니다: {c.get('points')!r}")

        if is_v2:
            if not isinstance(c.get('archive_bytes'), int) or c.get('archive_bytes', 0) <= 0:
                problems.append(
                    f"collections[{i}].archive_bytes 가 0보다 큰 정수가 아닙니다: {c.get('archive_bytes')!r}")
            for hk in ('archive_sha256', 'content_sha256'):
                v = c.get(hk)
                if not _is_valid_hex_digest(v, exact_len=64):
                    problems.append(
                        f"collections[{i}].{hk} 가 v2에서 요구하는 64자리 hex 다이제스트가 아닙니다: "
                        f"{v!r} (v2는 해시 누락/축약을 허용하지 않습니다)")
            if not isinstance(c.get('payload_indexes'), list):
                problems.append(
                    f"collections[{i}].payload_indexes 가 list가 아닙니다: {c.get('payload_indexes')!r}")
            if 'alias' in c and c['alias'] is not None and not isinstance(c['alias'], str):
                problems.append(f"collections[{i}].alias 가 문자열도 None도 아닙니다: {c['alias']!r}")
        else:
            for hk in ('archive_sha256', 'content_sha256'):
                v = c.get(hk)
                # v1 레거시 스냅샷은 content_sha256을 32자로 잘라 저장했었으므로
                # (진짜 sha256의 앞 32자) 8~64자 hex를 폭넓게 허용한다.
                if v is not None and not _is_valid_hex_digest(v):
                    problems.append(f"collections[{i}].{hk} 가 올바른 hex 다이제스트가 아닙니다: {v!r}")

        if not isinstance(c.get('vectors') or {}, dict):
            problems.append(f'collections[{i}].vectors 가 dict가 아닙니다')
        if not isinstance(c.get('sparse_vectors') or [], list):
            problems.append(f'collections[{i}].sparse_vectors 가 list가 아닙니다')

        file_val = c.get('file')
        if file_val is not None:
            if not _is_safe_snapshot_relative_path(file_val):
                problems.append(
                    f"collections[{i}].file 이 스냅샷 디렉토리 밖을 가리킬 수 있는 경로입니다"
                    f"(절대경로 또는 상위 디렉토리 이동 포함): {file_val!r}")
            elif file_val in seen_files:
                problems.append(f"중복된 file 이름: {file_val!r} (collections[{i}])")
            else:
                seen_files.add(file_val)

        coll_name = c.get('collection')
        if coll_name:
            if coll_name in seen_collections:
                problems.append(f"중복된 collection 이름: {coll_name!r} (collections[{i}])")
            else:
                seen_collections.add(coll_name)

        alias_val = c.get('alias')
        if alias_val:
            if alias_val in seen_aliases:
                problems.append(f"중복된 alias 이름: {alias_val!r} (collections[{i}])")
            else:
                seen_aliases.add(alias_val)

    return problems


def verify_snapshot_files(snapshot_dir, manifest):
    """(1) 파일 존재 (2) 크기 (3) archive_sha256 (4) content_sha256 을 확인.
    problems 리스트 반환(비어있으면 OK). manifest 구조는 이미 유효하다고 가정한다
    (validate_manifest_structure를 먼저 호출할 것)."""
    problems = []
    for c in manifest['collections']:
        path = os.path.join(snapshot_dir, c['file'])
        if not os.path.exists(path):
            problems.append(f"{c['collection']}: 파일 없음 ({path})")
            continue
        actual_bytes = os.path.getsize(path)
        expected_bytes = c.get('archive_bytes', c.get('bytes'))   # 'bytes'는 v1 레거시 필드명
        if expected_bytes is not None and actual_bytes != expected_bytes:
            problems.append(f"{c['collection']}: 파일 크기 불일치 (manifest {expected_bytes} "
                            f"!= 실제 {actual_bytes})")
            continue     # 크기부터 다르면 해시 계산은 의미 없음(변조 신호로 충분)
        if c.get('archive_sha256'):
            actual_hash = sha256_file(path)
            # manifest 쪽 해시가 잘려있으면(레거시) 그 길이만큼만 비교한다.
            stored = c['archive_sha256']
            if actual_hash[:len(stored)] != stored:
                problems.append(f"{c['collection']}: archive_sha256 불일치 (파일이 변조됐거나 "
                                f"손상됨 - manifest {stored[:16]}... != "
                                f"실제 {actual_hash[:16]}...)")
                continue
        if c.get('content_sha256'):
            h = hashlib.sha256()
            h_legacy = hashlib.sha256()   # v1 export_snapshot.py는 개행 없이 줄 내용만 해시했다(버그성
                                          # 관례) - 레거시 manifest를 오탐(false content-tamper)으로
                                          # 판정하지 않기 위해 두 관례를 모두 계산해 하나라도 맞으면 통과.
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
                        # 벡터 차원/이름을 manifest 선언과 대조(내용 변조가 해시로는
                        # 이미 잡히지만, 서로 다른 이유로 별도 검증도 남겨둔다 - 예:
                        # manifest의 vectors.size 자체가 조작된 경우 이 라인들이 실제
                        # 벡터 길이를 근거로 그 조작을 잡아낸다).
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            problems.append(f"{c['collection']}: {n}번째 줄 JSON 파싱 실패")
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
                problems.append(f"{c['collection']}: gzip 압축 해제 실패 - {type(exc).__name__}: {exc}")
                continue
            stored_content_hash = c['content_sha256']
            k = len(stored_content_hash)
            if h.hexdigest()[:k] != stored_content_hash and h_legacy.hexdigest()[:k] != stored_content_hash:
                problems.append(f"{c['collection']}: content_sha256 불일치 (내용이 바뀜 - "
                                f"압축은 정상이지만 압축 해제된 내용이 manifest와 다름, "
                                f"{n}줄 읽음)")
            if c.get('points') is not None and n != c['points']:
                problems.append(f"{c['collection']}: 줄 수({n}) != manifest.points({c['points']}) "
                                f"- 행 수는 같고 내용만 바뀐 변조일 수 있음")
            if dim_problems:
                problems.append(f"{c['collection']}: 실제 벡터 길이가 manifest.vectors 선언과 "
                                f"다른 포인트 {dim_problems}건 - manifest의 vector dimension이 "
                                f"조작됐을 수 있음")
    return problems


def validate_collection_schema(meta):
    """vectors/sparse_vectors/payload_indexes 선언 자체의 유효성(오탈자, 지원하지 않는 distance 등)."""
    problems = []
    for vn, vp in (meta.get('vectors') or {}).items():
        if not isinstance(vp.get('size'), int) or vp['size'] <= 0:
            problems.append(f"{meta['collection']}.vectors.{vn}.size 가 올바르지 않습니다: {vp.get('size')!r}")
        if not hasattr(models.Distance, str(vp.get('distance', '')).upper()):
            problems.append(f"{meta['collection']}.vectors.{vn}.distance 를 알 수 없습니다: {vp.get('distance')!r}")
    for idx in (meta.get('payload_indexes') or []):
        try:
            _schema_type_from_name(idx['schema'])
        except (AttributeError, KeyError):
            problems.append(f"{meta['collection']}.payload_indexes 스키마를 알 수 없습니다: {idx!r}")
    return problems


def preflight_check(snapshot_dir, manifest):
    """1~4단계를 순서대로 실행. 문제가 있으면 SnapshotIntegrityError(모든 문제 취합)를 던진다.
    이 함수가 예외 없이 반환하면 그 뒤로는 기존 컬렉션을 안전하게 변경해도 된다는 뜻이다."""
    problems = validate_manifest_structure(manifest)
    if problems:
        raise SnapshotIntegrityError('manifest 구조 검증 실패:\n  - ' + '\n  - '.join(problems))
    problems = verify_snapshot_files(snapshot_dir, manifest)
    if problems:
        raise SnapshotIntegrityError('스냅샷 파일 검증 실패(기존 컬렉션은 변경하지 않았습니다):\n  - '
                                     + '\n  - '.join(problems))
    schema_problems = []
    for c in manifest['collections']:
        schema_problems += validate_collection_schema(c)
    if schema_problems:
        raise SnapshotIntegrityError('컬렉션 스키마 검증 실패(기존 컬렉션은 변경하지 않았습니다):\n  - '
                                     + '\n  - '.join(schema_problems))


def verify_live_payload_schema(client, collection, index_specs):
    """서버에서 실제 조회한 payload schema가 manifest 선언과 일치하는지 확인한다.
    로컬 임베디드 모드에서는 payload_schema가 항상 비어 있다(효과 없음 - 이미
    create_payload_indexes()가 no_effect_local_mode로 분류함). 이 함수는 서버 모드에서만
    엄격하게 해석해야 하고, 로컬 모드에서 빈 결과가 나오는 것은 실패가 아니다."""
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
    """복원된 컬렉션에서 대표 dense/sparse 질의가 실제로 실행 가능한지 확인한다.
    점수의 정확성이 아니라 '검색이 동작하는가'만 본다(정확성은 인코더가 필요한
    evaluate_qdrant_retrieval.py/qdrant_search.py의 몫이고, 여기서는 인코더 없이도
    복원된 스키마로 검색이 실행되는지만 확인한다)."""
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


# ------------------------------------------------------------------------- 복원
def restore_snapshot(snapshot_dir, settings, client, collections=None, recreate=True,
                     batch_size=256, verify_search=True):
    """manifest.json 기준으로 스냅샷을 복원한다. 반환: report dict.
    실패(사전 검증 단계)는 SnapshotIntegrityError를 던지며 이때 기존 컬렉션은 손대지 않는다.
    """
    mpath = os.path.join(snapshot_dir, 'manifest.json')
    if not os.path.exists(mpath):
        raise SnapshotIntegrityError(f'{mpath} 가 없습니다.')
    with open(mpath, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    target = [c for c in manifest.get('collections', [])
             if collections is None or c['collection'] in collections]
    if not target:
        raise SnapshotIntegrityError('복원 대상 컬렉션이 manifest에 없습니다.')
    manifest = dict(manifest, collections=target)

    # 1~4단계: 여기서 실패하면 아래 어떤 컬렉션도 건드리지 않는다.
    preflight_check(snapshot_dir, manifest)

    local_mode = is_local_mode(settings)
    report = {'snapshot_dir': snapshot_dir, 'local_mode': local_mode, 'collections': []}

    for meta in manifest['collections']:
        name = meta['collection']
        path = os.path.join(snapshot_dir, meta['file'])
        coll_report = {'collection': name, 'expected_points': meta['points']}

        # 5) 기존 컬렉션 삭제/교체
        existed = collection_exists(client, name)
        if recreate and existed:
            client.delete_collection(name)
            existed = False
        coll_report['recreated'] = bool(recreate and existed is False)

        # 6) 컬렉션 생성
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

        # 7) 포인트 적재
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

        # 8) payload 인덱스 생성 - 로컬 임베디드 모드와 서버 모드를 다르게 판정한다.
        #    로컬: API 호출 자체가 실패(failed)하지 않는 한 통과(no_effect_local_mode는
        #    알려진 모드 제약이지 실패가 아니다 - 실패로 위장하지도, 성공으로 위장하지도 않는다).
        #    서버: failed가 비어있어야 하고, 실제로 조회한 payload schema가 manifest 선언과
        #    정확히 일치해야 한다(API 호출 성공과 실제 생성 확인을 구분).
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

        # 9) alias 생성(manifest에 기록된 경우만) - ensure_alias가 API 호출 후 실제
        #    재조회까지 해서 verified_exists/verified_target_matches를 채워 돌려준다.
        #    로컬 모드에서 alias API가 정말 지원되지 않는 백엔드라면 status='unsupported'를
        #    모드 제약으로 인정하지만, 그 외에는(로컬이든 서버든) 실제 검증 결과를 따른다.
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

        # 10) 최종 상태 검증
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

        # verify_search=True일 때만 대표 검색을 실제로 실행해 확인한다(인자를 받아만 놓고
        # 쓰지 않던 부분 - 사용하지 않으면 이 검사는 통과로 취급하고 all_ok에 영향을 주지 않는다).
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
