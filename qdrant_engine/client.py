"""Qdrant 클라이언트 팩토리 + 컬렉션/alias 관리.

URL이 설정되어 있으면 서버 모드, 없으면 로컬 임베디드 모드(파일 경로)를 쓴다.
어느 쪽이든 같은 API로 동작하므로 나머지 코드는 모드를 신경 쓰지 않는다.
"""
import os
import warnings

from qdrant_client import QdrantClient, models

from .config import load_settings
from .schemas import (
    PATENT_PAYLOAD_INDEXES, STRUCTURE_PAYLOAD_INDEXES,
    patent_vectors_config, patent_sparse_config, structure_sparse_config,
)


class LocalStorageUnusableError(RuntimeError):
    """로컬 임베디드 모드 저장소를 쓸 수 없을 때(예: SQLite가 동작하지 않는 파일시스템)."""


def check_local_storage_writable(path):
    """로컬 모드 저장소로 쓸 수 있는 경로인지 미리 확인한다.

    qdrant 로컬 모드는 SQLite를 쓰는데, 일부 네트워크/FUSE 마운트에서는 파일 잠금이
    지원되지 않아 'disk I/O error'가 난다. 그런 경로에 컬렉션 없는 빈 디렉토리만
    남기면 '적재 성공'처럼 보이므로, 미리 검사해서 명확히 실패시킨다.

    반환: (ok: bool, detail: str)
    """
    import sqlite3
    import tempfile
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        return False, f'디렉토리 생성 실패: {exc}'
    fd, probe = tempfile.mkstemp(prefix='.qdrant_sqlite_probe_', suffix='.db', dir=path)
    os.close(fd)
    try:
        con = sqlite3.connect(probe)
        con.execute('CREATE TABLE IF NOT EXISTS probe (id TEXT PRIMARY KEY, v BLOB)')
        con.execute("INSERT OR REPLACE INTO probe VALUES ('a', ?)", (b'x' * 64,))
        con.commit()
        con.close()
        return True, 'sqlite 쓰기/잠금 정상'
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


def get_client(settings=None, skip_storage_check=False):
    """설정에 따라 서버 클라이언트 또는 로컬 임베디드 클라이언트를 만든다."""
    settings = settings or load_settings()
    if settings.url:
        return QdrantClient(
            url=settings.url,
            api_key=settings.api_key(),
            prefer_grpc=settings.prefer_grpc,
            timeout=settings.timeout,
        )
    if not skip_storage_check:
        ok, detail = check_local_storage_writable(settings.local_path)
        if not ok:
            raise LocalStorageUnusableError(
                f"로컬 Qdrant 저장소를 쓸 수 없습니다: {settings.local_path}\n"
                f"  원인: {detail}\n"
                f"  이 경로의 파일시스템이 SQLite 파일 잠금을 지원하지 않습니다"
                f"(네트워크/FUSE 마운트에서 흔함).\n"
                f"  해결: 로컬 디스크 경로를 쓰거나(CHEMRAG_QDRANT_LOCAL_PATH),"
                f" 서버 모드(CHEMRAG_QDRANT_URL)를 사용하세요.\n"
                f"  다른 머신에서 만든 스냅샷이 있다면"
                f" scripts/qdrant_restore_snapshot.py 로 복원할 수 있습니다.")
    os.makedirs(settings.local_path, exist_ok=True)
    return QdrantClient(path=settings.local_path)


def is_local_mode(settings):
    return not bool(settings.url)


def collection_exists(client, name):
    try:
        return client.collection_exists(name)
    except Exception:
        return name in [c.name for c in client.get_collections().collections]


def create_payload_indexes(client, collection, index_specs, local_mode=False):
    """payload 인덱스 생성 시도. 로컬 모드에서는 효과가 없다는 경고만 나오므로
    성공/무효과 여부를 그대로 돌려준다(조용히 성공한 척하지 않는다)."""
    created, no_effect, failed = [], [], []
    for field_name, schema in index_specs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                client.create_payload_index(collection, field_name, field_schema=schema)
            # 로컬 임베디드 모드에서는 호출은 성공하지만 실제 인덱스가 만들어지지 않는다.
            # 성공한 척하지 않기 위해 모드에 따라 분류를 달리한다.
            (no_effect if local_mode else created).append(field_name)
        except Exception as exc:  # 이미 존재하는 인덱스 등
            failed.append(f'{field_name}({type(exc).__name__})')
    return created, no_effect, failed


def ensure_patent_collection(client, settings, dense_dim, recreate=False):
    name = settings.patents_collection
    if recreate and collection_exists(client, name):
        client.delete_collection(name)
    if not collection_exists(client, name):
        client.create_collection(
            name,
            vectors_config=patent_vectors_config(dense_dim),
            sparse_vectors_config=patent_sparse_config(),
        )
    created, no_effect, failed = create_payload_indexes(
        client, name, PATENT_PAYLOAD_INDEXES, local_mode=is_local_mode(settings))
    return {'collection': name, 'payload_indexes_created': created,
            'payload_indexes_no_effect_local_mode': no_effect, 'payload_indexes_failed': failed}


def ensure_structure_collection(client, settings, recreate=False):
    name = settings.structures_collection
    if recreate and collection_exists(client, name):
        client.delete_collection(name)
    if not collection_exists(client, name):
        client.create_collection(
            name,
            vectors_config={},                       # dense 벡터 없음 - 지문 sparse 벡터만 사용
            sparse_vectors_config=structure_sparse_config(),
        )
    created, no_effect, failed = create_payload_indexes(
        client, name, STRUCTURE_PAYLOAD_INDEXES, local_mode=is_local_mode(settings))
    return {'collection': name, 'payload_indexes_created': created,
            'payload_indexes_no_effect_local_mode': no_effect, 'payload_indexes_failed': failed}


_create_payload_indexes = create_payload_indexes  # 하위 호환 별칭


def verify_alias(client, alias_name, expected_collection):
    """alias 생성 API 호출이 성공했다는 것과 실제로 alias가 만들어졌다는 것은 다른 얘기다
    (API가 예외 없이 반환해도 일부 백엔드/버전에서는 실제로 아무 효과가 없을 수 있음).
    alias 목록을 다시 조회해 실제 존재 여부와 대상 컬렉션이 맞는지 확인한다.

    반환: {'exists': bool, 'target_collection': str|None, 'matches_expected': bool,
          'verify_error': str(있으면)}"""
    try:
        aliases = client.get_collection_aliases(expected_collection).aliases
    except Exception as exc:
        return {'exists': False, 'target_collection': None, 'matches_expected': False,
                'verify_error': f'{type(exc).__name__}: {exc}'}
    for a in aliases:
        if a.alias_name == alias_name:
            return {'exists': True, 'target_collection': a.collection_name,
                    'matches_expected': a.collection_name == expected_collection}
    return {'exists': False, 'target_collection': None, 'matches_expected': False}


def ensure_alias(client, alias_name, collection_name):
    """alias -> collection 연결. API 호출 성공만 믿지 않고, 생성을 시도한 뒤 실제 alias
    목록을 다시 조회해 존재/대상 일치 여부를 검증해서 함께 반환한다(API가 예외 없이
    반환했다고 해서 실제로 걸렸다고 단정하지 않기 위함)."""
    try:
        client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=collection_name, alias_name=alias_name)
                )
            ]
        )
        api_status, api_reason = 'created', None
    except Exception as exc:
        api_status, api_reason = 'unsupported', f'{type(exc).__name__}: {exc}'

    verification = verify_alias(client, alias_name, collection_name)
    result = {'alias': alias_name, 'collection': collection_name, 'status': api_status,
              'verified_exists': verification['exists'],
              'verified_target_matches': verification['matches_expected']}
    if api_reason:
        result['reason'] = api_reason
    if verification.get('verify_error'):
        result['verify_error'] = verification['verify_error']
    return result


def resolve_collection(client, alias_or_name):
    """alias가 있으면 실제 컬렉션명으로, 없으면 입력 그대로 돌려준다."""
    try:
        for a in client.get_collection_aliases(alias_or_name).aliases:
            if a.alias_name == alias_or_name:
                return a.collection_name
    except Exception:
        pass
    return alias_or_name
