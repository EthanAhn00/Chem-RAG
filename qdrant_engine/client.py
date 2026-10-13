"""Qdrant client factory and collection/alias management. Use server mode when a URL is configured, otherwise local embedded storage. Both expose the same API to the remaining code."""
import os
import warnings

from qdrant_client import QdrantClient, models

from .config import load_settings
from .schemas import (
    PATENT_PAYLOAD_INDEXES, STRUCTURE_PAYLOAD_INDEXES,
    patent_vectors_config, patent_sparse_config, structure_sparse_config,
)


class LocalStorageUnusableError(RuntimeError):
    """Local embedded storage is unavailable, e.g. SQLite is unsupported by the filesystem."""


def check_local_storage_writable(path):
    """Check whether a path supports local embedded storage. SQLite locking can fail on network/FUSE mounts with 'disk I/O error'. Fail explicitly instead of leaving an empty directory that could be mistaken for successful indexing. Return (ok: bool, detail: str)."""
    import sqlite3
    import tempfile
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        return False, f'Failed to create directory: {exc}'
    fd, probe = tempfile.mkstemp(prefix='.qdrant_sqlite_probe_', suffix='.db', dir=path)
    os.close(fd)
    try:
        con = sqlite3.connect(probe)
        con.execute('CREATE TABLE IF NOT EXISTS probe (id TEXT PRIMARY KEY, v BLOB)')
        con.execute("INSERT OR REPLACE INTO probe VALUES ('a', ?)", (b'x' * 64,))
        con.commit()
        con.close()
        return True, 'SQLite writes/locking OK'
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


def get_client(settings=None, skip_storage_check=False):
    """Create a server or local embedded client from settings."""
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
                f"Local Qdrant storage is unavailable: {settings.local_path}\n"
                f"  Cause: {detail}\n"
                f"  This filesystem does not support SQLite file locking"
                f" (common on network/FUSE mounts).\n"
                f"  Use a local disk path (CHEMRAG_QDRANT_LOCAL_PATH)"
                f" or server mode (CHEMRAG_QDRANT_URL).\n"
                f"  A snapshot exported on another machine can be restored with"
                f" scripts/qdrant_restore_snapshot.py.")
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
    """Attempt payload index creation. Local mode may only warn without creating indexes; report success/no-effect explicitly rather than implying success."""
    created, no_effect, failed = [], [], []
    for field_name, schema in index_specs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                client.create_payload_index(collection, field_name, field_schema=schema)
            # Embedded mode accepts the call but does not actually create payload indexes.
            # Classify the outcome by mode instead of implying successful index creation.
            (no_effect if local_mode else created).append(field_name)
        except Exception as exc:  # For example, an index already exists.
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
            vectors_config={},                       # No dense vectors; use sparse fingerprints only.
            sparse_vectors_config=structure_sparse_config(),
        )
    created, no_effect, failed = create_payload_indexes(
        client, name, STRUCTURE_PAYLOAD_INDEXES, local_mode=is_local_mode(settings))
    return {'collection': name, 'payload_indexes_created': created,
            'payload_indexes_no_effect_local_mode': no_effect, 'payload_indexes_failed': failed}


_create_payload_indexes = create_payload_indexes  # Backward-compatible aliases.


def verify_alias(client, alias_name, expected_collection):
    """An alias API call returning without error does not prove an alias was created; some backends/versions have no effect. Re-read aliases to verify existence and target. Return {'exists': bool, 'target_collection': str|None, 'matches_expected': bool, 'verify_error': str if present}."""
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
    """Link alias -> collection, then re-read aliases and report verified existence/target matching. An API call without exception is not sufficient evidence of an actual alias."""
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
    """Resolve an alias to its collection; otherwise return the input unchanged."""
    try:
        for a in client.get_collection_aliases(alias_or_name).aliases:
            if a.alias_name == alias_or_name:
                return a.collection_name
    except Exception:
        pass
    return alias_or_name
