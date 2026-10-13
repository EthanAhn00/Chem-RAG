"""Export portable Qdrant snapshots. Copying embedded SQLite storage can fail across filesystems; export IDs/vectors/payloads as gzip JSONL and reconstruct via scripts/qdrant/qdrant_restore_snapshot.py instead.
Thin CLI over qdrant_engine/snapshot.py export_collection/write_manifest; tests/test_integration.py round-trip checks call those functions directly, not CLI subprocesses.
Record both archive_sha256 and decompressed content_sha256 to detect transport corruption and content alteration separately.
Usage: python3 scripts/qdrant/qdrant_export_snapshot.py [--out-dir data/qdrant_snapshot].
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import DATA_DIR
from qdrant_engine.config import load_settings
from qdrant_engine.client import get_client
from qdrant_engine.snapshot import export_collection, write_manifest

DEFAULT_OUT = os.path.join(DATA_DIR, 'qdrant_snapshot')

# Choose the collection role/schema to record payload_indexes accurately.
COLLECTION_ROLES = {'patents': 'patents_collection', 'structures': 'structures_collection'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    args = ap.parse_args()

    settings = load_settings()
    client = get_client(settings)
    os.makedirs(args.out_dir, exist_ok=True)

    collections_meta = []
    for role, attr in COLLECTION_ROLES.items():
        name = getattr(settings, attr)
        alias = getattr(settings, attr.replace('_collection', '_alias'), None)
        print(f'Exporting: {name} (role={role})')
        meta = export_collection(client, name, args.out_dir, role=role, alias=alias)
        print(f'  {meta["points"]} points -> {meta["file"]} ({meta["archive_bytes"]/1e6:.1f}MB) '
              f'archive_sha256={meta["archive_sha256"][:16]}... '
              f'content_sha256={meta["content_sha256"][:16]}...')
        collections_meta.append(meta)

    mpath, manifest = write_manifest(args.out_dir, collections_meta, settings, ' '.join(sys.argv))
    print(json.dumps({c['collection']: {'points': c['points'], 'archive_sha256': c['archive_sha256'],
                                        'content_sha256': c['content_sha256']}
                      for c in manifest['collections']}, indent=2))
    print(f'\nSaved: {mpath} (manifest_version={manifest["manifest_version"]})')


if __name__ == '__main__':
    main()
