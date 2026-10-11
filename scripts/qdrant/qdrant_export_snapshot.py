"""Qdrant 컬렉션을 이식 가능한 스냅샷으로 내보낸다.

로컬 임베디드 저장소(SQLite)는 파일시스템에 따라 열리지 않을 수 있어서 그대로
복사하는 것이 안전하지 않다. 대신 포인트(ID/벡터/payload)를 gzip JSONL로 내보내고,
scripts/qdrant/qdrant_restore_snapshot.py 가 어느 머신에서든 동일한 컬렉션을 재구성한다.

핵심 로직(export_collection/write_manifest)은 qdrant_engine/snapshot.py에 있고
이 스크립트는 그 위에 얇은 CLI를 씌운 것뿐이다 - tests/test_integration.py의
roundtrip 테스트도 같은 함수를 직접 호출한다(이 CLI를 subprocess로 감싸 테스트하지 않음).

manifest에는 포인트 수뿐 아니라 압축 파일 자체의 해시(archive_sha256)와 압축 해제한
내용의 해시(content_sha256)를 둘 다 기록한다 - 전송 중 손상(archive)과 내용 변조(content)를
구분해서 검출하기 위함이다.

    python3 scripts/qdrant/qdrant_export_snapshot.py
    python3 scripts/qdrant/qdrant_export_snapshot.py --out-dir data/qdrant_snapshot
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

# 어떤 role(스키마)로 내보낼지 - manifest에 payload_indexes를 정확히 기록하기 위함
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
        print(f'내보내는 중: {name} (role={role})')
        meta = export_collection(client, name, args.out_dir, role=role, alias=alias)
        print(f'  {meta["points"]} points -> {meta["file"]} ({meta["archive_bytes"]/1e6:.1f}MB) '
              f'archive_sha256={meta["archive_sha256"][:16]}... '
              f'content_sha256={meta["content_sha256"][:16]}...')
        collections_meta.append(meta)

    mpath, manifest = write_manifest(args.out_dir, collections_meta, settings, ' '.join(sys.argv))
    print(json.dumps({c['collection']: {'points': c['points'], 'archive_sha256': c['archive_sha256'],
                                        'content_sha256': c['content_sha256']}
                      for c in manifest['collections']}, indent=2))
    print(f'\n저장됨: {mpath} (manifest_version={manifest["manifest_version"]})')


if __name__ == '__main__':
    main()
