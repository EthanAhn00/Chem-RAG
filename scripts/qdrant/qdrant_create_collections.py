"""Qdrant 컬렉션 및 alias 생성 스크립트.

    python3 scripts/qdrant/qdrant_create_collections.py
    python3 scripts/qdrant/qdrant_create_collections.py --recreate     # 기존 컬렉션 삭제 후 재생성

로컬 임베디드 모드에서는 payload 인덱스와 alias가 실제로 동작하지 않는다
(qdrant-client가 경고를 냄). 이 스크립트는 그 사실을 숨기지 않고 결과에 그대로 보고한다.
서버 모드(CHEMRAG_QDRANT_URL 설정)에서는 둘 다 정상 적용된다.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qdrant_engine.config import load_settings
from qdrant_engine.client import (
    get_client, is_local_mode, ensure_patent_collection, ensure_structure_collection, ensure_alias)
from embeddings.registry import build_dense_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recreate', action='store_true', help='기존 컬렉션을 삭제하고 새로 만든다')
    ap.add_argument('--dense-backend', default=None)
    ap.add_argument('--dense-model', default=None)
    args = ap.parse_args()

    settings = load_settings(overrides={'dense_backend': args.dense_backend, 'dense_model': args.dense_model})
    encoder = build_dense_encoder(settings)
    client = get_client(settings)

    report = {
        'mode': 'local(embedded)' if is_local_mode(settings) else f'server({settings.url})',
        'dense_encoder': encoder.describe(),
        'patents': ensure_patent_collection(client, settings, encoder.dim, recreate=args.recreate),
        'structures': ensure_structure_collection(client, settings, recreate=args.recreate),
        'aliases': [
            ensure_alias(client, settings.patents_alias, settings.patents_collection),
            ensure_alias(client, settings.structures_alias, settings.structures_collection),
        ],
    }
    if report['patents']['payload_indexes_no_effect_local_mode']:
        report['payload_index_note'] = (
            '로컬 임베디드 모드에서는 payload 인덱스가 생성되지 않고 브루트포스로 동작합니다. '
            '서버 모드(CHEMRAG_QDRANT_URL)에서 실제 인덱스가 적용됩니다.')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
