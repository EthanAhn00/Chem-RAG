"""Create Qdrant collections and aliases.
Usage: python3 scripts/qdrant/qdrant_create_collections.py [--recreate]. --recreate deletes existing collections before creation.
Embedded mode warns that payload indexes and aliases may not take effect; report this explicitly. Server mode uses CHEMRAG_QDRANT_URL and applies them.
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
    ap.add_argument('--recreate', action='store_true', help='Delete existing collections and recreate them')
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
            'Embedded mode does not create payload indexes and uses brute-force scans. '
            'Actual indexes apply in server mode (CHEMRAG_QDRANT_URL).')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
