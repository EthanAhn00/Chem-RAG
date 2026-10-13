"""Safely restore portable Qdrant collections from snapshots.
Order in qdrant_engine/snapshot.restore_snapshot:
1) manifest structure; 2) file existence; 3) archive/content size/hashes; 4) schema validity. Failure raises SnapshotIntegrityError, leaves existing collections unchanged, and exits 1.
5) delete old collections with --recreate; 6) create; 7) load points; 8) payload indexes; 9) aliases; 10) verify point counts/IDs/vector schema.
Embedded indexes/aliases may have no effect; record no_effect_local_mode/unsupported honestly, not as failure or verified creation.
Usage: python3 scripts/qdrant/qdrant_restore_snapshot.py [--snapshot-dir data/qdrant_snapshot --recreate].
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import DATA_DIR
from qdrant_engine.config import load_settings
from qdrant_engine.client import get_client
from qdrant_engine.snapshot import restore_snapshot, SnapshotIntegrityError
# Re-export names previously imported here for backward compatibility.
from qdrant_engine.snapshot import jsonable_to_vector  # noqa: F401

DEFAULT_SNAPSHOT = os.path.join(DATA_DIR, 'qdrant_snapshot')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot-dir', default=DEFAULT_SNAPSHOT)
    ap.add_argument('--recreate', action='store_true', help='Delete existing collections before restoring')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--collections', default=None,
                    help='Restore comma-separated collection names (default: all in manifest)')
    args = ap.parse_args()

    settings = load_settings()
    client = get_client(settings)
    print(f'Restore storage: {settings.url or settings.local_path}')

    collections = [c.strip() for c in args.collections.split(',')] if args.collections else None
    try:
        report = restore_snapshot(args.snapshot_dir, settings, client, collections=collections,
                                  recreate=args.recreate, batch_size=args.batch_size)
    except SnapshotIntegrityError as exc:
        print(f'\n[STOP] Pre-validation failed; existing collections unchanged.\n{exc}')
        return 1

    print(f"\nRestore mode: {report.get('restore_mode', '?')}")
    for c in report['collections']:
        print(f"\n{c['collection']}: loaded {c['upserted']} / expected {c['expected_points']} "
              f" -> actual {c['actual_points']} ({'OK' if c['points_match'] else 'MISMATCH'}, "
              f"{c['load_seconds']}s)")
        print(f"  Point-ID sets match: {c['point_id_set_match']} "
              f"(mismatches {c['point_id_mismatch_count']})")
        print(f"  Vector schemas match: {c['vector_schema_match']}")
        pidx = c['payload_indexes']
        print(f"  Payload indexes: created={pidx['created']} "
              f"local_no_effect={pidx['no_effect_local_mode']} failed={pidx['failed']} "
              f" -> verdict {'OK' if c['payload_indexes_ok'] else 'FAIL'}")
        if 'live_schema_check' in pidx:
            print(f"    (live server payload-schema comparison: {pidx['live_schema_check']})")
        print(f"  alias: {c['alias']} -> verdict {'OK' if c['alias_ok'] else 'FAIL'}")
        if 'representative_search' in c:
            print(f"  Representative search execution: {c['representative_search']} "
                  f" -> verdict {'OK' if c['representative_search_ok'] else 'FAIL'}")

    print(f"\nOverall result: {'OK' if report['all_ok'] else 'MISMATCH'}")
    out = os.path.join(args.snapshot_dir, 'last_restore_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'Detailed report: {out}')
    return 0 if report['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
