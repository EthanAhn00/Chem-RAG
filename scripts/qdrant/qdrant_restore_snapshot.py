"""스냅샷에서 Qdrant 컬렉션을 안전하게 복원한다(어느 머신에서든 동일 결과).

복원 순서(qdrant_engine/snapshot.restore_snapshot 참고):
  1) manifest 구조 검증  2) 파일 존재 확인  3) 파일 크기/해시(archive/content) 확인
  4) 컬렉션 스키마 유효성 확인
  -> 여기까지 하나라도 실패하면 SnapshotIntegrityError로 즉시 중단하고
     기존 컬렉션은 전혀 건드리지 않는다(종료 코드 1).
  5) 기존 컬렉션 삭제(--recreate)  6) 컬렉션 생성  7) 포인트 적재
  8) payload 인덱스 생성  9) alias 생성  10) 최종 상태 검증
     (포인트 수/ID 집합/벡터 스키마 일치, payload 인덱스·alias는 로컬 임베디드 모드에서
      효과가 없을 수 있음 - 그 경우 실패로 위장하지 않고 no_effect_local_mode/unsupported로 기록)

    python3 scripts/qdrant/qdrant_restore_snapshot.py
    python3 scripts/qdrant/qdrant_restore_snapshot.py --snapshot-dir data/qdrant_snapshot --recreate
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
# 하위 호환 - 다른 코드가 이 모듈에서 직접 import 하던 이름들을 재노출
from qdrant_engine.snapshot import jsonable_to_vector  # noqa: F401

DEFAULT_SNAPSHOT = os.path.join(DATA_DIR, 'qdrant_snapshot')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot-dir', default=DEFAULT_SNAPSHOT)
    ap.add_argument('--recreate', action='store_true', help='기존 컬렉션 삭제 후 복원')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--collections', default=None,
                    help='콤마로 구분된 컬렉션 이름만 복원(기본: manifest의 전부)')
    args = ap.parse_args()

    settings = load_settings()
    client = get_client(settings)
    print(f'복원 대상 저장소: {settings.url or settings.local_path}')

    collections = [c.strip() for c in args.collections.split(',')] if args.collections else None
    try:
        report = restore_snapshot(args.snapshot_dir, settings, client, collections=collections,
                                  recreate=args.recreate, batch_size=args.batch_size)
    except SnapshotIntegrityError as exc:
        print(f'\n[중단] 사전 검증 실패 - 기존 컬렉션은 변경되지 않았습니다.\n{exc}')
        return 1

    print(f"\n복원 모드: {report.get('restore_mode', '?')}")
    for c in report['collections']:
        print(f"\n{c['collection']}: 적재 {c['upserted']} / 기대 {c['expected_points']} "
              f"-> 실제 {c['actual_points']} ({'OK' if c['points_match'] else 'MISMATCH'}, "
              f"{c['load_seconds']}s)")
        print(f"  포인트 ID 집합 일치: {c['point_id_set_match']} "
              f"(불일치 {c['point_id_mismatch_count']}건)")
        print(f"  벡터 스키마 일치: {c['vector_schema_match']}")
        pidx = c['payload_indexes']
        print(f"  payload 인덱스: 생성={pidx['created']} "
              f"로컬모드_효과없음={pidx['no_effect_local_mode']} 실패={pidx['failed']} "
              f"-> 판정 {'OK' if c['payload_indexes_ok'] else 'FAIL'}")
        if 'live_schema_check' in pidx:
            print(f"    (서버 실측 payload schema 대조: {pidx['live_schema_check']})")
        print(f"  alias: {c['alias']} -> 판정 {'OK' if c['alias_ok'] else 'FAIL'}")
        if 'representative_search' in c:
            print(f"  대표 검색 실행 확인: {c['representative_search']} "
                  f"-> 판정 {'OK' if c['representative_search_ok'] else 'FAIL'}")

    print(f"\n전체 결과: {'OK' if report['all_ok'] else 'MISMATCH'}")
    out = os.path.join(args.snapshot_dir, 'last_restore_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'상세 리포트: {out}')
    return 0 if report['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
