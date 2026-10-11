"""특허-화학구조 관계를 chemrag_structures_v1 컬렉션에 적재한다.

입력: scripts/qdrant/qdrant_build_structure_metadata.py가 만든
      data/db_export/derived/structure_location_metadata.jsonl
      (원본 CSV를 직접 읽지 않고 검증된 파생 산출물을 쓴다)

  - (patent_id, mol_id) 중복은 메타데이터 생성 단계에서 이미 제거돼 있고,
    여기서도 한 번 더 확인해 중복 적재를 막는다.
  - 벡터: Morgan(ECFP4, radius=2, 2048bit) on-bit sparse 벡터.
    ** 이 벡터의 내적 점수는 Tanimoto가 아니다 ** - 후보 생성용이며 실제 유사도는
    RDKit exact Tanimoto로 다시 계산한다(qdrant_engine/structure_repository.py).
  - 포인트 ID: (patent_id + mol_id) 기반 UUID5 -> 멱등 upsert

사용법:
    python3 scripts/qdrant/qdrant_index_structures.py
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qdrant_client import models

from config.settings import DATA_DIR
from qdrant_engine.config import load_settings
from qdrant_engine.client import get_client, ensure_structure_collection
from qdrant_engine.schemas import structure_point_id
from qdrant_engine.structure_repository import morgan_bits

DERIVED_DIR = os.path.join(DATA_DIR, 'db_export', 'derived')
METADATA_PATH = os.path.join(DERIVED_DIR, 'structure_location_metadata.jsonl')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', default=METADATA_PATH)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--recreate', action='store_true')
    args = ap.parse_args()

    settings = load_settings(overrides={'batch_size': args.batch_size})

    if not os.path.exists(args.metadata):
        raise SystemExit(f'{args.metadata} 가 없습니다. 먼저 '
                         f'python3 scripts/qdrant/qdrant_build_structure_metadata.py 를 실행하세요.')

    print('[1/3] 구조 메타데이터 로딩...')
    records, seen = [], set()
    duplicate_skipped = 0
    with open(args.metadata, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['patent_id'], rec['mol_id'])
            if key in seen:
                duplicate_skipped += 1
                continue
            seen.add(key)
            records.append(rec)
    print(f'  고유 (patent_id, mol_id) {len(records)}건 (중복 스킵 {duplicate_skipped}건)')

    print('[2/3] 컬렉션 준비...')
    client = get_client(settings)
    ensure_structure_collection(client, settings, recreate=args.recreate)

    print('[3/3] 적재...')
    t0 = time.time()
    batch = settings.batch_size
    total = 0
    no_fingerprint = 0
    for start in range(0, len(records), batch):
        points = []
        for rec in records[start:start + batch]:
            bits = morgan_bits(rec.get('smiles'), settings.morgan_radius, settings.morgan_nbits)
            vector = {}
            fp_status = 'ok'
            if bits:
                vector[settings.fingerprint_vector_name] = models.SparseVector(
                    indices=bits, values=[1.0] * len(bits))
            else:
                fp_status = 'unparseable_smiles'
                no_fingerprint += 1

            payload = {
                'patent_id': rec['patent_id'],
                'mol_id': rec['mol_id'],
                'smiles': rec.get('smiles'),
                'canonical_smiles': rec.get('canonical_smiles'),
                'inchikey': rec.get('inchikey'),
                'source': rec.get('source'),
                'in_claim': rec.get('in_claim'),
                'in_figure': rec.get('in_figure'),
                'claim_occurrence_count': rec.get('claim_occurrence_count'),
                'claim_occurrence_status': rec.get('claim_occurrence_status'),
                'claim_unit_refs': rec.get('claim_unit_refs') or [],
                'figure_unit_refs': rec.get('figure_unit_refs') or [],
                # DB 내 중복 등장 횟수(patent_molecule.csv의 같은 (patent_id, mol_id) 행 수).
                # 청구항 본문 등장 횟수(claim_occurrence_count)와 다른 값이다.
                'db_occurrence_count': rec.get('db_occurrence_count'),
                # 이 구조가 속한 특허에 figure 행이 있는지. 구조가 도면에 그려졌는지(in_figure)와 다르다.
                'figure_exist': rec.get('figure_exist'),
                'figure_row_count': rec.get('figure_row_count'),
                'location_evidence': rec.get('location_evidence') or {},
                'location_status': rec.get('location_status'),
                'source_version': rec.get('source_version'),
                'fingerprint': {
                    'type': 'morgan_ecfp4',
                    'radius': settings.morgan_radius,
                    'nbits': settings.morgan_nbits,
                    'on_bits': len(bits) if bits else 0,
                    'status': fp_status,
                    'note': 'sparse 내적 점수는 Tanimoto가 아님 - 후보 생성용',
                },
            }
            points.append(models.PointStruct(
                id=structure_point_id(rec['patent_id'], rec['mol_id']),
                vector=vector, payload=payload))
        client.upsert(settings.structures_collection, points=points, wait=True)
        total += len(points)
        if (start // batch) % 10 == 0:
            print(f'  {total}/{len(records)} 적재 ({time.time() - t0:.1f}s)')

    count = client.count(settings.structures_collection).count
    summary = {
        'input_records': len(records), 'upserted': total, 'collection_count': count,
        'duplicate_pairs_skipped': duplicate_skipped,
        'points_without_fingerprint_unparseable_smiles': no_fingerprint,
        'elapsed_seconds': round(time.time() - t0, 1),
    }
    out = os.path.join(DERIVED_DIR, 'qdrant_index_structures_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n저장됨: {out}')


if __name__ == '__main__':
    main()
