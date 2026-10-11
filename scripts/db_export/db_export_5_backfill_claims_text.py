"""
patent_content.claims_text가 비어있는 특허 중, claim_element에 이미 확보된 청구항
텍스트가 있는 건을 찾아 재조립해서 채운다. db_export_4_ops_fulltext.py가 TARGET
특허의 청구항을 claim_element로만 저장하고 patent_content.claims_text에는
반영하지 않아서 생긴 공백을 메우는 작업.

claim_element는 청구항 단위(claim_no)로 여러 element_text 조각을 갖고 있어서,
같은 청구항의 조각을 element_no 순으로 합친 뒤 청구항 번호 순으로 이어 붙인다.

이어서 실행할 스크립트 없음 (patent_content.csv를 그 자리에서 갱신).

사용법 (맥 .venv):
    python3 scripts/db_export/db_export_5_backfill_claims_text.py

출력: data/db_export/patent_content.csv 갱신, coverage_report.json에
      claims_text_backfill 요약 추가
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR

OUT_DIR = os.path.join(DATA_DIR, 'db_export')


def load_csv(name):
    path = os.path.join(OUT_DIR, f'{name}.csv')
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(name, fieldnames, rows):
    path = os.path.join(OUT_DIR, f'{name}.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_claims_text(elements):
    """같은 patent_id의 claim_element 행들을 claim_no/element_no 순으로 합쳐
    하나의 청구항 전문 텍스트로 재구성한다."""
    by_claim = {}
    for el in elements:
        try:
            claim_no = int(el['claim_no'])
            element_no = int(el['element_no'])
        except (TypeError, ValueError):
            continue
        by_claim.setdefault(claim_no, []).append((element_no, el['element_text'] or ''))

    claim_texts = []
    for claim_no in sorted(by_claim):
        parts = [text for _, text in sorted(by_claim[claim_no])]
        claim_texts.append(' '.join(p.strip() for p in parts if p.strip()))
    return ' '.join(claim_texts).strip()


def main():
    patents = load_csv('patent_content')
    elements = load_csv('claim_element')

    elements_by_patent = {}
    for el in elements:
        elements_by_patent.setdefault(el['patent_id'], []).append(el)

    empty_before = [p['patent_id'] for p in patents if not p.get('claims_text')]
    filled, still_empty = [], []

    for p in patents:
        if p.get('claims_text'):
            continue
        els = elements_by_patent.get(p['patent_id'])
        if not els:
            still_empty.append(p['patent_id'])
            continue
        text = build_claims_text(els)
        if text:
            p['claims_text'] = text
            filled.append(p['patent_id'])
        else:
            still_empty.append(p['patent_id'])

    fieldnames = list(patents[0].keys())
    write_csv('patent_content', fieldnames, patents)

    coverage_path = os.path.join(OUT_DIR, 'coverage_report.json')
    coverage = {}
    if os.path.exists(coverage_path):
        with open(coverage_path, encoding='utf-8') as f:
            coverage = json.load(f)
    coverage['claims_text_backfill'] = {
        'empty_before': len(empty_before),
        'filled_from_claim_element': len(filled),
        'still_empty': len(still_empty),
        'still_empty_ids': still_empty,
        'note': 'claim_element는 OPS 전문 조회 대상(TARGET 특허)만 있어서, '
                'PRIOR로만 등장한 특허는 claim_element도 없으면 채울 수 없음',
    }
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    print(f'claims_text 결측 {len(empty_before)}건 중 {len(filled)}건 채움, {len(still_empty)}건 여전히 결측')
    print('여전히 결측:', still_empty)


if __name__ == '__main__':
    main()
