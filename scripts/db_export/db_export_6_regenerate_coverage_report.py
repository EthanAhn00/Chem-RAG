"""
coverage_report.json을 현재 data/db_export/*.csv 상태 그대로 다시 생성한다.
db_export_1이 처음 만든 coverage_report.json은 그 시점 스냅샷이라, 이후
db_export_2/3/4/5가 데이터를 채워도 값이 갱신되지 않고 남아있었다(예: claim_element
rows: 0, pub_date 채움 0건 등 — 전부 이후 스크립트가 채운 뒤로 사실과 다름).

이어서 실행할 스크립트 없음 (coverage_report.json 덮어씀).

사용법 (맥 .venv):
    python3 scripts/db_export/db_export_6_regenerate_coverage_report.py

출력: data/db_export/coverage_report.json 갱신
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR

OUT_DIR = os.path.join(DATA_DIR, 'db_export')

TABLE_NOTES = {
    'patent_molecule': 'location(claim/description/figure) 구분 정보 없음 - 특허 단위 매칭만 가능',
    'grounding_gold': 'claim_element_id는 TARGET 쪽 근사 조인이라 일부 부정확 - target_rel_claims 참고',
    'exam_case': 'Task 2/3 착수 시 작업 예정 (미착수)',
    'case_relevance': 'Task 2/3 착수 시 작업 예정 (미착수)',
    'description_text(patent_content)': '의도적으로 비움 - paragraph 테이블이 문단 단위로 대체',
}


def load(name):
    path = os.path.join(OUT_DIR, f'{name}.csv')
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    tables = ['patent_content', 'molecule', 'patent_molecule', 'examiner_citation', 'split',
              'paragraph', 'figure', 'claim_element', 'grounding_gold', 'exam_case', 'case_relevance']

    coverage = {'tables': {}}
    for name in tables:
        rows = load(name)
        if rows is None:
            coverage['tables'][name] = {'status': 'MISSING'}
            continue
        n = len(rows)
        entry = {'rows': n}
        if n:
            cols = rows[0].keys()
            entry['null_rate'] = {c: round(100 * sum(1 for r in rows if not r.get(c)) / n, 1) for c in cols}
        if name in TABLE_NOTES:
            entry['note'] = TABLE_NOTES[name]
        coverage['tables'][name] = entry

    # 자주 확인하는 몇 개 필드는 요약으로 따로 뽑아둠
    pc = load('patent_content')
    if pc:
        coverage['patent_content_summary'] = {
            'total': len(pc),
            'filled_pub_date': sum(1 for r in pc if r['pub_date']),
            'filled_app_date': sum(1 for r in pc if r['app_date']),
            'filled_family_id': sum(1 for r in pc if r['family_id']),
            'filled_title': sum(1 for r in pc if r['title']),
            'filled_claims_text': sum(1 for r in pc if r['claims_text']),
            'has_structure_true': sum(1 for r in pc if r['has_structure'] == 'True'),
            'ipc_codes_field_note': ('ipc_codes 컬럼명이 맞음 - 원본 데이터가 IPC(International '
                                      'Patent Classification) 코드임(예: C08L, A01K). 스펙 문서의 '
                                      '"cpc_codes" 표기가 오기(誤記)였음'),
        }

    ec = load('examiner_citation')
    if ec:
        from collections import Counter
        coverage['examiner_citation_summary'] = {
            'total': len(ec),
            'category_breakdown': dict(Counter(r['category'] for r in ec)),
            'filled_rejection_type': sum(1 for r in ec if r['rejection_type']),
        }

    with open(os.path.join(OUT_DIR, 'coverage_report.json'), 'w', encoding='utf-8') as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    print('coverage_report.json 갱신 완료')
    print(json.dumps(coverage.get('patent_content_summary', {}), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
