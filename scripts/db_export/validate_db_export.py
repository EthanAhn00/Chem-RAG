"""
data/db_export/ 아래 11개 테이블의 PK 중복, FK 정합성(참조하는 patent_id/mol_id가
실제로 존재하는지), 컬럼별 결측률을 점검해서 리포트로 남긴다. 새 계산 로직은 없고
기존 CSV를 그대로 읽어서 검사만 한다.

db_export_1~5 스크립트를 실행할 때마다 이 스크립트도 같이 돌려서, 문서(README/
coverage_report.json)와 실제 데이터가 어긋나는 걸 미리 잡는 용도.

known_issues.json에 승인해둔 기존 문제(정확한 수치까지 일치)는 "이미 아는 문제"로
표시만 하고 실패 처리하지 않는다. 새로 생기거나 수치가 달라진 문제만 exit code 1로
잡아내므로, CI/배포 게이트로 그대로 쓸 수 있다.

사용법 (맥 .venv):
    python3 scripts/db_export/validate_db_export.py

출력: data/db_export/validation_report.json, 콘솔에 요약 출력
    (승인 안 된 새 문제가 있으면 exit code 1)
"""
import os
import sys
import csv
import json
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR

OUT_DIR = os.path.join(DATA_DIR, 'db_export')
KNOWN_ISSUES_PATH = os.path.join(OUT_DIR, 'known_issues.json')

# 테이블별 (PK 컬럼, FK 컬럼 -> 참조 대상 테이블의 PK 컬럼)
TABLES = {
    'patent_content': {'pk': 'patent_id', 'fk': {}},
    'molecule': {'pk': 'mol_id', 'fk': {}},
    'patent_molecule': {'pk': None, 'fk': {'patent_id': 'patent_content', 'mol_id': 'molecule'}},
    'examiner_citation': {'pk': 'cite_id', 'fk': {'query_patent_id': 'patent_content',
                                                   'cited_patent_id': 'patent_content'}},
    'split': {'pk': 'id', 'fk': {'id': 'patent_content'}},
    'paragraph': {'pk': 'para_id', 'fk': {'patent_id': 'patent_content'}},
    'figure': {'pk': 'fig_id', 'fk': {'patent_id': 'patent_content'}},
    'claim_element': {'pk': 'element_id', 'fk': {'patent_id': 'patent_content'}},
    'grounding_gold': {'pk': 'ground_id', 'fk': {'query_patent_id': 'patent_content',
                                                  'cited_patent_id': 'patent_content'}},
    'exam_case': {'pk': 'case_id', 'fk': {'app_patent_id': 'patent_content'}},
    'case_relevance': {'pk': None, 'fk': {'query_patent_id': 'patent_content', 'case_id': 'exam_case'}},
}


def load(name):
    path = os.path.join(OUT_DIR, f'{name}.csv')
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def check_pk_duplicates(name, rows, pk):
    if not pk or not rows:
        return None
    counts = Counter(r[pk] for r in rows)
    dups = {k: v for k, v in counts.items() if v > 1}
    if not dups:
        return {'unique': True, 'dup_ids': 0, 'dup_rows': 0}
    dup_rows = sum(v - 1 for v in dups.values())
    diff_content = 0
    for pk_val in dups:
        matching = [r for r in rows if r[pk] == pk_val]
        other_cols = [c for c in matching[0] if c != pk]
        signatures = {tuple(r[c] for c in other_cols) for r in matching}
        if len(signatures) > 1:
            diff_content += 1
    return {'unique': False, 'dup_ids': len(dups), 'dup_rows': dup_rows,
            'dup_ids_with_differing_content': diff_content}


def check_fk(name, rows, fk_map, pk_index):
    if not rows or not fk_map:
        return {}
    result = {}
    for col, ref_table in fk_map.items():
        ref_ids = pk_index.get(ref_table)
        if ref_ids is None:
            result[col] = {'note': f'{ref_table} PK 인덱스 없음(참조 테이블 PK 미정의) - 점검 생략'}
            continue
        missing = Counter()
        checked = 0
        for r in rows:
            val = r.get(col)
            if not val:
                continue
            checked += 1
            if val not in ref_ids:
                missing[val] += 1
        result[col] = {'checked': checked, 'dangling_refs': len(missing),
                        'dangling_sample': list(missing)[:5]}
    return result


def null_rate(rows):
    if not rows:
        return {}
    n = len(rows)
    cols = rows[0].keys()
    return {c: {'empty': sum(1 for r in rows if not r.get(c)),
                'empty_pct': round(100 * sum(1 for r in rows if not r.get(c)) / n, 1)}
            for c in cols}


def load_known_issues():
    if not os.path.exists(KNOWN_ISSUES_PATH):
        return []
    with open(KNOWN_ISSUES_PATH, encoding='utf-8') as f:
        return json.load(f)


def main():
    tables = {name: load(name) for name in TABLES}
    pk_index = {name: set(r[TABLES[name]['pk']] for r in rows)
                for name, rows in tables.items() if rows and TABLES[name]['pk']}

    report = {}
    problems = []  # 각 항목: {fingerprint, message}
    for name, rows in tables.items():
        if rows is None:
            report[name] = {'status': 'MISSING'}
            continue
        pk_result = check_pk_duplicates(name, rows, TABLES[name]['pk'])
        fk_result = check_fk(name, rows, TABLES[name]['fk'], pk_index)
        report[name] = {
            'rows': len(rows),
            'pk_check': pk_result,
            'fk_check': fk_result,
            'null_rate': null_rate(rows),
        }
        if pk_result and not pk_result['unique']:
            problems.append({
                'fingerprint': {'type': 'pk_duplicate', 'table': name,
                                 'dup_ids': pk_result['dup_ids'], 'dup_rows': pk_result['dup_rows'],
                                 'dup_ids_with_differing_content': pk_result['dup_ids_with_differing_content']},
                'message': f"{name}: PK 중복 ID {pk_result['dup_ids']}개, 추가 중복 행 {pk_result['dup_rows']}개"
                           f"(그중 서로 다른 내용의 중복 ID {pk_result['dup_ids_with_differing_content']}개)",
            })
        for col, fk_res in fk_result.items():
            if fk_res.get('dangling_refs'):
                problems.append({
                    'fingerprint': {'type': 'dangling_fk', 'table': name, 'column': col,
                                     'dangling_refs': fk_res['dangling_refs']},
                    'message': f"{name}.{col}: 존재하지 않는 참조 {fk_res['dangling_refs']}건",
                })

    with open(os.path.join(OUT_DIR, 'validation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    known_issues = load_known_issues()
    known_set = {json.dumps(k, sort_keys=True) for k in known_issues}
    new_problems, known_matched = [], []
    for p in problems:
        key = json.dumps(p['fingerprint'], sort_keys=True)
        (known_matched if key in known_set else new_problems).append(p)

    print(f'{len(tables)}개 테이블 검사 완료 -> data/db_export/validation_report.json')
    if known_matched:
        print(f'\n이미 승인된 기존 문제 {len(known_matched)}건 (known_issues.json 기준, 실패로 안 잡음):')
        for p in known_matched:
            print(' -', p['message'])
    if new_problems:
        print(f'\n승인되지 않은 새 문제 {len(new_problems)}건:')
        for p in new_problems:
            print(' -', p['message'])
    if not problems:
        print('PK 중복/FK 미참조 문제 없음')

    sys.exit(1 if new_problems else 0)


if __name__ == '__main__':
    main()
