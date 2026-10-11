"""X+Y 통합 gold 생성 - 선행기술 검색 평가용 정답 집합.

배경:
    기존 gold(`data/db_export/examiner_citation.csv`, 1,254쌍)는 EPO search report
    category가 전부 X 계열이다(X / X,P / X,D / X,D,P). X는 단독으로 신규성·진보성을
    부정하는 가장 강한 선행기술이다.

    별도로 확보한 `epo_dataset/in_corpus_candidate_pairs.csv`(1,439쌍)는 위 집합과
    쌍 기준 교집합이 0이며, A(단순 기술 배경) 784쌍이 절반 이상을 차지한다.
    선행기술 검색 IR 관례(CLEF-IP 등)를 따라 **X와 Y만 relevant로 채택**하고
    A / E / T / L 은 제외한다.

산출물(원본 CSV는 읽기만 하고 절대 수정하지 않는다):
    data/db_export/derived/gold_xy_v1.csv
    data/db_export/derived/gold_xy_v1_summary.json

검증(하나라도 실패하면 종료 코드 1, 파일을 쓰지 않는다):
    - 두 소스의 쌍이 겹치지 않는지
    - 모든 특허가 코퍼스(patent_content.csv) 안에 있는지
    - 시간 필터: cited.pub_date <= query.app_date
    - family 필터: cited.family_id != query.family_id

    시간/family를 위반하는 쌍은 검색 파이프라인이 후보에서 제외하므로
    gold에 넣으면 '절대 찾을 수 없는 정답'이 되어 MAP 상한을 깎는다.

    python3 scripts/db_export/build_xy_gold.py
    python3 scripts/db_export/build_xy_gold.py --pairs-csv /경로/in_corpus_candidate_pairs.csv
"""
import os
import sys
import csv
import json
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import DATA_DIR, PROJECT_DIR

RELEVANT = ('X', 'Y')
GRADE = {'X': 2, 'Y': 1}

DERIVED_DIR = os.path.join(DATA_DIR, 'db_export', 'derived')
OUT_CSV = os.path.join(DERIVED_DIR, 'gold_xy_v1.csv')
OUT_JSON = os.path.join(DERIVED_DIR, 'gold_xy_v1_summary.json')
DEFAULT_PAIRS = os.path.join(PROJECT_DIR, '..', 'epo_dataset', 'in_corpus_candidate_pairs.csv')


def category_tokens(value):
    return {t.strip() for t in (value or '').split(',') if t.strip()}


def primary_grade(value):
    """category에 X가 있으면 2, Y만 있으면 1, 둘 다 없으면 None."""
    toks = category_tokens(value)
    if 'X' in toks:
        return GRADE['X']
    if 'Y' in toks:
        return GRADE['Y']
    return None


def load_corpus_meta():
    path = os.path.join(DATA_DIR, 'db_export', 'patent_content.csv')
    with open(path, newline='', encoding='utf-8') as f:
        return {r['patent_id']: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-csv', default=DEFAULT_PAIRS,
                    help='in_corpus_candidate_pairs.csv 경로(저장소 밖에 있을 수 있음)')
    ap.add_argument('--dry-run', action='store_true', help='파일을 쓰지 않고 통계만 출력')
    args = ap.parse_args()

    meta = load_corpus_meta()
    problems = []
    rows = []

    # --- 소스 1: examiner_citation.csv (기존 gold, X 계열) ---
    ec_path = os.path.join(DATA_DIR, 'db_export', 'examiner_citation.csv')
    with open(ec_path, newline='', encoding='utf-8') as f:
        ec = list(csv.DictReader(f))
    ec_kept = 0
    for r in ec:
        grade = primary_grade(r['category'])
        if grade is None:
            problems.append(f"examiner_citation에 X/Y 아닌 category: {r['category']}")
            continue
        rows.append({
            'query_patent_id': r['query_patent_id'],
            'cited_patent_id': r['cited_patent_id'],
            'category': r['category'],
            'relevance': grade,
            'source': 'examiner_citation',
        })
        ec_kept += 1

    # --- 소스 2: in_corpus_candidate_pairs.csv 중 X/Y 계열만 ---
    pairs_path = os.path.abspath(args.pairs_csv)
    if not os.path.exists(pairs_path):
        raise SystemExit(f'입력 파일 없음: {pairs_path}')
    with open(pairs_path, newline='', encoding='utf-8') as f:
        pr = list(csv.DictReader(f))
    ic_kept = 0
    ic_dropped = {}
    for r in pr:
        grade = primary_grade(r['category'])
        if grade is None:
            ic_dropped[r['category']] = ic_dropped.get(r['category'], 0) + 1
            continue
        rows.append({
            'query_patent_id': r['query_patent_id'],
            'cited_patent_id': r['cited_patent_id'],
            'category': r['category'],
            'relevance': grade,
            'source': 'in_corpus_candidate_pairs',
        })
        ic_kept += 1

    # --- 검증 ---
    seen = {}
    dup = 0
    for r in rows:
        key = (r['query_patent_id'], r['cited_patent_id'])
        if key in seen:
            dup += 1
            if seen[key] != r['source']:
                problems.append(f'두 소스에 중복된 쌍: {key}')
        seen[key] = r['source']

    out_of_corpus = time_violation = family_violation = missing_date = 0
    for r in rows:
        q, c = meta.get(r['query_patent_id']), meta.get(r['cited_patent_id'])
        if not q or not c:
            out_of_corpus += 1
            continue
        qa, cp = (q.get('app_date') or '').strip(), (c.get('pub_date') or '').strip()
        qf, cf = (q.get('family_id') or '').strip(), (c.get('family_id') or '').strip()
        if not (qa and cp):
            missing_date += 1
        elif cp > qa:
            time_violation += 1
        if qf and cf and qf == cf:
            family_violation += 1

    for label, n in (('코퍼스 밖 특허', out_of_corpus), ('시간 필터 위반', time_violation),
                     ('family 동일', family_violation), ('날짜 결측', missing_date),
                     ('중복 쌍', dup)):
        if n:
            problems.append(f'{label}: {n}건')

    queries = sorted({r['query_patent_id'] for r in rows})
    summary = {
        'generated_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'relevant_categories': list(RELEVANT),
        'grade_map': GRADE,
        'sources': {
            'examiner_citation': {'input_rows': len(ec), 'kept': ec_kept},
            'in_corpus_candidate_pairs': {
                'path': pairs_path, 'input_rows': len(pr), 'kept': ic_kept,
                'dropped_by_category': ic_dropped,
            },
        },
        'total_pairs': len(rows),
        'unique_queries': len(queries),
        'unique_cited': len({r['cited_patent_id'] for r in rows}),
        'grade_distribution': {
            'X(2)': sum(1 for r in rows if r['relevance'] == 2),
            'Y(1)': sum(1 for r in rows if r['relevance'] == 1),
        },
        'validation': {
            'out_of_corpus': out_of_corpus, 'time_violation': time_violation,
            'family_violation': family_violation, 'missing_date': missing_date,
            'duplicate_pairs': dup, 'passed': not problems,
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if problems:
        print('\n검증 실패:', file=sys.stderr)
        for p in problems[:20]:
            print(f'  - {p}', file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        print('\n--dry-run: 파일을 쓰지 않았습니다.')
        return 0

    os.makedirs(DERIVED_DIR, exist_ok=True)
    rows.sort(key=lambda r: (r['query_patent_id'], r['cited_patent_id']))
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['query_patent_id', 'cited_patent_id',
                                          'category', 'relevance', 'source'])
        w.writeheader()
        w.writerows(rows)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n저장: {OUT_CSV} ({len(rows)}쌍)')
    print(f'저장: {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
