"""Build X+Y gold for prior-art retrieval.
Legacy data/db_export/examiner_citation.csv has 1,254 X-category pairs (X / X,P / X,D / X,D,P); X is particularly relevant on its own. The separate epo_dataset/in_corpus_candidate_pairs.csv has 1,439 pairs, no pair overlap with legacy gold, and 784 background A pairs. Following prior-art IR conventions such as CLEF-IP, retain X/Y and exclude A/E/T/L.
Inputs are read-only. Outputs: data/db_export/derived/gold_xy_v1.csv and gold_xy_v1_summary.json.
Fail with exit code 1 and write nothing on overlapping source pairs, patents outside patent_content.csv, cited.pub_date > query.app_date, or matching cited/query family_id. Such filtered-out gold would be unretrievable and lower the attainable MAP.
Usage: python3 scripts/db_export/build_xy_gold.py [--pairs-csv /path/to/in_corpus_candidate_pairs.csv].
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
    """Return 2 for X, 1 for Y without X, or None for neither."""
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
                    help='Path to in_corpus_candidate_pairs.csv (may be outside the repository)')
    ap.add_argument('--dry-run', action='store_true', help='Print statistics without writing files')
    args = ap.parse_args()

    meta = load_corpus_meta()
    problems = []
    rows = []

    # --- Source 1: examiner_citation.csv (legacy X-category gold) ---
    ec_path = os.path.join(DATA_DIR, 'db_export', 'examiner_citation.csv')
    with open(ec_path, newline='', encoding='utf-8') as f:
        ec = list(csv.DictReader(f))
    ec_kept = 0
    for r in ec:
        grade = primary_grade(r['category'])
        if grade is None:
            problems.append(f"Non-X/Y category in examiner_citation: {r['category']}")
            continue
        rows.append({
            'query_patent_id': r['query_patent_id'],
            'cited_patent_id': r['cited_patent_id'],
            'category': r['category'],
            'relevance': grade,
            'source': 'examiner_citation',
        })
        ec_kept += 1

    # --- Source 2: X/Y pairs from in_corpus_candidate_pairs.csv ---
    pairs_path = os.path.abspath(args.pairs_csv)
    if not os.path.exists(pairs_path):
        raise SystemExit(f'Input file missing: {pairs_path}')
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

    # --- Validation ---
    seen = {}
    dup = 0
    for r in rows:
        key = (r['query_patent_id'], r['cited_patent_id'])
        if key in seen:
            dup += 1
            if seen[key] != r['source']:
                problems.append(f'Pairs overlapping between sources: {key}')
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

    for label, n in (('patents outside corpus', out_of_corpus), ('time-filter violations', time_violation),
                     ('same family', family_violation), ('missing dates', missing_date),
                     ('duplicate pairs', dup)):
        if n:
            problems.append(f'{label}: {n} records')

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
        print('\nValidation failed:', file=sys.stderr)
        for p in problems[:20]:
            print(f'  - {p}', file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        print('\n--dry-run: no files were written.')
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
    print(f'\nSaved: {OUT_CSV} ({len(rows)} pairs)')
    print(f'Saved: {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
