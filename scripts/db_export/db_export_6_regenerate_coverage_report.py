"""Regenerate coverage_report.json from current data/db_export/*.csv. The db_export_1 report is a historical snapshot not updated by db_export_2/3/4/5, leaving stale values such as claim_element rows: 0 and zero populated pub_date entries.
Run python3 scripts/db_export/db_export_6_regenerate_coverage_report.py; overwrites data/db_export/coverage_report.json. No follow-up script is specified here.
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR

OUT_DIR = os.path.join(DATA_DIR, 'db_export')

TABLE_NOTES = {
    'patent_molecule': 'No claim/description/figure location breakdown; patent-level matching only',
    'grounding_gold': 'Some TARGET claim_element_id joins are approximate; see target_rel_claims',
    'exam_case': 'Planned when Task 2/3 work begins (not started)',
    'case_relevance': 'Planned when Task 2/3 work begins (not started)',
    'description_text(patent_content)': 'Intentionally empty; paragraph table provides paragraph-level text',
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

    # Summarize frequently inspected fields separately.
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
            'ipc_codes_field_note': ('ipc_codes is the correct column name: source data uses IPC (International '
                                      'Patent Classification) codes, e.g. C08L, A01K. The specification\'s '
                                      '"cpc_codes" label was a typo'),
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

    print('coverage_report.json updated')
    print(json.dumps(coverage.get('patent_content_summary', {}), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
