"""Backfill empty patent_content.claims_text from available claim_element text. db_export_4_ops_fulltext.py stores TARGET claims in claim_element but not patent_content.claims_text. Join fragments by element_no within claim_no, then join claims by claim number.
Run python3 scripts/db_export/db_export_5_backfill_claims_text.py. Updates patent_content.csv in place and adds claims_text_backfill to coverage_report.json; no follow-up script is specified here.
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
    """Reconstruct full claims for one patent_id, ordered by claim_no/element_no."""
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
        'note': 'claim_element covers OPS full-text TARGET patents; '
                'PRIOR-only patents cannot be backfilled without claim_element rows',
    }
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    print(f'Missing claims_text: {len(empty_before)}; populated {len(filled)} filled; {len(still_empty)} still missing')
    print('Still missing:', still_empty)


if __name__ == '__main__':
    main()
