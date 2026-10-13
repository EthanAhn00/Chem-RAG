"""Populate 5 Task 1 tables from the 11-table DB schema: PatentContent, Molecule, PatentMolecule, ExaminerCitation, Split. Create header-only schemas for the other 6 Task 2/3 tables. Transfer existing pipeline values without new scoring logic.
Next: db_export_2_fill_citation_category.py fills citation categories; db_export_3_task2_tables.py fills paragraph/figure/claim_element/grounding_gold.
Usage: python3 run_chem_rag.py to refresh data/reference/epo_section_c_benchmark_dataset.json, then python3 scripts/db_export/db_export_1_core_tables.py.
Write 11 CSV tables and coverage_report.json under data/db_export/, summarizing rows and empty fields.
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR


def load_dataset():
    path = os.path.join(DATA_DIR, 'reference', 'epo_section_c_benchmark_dataset.json')
    if not os.path.exists(path):
        print(f'[FAIL] {path} is missing. Run python3 run_chem_rag.py first.')
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_split():
    split_path = os.path.join(DATA_DIR, 'reference', 'train_test_split.json')
    summary_path = os.path.join(DATA_DIR, 'reference', 'train_test_split_summary.json')
    split_map, cutoff_year = {}, None
    if os.path.exists(split_path):
        with open(split_path, 'r', encoding='utf-8') as f:
            split_map = json.load(f)
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            cutoff_year = json.load(f).get('cutoff_year')
    return split_map, cutoff_year


def write_csv(out_dir, name, fieldnames, rows):
    path = os.path.join(out_dir, f'{name}.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path, len(rows)


def build_patent_content(records):
    # The same patent may appear as target in one record and prior in another;
    # merge by patent_id.
    patents = {}

    def touch(pid, claim_text, ipc, has_structure):
        if not pid:
            return
        p = patents.setdefault(pid, {
            'patent_id': pid,
            'pub_date': None,          # Requires OPS /biblio integration (not yet implemented here).
            'app_date': None,          # Same as above.
            'family_id': None,         # Same as above.
            'ipc_codes': set(),
            'title': None,             # Requires OPS /biblio integration (not yet implemented here).
            'claims_text': None,
            'description_text': None,  # Original full specification text is unavailable.
            'has_structure': False,
        })
        for tok in str(ipc or '').replace(',', '|').split('|'):
            tok = tok.strip()
            if tok:
                p['ipc_codes'].add(tok)
        if claim_text and not p['claims_text']:
            p['claims_text'] = claim_text
        if has_structure:
            p['has_structure'] = True

    for r in records:
        t_struct = bool((r.get('structure_source') or {}).get('target'))
        p_struct = bool((r.get('structure_source') or {}).get('prior'))
        touch(r.get('target_patent_id'), r.get('target_claim'), r.get('target_ipc'), t_struct)
        touch(r.get('prior_patent_id'), r.get('prior_claim'), r.get('prior_ipc'), p_struct)

    rows = []
    for p in patents.values():
        p['ipc_codes'] = '|'.join(sorted(p['ipc_codes']))
        rows.append(p)
    fieldnames = ['patent_id', 'pub_date', 'app_date', 'family_id', 'ipc_codes',
                  'title', 'claims_text', 'description_text', 'has_structure']
    return fieldnames, rows


def build_molecule_and_patent_molecule(records):
    # Deduplicate by SMILES rather than mol_id; assign sequential IDs here.
    molecules = {}
    patent_molecule_rows = []

    def touch(patent_id, chem_meta):
        if not patent_id or not chem_meta:
            return
        for smi in (chem_meta.get('smiles_list') or []):
            if not smi:
                continue
            mol = molecules.setdefault(smi, {
                'mol_id': f'MOL_{len(molecules) + 1:06d}',
                'smiles': smi,
                'inchikey': chem_meta.get('inchikey'),
                'morgan_fp': None,  # Not precomputed (not yet implemented).
                'source': chem_meta.get('status', 'UNKNOWN'),
            })
            patent_molecule_rows.append({
                'patent_id': patent_id,
                'mol_id': mol['mol_id'],
                'location': None,  # Patent-level matching does not distinguish claim/description/figure locations.
                'unit_ref': None,
            })

    for r in records:
        touch(r.get('target_patent_id'), r.get('target_surechembl_metadata'))
        touch(r.get('prior_patent_id'), r.get('prior_surechembl_metadata'))

    mol_fieldnames = ['mol_id', 'smiles', 'inchikey', 'morgan_fp', 'source']
    pm_fieldnames = ['patent_id', 'mol_id', 'location', 'unit_ref']
    return mol_fieldnames, list(molecules.values()), pm_fieldnames, patent_molecule_rows


def build_examiner_citation(records):
    # Only ground_truth_label == 1 pairs are treated as actual cited prior-art gold.
    # db_export_2_fill_citation_category.py fills category/rejection_type.
    rows = []
    for i, r in enumerate(records):
        if r.get('ground_truth_label') != 1:
            continue
        if not r.get('target_patent_id') or not r.get('prior_patent_id'):
            continue
        rows.append({
            'cite_id': f'CITE_{i:06d}',
            'query_patent_id': r['target_patent_id'],
            'cited_patent_id': r['prior_patent_id'],
            'category': None,
            'rejection_type': None,
            'source': 'examiner',
        })
    fieldnames = ['cite_id', 'query_patent_id', 'cited_patent_id', 'category', 'rejection_type', 'source']
    return fieldnames, rows


def build_split(records, split_map, cutoff_year):
    # Expand record/filename-level split_map roles (target=query, prior=corpus)
    # to patent_id-level assignments.
    id_role_split = {}
    for r in records:
        rec_split = split_map.get(r.get('filename'))
        if not rec_split:
            continue
        for pid, role in ((r.get('target_patent_id'), 'query'), (r.get('prior_patent_id'), 'corpus')):
            if pid and pid not in id_role_split:
                id_role_split[pid] = {'id': pid, 'role': role, 'split': rec_split, 'time_cutoff': cutoff_year}

    fieldnames = ['id', 'role', 'split', 'time_cutoff']
    return fieldnames, list(id_role_split.values())


# Empty schemas until db_export_3_task2_tables.py populates them.
EMPTY_TABLES = {
    'claim_element': ['element_id', 'patent_id', 'claim_no', 'element_no', 'element_text', 'element_smiles'],
    'paragraph': ['para_id', 'patent_id', 'para_no', 'text'],
    'figure': ['fig_id', 'patent_id', 'fig_no', 'caption', 'image_path', 'ocsr_smiles'],
    'grounding_gold': ['ground_id', 'query_patent_id', 'claim_element_id', 'cited_patent_id',
                        'evidence_type', 'evidence_unit_id', 'source'],
    'exam_case': ['case_id', 'app_patent_id', 'claim_text', 'smiles_list', 'cited_ref_ids',
                  'rejection_type', 'outcome', 'decision_date'],
    'case_relevance': ['query_patent_id', 'case_id', 'relevance', 'basis'],
}


def main():
    records = load_dataset()
    split_map, cutoff_year = load_split()
    out_dir = os.path.join(DATA_DIR, 'db_export')
    os.makedirs(out_dir, exist_ok=True)

    coverage = {'source_dataset_record_count': len(records), 'tables': {}}

    fn, rows = build_patent_content(records)
    _, n = write_csv(out_dir, 'patent_content', fn, rows)
    coverage['tables']['patent_content'] = {
        'rows': n,
        'filled_pub_date': sum(1 for r in rows if r['pub_date']),
        'filled_title': sum(1 for r in rows if r['title']),
        'filled_has_structure_true': sum(1 for r in rows if r['has_structure']),
        'note': 'pub_date/app_date/family_id/title/description_text not populated (not yet implemented)',
    }

    mol_fn, mol_rows, pm_fn, pm_rows = build_molecule_and_patent_molecule(records)
    _, n_mol = write_csv(out_dir, 'molecule', mol_fn, mol_rows)
    _, n_pm = write_csv(out_dir, 'patent_molecule', pm_fn, pm_rows)
    coverage['tables']['molecule'] = {'rows': n_mol, 'note': 'morgan_fp not populated (not yet implemented)'}
    coverage['tables']['patent_molecule'] = {'rows': n_pm, 'note': 'No location breakdown available'}

    ec_fn, ec_rows = build_examiner_citation(records)
    _, n_ec = write_csv(out_dir, 'examiner_citation', ec_fn, ec_rows)
    coverage['tables']['examiner_citation'] = {
        'rows': n_ec,
        'filled_category': sum(1 for r in ec_rows if r['category']),
        'filled_rejection_type': sum(1 for r in ec_rows if r['rejection_type']),
        'note': 'category/rejection_type populated by db_export_2_fill_citation_category.py',
    }

    sp_fn, sp_rows = build_split(records, split_map, cutoff_year)
    _, n_sp = write_csv(out_dir, 'split', sp_fn, sp_rows)
    coverage['tables']['split'] = {'rows': n_sp}

    for name, fieldnames in EMPTY_TABLES.items():
        _, n = write_csv(out_dir, name, fieldnames, [])
        coverage['tables'][name] = {'rows': 0, 'note': 'Populated by db_export_3_task2_tables.py (except exam_case/case_relevance)'}

    coverage_path = os.path.join(out_dir, 'coverage_report.json')
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    print(f'Saved 11 CSV tables and coverage_report.json to {out_dir}.')
    print(json.dumps(coverage, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
