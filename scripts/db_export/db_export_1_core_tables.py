"""
11개 테이블 DB 스키마 중 Task 1(선행기술 검색)에 필요한 5개 테이블
(PatentContent, Molecule, PatentMolecule, ExaminerCitation, Split)을 채우고, 나머지
Task 2/3용 6개 테이블은 빈 스키마(헤더만)로 만든다. 새 계산 로직은 없고 파이프라인이
이미 만들어둔 값을 스키마 모양으로 옮기기만 함.

이어서 실행할 스크립트:
  db_export_2_fill_citation_category.py - examiner_citation의 category(X/Y/A) 채움
  db_export_3_task2_tables.py           - paragraph/figure/claim_element/grounding_gold 채움

사용법 (맥 .venv):
    python3 run_chem_rag.py                    # data/reference/epo_section_c_benchmark_dataset.json 최신화
    python3 scripts/db_export/db_export_1_core_tables.py

출력: data/db_export/ 아래 11개 테이블 CSV + coverage_report.json(테이블별 행 수/빈 열 요약)
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
        print(f'[FAIL] {path} 가 없습니다. 먼저 python3 run_chem_rag.py 를 실행해주세요.')
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
    # 같은 특허가 어떤 레코드에서는 target, 다른 레코드에서는 prior로 등장할 수 있어
    # patent_id 단위로 병합
    patents = {}

    def touch(pid, claim_text, ipc, has_structure):
        if not pid:
            return
        p = patents.setdefault(pid, {
            'patent_id': pid,
            'pub_date': None,          # OPS /biblio 연동 필요(미착수)
            'app_date': None,          # 위와 동일
            'family_id': None,         # 위와 동일
            'ipc_codes': set(),
            'title': None,             # OPS /biblio 연동 필요(미착수)
            'claims_text': None,
            'description_text': None,  # 원본 명세서 전문 없음
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
    # SMILES 문자열을 mol_id 대신 dedup 키로 씀 - 순번 id는 여기서 새로 채번
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
                'morgan_fp': None,  # 사전계산 안 됨(미착수)
                'source': chem_meta.get('status', 'UNKNOWN'),
            })
            patent_molecule_rows.append({
                'patent_id': patent_id,
                'mol_id': mol['mol_id'],
                'location': None,  # 특허 단위 매칭이라 claim/description/figure 구분 없음
                'unit_ref': None,
            })

    for r in records:
        touch(r.get('target_patent_id'), r.get('target_surechembl_metadata'))
        touch(r.get('prior_patent_id'), r.get('prior_surechembl_metadata'))

    mol_fieldnames = ['mol_id', 'smiles', 'inchikey', 'morgan_fp', 'source']
    pm_fieldnames = ['patent_id', 'mol_id', 'location', 'unit_ref']
    return mol_fieldnames, list(molecules.values()), pm_fieldnames, patent_molecule_rows


def build_examiner_citation(records):
    # ground_truth_label == 1 인 쌍만 "실제 인용된 선행특허"(gold)로 취급.
    # category/rejection_type은 db_export_2_fill_citation_category.py 가 채움.
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
    # split_map은 레코드(파일명) 단위 train/test - target은 query, prior는 corpus
    # 역할로 펼쳐서 patent_id 단위로 변환
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


# db_export_3_task2_tables.py 가 채우기 전까지는 빈 스키마만
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
        'note': 'pub_date/app_date/family_id/title/description_text 미채움 (미착수)',
    }

    mol_fn, mol_rows, pm_fn, pm_rows = build_molecule_and_patent_molecule(records)
    _, n_mol = write_csv(out_dir, 'molecule', mol_fn, mol_rows)
    _, n_pm = write_csv(out_dir, 'patent_molecule', pm_fn, pm_rows)
    coverage['tables']['molecule'] = {'rows': n_mol, 'note': 'morgan_fp 미채움 (미착수)'}
    coverage['tables']['patent_molecule'] = {'rows': n_pm, 'note': 'location 구분 정보 없음'}

    ec_fn, ec_rows = build_examiner_citation(records)
    _, n_ec = write_csv(out_dir, 'examiner_citation', ec_fn, ec_rows)
    coverage['tables']['examiner_citation'] = {
        'rows': n_ec,
        'filled_category': sum(1 for r in ec_rows if r['category']),
        'filled_rejection_type': sum(1 for r in ec_rows if r['rejection_type']),
        'note': 'category/rejection_type은 db_export_2_fill_citation_category.py 에서 채움',
    }

    sp_fn, sp_rows = build_split(records, split_map, cutoff_year)
    _, n_sp = write_csv(out_dir, 'split', sp_fn, sp_rows)
    coverage['tables']['split'] = {'rows': n_sp}

    for name, fieldnames in EMPTY_TABLES.items():
        _, n = write_csv(out_dir, name, fieldnames, [])
        coverage['tables'][name] = {'rows': 0, 'note': 'db_export_3_task2_tables.py 에서 채움(exam_case/case_relevance 제외)'}

    coverage_path = os.path.join(out_dir, 'coverage_report.json')
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    print(f'11개 테이블 CSV + coverage_report.json 을 {out_dir} 에 저장했습니다.')
    print(json.dumps(coverage, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
