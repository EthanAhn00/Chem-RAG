"""
Task 2용 4개 테이블(Paragraph/Figure/ClaimElement/GroundingGold)을 채운다.
db_export_1_core_tables.py, db_export_2_fill_citation_category.py를 먼저 실행해서
data/db_export/ 가 만들어져 있어야 함.

커버리지 요약:
  Paragraph, Figure : PRIOR 특허만 채움. config.settings.EPO_XML_DIR에는 PRIOR 쪽 전문
                       XML만 있고 TARGET 쪽은 로컬에 없음 - TARGET Paragraph/Figure는
                       미착수(필요하면 EPO_SRPT 원본 tar에서 별도 추출 필요).
  ClaimElement      : PRIOR은 XML의 <claim>/<claim-text> 구조에서 한정사항 단위로 정확히
                       분리(claim_no 포함). TARGET은 전문 XML이 없어 청구항 텍스트를
                       '- ' 구분자로 근사 분리(claim_no 불명, 정확도는 PRIOR보다 낮음).
  GroundingGold     : parsed_citations의 rel_claims/paragraphs_expanded/passage_type_grouped을
                       옮김. claim_element_id는 TARGET ClaimElement가 근사치라 정확히 조인이
                       안 돼 비워두고, 대신 target_rel_claims(청구항 번호 문자열)를 남김.
                       query/cited 특허 둘 다 우리 코퍼스(patent_content)에 있는 것만 채택 -
                       epo_dataset은 12개년 전체라 범위가 훨씬 넓은데, 필터링 없이 그대로
                       쓰면 다른 테이블엔 없는 특허가 섞여 정합성이 깨짐.

사용법 (맥 .venv):
    python3 scripts/db_export/db_export_3_task2_tables.py [--citations-dir /path/to/epo_dataset]
"""
import os
import sys
import re
import csv
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR, EPO_XML_DIR, PROJECT_DIR
from data_parser.patent_fulltext_parser import parse_paragraphs, parse_figures, parse_claim_elements

try:
    import openpyxl
except ImportError:
    openpyxl = None


def parse_prior_xml(xml_path):
    """PRIOR 특허 전문 XML 파일 하나를 읽어 data_parser.patent_fulltext_parser로 위임.
    실제 태그 파싱 로직은 OPS API 응답에도 그대로 재사용됨(같은 ep-patent-document DTD)."""
    with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    paragraphs = parse_paragraphs(text)
    figures = parse_figures(text)
    claim_elements = parse_claim_elements(text)
    return paragraphs, figures, claim_elements


def split_target_claim_heuristic(text):
    """TARGET은 전문 XML이 없어 '- ' 구분자로 근사 분리(claim_no 불명)"""
    if not text:
        return []
    return [p.strip() for p in re.split(r';?\s+-\s+', text) if p.strip()]


def build_paragraph_and_figure_and_claimelement(records):
    para_rows, fig_rows, ce_rows = [], [], []
    seen_prior = set()

    for r in records:
        prior_id, prior_xml = r.get('prior_patent_id'), r.get('prior_xml_file')
        if prior_id and prior_xml and prior_id not in seen_prior:
            seen_prior.add(prior_id)
            xml_path = os.path.join(EPO_XML_DIR, prior_xml)
            if os.path.exists(xml_path):
                paragraphs, figures, claim_elements = parse_prior_xml(xml_path)
                for p in paragraphs:
                    para_rows.append({'para_id': f"{prior_id}_p{p['para_no']}", 'patent_id': prior_id,
                                       'para_no': p['para_no'], 'text': p['text']})
                for fid, meta in figures.items():
                    fig_rows.append({'fig_id': f"{prior_id}_{fid}", 'patent_id': prior_id,
                                      'fig_no': meta['fig_no'], 'caption': meta['caption'],
                                      'image_path': meta['image_path'], 'ocsr_smiles': None})
                for claim_no, element_no, el_text in claim_elements:
                    ce_rows.append({'element_id': f"{prior_id}_c{claim_no}_e{element_no}", 'patent_id': prior_id,
                                     'claim_no': claim_no, 'element_no': element_no,
                                     'element_text': el_text, 'element_smiles': None})

        target_id = r.get('target_patent_id')
        if target_id:
            for i, el_text in enumerate(split_target_claim_heuristic(r.get('target_claim')), start=1):
                ce_rows.append({'element_id': f"{target_id}_TGT_e{i}", 'patent_id': target_id,
                                 'claim_no': None, 'element_no': i,
                                 'element_text': el_text, 'element_smiles': None})

    return para_rows, fig_rows, ce_rows


def build_grounding_gold(citations_dir, known_patent_ids):
    """known_patent_ids에 둘 다(query/cited) 속하는 행만 채택 - epo_dataset은 우리
    코퍼스보다 훨씬 넓은 범위(12개년 전체)라, 이 필터 없이는 다른 테이블(patent_content 등)에
    없는 특허까지 섞여 들어와 데이터 정합성이 깨짐. 코퍼스 밖 인용은 지금은 버리고,
    나중에 코퍼스 자체를 확장할 때 이 필터를 넓히면 됨."""
    if openpyxl is None:
        print('[경고] openpyxl이 없어 GroundingGold를 건너뜁니다: pip install openpyxl')
        return []
    rows = []
    skipped_out_of_corpus = 0
    for path in sorted(glob.glob(os.path.join(citations_dir, 'parsed_citations_*.xlsx'))):
        print(f'GroundingGold용으로 읽는 중: {os.path.basename(path)} ...')
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb[wb.sheetnames[0]]
        it = ws.iter_rows(values_only=True)
        header = next(it)
        idx = {h: i for i, h in enumerate(header)}
        needed = ('ID', 'dnum', 'country', 'rel_claims', 'paragraphs_expanded', 'passage_type_grouped', 'citation_id', 'file_year')
        if not all(c in idx for c in needed):
            wb.close()
            continue
        for row in it:
            if row[idx['country']] != 'EP':
                continue
            target_id, cited_id = row[idx['ID']], row[idx['dnum']]
            if not target_id or not cited_id:
                continue
            if target_id not in known_patent_ids or cited_id not in known_patent_ids:
                skipped_out_of_corpus += 1
                continue
            paragraphs_expanded = row[idx['paragraphs_expanded']]
            evidence_units = [u.strip('[]') for u in str(paragraphs_expanded).split(',') if u.strip()] \
                if paragraphs_expanded else []
            evidence_type = 'figure' if 'figure' in str(row[idx['passage_type_grouped']] or '').lower() \
                else ('paragraph' if evidence_units else 'unknown')
            rows.append({
                'ground_id': f"{row[idx['file_year']]}_{row[idx['citation_id']]}_{target_id}",
                'query_patent_id': target_id,
                'claim_element_id': None,  # TARGET ClaimElement가 근사치라 조인 불가 - target_rel_claims 참고
                'target_rel_claims': row[idx['rel_claims']],
                'cited_patent_id': cited_id,
                'evidence_type': evidence_type,
                'evidence_unit_id': '|'.join(evidence_units) if evidence_units else None,
                'source': 'office_action',
            })
        wb.close()
    print(f'코퍼스 밖(patent_content에 없는 특허) 인용이라 제외한 행: {skipped_out_of_corpus}건')
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'{path}: {len(rows)}행')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--citations-dir', default=os.path.join(os.path.dirname(PROJECT_DIR), 'epo_dataset'))
    args = parser.parse_args()

    dataset_path = os.path.join(DATA_DIR, 'reference', 'epo_section_c_benchmark_dataset.json')
    if not os.path.exists(dataset_path):
        print(f'[FAIL] {dataset_path} 가 없습니다. 먼저 python3 run_chem_rag.py 를 실행해주세요.')
        sys.exit(1)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    out_dir = os.path.join(DATA_DIR, 'db_export')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(EPO_XML_DIR):
        print(f'[경고] {EPO_XML_DIR} 를 찾을 수 없습니다 - 외장하드 연결을 확인해주세요.')

    # patent_content.csv와 정합성을 맞추기 위해, 우리 코퍼스에 실제로 등장하는
    # patent_id 집합을 계산 - grounding_gold를 이 범위로 필터링하는 데 씀
    known_patent_ids = set()
    for r in records:
        known_patent_ids.update(pid for pid in (r.get('target_patent_id'), r.get('prior_patent_id')) if pid)

    para_rows, fig_rows, ce_rows = build_paragraph_and_figure_and_claimelement(records)
    write_csv(os.path.join(out_dir, 'paragraph.csv'), ['para_id', 'patent_id', 'para_no', 'text'], para_rows)
    write_csv(os.path.join(out_dir, 'figure.csv'),
              ['fig_id', 'patent_id', 'fig_no', 'caption', 'image_path', 'ocsr_smiles'], fig_rows)
    write_csv(os.path.join(out_dir, 'claim_element.csv'),
              ['element_id', 'patent_id', 'claim_no', 'element_no', 'element_text', 'element_smiles'], ce_rows)

    gg_rows = build_grounding_gold(args.citations_dir, known_patent_ids)
    write_csv(os.path.join(out_dir, 'grounding_gold.csv'),
              ['ground_id', 'query_patent_id', 'claim_element_id', 'target_rel_claims',
               'cited_patent_id', 'evidence_type', 'evidence_unit_id', 'source'], gg_rows)

    print('\n완료. exam_case.csv / case_relevance.csv(Task 3)는 원본 심사결과/심사사례 데이터를 '
          '로컬에서 찾지 못해 여전히 비어 있습니다.')


if __name__ == '__main__':
    main()
