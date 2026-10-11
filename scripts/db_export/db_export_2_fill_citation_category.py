"""
data/db_export/examiner_citation.csv 의 category(X/Y/A) 열을 Project/epo_dataset/
parsed_citations_*.xlsx 로 채운다. db_export_1_core_tables.py를 먼저 실행해서
examiner_citation.csv가 있어야 함.

parsed_citations_*.xlsx에서 쓰는 컬럼:
  ID       - 타겟 특허의 EP 공개번호 (예: 'EP3253046A3')
  dnum     - 인용된 선행문헌 번호 (EP 외에도 US/JP/WO 등 섞여 있음)
  country  - dnum 발행국. EP인 것만 사용(현재 cited_patent_id가 전부 EP라서 조인 가능)
  category - EPO 표준 카테고리(X/Y/A) + 부가 플래그(D/P/T/E/L, 콤마 결합 가능). 가공 없이 그대로 저장.

rejection_type(102/103)은 채우지 않음 - USPTO 심사 조문 번호라 EPO의 X/Y/A 체계와
1:1 대응이 없고, 8/24 미팅에서 두 기준을 어떻게 정렬할지 아직 정해지지 않았음.

같은 (타겟,인용) 쌍이 여러 행으로 나오면 가장 심각한 카테고리(X>Y>A)를 채택.

사용법 (맥 .venv, openpyxl 필요 - 없으면 `pip install openpyxl`):
    python3 scripts/db_export/db_export_2_fill_citation_category.py
    # 기본 경로가 아니면: --citations-dir /path/to/epo_dataset
"""
import os
import sys
import csv
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR, PROJECT_DIR

try:
    import openpyxl
except ImportError:
    print('[FAIL] openpyxl이 설치되어 있지 않습니다: pip install openpyxl')
    sys.exit(1)

SEVERITY = {'X': 0, 'Y': 1, 'A': 2}


def severity_rank(category):
    primary = (category or '').split(',')[0].strip()
    return SEVERITY.get(primary, 99)


def build_category_map(citations_dir):
    xlsx_files = sorted(glob.glob(os.path.join(citations_dir, 'parsed_citations_*.xlsx')))
    if not xlsx_files:
        print(f'[FAIL] {citations_dir} 에서 parsed_citations_*.xlsx 파일을 찾지 못했습니다.')
        sys.exit(1)

    category_map = {}
    ep_row_count, total_rows = 0, 0

    for path in xlsx_files:
        print(f'읽는 중: {os.path.basename(path)} ...')
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        idx = {h: i for i, h in enumerate(header)}
        required = ('ID', 'dnum', 'country', 'category')
        if not all(c in idx for c in required):
            print(f'  [경고] {path}: 필요한 컬럼이 없어 건너뜀. 실제 컬럼: {header}')
            wb.close()
            continue

        for row in rows:
            total_rows += 1
            if row[idx['country']] != 'EP':
                continue
            target_id, cited_id, category = row[idx['ID']], row[idx['dnum']], row[idx['category']]
            if not target_id or not cited_id or not category:
                continue
            ep_row_count += 1
            key = (target_id, cited_id)
            existing = category_map.get(key)
            if existing is None or severity_rank(category) < severity_rank(existing):
                category_map[key] = category
        wb.close()

    print(f'\n전체 {len(xlsx_files)}개 연도 파일, {total_rows}행 스캔 완료.')
    print(f'EP 인용 행 {ep_row_count}건 -> 고유 (타겟,인용) 쌍 {len(category_map)}건.')
    return category_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--citations-dir', default=os.path.join(os.path.dirname(PROJECT_DIR), 'epo_dataset'))
    args = parser.parse_args()

    ec_path = os.path.join(DATA_DIR, 'db_export', 'examiner_citation.csv')
    if not os.path.exists(ec_path):
        print(f'[FAIL] {ec_path} 가 없습니다. 먼저 python3 scripts/db_export/db_export_1_core_tables.py 를 실행해주세요.')
        sys.exit(1)

    category_map = build_category_map(args.citations_dir)

    with open(ec_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    filled = 0
    category_value_counts = {}
    for row in rows:
        category = category_map.get((row['query_patent_id'], row['cited_patent_id']))
        if category:
            row['category'] = category
            filled += 1
            category_value_counts[category] = category_value_counts.get(category, 0) + 1

    with open(ec_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n{ec_path} 갱신 완료: 전체 {len(rows)}건 중 {filled}건에 category 채움 '
          f'({filled / len(rows) * 100:.1f}%).')
    print('채워진 category 값 분포:', category_value_counts)
    unmatched = len(rows) - filled
    if unmatched:
        print(f'\n[참고] {unmatched}건은 매칭 안 됨 - 연도 범위(2012~2023) 밖이거나 kind code 차이일 수 있음.')


if __name__ == '__main__':
    main()
