"""
OPS API로 patent_content의 서지 정보(pub_date/app_date/family_id/title, 전체 특허 대상)와
TARGET 특허의 전문(paragraph/figure/claim_element)을 채운다.
db_export_1/2/3_*.py를 먼저 실행해서 data/db_export/가 만들어져 있어야 함.

범위:
  biblio(pub_date/app_date/family_id/title) : patent_content의 전체 특허(TARGET+PRIOR)
  claims/description(paragraph/figure/claim_element) : TARGET 특허만
    - PRIOR은 이미 로컬 전문 XML(db_export_3)로 정확하게 채워져 있어 재작업 불필요
    - 기존 TARGET claim_element는 '- ' 구분자 근사치(claim_no 없음)였는데, 이번에
      OPS 기반(claim_no 있음)으로 교체됨

재실행 안전(resumable):
  data/db_export/ops_cache/<patent_id>.json 에 특허별 fetch 결과를 캐시함.
  이미 캐시된 특허는 재요청하지 않아서 중간에 끊겨도 다시 실행하면 이어서 진행됨.

주의: 무료(Non-paying) 등급은 요청 빈도 제한이 있어서 --sleep으로 호출 사이 대기 시간을
둠. 4,123개 전체를 한 번에 돌리기 전에 --limit으로 소량 먼저 테스트해볼 것을 권장.

사용법 (맥 .venv):
    export EPO_OPS_CONSUMER_KEY=... EPO_OPS_CONSUMER_SECRET=...
    python3 scripts/db_export/db_export_4_ops_fulltext.py --limit 20      # 먼저 20개만 테스트
    python3 scripts/db_export/db_export_4_ops_fulltext.py                 # 전체 실행(오래 걸림)
"""
import os
import sys
import csv
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR
from vision_ocsr.epo_ops_fetcher import EPOOPSFetcher


def read_csv_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_cache(cache_dir, patent_id):
    path = os.path.join(cache_dir, f'{patent_id}.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_cache(cache_dir, patent_id, data):
    with open(os.path.join(cache_dir, f'{patent_id}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def _strip_debug_keys(d):
    return {k: v for k, v in d.items() if k not in ('raw_xml', 'debug_raw_saved_at')}


def fetch_one(fetcher, patent_id, need_fulltext, sleep_sec):
    """biblio는 모든 특허, claims/description은 TARGET 특허(need_fulltext)일 때만 조회"""
    entry = {'patent_id': patent_id}

    entry['biblio'] = _strip_debug_keys(fetcher.fetch_biblio_data(patent_id))
    time.sleep(sleep_sec)

    if need_fulltext:
        entry['claims'] = _strip_debug_keys(fetcher.fetch_claims_data(patent_id))
        time.sleep(sleep_sec)
        entry['description'] = _strip_debug_keys(fetcher.fetch_description_data(patent_id))
        time.sleep(sleep_sec)

    return entry


def is_complete(cached, need_fulltext):
    if not cached or cached.get('biblio', {}).get('status') != 'SUCCESS':
        return False
    if need_fulltext and cached.get('claims', {}).get('status') != 'SUCCESS':
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='테스트용 - 앞에서 N개 특허만 처리')
    parser.add_argument('--sleep', type=float, default=0.5, help='API 호출 사이 대기 시간(초, rate limit 보호)')
    args = parser.parse_args()

    out_dir = os.path.join(DATA_DIR, 'db_export')
    cache_dir = os.path.join(out_dir, 'ops_cache')
    os.makedirs(cache_dir, exist_ok=True)

    pc_path = os.path.join(out_dir, 'patent_content.csv')
    split_path = os.path.join(out_dir, 'split.csv')
    if not os.path.exists(pc_path) or not os.path.exists(split_path):
        print('[FAIL] patent_content.csv / split.csv가 없습니다. db_export_1_core_tables.py 먼저 실행해주세요.')
        sys.exit(1)

    patents = read_csv_rows(pc_path)
    target_ids = {r['id'] for r in read_csv_rows(split_path) if r['role'] == 'query'}
    print(f'전체 특허 {len(patents)}개, 그 중 TARGET(전문까지 조회) {len(target_ids)}개')

    fetcher = EPOOPSFetcher()
    if not fetcher.enabled:
        print('[FAIL] Consumer Key/Secret이 설정되지 않았습니다: '
              'export EPO_OPS_CONSUMER_KEY=... / EPO_OPS_CONSUMER_SECRET=...')
        sys.exit(1)

    patent_ids = [p['patent_id'] for p in patents]
    if args.limit:
        patent_ids = patent_ids[:args.limit]

    n_new, n_cached, n_biblio_failed = 0, 0, 0
    for i, pid in enumerate(patent_ids, start=1):
        need_fulltext = pid in target_ids
        cached = load_cache(cache_dir, pid)
        if is_complete(cached, need_fulltext):
            n_cached += 1
        else:
            entry = fetch_one(fetcher, pid, need_fulltext, args.sleep)
            save_cache(cache_dir, pid, entry)
            n_new += 1
            if entry['biblio'].get('status') != 'SUCCESS':
                n_biblio_failed += 1
        if i % 50 == 0 or i == len(patent_ids):
            print(f'  {i}/{len(patent_ids)} 처리 (신규조회 {n_new}, 캐시재사용 {n_cached}, biblio실패 {n_biblio_failed})')

    print(f'\n조회 완료 - 신규 {n_new}건, 캐시 재사용 {n_cached}건, biblio 실패 {n_biblio_failed}건')

    # --- patent_content.csv: 이미 값이 있으면 덮어쓰지 않고 빈 값만 채움 ---
    filled = {'pub_date': 0, 'app_date': 0, 'family_id': 0, 'title': 0}
    for p in patents:
        cached = load_cache(cache_dir, p['patent_id'])
        biblio = (cached or {}).get('biblio', {})
        if biblio.get('status') != 'SUCCESS':
            continue
        field_map = {'pub_date': 'publication_date', 'app_date': 'application_date',
                     'family_id': 'family_id', 'title': 'title'}
        for csv_field, biblio_field in field_map.items():
            if biblio.get(biblio_field) and not p[csv_field]:
                p[csv_field] = biblio[biblio_field]
                filled[csv_field] += 1

    write_csv(pc_path, list(patents[0].keys()), patents)
    print(f'patent_content.csv 갱신: pub_date +{filled["pub_date"]}, app_date +{filled["app_date"]}, '
          f'family_id +{filled["family_id"]}, title +{filled["title"]}')

    # --- TARGET paragraph/figure/claim_element: 기존 TARGET 행(근사치 포함)은 버리고 OPS로 다시 채움 ---
    para_path = os.path.join(out_dir, 'paragraph.csv')
    fig_path = os.path.join(out_dir, 'figure.csv')
    ce_path = os.path.join(out_dir, 'claim_element.csv')

    para_rows = [r for r in read_csv_rows(para_path) if r['patent_id'] not in target_ids]
    fig_rows = [r for r in read_csv_rows(fig_path) if r['patent_id'] not in target_ids]
    ce_rows = [r for r in read_csv_rows(ce_path) if r['patent_id'] not in target_ids]

    n_para, n_fig, n_ce = 0, 0, 0
    for pid in target_ids:
        cached = load_cache(cache_dir, pid)
        if not cached:
            continue

        claims = cached.get('claims', {})
        if claims.get('status') == 'SUCCESS':
            for claim_no, element_no, text in claims.get('claim_elements', []):
                ce_rows.append({'element_id': f'{pid}_c{claim_no}_e{element_no}', 'patent_id': pid,
                                 'claim_no': claim_no, 'element_no': element_no,
                                 'element_text': text, 'element_smiles': None})
                n_ce += 1

        desc = cached.get('description', {})
        if desc.get('status') == 'SUCCESS':
            for j, para in enumerate(desc.get('paragraphs', []), start=1):
                para_no = para['para_no'] or f'h{j}'  # 번호 없는 소제목 문단은 hN으로 대체
                para_rows.append({'para_id': f'{pid}_p{para_no}', 'patent_id': pid,
                                   'para_no': para['para_no'], 'text': para['text']})
                n_para += 1
            for fid, meta in desc.get('figures', {}).items():
                fig_rows.append({'fig_id': f'{pid}_{fid}', 'patent_id': pid,
                                  'fig_no': meta['fig_no'], 'caption': meta['caption'],
                                  'image_path': meta['image_path'], 'ocsr_smiles': None})
                n_fig += 1

    write_csv(para_path, ['para_id', 'patent_id', 'para_no', 'text'], para_rows)
    write_csv(fig_path, ['fig_id', 'patent_id', 'fig_no', 'caption', 'image_path', 'ocsr_smiles'], fig_rows)
    write_csv(ce_path, ['element_id', 'patent_id', 'claim_no', 'element_no', 'element_text', 'element_smiles'], ce_rows)

    print(f'TARGET 행 추가: paragraph +{n_para}, figure +{n_fig}, claim_element +{n_ce}')
    print(f'(도면 image_path는 OPS 응답에 파일 정보가 없어 전부 None - 알려진 한계)')


if __name__ == '__main__':
    main()
