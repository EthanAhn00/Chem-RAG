"""
OCSR 폴백 사전 계산 스크립트 - 반드시 별도의 OCSR 전용 가상환경(Python 3.10~3.13,
decimer 설치됨)에서 실행할 것. 메인 파이프라인 .venv(Python 3.14)에는
decimer/tensorflow가 설치되지 않으므로 여기서 실행하면 안 됨

동작: SureChEMBL에 구조가 없고 PatCID bbox 어노테이션(실제 화학구조가 있는
페이지/좌표)이 있는 특허번호들을 모아 OCSRPipeline.resolve_smiles()로 하나씩
처리 - 결과는 data/ocsr_cache/EP*.json에 저장되며, 이후 메인 .venv의
run_chem_rag.py는 이 캐시를 읽기만 함 (vision_ocsr/ocsr_cache_lookup.py 참고)

PatCID 매핑이 없는 특허는 애초에 후보에서 제외됨 - 범용 윤곽선 검출로 대충 잘라
억지로 처리하지 않음(vision_ocsr/ocsr_pipeline.py 참고. 초기 버전은 이 방식으로
텍스트/화살표 등을 화학구조로 오인해 무의미한 SMILES를 만들어낸 바 있음)

사용법:
    python3 scripts/ocsr/run_ocsr_fallback.py --max-samples 10000 --limit 50
    (--limit: OPS 쿼터 보호용 - 이번 실행에서 새로 처리할 특허 수 상한. 이미
     캐시된 특허는 제외하고 계산되므로 여러 번 나눠 돌려도 됨)
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_parser.epo_stream_parser import EPOStreamParser
from data_parser.patcid_surechembl_connector import PatCIDSureChEMBLConnector, normalize_patent_number
from vision_ocsr.ocsr_pipeline import OCSRPipeline
from config.settings import OCSR_CACHE_DIR


def collect_candidates(max_samples, surechembl_connector):
    # SureChEMBL 미매칭 + PatCID bbox 어노테이션 존재 특허번호만 후보로 수집
    # (target/prior 통합, 중복 제거) - XML <img> 태그 존재 여부(has_drawing)는
    # 이미지 자체가 로컬에 없어 크롭 가능 여부와 무관하므로 더 이상 필터로 안 씀
    stream_parser = EPOStreamParser()
    records = stream_parser.stream_section_c_records(max_samples=max_samples)

    candidates = set()
    for r in records:
        surechembl_connector.enrich_patent_record(r)
        target_has_smiles = bool((r.get('target_surechembl_metadata') or {}).get('smiles_list'))
        prior_has_smiles = bool((r.get('prior_surechembl_metadata') or {}).get('smiles_list'))

        for role, patent_id, has_smiles in (
            ('target', r.get('target_patent_id'), target_has_smiles),
            ('prior', r.get('prior_patent_id'), prior_has_smiles),
        ):
            if has_smiles or not patent_id:
                continue
            clean_id = normalize_patent_number(patent_id)
            patcid_entry = surechembl_connector.patcid_annotations_map.get(clean_id)
            if patcid_entry and patcid_entry.get('figures'):
                candidates.add(patent_id)

    return candidates


def already_cached(pipeline, ep_number):
    return os.path.exists(pipeline._cache_path(ep_number))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-samples', type=int, default=10000)
    ap.add_argument('--limit', type=int, default=50, help='이번 실행에서 새로 처리할 특허 수 상한(쿼터 보호)')
    args = ap.parse_args()

    print('후보 특허번호 수집 중 (SureChEMBL 미매칭 + PatCID bbox 존재)...')
    surechembl_connector = PatCIDSureChEMBLConnector()
    candidates = collect_candidates(args.max_samples, surechembl_connector)
    print(f'후보 {len(candidates)}건 발견')

    pipeline = OCSRPipeline(patcid_annotations_map=surechembl_connector.patcid_annotations_map)
    if not pipeline.fetcher.enabled:
        print('[중단] OPS Consumer Key/Secret이 설정되지 않았습니다 (config/settings.py 확인).')
        sys.exit(1)

    todo = [c for c in candidates if not already_cached(pipeline, c)]
    print(f'이미 캐시된 것 제외 - 이번에 처리할 후보 {len(todo)}건 (상한 {args.limit}건)')

    processed, recovered = 0, 0
    for ep_number in todo[:args.limit]:
        res = pipeline.resolve_smiles(ep_number)
        processed += 1
        status = res.get('status')
        if res.get('smiles'):
            recovered += 1
            print(f'  [{processed}] {ep_number}: SUCCESS ({res.get("crop_method")}) - {res["smiles"][:40]}')
        else:
            print(f'  [{processed}] {ep_number}: {status} - {res.get("detail", "")}')

    print(f'\n완료 - 처리 {processed}건 중 구조 인식 성공 {recovered}건')
    print(f'캐시 위치: {OCSR_CACHE_DIR}')
    if len(todo) > args.limit:
        print(f'남은 미처리 후보 {len(todo) - args.limit}건 - --limit을 늘려 다시 실행하면 이어서 처리됩니다.')


if __name__ == '__main__':
    main()
