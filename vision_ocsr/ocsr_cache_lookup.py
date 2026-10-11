import os
import json
import re

from config.settings import OCSR_CACHE_DIR


def _clean_ep_number(ep_doc_number):
    # epo_ops_fetcher.EPOOPSFetcher.clean_ep_number()과 동일한 정규화 로직을
    # 별도 모듈로 복제 - cv2/decimer 없이도 캐시 조회가 가능하도록 의존성을 분리
    num = ep_doc_number.replace('EP', '').replace('-', '').strip()
    num = re.sub(r'(NWA|NWB|NW)\d*$', '', num)
    num = re.sub(r'[A-Z]\d*$', '', num)
    return num


def get_cached_smiles(ep_doc_number):
    """scripts/ocsr/run_ocsr_fallback.py(별도 OCSR 전용 가상환경에서 실행)가 미리 계산해둔
    OCSR 결과 캐시를 조회. cv2/decimer를 임포트하지 않는 순수 표준 라이브러리 함수라
    run_chem_rag.py(메인 파이프라인 .venv)에서 안전하게 쓸 수 있음.
    캐시가 없으면 None 반환 - 아직 OCSR을 돌리지 않았다는 뜻(구조 없음과는 다름)"""
    if not ep_doc_number:
        return None
    ep_number = _clean_ep_number(ep_doc_number)
    cache_path = os.path.join(OCSR_CACHE_DIR, f'EP{ep_number}.json')
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result.get('smiles')
