import os
import json

from config.settings import OCSR_CACHE_DIR
from vision_ocsr.epo_ops_fetcher import EPOOPSFetcher
from vision_ocsr.patcid_bbox_cropper import crop_by_patcid_bbox
from vision_ocsr.ocsr_smiles_converter import OCSRSMILESConverter
from data_parser.patcid_surechembl_connector import get_best_clean_smiles


class OCSRPipeline:
    """SureChEMBL 매칭이 실패한 특허에 한해, 화학구조 데이터를 보강하는 폴백 파이프라인.
    두 소스를 순서대로 시도:

    1) PatCID 자체 인식 SMILES (최우선) - PatCID(IBM Deep Search, Nature 2024)가 도면
       분석 시 이미 계산해둔 고신뢰 SMILES(class='Clean'). 있으면 이걸 그대로 매칭해
       쓰고 아래 2)는 시도하지 않음.

    2) OPS 페이지 조회 -> PatCID bbox 크롭 -> DECIMER OCSR 인식
       기본 비활성화(ENABLE_BBOX_CROP_FALLBACK=False) - bbox 크롭 결과를 검증해보니
       PatCID의 'page' 필드 의미가 우리 가정과 달라 신뢰할 수 없는 크롭이 나왔음.
       해소 전까지는 억지 결과 대신 NOT_FOUND 반환.

    결과는 특허 단위로 캐싱(data/ocsr_cache/{ep_number}.json)해 재조회 시 OPS 쿼터와
    OCSR 연산을 재사용하지 않도록 함
    """

    ENABLE_BBOX_CROP_FALLBACK = False  # 사유: 클래스 docstring 2) 참고

    def __init__(self, patcid_annotations_map=None):
        self.fetcher = EPOOPSFetcher()
        self.converter = OCSRSMILESConverter()
        # {clean_ep_number: {'figures': [...], 'page_dimensions': [...]}} -
        # data_parser/patcid_surechembl_connector.py의 patcid_annotations_map을 그대로 주입받아
        # 231k건 규모의 맵을 이 클래스에서 다시 로드하지 않도록 함
        self.patcid_annotations_map = patcid_annotations_map or {}

    def _cache_path(self, ep_doc_number):
        ep_number = self.fetcher.clean_ep_number(ep_doc_number)
        return os.path.join(OCSR_CACHE_DIR, f'EP{ep_number}.json')

    def resolve_smiles(self, ep_doc_number):
        """주어진 EP 특허번호의 도면에서 SMILES를 인식. 실패 시 None 반환
        (SureChEMBL의 NOT_FOUND와 동일하게, "인식 실패"와 "유사도 0"을 구분하기 위해
        임의의 값을 대신 채우지 않음)"""
        if not ep_doc_number:
            return {'status': 'NO_EP_NUMBER', 'smiles': None}

        cache_path = self._cache_path(ep_doc_number)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        result = self._resolve_uncached(ep_doc_number)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
        return result

    def _resolve_uncached(self, ep_doc_number):
        ep_number = self.fetcher.clean_ep_number(ep_doc_number)
        patcid_entry = self.patcid_annotations_map.get(ep_number)
        if not patcid_entry or not patcid_entry.get('figures'):
            return {'status': 'NO_PATCID_BBOX', 'smiles': None,
                    'detail': 'PatCID 좌표 매핑이 없어 신뢰 가능한 크롭 불가 (윤곽선 추정으로 대체하지 않음)'}

        # 1순위: PatCID 자체 인식 SMILES - OPS 조회 없이 바로 반환 (클래스 docstring 참고)
        best_clean = get_best_clean_smiles(patcid_entry)
        if best_clean:
            return {
                'status': 'SUCCESS', 'smiles': best_clean['smiles'],
                'confidence': best_clean.get('confidence'), 'source_page': best_clean.get('page'),
                'crop_method': 'patcid_precomputed', 'engine': 'PatCID Nature 2024 (IBM Deep Search)',
            }

        if not self.ENABLE_BBOX_CROP_FALLBACK:
            return {'status': 'NOT_FOUND', 'smiles': None,
                    'detail': 'PatCID 자체 인식 SMILES 없음(Markush/Trash만 존재) - '
                               'bbox 크롭 폴백은 page 필드 의미 미검증으로 비활성화됨'}

        page_dims_by_page = {d['page']: d for d in patcid_entry.get('page_dimensions', [])}
        figures = patcid_entry['figures']
        page_numbers = sorted({fig['page'] for fig in figures if 'page' in fig})

        # PatCID가 실제로 분석한 공보(patent_id, kind code 포함)의 kind code를 명시적으로
        # 지정해 그 kind code의 FullDocument만 정확히 조회 - OPS images 엔드포인트는
        # 같은 특허번호라도 kind code(A1/B1 등)마다 완전히 다른 페이지 구성의 별도
        # 공보를 갖고 있어서, kind code를 명시하지 않으면 bbox 좌표가 엉뚱한 페이지
        # 내용(본문 텍스트 등)을 가리키게 됨 - epo_ops_fetcher.fetch_full_document_pages() 참고
        patcid_reference = patcid_entry.get('patent_id') or ep_doc_number
        desired_kind = self.fetcher.extract_kind_code(patcid_reference)
        fetch_res = self.fetcher.fetch_full_document_pages(patcid_reference, page_numbers, desired_kind=desired_kind)
        if fetch_res['status'] != 'SUCCESS' or not fetch_res['image_paths']:
            return {'status': fetch_res['status'], 'smiles': None, 'detail': fetch_res.get('message')}

        for fig in figures:
            page = fig.get('page')
            page_image_path = fetch_res['image_paths'].get(page)
            dims = page_dims_by_page.get(page)
            if not page_image_path or not dims or 'bbox' not in fig:
                continue

            crop_path = os.path.join(
                self.fetcher.output_dir,
                f"EP{ep_number}_p{page}_bbox_{'_'.join(str(round(c, 2)) for c in fig['bbox'])}.png"
            )
            cropped = crop_by_patcid_bbox(
                page_image_path, fig['bbox'], dims.get('width'), dims.get('height'), crop_path
            )
            if not cropped:
                continue

            ocsr_res = self.converter.convert_image_to_smiles(cropped)
            if ocsr_res.get('status') == 'SUCCESS' and ocsr_res.get('smiles'):
                return {
                    'status': 'SUCCESS', 'smiles': ocsr_res['smiles'],
                    'source_page': page_image_path, 'source_crop': cropped,
                    'crop_method': 'patcid_bbox', 'engine': ocsr_res.get('engine'),
                }

        return {'status': 'NOT_FOUND', 'smiles': None, 'detail': 'PatCID bbox 영역에서 유효한 구조를 인식하지 못함'}
