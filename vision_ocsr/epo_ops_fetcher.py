import os
import re
import base64
import time
import ssl
import urllib.request
import urllib.error
import json

from config.settings import EPO_OPS_CONSUMER_KEY, EPO_OPS_CONSUMER_SECRET, OCSR_MAX_PAGES_PER_PATENT
from data_parser.patent_fulltext_parser import (
    parse_ops_paragraphs, parse_ops_figures_heuristic, parse_ops_claim_elements,
)


def _build_ssl_context():
    # python.org에서 받은 macOS Python은 시스템 인증서 저장소를 안 쓰기 때문에
    # CERTIFICATE_VERIFY_FAILED가 흔히 발생 - certifi의 CA 번들을 명시적으로 사용
    # (pip install certifi로 설치돼 있으면 자동 사용, 없으면 기본 컨텍스트로 폴백)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class EPOOPSFetcher:
    """EPO Open Patent Services(OPS) REST API로 공개 도면 이미지를 가져오는 클래스

    로컬 XML 코퍼스에는 도면 파일명만 참조돼 있고 실제 이미지가 없어서(원본 대용량
    아카이브 미보유), OPS API로 필요한 특허의 도면 페이지만 개별 조회해서 보충함.
    무료(Non-paying) 등급 기준 주당 3.5GB 트래픽 제한 - OCSR_MAX_PAGES_PER_PATENT로
    특허 1건당 받아오는 페이지 수를 제한해 쿼터를 보호함

    인증: OAuth2 client_credentials 플로우(Consumer Key/Secret -> access_token,
    토큰은 약 20분 후 만료되므로 만료 임박 시 자동 재발급)
    """

    AUTH_URL = 'https://ops.epo.org/3.2/auth/accesstoken'
    BASE_URL = 'https://ops.epo.org/3.2/rest-services'

    def __init__(self, output_dir=None, consumer_key=None, consumer_secret=None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fetched_drawings')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.consumer_key = consumer_key or EPO_OPS_CONSUMER_KEY
        self.consumer_secret = consumer_secret or EPO_OPS_CONSUMER_SECRET
        self.enabled = bool(self.consumer_key and self.consumer_secret)
        self._ssl_context = _build_ssl_context()

        self._access_token = None
        self._token_expires_at = 0

    def _get_access_token(self):
        # 만료 60초 전이면 미리 재발급(요청 도중 만료되는 것 방지)
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        credentials = f'{self.consumer_key}:{self.consumer_secret}'
        auth_header = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
        req = urllib.request.Request(
            self.AUTH_URL,
            data=b'grant_type=client_credentials',
            headers={
                'Authorization': f'Basic {auth_header}',
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=15, context=self._ssl_context) as resp:
            body = json.loads(resp.read().decode('utf-8'))

        self._access_token = body['access_token']
        self._token_expires_at = time.time() + int(body.get('expires_in', 1200))
        return self._access_token

    def clean_ep_number(self, ep_doc_number):
        # kind code까지 제거한 순수 숫자만 추출 - 캐시 키/파일명 전용, OPS 조회에는
        # 절대 쓰지 말 것(kind code를 지우면 안 되는 이유는 epodoc_reference() 참고)
        # 예: 'EP07075413NWB1' -> '07075413', 'EP-4013497-B1' -> '4013497'
        num = ep_doc_number.replace('EP', '').replace('-', '').strip()
        num = re.sub(r'(NWA|NWB|NW)\d*$', '', num)
        num = re.sub(r'[A-Z]\d*$', '', num)
        return num

    def epodoc_reference(self, ep_doc_number):
        # OPS 조회용 참조 - kind code(A1/B1 등)는 유지하되 대시는 제거.
        # kind code를 지우면 OPS가 페이지 구성이 다른 엉뚱한 공보를 돌려줄 수 있어
        # bbox가 잘못된 내용을 가리키게 됨(실제로 이 문제로 본문 텍스트가 크롭된 적
        # 있음). 대시는 PatCID identifiers의 SureChEMBL 포맷('EP-3705311-A1')에
        # 섞여 있는데 남기면 OPS가 404를 반환해서 제거 필요.
        ref = ep_doc_number.replace('EP', '').replace('-', '').strip()
        return ref

    def extract_kind_code(self, ep_doc_number):
        # 식별자에서 kind code만 추출. 'EP-3705311-A1' -> 'A1', 'EP3705311' -> None
        s = str(ep_doc_number).strip().replace('EP', '').replace('-', '')
        base = re.sub(r'(NWA|NWB|NW)\d*$', '', s)
        base = re.sub(r'[A-Z]\d*$', '', base)
        if base != s and base:
            return s[len(base):]
        return None

    def list_available_images(self, ep_doc_number):
        # 이미지 인벤토리 조회 - OPS images 엔드포인트는 kind code를 붙이면 404를
        # 반환하므로 항상 bare 번호로 조회. 대신 해당 특허의 모든 공보(A1, B1, ...)를
        # <ops:inquiry-result> 블록별로 한꺼번에 돌려주므로, 블록의 <kind> 태그로
        # 구분해 instances_by_kind로 분리 반환 - 실제 kind code 선택은
        # fetch_full_document_pages()에서 수행(공보마다 페이지 구성이 다르므로 아무거나
        # 쓰면 안 됨)
        if not self.enabled:
            return {'status': 'DISABLED', 'instances_by_kind': {}}

        cache_tag = self.clean_ep_number(ep_doc_number)
        bare_ref = self.clean_ep_number(ep_doc_number)
        url = f'{self.BASE_URL}/published-data/publication/epodoc/EP{bare_ref}/images'
        try:
            token = self._get_access_token()
            req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
            with urllib.request.urlopen(req, timeout=15, context=self._ssl_context) as resp:
                text = resp.read().decode('utf-8', errors='ignore')

            debug_path = os.path.join(self.output_dir, f'_debug_images_response_{cache_tag}.xml')
            try:
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception:
                pass

            # <ops:inquiry-result>...</ops:inquiry-result> 블록 단위로 잘라서 블록마다
            # kind code와 그 안의 document-instance들을 같이 묶어 파싱 (블록 경계를 넘어
            # 다른 kind code의 인스턴스와 섞이지 않도록)
            instances_by_kind = {}
            for block_match in re.finditer(r'<ops:inquiry-result\b.*?</ops:inquiry-result>', text, re.DOTALL):
                block = block_match.group(0)
                kind_m = re.search(r'<kind>([^<]+)</kind>', block)
                kind = kind_m.group(1).strip() if kind_m else 'UNKNOWN'
                instances = []
                for tag_match in re.finditer(r'<ops:document-instance\b[^>]*>', block):
                    tag = tag_match.group(0)
                    desc_m = re.search(r'desc="([^"]+)"', tag)
                    pages_m = re.search(r'number-of-pages="(\d+)"', tag)
                    link_m = re.search(r'link="([^"]+)"', tag)
                    if desc_m and link_m:
                        instances.append({
                            'desc': desc_m.group(1),
                            'number_of_pages': int(pages_m.group(1)) if pages_m else 1,
                            'link': link_m.group(1),
                        })
                if instances:
                    instances_by_kind.setdefault(kind, []).extend(instances)

            return {'status': 'SUCCESS', 'instances_by_kind': instances_by_kind, 'debug_raw_saved_at': debug_path}
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode('utf-8', errors='ignore')
            except Exception:
                err_body = ''
            debug_path = os.path.join(self.output_dir, f'_debug_error_{cache_tag}.txt')
            try:
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(f'URL: {url}\nHTTP {e.code}\n\n{err_body}')
            except Exception:
                pass
            return {'status': f'HTTP_ERROR_{e.code}', 'instances_by_kind': {},
                    'requested_url': url, 'error_body': err_body[:300], 'debug_raw_saved_at': debug_path}
        except Exception as e:
            return {'status': f'ERROR: {e}', 'instances_by_kind': {}, 'requested_url': url}

    def fetch_drawing_pages(self, ep_doc_number, max_pages=None):
        # 미사용 - fetch_full_document_pages()로 대체됨(kind code 선택 로직이 없어서
        # PatCID bbox와 짝을 맞출 수 없음). desc='Drawing' 인스턴스만 찾아 다운로드.
        if not self.enabled:
            return {'status': 'DISABLED', 'image_paths': [], 'message': 'EPO OPS Consumer Key/Secret이 설정되지 않음.'}

        max_pages = max_pages or OCSR_MAX_PAGES_PER_PATENT
        inventory = self.list_available_images(ep_doc_number)
        if inventory['status'] != 'SUCCESS':
            return {'status': inventory['status'], 'image_paths': [], 'message': '이미지 인벤토리 조회 실패'}

        all_instances = [i for kind_list in inventory['instances_by_kind'].values() for i in kind_list]
        # desc 값이 문서마다 'Drawing', 'FullDocumentDrawing' 등으로 다를 수 있어 'drawing'
        # 포함 여부로 느슨하게 매칭 (list_available_images가 남긴 debug XML로 실제 값 확인 가능)
        drawing_instance = next((i for i in all_instances if 'drawing' in i['desc'].lower()), None)
        if not drawing_instance:
            found_descs = [i['desc'] for i in all_instances]
            return {'status': 'NO_DRAWING_INSTANCE', 'image_paths': [],
                    'message': f'이 특허는 OPS 상 도면 인스턴스가 없음 (발견된 desc 목록: {found_descs})'}

        n_pages = min(drawing_instance['number_of_pages'], max_pages)
        ep_number = self.clean_ep_number(ep_doc_number)
        image_paths = []
        try:
            token = self._get_access_token()
            for page in range(1, n_pages + 1):
                save_path = os.path.join(self.output_dir, f'EP{ep_number}_drawing_p{page}.tiff')
                if os.path.exists(save_path):
                    image_paths.append(save_path)
                    continue
                url = f"{self.BASE_URL}/{drawing_instance['link']}.tiff?Range={page}"
                req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'Accept': 'application/tiff'})
                with urllib.request.urlopen(req, timeout=20, context=self._ssl_context) as resp, open(save_path, 'wb') as f:
                    f.write(resp.read())
                image_paths.append(save_path)
            return {'status': 'SUCCESS', 'image_paths': image_paths, 'message': f'{len(image_paths)}페이지 다운로드 완료'}
        except urllib.error.HTTPError as e:
            return {'status': f'HTTP_ERROR_{e.code}', 'image_paths': image_paths, 'message': str(e)}
        except Exception as e:
            return {'status': f'ERROR: {e}', 'image_paths': image_paths, 'message': str(e)}

    def fetch_full_document_pages(self, ep_doc_number, page_numbers, desired_kind=None):
        """FullDocument 인스턴스에서 PatCID bbox가 가리키는 절대 페이지 번호들만 다운로드
        (Drawing 전용 인스턴스의 자체 페이지 번호와는 다름 - vision_ocsr/patcid_bbox_cropper.py 참고)

        desired_kind: PatCID bbox가 실제로 계산된 공보의 kind code(예: 'A1'). 명시하지
        않으면 다른 kind code의 FullDocument를 받아와 bbox가 엉뚱한 페이지를 가리킬 수
        있어, list_available_images()가 kind code별로 분리해 돌려준 인벤토리에서 정확히
        이 kind code만 선택함

        반환: {page_number: image_path}
        """
        if not self.enabled:
            return {'status': 'DISABLED', 'image_paths': {}, 'message': 'EPO OPS Consumer Key/Secret이 설정되지 않음.'}

        inventory = self.list_available_images(ep_doc_number)
        if inventory['status'] != 'SUCCESS':
            detail_bits = []
            if inventory.get('requested_url'):
                detail_bits.append(f"URL: {inventory['requested_url']}")
            if inventory.get('error_body'):
                detail_bits.append(f"응답: {inventory['error_body']}")
            detail = (' | ' + ' / '.join(detail_bits)) if detail_bits else ''
            return {'status': inventory['status'], 'image_paths': {},
                    'message': f'이미지 인벤토리 조회 실패{detail}'}

        instances_by_kind = inventory['instances_by_kind']

        def _full_doc_in(kind_list):
            return next((i for i in kind_list if 'fulldocument' in i['desc'].lower()), None)

        full_doc_instance = None
        used_kind = None
        if desired_kind and desired_kind in instances_by_kind:
            full_doc_instance = _full_doc_in(instances_by_kind[desired_kind])
            used_kind = desired_kind
        elif not desired_kind:
            # PatCID identifiers에 kind code가 아예 없어 원하는 kind code를 알 수 없는
            # 경우에 한해서만 'A1'(최초 공개공보)을 기본값으로 시도 - PatCID가 화학구조
            # bbox를 계산할 때 주로 최초 공개된 A1 문서를 사용하는 경향이 있어서임
            # (desired_kind가 명시됐는데 못 찾은 경우는 아래에서 그냥 실패 처리 -
            # 임의의 다른 kind code로 조용히 대체하지 않음)
            if 'A1' in instances_by_kind:
                full_doc_instance = _full_doc_in(instances_by_kind['A1'])
                used_kind = 'A1'

        if not full_doc_instance:
            # 원하는 kind code에 FullDocument 인스턴스가 없으면(그 공보엔 이미지가 아예
            # 없는 경우) - 대신 신뢰할 수 있는 대체가 없다는 뜻이므로 절대 임의의 다른
            # kind code로 조용히 넘어가지 않고 실패로 보고 (bbox가 다른 레이아웃을
            # 가리키게 되는 문제를 방지) - 상위(ocsr_pipeline)에서 NOT_FOUND 처리
            available_kinds = list(instances_by_kind.keys())
            return {'status': 'DESIRED_KIND_NOT_AVAILABLE', 'image_paths': {},
                    'message': f'요청한 kind code({desired_kind})의 FullDocument 인스턴스가 없음 '
                                f'(사용 가능한 kind code: {available_kinds})'}

        ep_number = self.clean_ep_number(ep_doc_number)
        image_paths = {}
        try:
            token = self._get_access_token()
            for page in page_numbers:
                if page < 1 or page > full_doc_instance['number_of_pages']:
                    continue
                save_path = os.path.join(self.output_dir, f'EP{ep_number}_{used_kind}_fulldoc_p{page}.tiff')
                if os.path.exists(save_path):
                    image_paths[page] = save_path
                    continue
                url = f"{self.BASE_URL}/{full_doc_instance['link']}.tiff?Range={page}"
                req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'Accept': 'application/tiff'})
                with urllib.request.urlopen(req, timeout=20, context=self._ssl_context) as resp, open(save_path, 'wb') as f:
                    f.write(resp.read())
                image_paths[page] = save_path
            return {'status': 'SUCCESS', 'image_paths': image_paths, 'message': f'{len(image_paths)}페이지 다운로드 완료',
                    'used_kind': used_kind}
        except urllib.error.HTTPError as e:
            return {'status': f'HTTP_ERROR_{e.code}', 'image_paths': image_paths, 'message': str(e), 'used_kind': used_kind}
        except Exception as e:
            return {'status': f'ERROR: {e}', 'image_paths': image_paths, 'message': str(e), 'used_kind': used_kind}

    def fetch_biblio_data(self, ep_doc_number):
        """OPS published-data/biblio 엔드포인트로 서지 메타데이터 조회 - 특허번호만
        있으면 되고 원본 도면/전문 데이터는 필요 없음. PatCID/SureChEMBL과 달리
        여기에만 있는 정보: 발명 명칭(title), 출원인(applicants), 발명자(inventors),
        공개일(publication date), IPC 분류(classification). 필요하면 청구항 전문은
        같은 kind code로 /claims 엔드포인트를, 명세서 전문은 /description 엔드포인트를
        별도 호출해서 받을 수 있음(이 메서드는 biblio만 처리)

        kind code를 유지한 epodoc 참조로 먼저 시도하고, 404가 나면 images
        엔드포인트에서 이미 겪은 것과 같은 문제(kind code 붙은 조회 거부)일 수
        있어 kind code 없는 bare 참조로 한 번 더 시도함 - 둘 다 실패하면 그
        특허번호 자체가 없거나 형식이 잘못된 것
        """
        if not self.enabled:
            return {'status': 'DISABLED', 'message': 'EPO OPS Consumer Key/Secret이 설정되지 않음.'}

        ref = self.epodoc_reference(ep_doc_number)
        bare_ref = self.clean_ep_number(ep_doc_number)
        cache_tag = bare_ref

        attempts = [ref] if ref == bare_ref else [ref, bare_ref]
        last_error = None
        for attempt_ref in attempts:
            url = f'{self.BASE_URL}/published-data/publication/epodoc/EP{attempt_ref}/biblio'
            try:
                token = self._get_access_token()
                req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
                with urllib.request.urlopen(req, timeout=15, context=self._ssl_context) as resp:
                    text = resp.read().decode('utf-8', errors='ignore')

                debug_path = os.path.join(self.output_dir, f'_debug_biblio_{cache_tag}.xml')
                try:
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                except Exception:
                    pass

                # 정식 XML 파서 없이 정규식으로 필요한 필드만 뽑음(파일 전체의 다른
                # 메서드와 일관된 스타일) - 다국어 title 중 첫 번째, applicant/inventor는
                # 여러 명일 수 있어 findall로 전부 수집. EPO는 같은 이름을 대괄호 표기와
                # DOCDB 표기로 중복 기재하는 경우가 많아 순서를 유지한 채 중복 제거
                title_m = re.search(r'<invention-title[^>]*>([^<]+)</invention-title>', text)
                applicants = _dedupe_preserve_order(re.findall(r'<applicant-name>\s*<name>([^<]+)</name>', text))
                inventors = _dedupe_preserve_order(re.findall(r'<inventor-name>\s*<name>([^<]+)</name>', text))
                pub_date_m = re.search(r'<publication-reference>.*?<date>(\d{8})</date>', text, re.DOTALL)
                # <application-reference doc-id="..."> 처럼 속성이 붙어 있어 태그를 [^>]*로 매칭
                # (publication-reference와 다르게 항상 속성이 붙어 있음 - 실사 확인됨)
                app_date_m = re.search(r'<application-reference[^>]*>.*?<date>(\d{8})</date>', text, re.DOTALL)
                # patent family id는 <exchange-document family-id="..."> 속성으로 옴(본문 태그가 아님)
                family_id_m = re.search(r'<exchange-document[^>]*\bfamily-id="([^"]+)"', text)
                # <classification-ipcr sequence="N"> 처럼 속성이 붙어 있어 태그를 [^>]*로 매칭
                ipc_codes = _dedupe_preserve_order(re.findall(r'<classification-ipcr[^>]*>\s*<text>([^<]+)</text>', text))

                return {
                    'status': 'SUCCESS',
                    'used_reference': f'EP{attempt_ref}',
                    'title': title_m.group(1).strip() if title_m else None,
                    'applicants': applicants,
                    'inventors': inventors,
                    'publication_date': pub_date_m.group(1) if pub_date_m else None,
                    'application_date': app_date_m.group(1) if app_date_m else None,
                    'family_id': family_id_m.group(1) if family_id_m else None,
                    'ipc_classification': [c.strip() for c in ipc_codes],
                    'debug_raw_saved_at': debug_path,
                }
            except urllib.error.HTTPError as e:
                try:
                    err_body = e.read().decode('utf-8', errors='ignore')
                except Exception:
                    err_body = ''
                last_error = {'status': f'HTTP_ERROR_{e.code}', 'requested_url': url, 'error_body': err_body[:300]}
                if e.code != 404:
                    return last_error  # 404 외의 에러는 재시도해도 의미 없음
            except Exception as e:
                return {'status': f'ERROR: {e}', 'requested_url': url}

        return last_error

    def _fetch_fulltext_xml(self, endpoint, ep_doc_number):
        """/claims, /description처럼 전문 XML을 돌려주는 엔드포인트 공용 호출부.
        fetch_biblio_data()와 같은 kind-code-유지 -> bare 순 재시도 패턴을 씀.
        원본 XML 텍스트와 디버그 저장 경로만 반환하고, 태그 파싱은 호출자(fetch_claims_data/
        fetch_description_data)가 함 - OPS 응답은 로컬 XML 코퍼스와 태그 구조가 달라서
        (실사 확인됨: id/num 속성 없이 <claim-text>가 쭉 나열되는 'text-only' 포맷) 로컬용
        parse_paragraphs/parse_figures/parse_claim_elements가 아니라 OPS 전용 파서
        (parse_ops_*)를 씀
        """
        if not self.enabled:
            return {'status': 'DISABLED', 'message': 'EPO OPS Consumer Key/Secret이 설정되지 않음.'}

        ref = self.epodoc_reference(ep_doc_number)
        bare_ref = self.clean_ep_number(ep_doc_number)
        cache_tag = bare_ref

        attempts = [ref] if ref == bare_ref else [ref, bare_ref]
        last_error = None
        for attempt_ref in attempts:
            url = f'{self.BASE_URL}/published-data/publication/epodoc/EP{attempt_ref}/{endpoint}'
            try:
                token = self._get_access_token()
                req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
                with urllib.request.urlopen(req, timeout=15, context=self._ssl_context) as resp:
                    text = resp.read().decode('utf-8', errors='ignore')

                debug_path = os.path.join(self.output_dir, f'_debug_{endpoint}_{cache_tag}.xml')
                try:
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                except Exception:
                    pass

                return {'status': 'SUCCESS', 'used_reference': f'EP{attempt_ref}', 'raw_xml': text,
                        'debug_raw_saved_at': debug_path}
            except urllib.error.HTTPError as e:
                try:
                    err_body = e.read().decode('utf-8', errors='ignore')
                except Exception:
                    err_body = ''
                last_error = {'status': f'HTTP_ERROR_{e.code}', 'requested_url': url, 'error_body': err_body[:300]}
                if e.code != 404:
                    return last_error
            except Exception as e:
                return {'status': f'ERROR: {e}', 'requested_url': url}

        return last_error

    def fetch_claims_data(self, ep_doc_number):
        """OPS /claims 엔드포인트로 청구항 전문을 받아 claim_element 단위로 파싱.
        로컬 XML 코퍼스가 없는 TARGET 특허의 claim_no를 채우는 용도(기존엔 claim_no 없이
        '- ' 구분자 근사 분리만 가능했음) - element_no 세분화는 여전히 '- ' 구분자 근사"""
        result = self._fetch_fulltext_xml('claims', ep_doc_number)
        if result['status'] != 'SUCCESS':
            return result
        result['claim_elements'] = parse_ops_claim_elements(result['raw_xml'])
        return result

    def fetch_description_data(self, ep_doc_number):
        """OPS /description 엔드포인트로 명세서 전문을 받아 문단/도면을 파싱.
        TARGET 특허의 paragraph/figure를 채우는 용도(기존엔 PRIOR만 로컬 XML로 가능했음).
        도면은 OPS 응답에 <figure> 태그 자체가 없어 "Fig. N ..." 문장 매칭 근사치이고
        image_path는 항상 None(원본 도면 파일 정보가 없음)"""
        result = self._fetch_fulltext_xml('description', ep_doc_number)
        if result['status'] != 'SUCCESS':
            return result
        result['paragraphs'] = parse_ops_paragraphs(result['raw_xml'])
        result['figures'] = parse_ops_figures_heuristic(result['raw_xml'])
        return result
