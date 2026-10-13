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
    # macOS Python from python.org does not use the system certificate store;
    # use certifi's CA bundle explicitly to avoid CERTIFICATE_VERIFY_FAILED.
    # Use certifi when installed; otherwise fall back to the default SSL context.
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
    """Retrieve public drawing images via EPO Open Patent Services (OPS) REST API. Local XML contains drawing filenames but not image files because the original archive is unavailable. Download required pages individually. The non-paying tier's 3.5GB weekly traffic allowance is protected by OCSR_MAX_PAGES_PER_PATENT.
    OAuth2 client_credentials exchanges Consumer Key/Secret for an access_token; tokens expire after about 20 minutes and are refreshed before expiration.
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
        # Refresh 60 seconds before expiration to avoid expiry during a request.
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
        # Strip to numeric identifiers only for cache keys/filenames, never OPS lookup;
        # epodoc_reference() explains why OPS references retain kind codes.
        # Examples: 'EP07075413NWB1' -> '07075413', 'EP-4013497-B1' -> '4013497'.
        num = ep_doc_number.replace('EP', '').replace('-', '').strip()
        num = re.sub(r'(NWA|NWB|NW)\d*$', '', num)
        num = re.sub(r'[A-Z]\d*$', '', num)
        return num

    def epodoc_reference(self, ep_doc_number):
        # OPS reference: retain kind code (A1/B1 etc.), but remove hyphens.
        # Removing kind codes can retrieve a different publication/page layout,
        # causing incorrect bbox crops; body text was cropped in a previous incident.
        # PatCID identifiers may use SureChEMBL notation ('EP-3705311-A1');
        # retaining those hyphens causes OPS 404 responses.
        ref = ep_doc_number.replace('EP', '').replace('-', '').strip()
        return ref

    def extract_kind_code(self, ep_doc_number):
        # Extract kind code: 'EP-3705311-A1' -> 'A1', 'EP3705311' -> None.
        s = str(ep_doc_number).strip().replace('EP', '').replace('-', '')
        base = re.sub(r'(NWA|NWB|NW)\d*$', '', s)
        base = re.sub(r'[A-Z]\d*$', '', base)
        if base != s and base:
            return s[len(base):]
        return None

    def list_available_images(self, ep_doc_number):
        # OPS images rejects kind-coded references with 404; query the bare number.
        # The response includes publications A1, B1, etc. in separate
        # <ops:inquiry-result> blocks; split by each block's <kind> tag
        # into instances_by_kind. Select the intended kind code in
        # fetch_full_document_pages(); publication layouts differ and are not
        # interchangeable.
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

            # Parse each <ops:inquiry-result> block together with its own kind code
            # and document-instance entries to avoid mixing publications
            # across block boundaries.
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
        # Unused: replaced by fetch_full_document_pages(). Without kind selection,
        # this method cannot align PatCID boxes; it downloads only desc='Drawing'.
        if not self.enabled:
            return {'status': 'DISABLED', 'image_paths': [], 'message': 'EPO OPS Consumer Key/Secret is not configured.'}

        max_pages = max_pages or OCSR_MAX_PAGES_PER_PATENT
        inventory = self.list_available_images(ep_doc_number)
        if inventory['status'] != 'SUCCESS':
            return {'status': inventory['status'], 'image_paths': [], 'message': 'Image inventory lookup failed'}

        all_instances = [i for kind_list in inventory['instances_by_kind'].values() for i in kind_list]
        # Match 'drawing' loosely because desc can be 'Drawing', 'FullDocumentDrawing',
        # etc.; inspect actual values in list_available_images debug XML.
        drawing_instance = next((i for i in all_instances if 'drawing' in i['desc'].lower()), None)
        if not drawing_instance:
            found_descs = [i['desc'] for i in all_instances]
            return {'status': 'NO_DRAWING_INSTANCE', 'image_paths': [],
                    'message': f'No OPS drawing instances for this patent (available desc values: {found_descs})'}

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
            return {'status': 'SUCCESS', 'image_paths': image_paths, 'message': f'{len(image_paths)}page download complete'}
        except urllib.error.HTTPError as e:
            return {'status': f'HTTP_ERROR_{e.code}', 'image_paths': image_paths, 'message': str(e)}
        except Exception as e:
            return {'status': f'ERROR: {e}', 'image_paths': image_paths, 'message': str(e)}

    def fetch_full_document_pages(self, ep_doc_number, page_numbers, desired_kind=None):
        """Download FullDocument pages using PatCID's absolute page numbers, not Drawing-instance page numbers (see vision_ocsr/patcid_bbox_cropper.py). desired_kind specifies the publication used for PatCID bbox computation, e.g. 'A1'. Select only that kind from list_available_images(), since another publication can have a different page layout. Return {page_number: image_path}."""
        if not self.enabled:
            return {'status': 'DISABLED', 'image_paths': {}, 'message': 'EPO OPS Consumer Key/Secret is not configured.'}

        inventory = self.list_available_images(ep_doc_number)
        if inventory['status'] != 'SUCCESS':
            detail_bits = []
            if inventory.get('requested_url'):
                detail_bits.append(f"URL: {inventory['requested_url']}")
            if inventory.get('error_body'):
                detail_bits.append(f"Response: {inventory['error_body']}")
            detail = (' | ' + ' / '.join(detail_bits)) if detail_bits else ''
            return {'status': inventory['status'], 'image_paths': {},
                    'message': f'Image inventory lookup failed{detail}'}

        instances_by_kind = inventory['instances_by_kind']

        def _full_doc_in(kind_list):
            return next((i for i in kind_list if 'fulldocument' in i['desc'].lower()), None)

        full_doc_instance = None
        used_kind = None
        if desired_kind and desired_kind in instances_by_kind:
            full_doc_instance = _full_doc_in(instances_by_kind[desired_kind])
            used_kind = desired_kind
        elif not desired_kind:
            # Only when PatCID identifiers contain no kind code and the intended kind
            # is unknown, try 'A1' (first publication) as the default;
            # PatCID commonly computes boxes from the initial A1 publication.
            # If an explicitly requested desired_kind is absent, fail below rather than
            # silently substituting another kind code.
            if 'A1' in instances_by_kind:
                full_doc_instance = _full_doc_in(instances_by_kind['A1'])
                used_kind = 'A1'

        if not full_doc_instance:
            # If the requested publication lacks a FullDocument instance, there is no
            # trustworthy substitute. Never silently select a different kind code;
            # report failure to avoid using bbox coordinates on another layout.
            # The upstream ocsr_pipeline treats this as NOT_FOUND.
            available_kinds = list(instances_by_kind.keys())
            return {'status': 'DESIRED_KIND_NOT_AVAILABLE', 'image_paths': {},
                    'message': f'No FullDocument instance for requested kind code ({desired_kind}) '
                                f'(available kind codes: {available_kinds})'}

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
            return {'status': 'SUCCESS', 'image_paths': image_paths, 'message': f'{len(image_paths)}page download complete',
                    'used_kind': used_kind}
        except urllib.error.HTTPError as e:
            return {'status': f'HTTP_ERROR_{e.code}', 'image_paths': image_paths, 'message': str(e), 'used_kind': used_kind}
        except Exception as e:
            return {'status': f'ERROR: {e}', 'image_paths': image_paths, 'message': str(e), 'used_kind': used_kind}

    def fetch_biblio_data(self, ep_doc_number):
        """Fetch bibliographic metadata from OPS published-data/biblio using only a patent number. Retrieves title, applicants, inventors, publication date, and IPC classification not provided by the local PatCID/SureChEMBL connection. This method processes biblio only; claims and description require separate endpoints with the same kind code.
        Try a kind-preserving epodoc reference first, then a bare reference on 404, since kind-coded references may be rejected as with images. Failure of both suggests an absent or malformed patent identifier.
        """
        if not self.enabled:
            return {'status': 'DISABLED', 'message': 'EPO OPS Consumer Key/Secret is not configured.'}

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

                # Extract needed fields with regex, matching the style of other methods.
                # Take the first multilingual title; collect all applicants/inventors with
                # findall, then deduplicate EPO bracket/DOCDB name variants
                # while preserving their original order.
                title_m = re.search(r'<invention-title[^>]*>([^<]+)</invention-title>', text)
                applicants = _dedupe_preserve_order(re.findall(r'<applicant-name>\s*<name>([^<]+)</name>', text))
                inventors = _dedupe_preserve_order(re.findall(r'<inventor-name>\s*<name>([^<]+)</name>', text))
                pub_date_m = re.search(r'<publication-reference>.*?<date>(\d{8})</date>', text, re.DOTALL)
                # Match attributes on <application-reference doc-id="..."> with [^>]*;
                # inspected responses consistently include them, unlike publication-reference.
                app_date_m = re.search(r'<application-reference[^>]*>.*?<date>(\d{8})</date>', text, re.DOTALL)
                # family_id is an <exchange-document family-id="..."> attribute, not a body tag.
                family_id_m = re.search(r'<exchange-document[^>]*\bfamily-id="([^"]+)"', text)
                # Match attributes such as <classification-ipcr sequence="N"> with [^>]*.
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
                    return last_error  # Retrying non-404 errors is not useful here.
            except Exception as e:
                return {'status': f'ERROR: {e}', 'requested_url': url}

        return last_error

    def _fetch_fulltext_xml(self, endpoint, ep_doc_number):
        """Shared caller for full-text XML endpoints such as /claims and /description. Retry kind-preserving then bare references as in fetch_biblio_data(). Return raw XML and its debug path; fetch_claims_data/fetch_description_data parse it. Inspected OPS responses are text-only, with consecutive <claim-text> entries lacking id/num attributes, unlike local XML. Use OPS-specific parse_ops_* rather than local parse_paragraphs/parse_figures/parse_claim_elements."""
        if not self.enabled:
            return {'status': 'DISABLED', 'message': 'EPO OPS Consumer Key/Secret is not configured.'}

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
        """Fetch OPS /claims and parse claim elements for TARGET patents without local XML. Recover claim_no from numbering; element_no still uses approximate '- ' splitting, which previously operated without claim numbers."""
        result = self._fetch_fulltext_xml('claims', ep_doc_number)
        if result['status'] != 'SUCCESS':
            return result
        result['claim_elements'] = parse_ops_claim_elements(result['raw_xml'])
        return result

    def fetch_description_data(self, ep_doc_number):
        """Fetch OPS /description and populate TARGET paragraphs/figures, previously available only for PRIOR local XML. OPS lacks <figure> tags, so captions approximate 'Fig. N ...' sentences; image_path remains None without source drawing-file metadata."""
        result = self._fetch_fulltext_xml('description', ep_doc_number)
        if result['status'] != 'SUCCESS':
            return result
        result['paragraphs'] = parse_ops_paragraphs(result['raw_xml'])
        result['figures'] = parse_ops_figures_heuristic(result['raw_xml'])
        return result
