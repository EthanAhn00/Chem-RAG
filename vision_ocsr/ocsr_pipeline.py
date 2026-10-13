import os
import json

from config.settings import OCSR_CACHE_DIR
from vision_ocsr.epo_ops_fetcher import EPOOPSFetcher
from vision_ocsr.patcid_bbox_cropper import crop_by_patcid_bbox
from vision_ocsr.ocsr_smiles_converter import OCSRSMILESConverter
from data_parser.patcid_surechembl_connector import get_best_clean_smiles


class OCSRPipeline:
    """Structure fallback only after SureChEMBL matching fails.
    1) Prefer PatCID-recognized high-confidence SMILES (class='Clean', IBM Deep Search / Nature 2024); use directly and skip step 2 when available.
    2) OPS page -> PatCID bbox -> DECIMER recognition. Disabled by default (ENABLE_BBOX_CROP_FALLBACK=False): inspected crops were unreliable because PatCID's page field differed from assumptions. Return NOT_FOUND rather than fabricating results until resolved.
    Cache per-patent results in data/ocsr_cache/{ep_number}.json to avoid repeated OPS quota use and OCSR computation.
    """

    ENABLE_BBOX_CROP_FALLBACK = False  # Rationale: see item 2 in the class docstring.

    def __init__(self, patcid_annotations_map=None):
        self.fetcher = EPOOPSFetcher()
        self.converter = OCSRSMILESConverter()
        # {clean_ep_number: {'figures': [...], 'page_dimensions': [...]}} -
        # Inject patcid_annotations_map from data_parser/patcid_surechembl_connector.py
        # rather than reloading the 231k-entry map in this class.
        self.patcid_annotations_map = patcid_annotations_map or {}

    def _cache_path(self, ep_doc_number):
        ep_number = self.fetcher.clean_ep_number(ep_doc_number)
        return os.path.join(OCSR_CACHE_DIR, f'EP{ep_number}.json')

    def resolve_smiles(self, ep_doc_number):
        """Recognize SMILES from drawings for an EP patent number. Return None on failure; do not fabricate a substitute, distinguishing failed recognition from zero similarity as with SureChEMBL NOT_FOUND."""
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
                    'detail': 'No PatCID coordinate mapping for a trustworthy crop (no contour-based substitute)'}

        # First choice: return PatCID-recognized SMILES without OPS lookup (see class docstring).
        best_clean = get_best_clean_smiles(patcid_entry)
        if best_clean:
            return {
                'status': 'SUCCESS', 'smiles': best_clean['smiles'],
                'confidence': best_clean.get('confidence'), 'source_page': best_clean.get('page'),
                'crop_method': 'patcid_precomputed', 'engine': 'PatCID Nature 2024 (IBM Deep Search)',
            }

        if not self.ENABLE_BBOX_CROP_FALLBACK:
            return {'status': 'NOT_FOUND', 'smiles': None,
                    'detail': 'No PatCID-recognized SMILES (only Markush/Trash) - '
                               'bbox fallback disabled because page-field semantics are unverified'}

        page_dims_by_page = {d['page']: d for d in patcid_entry.get('page_dimensions', [])}
        figures = patcid_entry['figures']
        page_numbers = sorted({fig['page'] for fig in figures if 'page' in fig})

        # Specify the kind code of the exact publication analyzed by PatCID
        # and retrieve only its FullDocument instance. OPS images contains
        # different page layouts for A1/B1 etc. even with the same patent number.
        # Without explicit kind selection, bbox coordinates may point to unrelated
        # content such as body text; see epo_ops_fetcher.fetch_full_document_pages().
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

        return {'status': 'NOT_FOUND', 'smiles': None, 'detail': 'No valid structure recognized in the PatCID bbox region'}
