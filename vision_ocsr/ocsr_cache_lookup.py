import os
import json
import re

from config.settings import OCSR_CACHE_DIR


def _clean_ep_number(ep_doc_number):
    # Duplicate EPOOPSFetcher.clean_ep_number() normalization in a separate module
    # to allow cache lookup without importing cv2/DECIMER.
    num = ep_doc_number.replace('EP', '').replace('-', '').strip()
    num = re.sub(r'(NWA|NWB|NW)\d*$', '', num)
    num = re.sub(r'[A-Z]\d*$', '', num)
    return num


def get_cached_smiles(ep_doc_number):
    """Read OCSR results prepared by scripts/ocsr/run_ocsr_fallback.py in its separate environment. Standard-library-only lookup avoids cv2/DECIMER imports in run_chem_rag.py's main .venv. Missing cache returns None: OCSR has not run, not proof that no structure exists."""
    if not ep_doc_number:
        return None
    ep_number = _clean_ep_number(ep_doc_number)
    cache_path = os.path.join(OCSR_CACHE_DIR, f'EP{ep_number}.json')
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result.get('smiles')
