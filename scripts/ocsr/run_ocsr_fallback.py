"""Precompute OCSR fallback in a separate Python 3.10~3.13 environment with DECIMER, not the main Python 3.14 .venv without DECIMER/TensorFlow.
Collect patents missing SureChEMBL structures but having PatCID bbox annotations, then call OCSRPipeline.resolve_smiles(). Write data/ocsr_cache/EP*.json; run_chem_rag.py later reads the cache via vision_ocsr/ocsr_cache_lookup.py.
Exclude patents without PatCID mapping rather than fabricating contour crops; earlier contour-based attempts mistook text/arrows for structures.
Usage: python3 scripts/ocsr/run_ocsr_fallback.py --max-samples 10000 --limit 50. --limit bounds newly processed patents for quota protection; cached patents do not count and processing can be resumed.
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
    # Collect only patents missing SureChEMBL matches but having PatCID bbox annotations.
    # Deduplicate across targets/priors. XML has_drawing flags only indicate tags;
    # without local image files they do not imply crop availability, so are not filters.
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
    ap.add_argument('--limit', type=int, default=50, help='Maximum newly processed patents in this run (quota protection)')
    args = ap.parse_args()

    print('Collecting patent IDs (no SureChEMBL match + PatCID bbox available)...')
    surechembl_connector = PatCIDSureChEMBLConnector()
    candidates = collect_candidates(args.max_samples, surechembl_connector)
    print(f'Candidates: {len(candidates)} found')

    pipeline = OCSRPipeline(patcid_annotations_map=surechembl_connector.patcid_annotations_map)
    if not pipeline.fetcher.enabled:
        print('[STOP] OPS Consumer Key/Secret is not configured (see config/settings.py).')
        sys.exit(1)

    todo = [c for c in candidates if not already_cached(pipeline, c)]
    print(f'Excluding cached patents - candidates for this run: {len(todo)} (limit {args.limit})')

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

    print(f'\nDone - processed {processed}; successful structure recognition: {recovered} records')
    print(f'Cache path: {OCSR_CACHE_DIR}')
    if len(todo) > args.limit:
        print(f'Unprocessed candidates remaining: {len(todo) - args.limit}; increase --limit and rerun to resume.')


if __name__ == '__main__':
    main()
