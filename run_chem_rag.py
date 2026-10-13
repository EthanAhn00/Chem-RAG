import os
import json
import time
from data_parser.epo_stream_parser import EPOStreamParser
from data_parser.epo_xml_parser import EPOXMLParser
from data_parser.patcid_surechembl_connector import PatCIDSureChEMBLConnector
from rag_engine.multimodal_ranker import MultimodalChemicalRanker
from benchmark.evaluator import ChemicalRAGEvaluator
from config.settings import (
    BENCHMARK_OUTPUT_DIR, DATA_DIR, TRAIN_TEST_SPLIT_PATH,
    DEFAULT_TEXT_WEIGHT, DEFAULT_TANIMOTO_WEIGHT, DEFAULT_VISION_WEIGHT,
    ENABLE_OCSR_FALLBACK,
)
# TensorFlow constraints require cv2/DECIMER in a separate OCSR environment
# (Python 3.10~3.13), not the main .venv (Python 3.14). Run scripts/run_ocsr_fallback.py
# there first to populate the cache; this pipeline only reads it using a
# lightweight standard-library lookup, without importing OCSRPipeline.
from vision_ocsr.ocsr_cache_lookup import get_cached_smiles


def _load_train_test_split(path):
    # A missing split file is nonfatal; run normally but omit leak-free test metrics.
    if not path or not os.path.exists(path):
        print(f'[run_chem_rag] train_test_split not found at {path} - skipping leak-free test-set evaluation. '
              f'Run scripts/build_time_family_split.py first to enable it.')
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_chem_rag_pipeline(max_samples=10000, allow_remote_surechembl=False):
    """Chemical RAG entry point with dynamic multimodal weights.
    Four stages for EPO Section C records:
    1) EPOStreamParser reads labeled JSON and attaches actual EP identifiers.
    2) Check XML drawings and SureChEMBL SMILES; when unmatched and ENABLE_OCSR_FALLBACK=True, read the OCSR cache prepared in a separate environment by scripts/run_ocsr_fallback.py (vision_ocsr/ocsr_cache_lookup.py).
    3) MultimodalChemicalRanker fuses text/structure/drawing scores (rag_engine/multimodal_ranker.py).
    4) ChemicalRAGEvaluator computes MAP/Hit@K (benchmark/evaluator.py).
    """
    print('=== STARTING FULL BULK IPC SECTION C CHEMICAL RAG PIPELINE ===')
    print('Target Scope: All Section C (Chemistry & Metallurgy C01 ~ C99 / A61K / A61P) Patents')

    # Step 1: stream EPO Section C JSON, including actual patent/XML filename links.
    stream_parser = EPOStreamParser()
    records = stream_parser.stream_section_c_records(max_samples=max_samples)

    if not records:
        print('No Section C records extracted.')
        return

    print(f'Successfully extracted {len(records)} IPC Section C Chemical Patent Pairs!')

    # Step 2: initialize XML drawing checks, SureChEMBL/PatCID connectors, and ranker.
    xml_parser = EPOXMLParser()
    surechembl_connector = PatCIDSureChEMBLConnector()
    ranker = MultimodalChemicalRanker(
        text_weight=DEFAULT_TEXT_WEIGHT,
        tanimoto_weight=DEFAULT_TANIMOTO_WEIGHT,
        vision_weight=DEFAULT_VISION_WEIGHT,
    )
    evaluator = ChemicalRAGEvaluator()
    ocsr_recovered_count = 0

    print('Executing Hybrid Chemical RAG & Dynamic Multimodal Weight Ranking...')
    start_time = time.time()

    scored_records = []
    drawing_count_total = 0
    structure_count_total = 0

    for r in records:
        # Drawing lookup uses the PRIOR document's actual XML filename,
        # recovered from the ID -> patent-number mapping, not sequential record filenames.
        prior_xml_file = r.get('prior_xml_file')
        xml_meta = xml_parser.inspect_drawing_metadata(prior_xml_file) if prior_xml_file else \
            {'has_drawing': False, 'figure_count': 0, 'figure_files': []}

        has_drawing = xml_meta['has_drawing']
        fig_count = xml_meta['figure_count']
        if has_drawing:
            drawing_count_total += 1

        # Query SureChEMBL by actual EP patent number, not claim text.
        surechembl_connector.enrich_patent_record(r, allow_remote=allow_remote_surechembl)
        target_smiles_list = (r.get('target_surechembl_metadata') or {}).get('smiles_list') or []
        prior_smiles_list = (r.get('prior_surechembl_metadata') or {}).get('smiles_list') or []
        target_smiles = target_smiles_list[0] if target_smiles_list else None
        prior_smiles = prior_smiles_list[0] if prior_smiles_list else None
        structure_source = {'target': 'surechembl' if target_smiles else None,
                             'prior': 'surechembl' if prior_smiles else None}

        # OCSR fallback only reads the cache prepared by scripts/run_ocsr_fallback.py
        # in its separate environment. A missing cache leaves the result None.
        # Read only with config.settings.ENABLE_OCSR_FALLBACK=True; default False.
        if ENABLE_OCSR_FALLBACK and has_drawing:
            if not target_smiles and r.get('target_patent_id'):
                cached = get_cached_smiles(r['target_patent_id'])
                if cached:
                    target_smiles = cached
                    structure_source['target'] = 'ocsr'
                    ocsr_recovered_count += 1
            if not prior_smiles and r.get('prior_patent_id'):
                cached = get_cached_smiles(r['prior_patent_id'])
                if cached:
                    prior_smiles = cached
                    structure_source['prior'] = 'ocsr'
                    ocsr_recovered_count += 1
        r['structure_source'] = structure_source

        ranking_res = ranker.rank_patent_pair(
            target_claim=r['target_claim'],
            prior_claim=r['prior_claim'],
            target_smiles=target_smiles,
            prior_smiles=prior_smiles,
            has_drawing=has_drawing,
            drawing_count=fig_count
        )
        if ranking_res.get('has_structure'):
            structure_count_total += 1

        r['xml_metadata'] = xml_meta
        r['ranking_res'] = ranking_res
        scored_records.append(r)

    elapsed = time.time() - start_time
    print(f'Ranking completed in {elapsed:.2f}s! Records with XML drawings: {drawing_count_total}/{len(records)}, '
          f'records with resolved chemical structure: {structure_count_total}/{len(records)}')

    # Step 3: compute metrics and save the benchmark report/dataset.
    # Report both legacy micro AP (all pairs in one list) and standard query-level MAP
    # (average AP grouped by target_patent_id).
    eval_metrics = evaluator.evaluate_ranking_performance(scored_records, label='full')
    eval_metrics_per_query = evaluator.evaluate_ranking_performance_per_query(scored_records, label='full')

    print('=== BENCHMARK EVALUATION METRICS (full set, micro/legacy) ===')
    for k, v in eval_metrics.items():
        print(f'  {k}: {v}')
    print('=== BENCHMARK EVALUATION METRICS (full set, per-query MAP - standard IR definition) ===')
    for k, v in eval_metrics_per_query.items():
        print(f'  {k}: {v}')

    # Step 3b: if scripts/build_time_family_split.py provides a split, also evaluate
    # test records after cutoff_year with no patent-number overlap with train.
    split_map = _load_train_test_split(TRAIN_TEST_SPLIT_PATH)
    eval_metrics_test_leakfree = None
    eval_metrics_test_leakfree_per_query = None
    split_counts = None
    if split_map:
        test_records = [r for r in scored_records if split_map.get(r['filename']) == 'test']
        split_counts = {
            'train': sum(1 for r in scored_records if split_map.get(r['filename']) == 'train'),
            'test': len(test_records),
            'train_unresolved': sum(1 for r in scored_records if split_map.get(r['filename']) == 'train_unresolved'),
            'not_in_split_file': sum(1 for r in scored_records if r['filename'] not in split_map),
        }
        eval_metrics_test_leakfree = evaluator.evaluate_ranking_performance(test_records, label='test_leakfree')
        eval_metrics_test_leakfree_per_query = evaluator.evaluate_ranking_performance_per_query(test_records, label='test_leakfree')
        print(f'=== BENCHMARK EVALUATION METRICS (leak-free test subset, n={len(test_records)}, micro/legacy) ===')
        for k, v in eval_metrics_test_leakfree.items():
            print(f'  {k}: {v}')
        print(f'=== BENCHMARK EVALUATION METRICS (leak-free test subset, n={len(test_records)}, per-query MAP) ===')
        for k, v in eval_metrics_test_leakfree_per_query.items():
            print(f'  {k}: {v}')

    # Save the filtered dataset.
    dataset_output_path = os.path.join(DATA_DIR, 'reference', 'epo_section_c_benchmark_dataset.json')
    with open(dataset_output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_records, f, indent=2, ensure_ascii=False)
    print(f'Filtered Section C Dataset saved to: {dataset_output_path}')

    # Save the benchmark evaluation report.
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'project': 'PANORAMA-Chem-RAG',
        'pipeline_version': 'SOTA Chemical RAG + Dynamic Multimodal Vision Weight v2.0 (patent-number linkage restored)',
        'evaluated_sample_count': len(records),
        'records_with_drawing': drawing_count_total,
        'drawing_ratio_pct': round((drawing_count_total / len(records)) * 100, 2),
        'records_with_structure': structure_count_total,
        'structure_ratio_pct': round((structure_count_total / len(records)) * 100, 2),
        'ocsr_fallback_enabled': ENABLE_OCSR_FALLBACK,
        'ocsr_recovered_smiles_count': ocsr_recovered_count,
        'weights': {
            'text_weight': DEFAULT_TEXT_WEIGHT,
            'tanimoto_weight': DEFAULT_TANIMOTO_WEIGHT,
            'vision_weight': DEFAULT_VISION_WEIGHT,
        },
        'elapsed_seconds': round(elapsed, 2),
        'evaluation_metrics': eval_metrics,
        'evaluation_metrics_per_query': eval_metrics_per_query,
        'split_counts': split_counts,
        'evaluation_metrics_test_leakfree': eval_metrics_test_leakfree,
        'evaluation_metrics_test_leakfree_per_query': eval_metrics_test_leakfree_per_query,
        'dataset_saved_at': dataset_output_path
    }

    output_report_path = os.path.join(BENCHMARK_OUTPUT_DIR, 'chem_rag_benchmark_report.json')
    with open(output_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'Benchmark Report saved to: {output_report_path}')
    return report


if __name__ == '__main__':
    run_chem_rag_pipeline(max_samples=10000)
