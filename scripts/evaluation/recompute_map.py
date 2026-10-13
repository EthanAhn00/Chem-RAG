"""Recompute MAP from data/reference/epo_section_c_benchmark_dataset.json after scoring, without rerunning EPOStreamParser, SureChEMBL/OCSR lookup, or ranking; useful after evaluator-definition changes. The dataset requires at least one run_chem_rag.py run. To include OCSR fallback, first regenerate with ENABLE_OCSR_FALLBACK=1.
Usage: python3 scripts/evaluation/recompute_map.py.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.evaluator import ChemicalRAGEvaluator
from config.settings import DATA_DIR, TRAIN_TEST_SPLIT_PATH


def main():
    dataset_path = os.path.join(DATA_DIR, 'reference', 'epo_section_c_benchmark_dataset.json')
    if not os.path.exists(dataset_path):
        print(f'[STOP] {dataset_path} missing - run run_chem_rag.py at least once first.')
        sys.exit(1)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        scored_records = json.load(f)
    print(f'{len(scored_records)} records loaded: {dataset_path}')

    evaluator = ChemicalRAGEvaluator()

    print('\n=== Full dataset ===')
    micro = evaluator.evaluate_ranking_performance(scored_records, label='full')
    per_query = evaluator.evaluate_ranking_performance_per_query(scored_records, label='full')
    print('-- micro (legacy, without query separation) --')
    for k, v in micro.items():
        print(f'  {k}: {v}')
    print('-- per-query MAP (standard IR definition) --')
    for k, v in per_query.items():
        print(f'  {k}: {v}')

    if os.path.exists(TRAIN_TEST_SPLIT_PATH):
        with open(TRAIN_TEST_SPLIT_PATH, 'r', encoding='utf-8') as f:
            split_map = json.load(f)
        test_records = [r for r in scored_records if split_map.get(r['filename']) == 'test']
        if test_records:
            print(f'\n=== Leak-free test subset (n={len(test_records)}) ===')
            micro_test = evaluator.evaluate_ranking_performance(test_records, label='test_leakfree')
            per_query_test = evaluator.evaluate_ranking_performance_per_query(test_records, label='test_leakfree')
            print('-- micro (legacy) --')
            for k, v in micro_test.items():
                print(f'  {k}: {v}')
            print('-- per-query MAP (standard IR definition) --')
            for k, v in per_query_test.items():
                print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
