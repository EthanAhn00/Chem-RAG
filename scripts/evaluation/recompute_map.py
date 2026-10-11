"""저장된 벤치마크 데이터셋(data/reference/epo_section_c_benchmark_dataset.json)을 다시 읽어
MAP을 재계산하는 스크립트 - 파이프라인(EPOStreamParser 스캔 + SureChEMBL/OCSR 조회 +
랭킹)을 처음부터 다시 돌릴 필요 없이, 이미 채점까지 끝난 레코드에 대해 평가 지표만
다시 계산하고 싶을 때 사용 (예: evaluator.py의 MAP 정의를 바꾼 뒤 재검증)

주의: 이 파일은 run_chem_rag.py를 최소 한 번 실행해야 존재함. OCSR 폴백 결과를
반영하려면 먼저 ENABLE_OCSR_FALLBACK=1로 run_chem_rag.py를 실행해 데이터셋을
새로 저장한 뒤 이 스크립트를 돌릴 것.

사용법:
    python3 scripts/evaluation/recompute_map.py
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
        print(f'[중단] {dataset_path} 없음 - run_chem_rag.py를 먼저 한 번 실행해야 함.')
        sys.exit(1)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        scored_records = json.load(f)
    print(f'{len(scored_records)}개 레코드 로드 완료: {dataset_path}')

    evaluator = ChemicalRAGEvaluator()

    print('\n=== 전체 데이터셋 ===')
    micro = evaluator.evaluate_ranking_performance(scored_records, label='full')
    per_query = evaluator.evaluate_ranking_performance_per_query(scored_records, label='full')
    print('-- micro(레거시, 쿼리 구분 없음) --')
    for k, v in micro.items():
        print(f'  {k}: {v}')
    print('-- per-query MAP(표준 IR 정의) --')
    for k, v in per_query.items():
        print(f'  {k}: {v}')

    if os.path.exists(TRAIN_TEST_SPLIT_PATH):
        with open(TRAIN_TEST_SPLIT_PATH, 'r', encoding='utf-8') as f:
            split_map = json.load(f)
        test_records = [r for r in scored_records if split_map.get(r['filename']) == 'test']
        if test_records:
            print(f'\n=== leak-free 테스트 서브셋 (n={len(test_records)}) ===')
            micro_test = evaluator.evaluate_ranking_performance(test_records, label='test_leakfree')
            per_query_test = evaluator.evaluate_ranking_performance_per_query(test_records, label='test_leakfree')
            print('-- micro(레거시) --')
            for k, v in micro_test.items():
                print(f'  {k}: {v}')
            print('-- per-query MAP(표준 IR 정의) --')
            for k, v in per_query_test.items():
                print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
