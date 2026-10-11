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
# cv2/decimer는 TensorFlow 버전 제약 때문에 메인 파이프라인 .venv(Python 3.14)와
# 별도의 OCSR 전용 가상환경(Python 3.10~3.13)에 설치됨 - scripts/run_ocsr_fallback.py를
# 그 가상환경에서 먼저 실행해 캐시를 채워두면, 여기서는 표준 라이브러리만 쓰는
# 가벼운 캐시 조회 함수로 결과를 읽기만 함 (OCSRPipeline을 직접 임포트하지 않음)
from vision_ocsr.ocsr_cache_lookup import get_cached_smiles


def _load_train_test_split(path):
    # 파일이 없어도 치명적이지 않음 - 파이프라인은 그대로 실행되고 leak-free 테스트셋 지표만 생략
    if not path or not os.path.exists(path):
        print(f'[run_chem_rag] train_test_split not found at {path} - skipping leak-free test-set evaluation. '
              f'Run scripts/build_time_family_split.py first to enable it.')
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_chem_rag_pipeline(max_samples=10000, allow_remote_surechembl=False):
    """Chemical RAG 파이프라인 메인 실행 함수 - 동적 멀티모달 가중치 방식
    EPO Section C 레코드를 스캔·추출·채점·평가하는 4단계로 구성:
      1) EPOStreamParser - 골드 라벨 JSON 레코드 스캔 및 실제 EP 특허번호 연결
      2) XML 도면 유무, SureChEMBL 화학구조(SMILES) 매칭 여부 조회
         (SureChEMBL 미매칭 + ENABLE_OCSR_FALLBACK=True면 scripts/run_ocsr_fallback.py가
         별도 가상환경에서 미리 채워둔 OCSR 캐시도 함께 조회 - vision_ocsr/ocsr_cache_lookup.py)
      3) MultimodalChemicalRanker - 텍스트·구조·도면 점수 융합 (rag_engine/multimodal_ranker.py 참고)
      4) ChemicalRAGEvaluator - MAP/Hit@K 계산 (benchmark/evaluator.py 참고)
    """
    print('=== STARTING FULL BULK IPC SECTION C CHEMICAL RAG PIPELINE ===')
    print('Target Scope: All Section C (Chemistry & Metallurgy C01 ~ C99 / A61K / A61P) Patents')

    # Step 1: EPO Section C JSON 레코드 스트림 파싱(실제 특허번호/XML 파일명 연결 포함)
    stream_parser = EPOStreamParser()
    records = stream_parser.stream_section_c_records(max_samples=max_samples)

    if not records:
        print('No Section C records extracted.')
        return

    print(f'Successfully extracted {len(records)} IPC Section C Chemical Patent Pairs!')

    # Step 2: XML 도면 검사기, SureChEMBL/PatCID 커넥터, 멀티모달 랭커 초기화
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
        # 도면 조회는 순차 레코드 파일명이 아니라 PRIOR 문서의 실제 XML 파일명 사용
        # (id -> 특허번호 맵으로 복원)
        prior_xml_file = r.get('prior_xml_file')
        xml_meta = xml_parser.inspect_drawing_metadata(prior_xml_file) if prior_xml_file else \
            {'has_drawing': False, 'figure_count': 0, 'figure_files': []}

        has_drawing = xml_meta['has_drawing']
        fig_count = xml_meta['figure_count']
        if has_drawing:
            drawing_count_total += 1

        # SureChEMBL 구조 조회는 청구항 텍스트가 아니라 실제 EP 특허번호 기준
        surechembl_connector.enrich_patent_record(r, allow_remote=allow_remote_surechembl)
        target_smiles_list = (r.get('target_surechembl_metadata') or {}).get('smiles_list') or []
        prior_smiles_list = (r.get('prior_surechembl_metadata') or {}).get('smiles_list') or []
        target_smiles = target_smiles_list[0] if target_smiles_list else None
        prior_smiles = prior_smiles_list[0] if prior_smiles_list else None
        structure_source = {'target': 'surechembl' if target_smiles else None,
                             'prior': 'surechembl' if prior_smiles else None}

        # OCSR 폴백 - scripts/run_ocsr_fallback.py(별도 가상환경)가 미리 채워둔 캐시를
        # 조회만 함. 캐시가 없으면(아직 OCSR을 안 돌렸으면) 그냥 None 유지
        # (config.settings.ENABLE_OCSR_FALLBACK=True일 때만 조회 - 기본 False)
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

    # Step 3: 평가 지표 계산 및 벤치마크 리포트 & 데이터셋 저장
    # micro(레거시, 쿼리 구분 없이 전체를 한 리스트로 취급) / per_query(표준 IR MAP,
    # target_patent_id 쿼리별 AP 평균) 두 정의를 모두 계산해서 같이 보고
    eval_metrics = evaluator.evaluate_ranking_performance(scored_records, label='full')
    eval_metrics_per_query = evaluator.evaluate_ranking_performance_per_query(scored_records, label='full')

    print('=== BENCHMARK EVALUATION METRICS (full set, micro/legacy) ===')
    for k, v in eval_metrics.items():
        print(f'  {k}: {v}')
    print('=== BENCHMARK EVALUATION METRICS (full set, per-query MAP - 표준 IR 정의) ===')
    for k, v in eval_metrics_per_query.items():
        print(f'  {k}: {v}')

    # Step 3b: split이 있으면(scripts/build_time_family_split.py) family/time 누출 없는
    # 테스트셋 지표도 계산 - train과 특허번호가 겹치지 않고 cutoff_year 이후인 레코드만 포함
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

    # 필터링된 데이터셋 저장
    dataset_output_path = os.path.join(DATA_DIR, 'reference', 'epo_section_c_benchmark_dataset.json')
    with open(dataset_output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_records, f, indent=2, ensure_ascii=False)
    print(f'Filtered Section C Dataset saved to: {dataset_output_path}')

    # 벤치마크 평가 리포트 저장
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
