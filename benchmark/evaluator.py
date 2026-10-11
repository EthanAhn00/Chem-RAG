import os
import json
import time

def _average_precision(sorted_records):
    """정렬된(내림차순 final_hybrid_score) 레코드 리스트 하나에 대한 AP 계산 -
    정답(label==1)을 만날 때마다 그 지점까지의 precision을 기록하고 평균"""
    precisions = []
    num_correct = 0
    for i, r in enumerate(sorted_records):
        if r['ground_truth_label'] == 1:
            num_correct += 1
            precisions.append(num_correct / float(i + 1))
    return sum(precisions) / float(len(precisions)) if precisions else None  # None = 정답이 없어 AP 정의 불가


class ChemicalRAGEvaluator:
    """정답 인용 라벨을 기준으로 Chemical RAG 검색·랭킹 성능 지표(MAP, Hit@K)를 평가

    두 가지 MAP 정의를 모두 제공:
      - evaluate_ranking_performance(): "micro" 방식 - 쿼리 구분 없이 전체
        (target, prior) 쌍을 하나의 랭킹 리스트로 취급해 계산한 단일 AP.
        표준 IR 정의가 아님(쿼리별 난이도가 섞여버려 특정 target에 후보가 몰려
        있으면 그쪽에 결과가 쏠림) - 참고용으로만 남겨둠(레거시 호환)
      - evaluate_ranking_performance_per_query(): 표준 IR MAP 정의 - target
        특허(쿼리)별로 그룹을 나눠 그룹 안에서 prior 후보들을 랭킹, 쿼리마다
        AP를 구한 뒤 "정답이 하나 이상 있는 쿼리"만 대상으로 평균. 이게 논문/
        미팅에 보고할 "진짜" MAP
    """

    def __init__(self):
        pass

    def evaluate_ranking_performance(self, scored_records, label=None):
        # MAP(micro, 레거시), Hit@1, Hits@5 계산. `label`은 결과 구분용 설명 값(예: 'full' / 'test_leakfree')
        if not scored_records:
            return {'label': label, 'evaluated_records': 0, 'note': 'no records in this subset'}

        # 최종 하이브리드 점수 내림차순 정렬
        sorted_records = sorted(scored_records, key=lambda x: x['ranking_res']['final_hybrid_score'], reverse=True)

        total_positives = sum(1 for r in sorted_records if r['ground_truth_label'] == 1)
        total_negatives = sum(1 for r in sorted_records if r['ground_truth_label'] == 0)

        if total_positives == 0:
            return {'label': label, 'error': 'No positive ground truth citations found in dataset.'}

        hits_at_1 = 1 if sorted_records[0]['ground_truth_label'] == 1 else 0
        hits_at_5 = sum(1 for r in sorted_records[:5] if r['ground_truth_label'] == 1)

        ap = _average_precision(sorted_records) or 0.0

        return {
            'label': label,
            'evaluated_records': len(sorted_records),
            'positive_citations': total_positives,
            'negative_citations': total_negatives,
            'hit_at_1': hits_at_1,
            'hits_at_5': hits_at_5,
            'micro_average_precision_map_legacy': round(ap, 4),
            'top_ranked_sample': {
                'filename': sorted_records[0]['filename'],
                'target_ipc': sorted_records[0]['target_ipc'],
                'hybrid_score': sorted_records[0]['ranking_res']['final_hybrid_score'],
                'ground_truth_label': sorted_records[0]['ground_truth_label']
            }
        }

    def evaluate_ranking_performance_per_query(self, scored_records, label=None):
        """표준 IR 정의의 MAP - target_patent_id(쿼리)별로 그룹핑해 쿼리별 AP를
        구한 뒤 평균. target_patent_id가 없는 레코드는(EP 특허번호 복원 실패)
        쿼리를 특정할 수 없어 집계에서 제외하고 별도로 카운트만 보고함 - 임의로
        filename 등으로 대체 키를 만들어 억지로 포함시키지 않음(잘못된 그룹핑으로
        수치를 왜곡하지 않기 위해)"""
        if not scored_records:
            return {'label': label, 'evaluated_records': 0, 'note': 'no records in this subset'}

        excluded_no_query_id = 0
        groups = {}
        for r in scored_records:
            query_id = r.get('target_patent_id')
            if not query_id:
                excluded_no_query_id += 1
                continue
            groups.setdefault(query_id, []).append(r)

        if not groups:
            return {'label': label, 'error': 'No records with a resolvable target_patent_id (query id) found.',
                    'excluded_no_query_id': excluded_no_query_id}

        # 쿼리 그룹 크기 분포(디버그/검증용) - 그룹당 후보가 1개뿐이면(=target마다 prior가
        # 정확히 하나) 표준 MAP이 사실상 accuracy와 비슷해지므로, 실제 구조를 눈으로 확인할 수 있게 남김
        group_sizes = sorted(len(v) for v in groups.values())
        n = len(group_sizes)
        size_distribution = {
            'num_queries_total': n,
            'min_candidates_per_query': group_sizes[0],
            'median_candidates_per_query': group_sizes[n // 2],
            'max_candidates_per_query': group_sizes[-1],
            'queries_with_single_candidate': sum(1 for s in group_sizes if s == 1),
        }

        per_query_aps = []
        # 후보가 2개 이상인 쿼리만 따로 모음 - 후보가 1개뿐인 쿼리는 랭킹을 전혀 하지
        # 않고(비교 대상이 없음) 그 1개가 정답이면 AP가 자동으로 1.0이 되므로, 이런
        # 쿼리가 대다수면 전체 MAP이 "모델이 잘 랭킹해서"가 아니라 "애초에 비교할 게
        # 없어서" 높게 나오는 착시가 생김 - multi_candidate 지표로 이 착시를 분리해서 보고
        per_query_aps_multi = []
        hits_at_1_count = 0
        hits_at_5_count = 0
        hits_at_1_count_multi = 0
        queries_no_positive = 0
        for query_id, records in groups.items():
            sorted_records = sorted(records, key=lambda x: x['ranking_res']['final_hybrid_score'], reverse=True)
            ap = _average_precision(sorted_records)
            if ap is None:
                queries_no_positive += 1  # 이 쿼리엔 정답 라벨이 없어 AP 정의 불가 - 평균에서 제외
                continue
            per_query_aps.append(ap)
            is_hit_1 = sorted_records[0]['ground_truth_label'] == 1
            if is_hit_1:
                hits_at_1_count += 1
            hits_at_5_count += sum(1 for r in sorted_records[:5] if r['ground_truth_label'] == 1)
            if len(sorted_records) > 1:
                per_query_aps_multi.append(ap)
                if is_hit_1:
                    hits_at_1_count_multi += 1

        if not per_query_aps:
            return {'label': label, 'error': 'No query had a positive ground-truth citation to compute AP against.',
                    'excluded_no_query_id': excluded_no_query_id, 'query_size_distribution': size_distribution}

        mean_ap = sum(per_query_aps) / len(per_query_aps)
        mean_ap_multi = sum(per_query_aps_multi) / len(per_query_aps_multi) if per_query_aps_multi else None

        return {
            'label': label,
            'evaluated_records': len(scored_records),
            'excluded_no_query_id': excluded_no_query_id,
            'query_size_distribution': size_distribution,
            'num_queries_evaluated': len(per_query_aps),
            'queries_without_positive_excluded': queries_no_positive,
            'hit_at_1_rate': round(hits_at_1_count / len(per_query_aps), 4),
            'avg_hits_at_5': round(hits_at_5_count / len(per_query_aps), 4),
            'mean_average_precision_map': round(mean_ap, 4),
            # 후보 2개 이상(=실제 랭킹이 일어나는) 쿼리만 대상으로 한 MAP - 전체 MAP과
            # 이 값의 차이가 크면, 전체 MAP이 단일 후보 쿼리들 때문에 부풀려진 것
            'num_queries_multi_candidate': len(per_query_aps_multi),
            'hit_at_1_rate_multi_candidate': round(hits_at_1_count_multi / len(per_query_aps_multi), 4) if per_query_aps_multi else None,
            'mean_average_precision_map_multi_candidate_only': round(mean_ap_multi, 4) if mean_ap_multi is not None else None,
        }
