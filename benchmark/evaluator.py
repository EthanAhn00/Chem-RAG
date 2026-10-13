import os
import json
import time

def _average_precision(sorted_records):
    """Compute AP for records sorted by descending final_hybrid_score; average precision at each positive (label==1)."""
    precisions = []
    num_correct = 0
    for i, r in enumerate(sorted_records):
        if r['ground_truth_label'] == 1:
            num_correct += 1
            precisions.append(num_correct / float(i + 1))
    return sum(precisions) / float(len(precisions)) if precisions else None  # None means AP is undefined because there are no positives.


class ChemicalRAGEvaluator:
    """Evaluate Chemical RAG retrieval/ranking using citation labels (MAP, Hit@K).

    Two definitions are retained:
    - evaluate_ranking_performance(): legacy micro AP over one ranking of all (target, prior) pairs, without separating queries. This is not standard IR MAP: query difficulty and unequal candidate counts can bias the result. Retained for reference and backward compatibility.
    - evaluate_ranking_performance_per_query(): group by target patent, rank prior candidates within each query, compute AP, and average over queries with at least one positive. This is the query-level MAP used for reporting.
    """

    def __init__(self):
        pass

    def evaluate_ranking_performance(self, scored_records, label=None):
        # Compute legacy micro MAP, Hit@1 and Hits@5. `label` describes the result subset (e.g. 'full' / 'test_leakfree').
        if not scored_records:
            return {'label': label, 'evaluated_records': 0, 'note': 'no records in this subset'}

        # Sort by descending final hybrid score.
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
        """Compute query-level IR MAP grouped by target_patent_id. Exclude and count records missing target_patent_id (unresolved EP identifiers); do not invent filename-based replacement keys that would distort query grouping."""
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

        # Inspect query-group sizes; with only one candidate per target,
        # MAP effectively resembles accuracy. Keep the distribution visible for diagnostics.
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
        # Also report queries with at least 2 candidates. A single-candidate query
        # requires no ranking; if that candidate is positive, AP is automatically 1.0.
        # Many such queries can inflate MAP through lack of competition rather than
        # ranking quality. Report multi_candidate metrics separately to expose this effect.
        per_query_aps_multi = []
        hits_at_1_count = 0
        hits_at_5_count = 0
        hits_at_1_count_multi = 0
        queries_no_positive = 0
        for query_id, records in groups.items():
            sorted_records = sorted(records, key=lambda x: x['ranking_res']['final_hybrid_score'], reverse=True)
            ap = _average_precision(sorted_records)
            if ap is None:
                queries_no_positive += 1  # No positive labels: AP is undefined; exclude this query from the mean.
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
            # MAP for queries with at least 2 candidates, where ranking actually occurs.
            # A large gap from overall MAP indicates inflation by single-candidate queries.
            'num_queries_multi_candidate': len(per_query_aps_multi),
            'hit_at_1_rate_multi_candidate': round(hits_at_1_count_multi / len(per_query_aps_multi), 4) if per_query_aps_multi else None,
            'mean_average_precision_map_multi_candidate_only': round(mean_ap_multi, 4) if mean_ap_multi is not None else None,
        }
