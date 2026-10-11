#!/usr/bin/env python3
"""T1 Top-N 후보만 정규화된 T1·T2 점수로 재정렬한다.

Top-N과 alpha는 development query에서 모두 보고하며, Top-N 설정은
사후에 하나를 고르지 않는 민감도 실험으로 유지한다. 각 Top-N의
alpha는 development Citation MRR@100으로 선택하고 test에서 고정한다.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .evaluate_t2_structure_retrieval import evaluate_ranking
    from .evaluate_t3_rrf import (
        METRICS,
        TASKS,
        load_ground_truth,
        load_per_query_index,
        paired_summary,
        sha256_file,
        write_csv_atomic,
        write_json_atomic,
        write_jsonl_atomic,
    )
    from .evaluate_t3_weighted_fusion import (
        MINIMUM_INFERENCE_QUERIES,
        PRIMARY_SELECTION_METRIC,
        PRIMARY_TASK,
        adjust_holm_for_allowed_rows,
        best_gold_rank,
        load_query_splits,
        load_score_rankings,
        parse_alpha_grid,
        require_full_score_rankings,
        select_alpha,
    )
except ImportError:
    from evaluate_t2_structure_retrieval import evaluate_ranking
    from evaluate_t3_rrf import (
        METRICS,
        TASKS,
        load_ground_truth,
        load_per_query_index,
        paired_summary,
        sha256_file,
        write_csv_atomic,
        write_json_atomic,
        write_jsonl_atomic,
    )
    from evaluate_t3_weighted_fusion import (
        MINIMUM_INFERENCE_QUERIES,
        PRIMARY_SELECTION_METRIC,
        PRIMARY_TASK,
        adjust_holm_for_allowed_rows,
        best_gold_rank,
        load_query_splits,
        load_score_rankings,
        parse_alpha_grid,
        require_full_score_rankings,
        select_alpha,
    )


def parse_top_n_values(value: str) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Top-N은 정수여야 합니다") from exc
    if not values or any(item < 100 for item in values):
        raise argparse.ArgumentTypeError("Top-N은 100 이상의 정수여야 합니다")
    return values


def zscore_or_zero(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """분산이 없으면 정보가 없는 신호로 보고 0 벡터를 반환한다."""
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    if not math.isfinite(mean) or not math.isfinite(standard_deviation):
        raise ValueError("점수 평균 또는 표준편차가 유한수가 아닙니다")
    if standard_deviation <= 1e-15:
        return np.zeros_like(values, dtype=np.float64), True
    return (values - mean) / standard_deviation, False


def rerank_text_top_n(
    text_ranking: list[tuple[str, float]],
    structure_ranking: list[tuple[str, float]],
    top_n: int,
    alpha: float,
) -> tuple[list[tuple[str, float]], bool, bool]:
    """T1 Top-N 멤버십은 유지하고 그 안의 순서만 변경한다."""
    text_ids = [patent_id for patent_id, _ in text_ranking]
    structure_ids = [patent_id for patent_id, _ in structure_ranking]
    if set(text_ids) != set(structure_ids):
        raise ValueError("T1과 T2 ranking의 후보 집합이 다릅니다")
    if top_n > len(text_ranking):
        raise ValueError(f"Top-N이 후보 수보다 큽니다: {top_n}>{len(text_ranking)}")

    head = text_ranking[:top_n]
    tail = text_ranking[top_n:]
    head_ids = [patent_id for patent_id, _ in head]
    structure_scores = dict(structure_ranking)
    text_values = np.asarray([score for _, score in head], dtype=np.float64)
    structure_values = np.asarray(
        [structure_scores[patent_id] for patent_id in head_ids],
        dtype=np.float64,
    )
    text_z, text_zero_variance = zscore_or_zero(text_values)
    structure_z, structure_zero_variance = zscore_or_zero(structure_values)
    fused_scores = alpha * text_z + (1.0 - alpha) * structure_z

    if math.isclose(alpha, 1.0, abs_tol=1e-15):
        order = np.arange(top_n)
    else:
        patent_array = np.asarray(head_ids, dtype=str)
        text_rank = np.arange(top_n)
        order = np.lexsort((patent_array, text_rank, -fused_scores))

    reranked_head = [
        (head_ids[index], float(fused_scores[index]))
        for index in order
    ]
    # 평가 함수는 tail 점수를 사용하지 않으며, 원 T1 순서 보존을 위해 그대로 붙인다.
    reranked_tail = [
        (patent_id, float(score))
        for patent_id, score in tail
    ]
    return (
        reranked_head + reranked_tail,
        text_zero_variance,
        structure_zero_variance,
    )


def run_self_test() -> None:
    text = [
        ("EP1A1", 0.9),
        ("EP2A1", 0.8),
        ("EP3A1", 0.7),
        ("EP4A1", 0.6),
    ]
    structure = [
        ("EP3A1", 0.9),
        ("EP2A1", 0.8),
        ("EP1A1", 0.1),
        ("EP4A1", 0.0),
    ]
    fused, _, _ = rerank_text_top_n(text, structure, 3, 0.0)
    assert [patent_id for patent_id, _ in fused] == [
        "EP3A1",
        "EP2A1",
        "EP1A1",
        "EP4A1",
    ]
    text_only, _, _ = rerank_text_top_n(text, structure, 3, 1.0)
    assert [patent_id for patent_id, _ in text_only] == [
        patent_id for patent_id, _ in text
    ]
    assert set(patent_id for patent_id, _ in fused[:3]) == {
        "EP1A1",
        "EP2A1",
        "EP3A1",
    }
    assert [patent_id for patent_id, _ in fused[3:]] == ["EP4A1"]
    mixed, _, _ = rerank_text_top_n(text, structure, 3, 0.5)
    assert len(mixed) == len(text)
    assert {patent_id for patent_id, _ in mixed} == {
        patent_id for patent_id, _ in text
    }
    try:
        rerank_text_top_n(text, structure[:-1], 3, 0.5)
    except ValueError as exc:
        assert "후보 집합" in str(exc)
    else:
        raise AssertionError("후보 집합 불일치가 감지되지 않았습니다")
    try:
        rerank_text_top_n(text, structure, 5, 0.5)
    except ValueError as exc:
        assert "후보 수" in str(exc)
    else:
        raise AssertionError("후보 수보다 큰 Top-N이 감지되지 않았습니다")
    assert parse_top_n_values("500,100") == [100, 500]
    zero, collapsed = zscore_or_zero(np.asarray([1.0, 1.0]))
    assert collapsed and np.array_equal(zero, np.zeros(2))
    print("자체 검증을 통과했습니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--query-split", type=Path)
    parser.add_argument("--text-per-query", type=Path)
    parser.add_argument("--text-rankings", type=Path)
    parser.add_argument("--structure-per-query", type=Path)
    parser.add_argument("--structure-rankings", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--top-n-values",
        type=parse_top_n_values,
        default=parse_top_n_values("100,500"),
    )
    parser.add_argument(
        "--alpha-grid",
        type=parse_alpha_grid,
        default=parse_alpha_grid("0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"),
    )
    parser.add_argument("--development-label", default="train")
    parser.add_argument("--test-label", default="test")
    parser.add_argument("--ranking-output-limit", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--conflict-field",
        choices=("conflict_patents", "x_category_patents"),
        default="conflict_patents",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    required = {
        "--ground-truth": args.ground_truth,
        "--query-split": args.query_split,
        "--text-per-query": args.text_per_query,
        "--text-rankings": args.text_rankings,
        "--structure-per-query": args.structure_per_query,
        "--structure-rankings": args.structure_rankings,
        "--output-dir": args.output_dir,
    }
    missing_arguments = [name for name, value in required.items() if value is None]
    if missing_arguments:
        parser.error(f"필수 인자가 없습니다: {', '.join(missing_arguments)}")
    missing_files = [
        str(path)
        for name, path in required.items()
        if name != "--output-dir" and not path.is_file()
    ]
    if missing_files:
        raise SystemExit(f"입력 파일이 없습니다: {missing_files}")
    if args.ranking_output_limit < 100:
        parser.error("--ranking-output-limit은 100 이상이어야 합니다")
    if args.iterations < 1000:
        parser.error("--iterations는 1000 이상이어야 합니다")

    started_at = time.perf_counter()
    ground_truth = load_ground_truth(args.ground_truth, args.conflict_field)
    query_split, observed_split_counts = load_query_splits(
        args.query_split,
        args.development_label,
        args.test_label,
    )
    text_per_query, text_models = load_per_query_index(args.text_per_query, "source")
    structure_per_query, structure_models = load_per_query_index(
        args.structure_per_query,
        "method",
    )
    if len(structure_models) != 1:
        raise SystemExit(
            f"reranking 1차 실험은 T2 모델 하나만 허용합니다: "
            f"{sorted(structure_models)}"
        )
    structure_model = next(iter(structure_models))
    text_rankings, text_ranking_models = load_score_rankings(args.text_rankings, "source")
    structure_rankings, structure_ranking_models = load_score_rankings(
        args.structure_rankings,
        "method",
    )
    if text_models != text_ranking_models:
        raise SystemExit("T1 query 결과와 ranking의 source 집합이 다릅니다")
    if structure_models != structure_ranking_models:
        raise SystemExit("T2 query 결과와 ranking의 method 집합이 다릅니다")
    require_full_score_rankings(text_per_query, text_rankings, text_models, "T1")
    require_full_score_rankings(
        structure_per_query,
        structure_rankings,
        structure_models,
        "T2",
    )
    candidate_counts = {
        len(ranking)
        for model in text_models
        for ranking in text_rankings[model].values()
    }
    if len(candidate_counts) != 1:
        raise SystemExit(f"T1 query별 후보 수가 다릅니다: {sorted(candidate_counts)}")
    candidate_count = next(iter(candidate_counts))
    if max(args.top_n_values) > candidate_count:
        raise SystemExit(
            f"Top-N이 후보 수보다 큽니다: "
            f"{max(args.top_n_values)}>{candidate_count}"
        )

    split_labels = {
        "development": args.development_label,
        "test": args.test_label,
    }
    eligible_queries: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    excluded_query_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for text_model in sorted(text_models):
        for task in TASKS:
            text_queries = set(text_per_query[text_model].get(task, {}))
            structure_queries = set(structure_per_query[structure_model].get(task, {}))
            if text_queries != structure_queries:
                raise SystemExit(
                    f"T1과 T2의 평가 query 집합이 다릅니다: "
                    f"source={text_model}, task={task}"
                )
            assigned: set[str] = set()
            for split_name, label in split_labels.items():
                selected = sorted(
                    query_id
                    for query_id in text_queries
                    if query_split.get(query_id) == label
                )
                eligible_queries[text_model][task][split_name] = selected
                assigned.update(selected)
            excluded_query_counts[text_model][task] = len(text_queries - assigned)

    tuning_rows: list[dict[str, Any]] = []
    selected_alphas: dict[str, dict[int, float]] = defaultdict(dict)
    zero_variance_counts: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"text": 0, "structure": 0})
    )
    for text_model in sorted(text_models):
        development_queries = eligible_queries[text_model][PRIMARY_TASK]["development"]
        for top_n in args.top_n_values:
            metric_values = {
                alpha: {metric: [] for metric in METRICS}
                for alpha in args.alpha_grid
            }
            for query_id in development_queries:
                gold = ground_truth[query_id][PRIMARY_TASK]
                variance_recorded = False
                for alpha in args.alpha_grid:
                    try:
                        reranked, text_zero, structure_zero = rerank_text_top_n(
                            text_rankings[text_model][query_id],
                            structure_rankings[structure_model][query_id],
                            top_n,
                            alpha,
                        )
                    except ValueError as exc:
                        raise SystemExit(
                            f"{exc}: source={text_model}, query={query_id}, top_n={top_n}"
                        ) from exc
                    if not variance_recorded:
                        zero_variance_counts[text_model][top_n]["text"] += int(text_zero)
                        zero_variance_counts[text_model][top_n]["structure"] += int(
                            structure_zero
                        )
                        variance_recorded = True
                    metrics = evaluate_ranking(reranked, gold)
                    for metric in METRICS:
                        metric_values[alpha][metric].append(metrics[metric])
            source_rows: list[dict[str, Any]] = []
            for alpha in args.alpha_grid:
                row = {
                    "text_source": text_model,
                    "structure_method": structure_model,
                    "top_n": top_n,
                    "selection_split": "development",
                    "selection_task": PRIMARY_TASK,
                    "query_count": len(development_queries),
                    "alpha": alpha,
                    **{
                        metric: float(np.mean(metric_values[alpha][metric]))
                        for metric in METRICS
                    },
                }
                source_rows.append(row)
                tuning_rows.append(row)
            selected_alpha, _ = select_alpha(source_rows)
            selected_alphas[text_model][top_n] = selected_alpha
            for row in source_rows:
                row["selected"] = float(row["alpha"]) == selected_alpha

    result_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    comparison_inputs: dict[
        tuple[str, str, str],
        tuple[list[float], list[float]],
    ] = {}
    for text_model in sorted(text_models):
        for top_n in args.top_n_values:
            alpha = selected_alphas[text_model][top_n]
            method = (
                f"text_top{top_n}_zscore_alpha{alpha:g}__"
                f"{text_model}__{structure_model}"
            )
            ranking_written: set[str] = set()
            for task in TASKS:
                for split_name in ("development", "test"):
                    queries = eligible_queries[text_model][task][split_name]
                    text_metric_values = {metric: [] for metric in METRICS}
                    fusion_metric_values = {metric: [] for metric in METRICS}
                    for query_id in queries:
                        gold = ground_truth[query_id][task]
                        if not gold:
                            raise SystemExit(
                                f"평가 query의 gold가 비어 있습니다: {task}:{query_id}"
                            )
                        text_meta = text_per_query[text_model][task][query_id]
                        structure_meta = structure_per_query[structure_model][task][query_id]
                        if (
                            text_meta["gold_count"] != len(gold)
                            or structure_meta["gold_count"] != len(gold)
                        ):
                            raise SystemExit(f"gold 수가 입력과 다릅니다: {task}:{query_id}")
                        try:
                            reranked, _, _ = rerank_text_top_n(
                                text_rankings[text_model][query_id],
                                structure_rankings[structure_model][query_id],
                                top_n,
                                alpha,
                            )
                        except ValueError as exc:
                            raise SystemExit(
                                f"{exc}: source={text_model}, query={query_id}, top_n={top_n}"
                            ) from exc
                        text_original = text_rankings[text_model][query_id]
                        text_metrics = evaluate_ranking(text_original, gold)
                        fusion_metrics = evaluate_ranking(reranked, gold)
                        for metric in METRICS:
                            text_metric_values[metric].append(text_metrics[metric])
                            fusion_metric_values[metric].append(fusion_metrics[metric])
                        per_query_rows.append(
                            {
                                "method": method,
                                "text_source": text_model,
                                "structure_method": structure_model,
                                "top_n": top_n,
                                "alpha": alpha,
                                "evaluation_split": split_name,
                                "query_patent_id": query_id,
                                "task": task,
                                "gold_count": len(gold),
                                "candidate_count_ranked": len(reranked),
                                "ranking_rows_written": min(
                                    args.ranking_output_limit,
                                    len(reranked),
                                ),
                                "text_best_gold_rank": best_gold_rank(text_original, gold) or "",
                                "fusion_best_gold_rank": best_gold_rank(reranked, gold) or "",
                                **fusion_metrics,
                            }
                        )
                        if query_id not in ranking_written:
                            ranking_rows.append(
                                {
                                    "method": method,
                                    "text_source": text_model,
                                    "structure_method": structure_model,
                                    "top_n": top_n,
                                    "alpha": alpha,
                                    "evaluation_split": split_name,
                                    "query_patent_id": query_id,
                                    "ranking": [
                                        {
                                            "rank": rank,
                                            "patent_id": patent_id,
                                            "score": round(score, 12),
                                        }
                                        for rank, (patent_id, score) in enumerate(
                                            reranked[: args.ranking_output_limit],
                                            start=1,
                                        )
                                    ],
                                }
                            )
                            ranking_written.add(query_id)
                    result_rows.append(
                        {
                            "method": method,
                            "text_source": text_model,
                            "structure_method": structure_model,
                            "top_n": top_n,
                            "alpha": alpha,
                            "evaluation_split": split_name,
                            "task": task,
                            "Queries": len(queries),
                            **{
                                metric: (
                                    float(np.mean(fusion_metric_values[metric]))
                                    if queries
                                    else ""
                                )
                                for metric in METRICS
                            },
                        }
                    )
                    if split_name == "test" and queries:
                        for metric in METRICS:
                            comparison_inputs[(method, task, metric)] = (
                                text_metric_values[metric],
                                fusion_metric_values[metric],
                            )

    rng = np.random.default_rng(args.seed)
    comparison_rows: list[dict[str, Any]] = []
    for (method, task, metric), (text_values, fusion_values) in sorted(
        comparison_inputs.items()
    ):
        text_array = np.asarray(text_values, dtype=np.float64)
        fusion_array = np.asarray(fusion_values, dtype=np.float64)
        parts = method.split("__", 2)
        text_model = parts[1]
        top_n = int(parts[0].split("_", 2)[1][3:])
        inference_allowed = len(text_array) >= MINIMUM_INFERENCE_QUERIES
        row: dict[str, Any] = {
            "method": method,
            "text_source": text_model,
            "top_n": top_n,
            "task": task,
            "metric": metric,
            "query_count": len(text_array),
            "text_mean": float(text_array.mean()),
            "fusion_mean": float(fusion_array.mean()),
            "mean_difference_fusion_minus_text": float(
                (fusion_array - text_array).mean()
            ),
            "inference_allowed": inference_allowed,
            "bootstrap_ci_low": "",
            "bootstrap_ci_high": "",
            "sign_flip_p_value": "",
            "holm_adjusted_p_value": "",
            "fusion_wins": int(np.count_nonzero(fusion_array > text_array + 1e-15)),
            "ties": int(np.count_nonzero(np.abs(fusion_array - text_array) <= 1e-15)),
            "fusion_losses": int(np.count_nonzero(fusion_array < text_array - 1e-15)),
        }
        if inference_allowed:
            row.update(paired_summary(text_array, fusion_array, args.iterations, rng))
            row["holm_adjusted_p_value"] = ""
        comparison_rows.append(row)
    adjust_holm_for_allowed_rows(comparison_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = args.output_dir / "t3_reranking_alpha_tuning.csv"
    result_path = args.output_dir / "t3_reranking_results.csv"
    per_query_path = args.output_dir / "t3_reranking_per_query.csv"
    comparison_path = args.output_dir / "t3_reranking_text_comparison.csv"
    ranking_path = (
        args.output_dir
        / f"t3_reranking_rankings_top{args.ranking_output_limit}.jsonl"
    )
    manifest_path = args.output_dir / "t3_reranking_manifest.json"
    write_csv_atomic(
        tuning_path,
        [
            "text_source",
            "structure_method",
            "top_n",
            "selection_split",
            "selection_task",
            "query_count",
            "alpha",
            "selected",
            *METRICS,
        ],
        tuning_rows,
    )
    write_csv_atomic(
        result_path,
        [
            "method",
            "text_source",
            "structure_method",
            "top_n",
            "alpha",
            "evaluation_split",
            "task",
            "Queries",
            *METRICS,
        ],
        result_rows,
    )
    write_csv_atomic(
        per_query_path,
        [
            "method",
            "text_source",
            "structure_method",
            "top_n",
            "alpha",
            "evaluation_split",
            "query_patent_id",
            "task",
            "gold_count",
            "candidate_count_ranked",
            "ranking_rows_written",
            "text_best_gold_rank",
            "fusion_best_gold_rank",
            *METRICS,
        ],
        per_query_rows,
    )
    write_csv_atomic(
        comparison_path,
        [
            "method",
            "text_source",
            "top_n",
            "task",
            "metric",
            "query_count",
            "text_mean",
            "fusion_mean",
            "mean_difference_fusion_minus_text",
            "inference_allowed",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "sign_flip_p_value",
            "holm_adjusted_p_value",
            "fusion_wins",
            "ties",
            "fusion_losses",
        ],
        comparison_rows,
    )
    write_jsonl_atomic(ranking_path, ranking_rows)

    input_paths = (
        args.ground_truth,
        args.query_split,
        args.text_per_query,
        args.text_rankings,
        args.structure_per_query,
        args.structure_rankings,
    )
    manifest = {
        "status": "completed",
        "method": "text_first_structure_weighted_reranking",
        "candidate_policy": (
            "T1 Top-N 멤버십만 구조 점수 결합 대상으로 사용하고, "
            "Top-N 밖 후보는 원 T1 순서와 위치를 유지"
        ),
        "score_definition": (
            "Top-N 내부 alpha * z(T1 score) + (1 - alpha) * z(T2 score)"
        ),
        "normalization_scope": "각 query와 Top-N 설정 내부",
        "top_n_values": args.top_n_values,
        "top_n_selection_policy": "Top-N별 결과를 모두 보고하며 test 결과로 하나를 선택하지 않음",
        "alpha_grid": args.alpha_grid,
        "alpha_selection": {
            "split": "development",
            "source_label": args.development_label,
            "task": PRIMARY_TASK,
            "metric": PRIMARY_SELECTION_METRIC,
            "per_text_source_and_top_n": True,
            "tie_break_policy": (
                "development MRR@100 동률이면 alpha=0.5에 가까운 값을 우선하고, "
                "다시 동률이면 text 비중이 큰 alpha를 선택"
            ),
            "selected_alphas": {
                source: {str(top_n): alpha for top_n, alpha in values.items()}
                for source, values in selected_alphas.items()
            },
            "test_labels_not_used_for_selection": True,
        },
        "zero_variance_policy": (
            "Top-N 내부 점수 표준편차가 0이면 해당 모달리티 z-score를 0으로 처리"
        ),
        "zero_variance_development_query_counts": {
            source: {str(top_n): counts for top_n, counts in values.items()}
            for source, values in zero_variance_counts.items()
        },
        "evaluation_split": {
            "development_label": args.development_label,
            "test_label": args.test_label,
            "split_file_observed_query_counts": observed_split_counts,
            "queries_without_selected_split_are_excluded": True,
            "excluded_query_counts": excluded_query_counts,
        },
        "minimum_inference_queries": MINIMUM_INFERENCE_QUERIES,
        "small_sample_policy": (
            "test query가 30개 미만이면 평균과 win/tie/loss만 기록하고 "
            "bootstrap CI와 p-value를 산출하지 않음"
        ),
        "text_sources": sorted(text_models),
        "structure_method": structure_model,
        "candidate_count": candidate_count,
        "ranking_output_limit": args.ranking_output_limit,
        "paired_inference": {
            "evaluation_split": "test only",
            "unit": "query",
            "comparison": "각 reranking과 해당 T1 원본",
            "bootstrap_iterations": args.iterations,
            "sign_flip_iterations": args.iterations,
            "seed": args.seed,
            "confidence_interval": "percentile paired bootstrap 95% CI",
            "multiple_comparison_policy": (
                "task·metric별 T1 source와 Top-N 조합 전체에 Holm 보정"
            ),
        },
        "ground_truth_field_mapping": {
            "Citation": "all_cited_patents",
            "Conflict-X": args.conflict_field,
        },
        "interpretation_limit": (
            "현재 oracle-derived closed candidate pool의 운영형 모사 결과이며 "
            "open-corpus 계산량 또는 일반화 성능의 실측치로 해석하지 않음"
        ),
        "provenance": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "code_dependencies": {
                str(path.resolve()): sha256_file(path.resolve())
                for path in (
                    Path(__file__).with_name("evaluate_t2_structure_retrieval.py"),
                    Path(__file__).with_name("evaluate_t3_rrf.py"),
                    Path(__file__).with_name("evaluate_t3_weighted_fusion.py"),
                )
            },
        },
        "input_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        },
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "output_files": {},
    }
    for path in (
        tuning_path,
        result_path,
        per_query_path,
        comparison_path,
        ranking_path,
    ):
        manifest["output_files"][str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
