#!/usr/bin/env python3
"""Rerank T1 Top-N using normalized T1/T2 scores. Report Top-N and alpha on development queries, retaining all Top-N sensitivity settings rather than selecting one post hoc. Choose each alpha by development Citation MRR@100 and freeze it on test."""

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
        raise argparse.ArgumentTypeError("Top-N must be an integer") from exc
    if not values or any(item < 100 for item in values):
        raise argparse.ArgumentTypeError("Top-N must be an integer of at least 100")
    return values


def zscore_or_zero(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """Treat zero variance as an uninformative signal and return a zero vector."""
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    if not math.isfinite(mean) or not math.isfinite(standard_deviation):
        raise ValueError("Score mean or standard deviation is nonfinite")
    if standard_deviation <= 1e-15:
        return np.zeros_like(values, dtype=np.float64), True
    return (values - mean) / standard_deviation, False


def rerank_text_top_n(
    text_ranking: list[tuple[str, float]],
    structure_ranking: list[tuple[str, float]],
    top_n: int,
    alpha: float,
) -> tuple[list[tuple[str, float]], bool, bool]:
    """Preserve T1 Top-N membership and change only its internal ordering."""
    text_ids = [patent_id for patent_id, _ in text_ranking]
    structure_ids = [patent_id for patent_id, _ in structure_ranking]
    if set(text_ids) != set(structure_ids):
        raise ValueError("T1/T2 ranking candidate sets differ")
    if top_n > len(text_ranking):
        raise ValueError(f"Top-N exceeds candidate count: {top_n}>{len(text_ranking)}")

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
    # Evaluation ignores tail scores; append the tail unchanged to preserve original T1 order.
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
        assert "candidate set" in str(exc)
    else:
        raise AssertionError("Candidate-set mismatch was not detected")
    try:
        rerank_text_top_n(text, structure, 5, 0.5)
    except ValueError as exc:
        assert "candidate count" in str(exc)
    else:
        raise AssertionError("Top-N exceeding candidate count was not detected")
    assert parse_top_n_values("500,100") == [100, 500]
    zero, collapsed = zscore_or_zero(np.asarray([1.0, 1.0]))
    assert collapsed and np.array_equal(zero, np.zeros(2))
    print("Self-check passed.")


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
        parser.error(f"Missing required arguments: {', '.join(missing_arguments)}")
    missing_files = [
        str(path)
        for name, path in required.items()
        if name != "--output-dir" and not path.is_file()
    ]
    if missing_files:
        raise SystemExit(f"Input file missing: {missing_files}")
    if args.ranking_output_limit < 100:
        parser.error("--ranking-output-limit must be at least 100")
    if args.iterations < 1000:
        parser.error("--iterations must be at least 1000")

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
            f"Initial reranking experiment allows only one T2 model: "
            f"{sorted(structure_models)}"
        )
    structure_model = next(iter(structure_models))
    text_rankings, text_ranking_models = load_score_rankings(args.text_rankings, "source")
    structure_rankings, structure_ranking_models = load_score_rankings(
        args.structure_rankings,
        "method",
    )
    if text_models != text_ranking_models:
        raise SystemExit("T1 query-result and ranking source sets differ")
    if structure_models != structure_ranking_models:
        raise SystemExit("T2 query-result and ranking method sets differ")
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
        raise SystemExit(f"T1 candidate counts differ across queries: {sorted(candidate_counts)}")
    candidate_count = next(iter(candidate_counts))
    if max(args.top_n_values) > candidate_count:
        raise SystemExit(
            f"Top-N exceeds candidate count: "
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
                    f"T1/T2 evaluation query sets differ: "
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
                                f"Evaluation query has empty gold: {task}:{query_id}"
                            )
                        text_meta = text_per_query[text_model][task][query_id]
                        structure_meta = structure_per_query[structure_model][task][query_id]
                        if (
                            text_meta["gold_count"] != len(gold)
                            or structure_meta["gold_count"] != len(gold)
                        ):
                            raise SystemExit(f"Gold count differs from input: {task}:{query_id}")
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
            "Fuse structure scores only within T1 Top-N membership; "
            "retain original T1 order and positions outside Top-N"
        ),
        "score_definition": (
            "Within Top-N: alpha * z(T1 score) + (1 - alpha) * z(T2 score)"
        ),
        "normalization_scope": "Within each query and Top-N setting",
        "top_n_values": args.top_n_values,
        "top_n_selection_policy": "Report all Top-N settings without choosing one by test results",
        "alpha_grid": args.alpha_grid,
        "alpha_selection": {
            "split": "development",
            "source_label": args.development_label,
            "task": PRIMARY_TASK,
            "metric": PRIMARY_SELECTION_METRIC,
            "per_text_source_and_top_n": True,
            "tie_break_policy": (
                "For tied development MRR@100, prefer alpha nearest 0.5; "
                "then prefer larger alpha (more text weight)"
            ),
            "selected_alphas": {
                source: {str(top_n): alpha for top_n, alpha in values.items()}
                for source, values in selected_alphas.items()
            },
            "test_labels_not_used_for_selection": True,
        },
        "zero_variance_policy": (
            "Zero within-Top-N score standard deviation gives zero modality z-scores"
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
            "With fewer than 30 test queries, report only means and win/tie/loss, "
            "without bootstrap CI or p-values"
        ),
        "text_sources": sorted(text_models),
        "structure_method": structure_model,
        "candidate_count": candidate_count,
        "ranking_output_limit": args.ranking_output_limit,
        "paired_inference": {
            "evaluation_split": "test only",
            "unit": "query",
            "comparison": "Each reranking versus its corresponding original T1",
            "bootstrap_iterations": args.iterations,
            "sign_flip_iterations": args.iterations,
            "seed": args.seed,
            "confidence_interval": "percentile paired bootstrap 95% CI",
            "multiple_comparison_policy": (
                "Holm correction over all T1-source/Top-N combinations per task/metric"
            ),
        },
        "ground_truth_field_mapping": {
            "Citation": "all_cited_patents",
            "Conflict-X": args.conflict_field,
        },
        "interpretation_limit": (
            "Operational simulation on the current oracle-derived closed candidate pool; "
            "not measured open-corpus computation or generalization performance"
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
