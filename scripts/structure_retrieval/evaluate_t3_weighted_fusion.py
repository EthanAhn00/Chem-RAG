#!/usr/bin/env python3
"""Evaluate weighted T1/T2 fusion using query-normalized scores. Select alpha only on time-split development queries and freeze it on test. Z-score each modality over the complete common candidate pool per query."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .evaluate_t3_rrf import (
        METRICS,
        TASKS,
        load_ground_truth,
        load_per_query_index,
        normalize_exact_ep,
        paired_summary,
        sha256_file,
        write_csv_atomic,
        write_json_atomic,
        write_jsonl_atomic,
    )
    from .evaluate_t2_structure_retrieval import evaluate_ranking
except ImportError:
    from evaluate_t3_rrf import (
        METRICS,
        TASKS,
        load_ground_truth,
        load_per_query_index,
        normalize_exact_ep,
        paired_summary,
        sha256_file,
        write_csv_atomic,
        write_json_atomic,
        write_jsonl_atomic,
    )
    from evaluate_t2_structure_retrieval import evaluate_ranking


PRIMARY_TASK = "Citation"
PRIMARY_SELECTION_METRIC = "MRR@100"
MINIMUM_INFERENCE_QUERIES = 30


def parse_alpha_grid(value: str) -> list[float]:
    """Validate and sort a comma-separated alpha grid."""
    try:
        alphas = sorted({float(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Alpha grid must be numeric") from exc
    if not alphas or any(not math.isfinite(alpha) for alpha in alphas):
        raise argparse.ArgumentTypeError("Alpha grid is empty or contains nonfinite values")
    if alphas[0] < 0.0 or alphas[-1] > 1.0:
        raise argparse.ArgumentTypeError("Alpha must be between 0 and 1")
    if 0.0 not in alphas or 1.0 not in alphas:
        raise argparse.ArgumentTypeError("Alpha grid must include both 0.0 and 1.0")
    return alphas


def load_query_splits(
    path: Path,
    development_label: str,
    test_label: str,
) -> tuple[dict[str, str], dict[str, int]]:
    """Read development/test labels for patents with the query role."""
    output: dict[str, str] = {}
    observed_labels: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "role", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Split file lacks required columns: {path}:{sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if str(row.get("role") or "").strip() != "query":
                continue
            query_id = normalize_exact_ep(row.get("id"))
            label = str(row.get("split") or "").strip()
            if query_id is None or not label:
                raise SystemExit(f"Invalid split-query row: {path}:{row_number}")
            if query_id in output and output[query_id] != label:
                raise SystemExit(f"Conflicting query splits: {query_id}")
            output[query_id] = label
            observed_labels[label] += 1
    if development_label == test_label:
        raise SystemExit("Development and test labels must differ")
    for label in (development_label, test_label):
        if label not in observed_labels:
            raise SystemExit(f"Requested labels absent from split file: {label}")
    return output, dict(sorted(observed_labels.items()))


def load_score_rankings(
    path: Path,
    model_column: str,
) -> tuple[dict[str, dict[str, list[tuple[str, float]]]], set[str]]:
    """Read complete candidate rankings and original scores together."""
    output: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
    models: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            model = str(row.get(model_column) or "").strip()
            query_id = normalize_exact_ep(row.get("query_patent_id"))
            ranking = row.get("ranking")
            if not model or query_id is None or not isinstance(ranking, list):
                raise SystemExit(f"Invalid ranking row: {path}:{row_number}")
            if query_id in output[model]:
                raise SystemExit(f"Duplicate model/query ranking: {path}:{row_number}")
            parsed: list[tuple[str, float]] = []
            for expected_rank, item in enumerate(ranking, start=1):
                if not isinstance(item, dict) or item.get("rank") != expected_rank:
                    raise SystemExit(
                        f"Ranking ranks are not contiguous: "
                        f"{path}:{row_number}:{expected_rank}"
                    )
                patent_id = normalize_exact_ep(item.get("patent_id"))
                try:
                    score = float(item.get("score"))
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        f"Invalid ranking score: {path}:{row_number}"
                    ) from exc
                if patent_id is None or patent_id == query_id or not math.isfinite(score):
                    raise SystemExit(f"Invalid ranking entry: {path}:{row_number}")
                parsed.append((patent_id, score))
            patent_ids = [patent_id for patent_id, _ in parsed]
            if len(patent_ids) != len(set(patent_ids)):
                raise SystemExit(f"Ranking contains duplicate candidates: {path}:{row_number}")
            output[model][query_id] = parsed
            models.add(model)
    if not models:
        raise SystemExit(f"Ranking missing: {path}")
    return output, models


def require_full_score_rankings(
    per_query: dict[str, dict[str, dict[str, dict[str, Any]]]],
    rankings: dict[str, dict[str, list[tuple[str, float]]]],
    models: set[str],
    label: str,
) -> None:
    """Verify full candidate-score coverage for every evaluation query."""
    for model in sorted(models):
        expected_queries = set().union(
            *(set(per_query[model].get(task, {})) for task in TASKS)
        )
        available_queries = set(rankings.get(model, {}))
        missing_queries = expected_queries - available_queries
        if missing_queries:
            raise SystemExit(
                f"{label} evaluation queries lack rankings: "
                f"{model}:count={len(missing_queries)}, "
                f"examples={sorted(missing_queries)[:5]}"
            )
        for query_id in sorted(expected_queries):
            task_rows = [
                per_query[model][task][query_id]
                for task in TASKS
                if query_id in per_query[model].get(task, {})
            ]
            expected_counts = {row["candidate_count_ranked"] for row in task_rows}
            written_counts = {row["ranking_rows_written"] for row in task_rows}
            if len(expected_counts) != 1 or len(written_counts) != 1:
                raise SystemExit(f"{label} candidate counts differ across tasks: {model}:{query_id}")
            expected_count = next(iter(expected_counts))
            written_count = next(iter(written_counts))
            actual_count = len(rankings[model][query_id])
            if written_count != expected_count or actual_count != expected_count:
                raise SystemExit(
                    f"{label} ranking lacks complete candidate scores: "
                    f"{model}:{query_id}:expected={expected_count}, "
                    f"declared={written_count}, actual={actual_count}"
                )


def zscore(values: np.ndarray, label: str) -> np.ndarray:
    """Standardize scores within each query using population standard deviation."""
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    if not math.isfinite(mean) or not math.isfinite(standard_deviation):
        raise ValueError(f"{label} score mean or standard deviation is nonfinite")
    if standard_deviation <= 1e-15:
        raise ValueError(f"{label} score standard deviation is 0")
    return (values - mean) / standard_deviation


def prepare_query_scores(
    text_ranking: list[tuple[str, float]],
    structure_ranking: list[tuple[str, float]],
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Align both score arrays with T1 order and z-score them separately."""
    text_ids = [patent_id for patent_id, _ in text_ranking]
    structure_ids = [patent_id for patent_id, _ in structure_ranking]
    if set(text_ids) != set(structure_ids):
        raise ValueError("T1/T2 ranking candidate sets differ")
    structure_score = dict(structure_ranking)
    text_values = np.asarray([score for _, score in text_ranking], dtype=np.float64)
    structure_values = np.asarray(
        [structure_score[patent_id] for patent_id in text_ids],
        dtype=np.float64,
    )
    return (
        text_ids,
        zscore(text_values, "T1"),
        zscore(structure_values, "T2"),
        structure_ids,
    )


def fuse_weighted_scores(
    patent_ids: list[str],
    text_z: np.ndarray,
    structure_z: np.ndarray,
    structure_order: list[str],
    alpha: float,
) -> list[tuple[str, float]]:
    """Weighted sum of normalized scores, sorted descending."""
    fused_scores = alpha * text_z + (1.0 - alpha) * structure_z
    if math.isclose(alpha, 1.0, abs_tol=1e-15):
        order = np.arange(len(patent_ids))
    elif math.isclose(alpha, 0.0, abs_tol=1e-15):
        patent_to_position = {patent_id: index for index, patent_id in enumerate(patent_ids)}
        order = np.asarray(
            [patent_to_position[patent_id] for patent_id in structure_order],
            dtype=np.int64,
        )
    else:
        ids = np.asarray(patent_ids, dtype=str)
        order = np.lexsort((ids, -fused_scores))
    return [(patent_ids[index], float(fused_scores[index])) for index in order]


def best_gold_rank(ranking: list[tuple[str, float]], gold: set[str]) -> int | None:
    return next(
        (rank for rank, (patent_id, _) in enumerate(ranking, start=1) if patent_id in gold),
        None,
    )


def select_alpha(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Choose alpha maximizing development Citation MRR@100 by predefined rules."""
    if not rows:
        raise SystemExit("No development results for alpha selection")
    best_value = max(float(row[PRIMARY_SELECTION_METRIC]) for row in rows)
    tied = [
        float(row["alpha"])
        for row in rows
        if math.isclose(
            float(row[PRIMARY_SELECTION_METRIC]),
            best_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    # Exact ties prefer values nearest 0.5, then larger alpha (more T1 weight).
    selected = min(tied, key=lambda alpha: (abs(alpha - 0.5), -alpha))
    return selected, best_value


def adjust_holm_for_allowed_rows(rows: list[dict[str, Any]]) -> None:
    """Apply task/metric Holm correction only to rows eligible for inference."""
    for task in TASKS:
        for metric in METRICS:
            family = [
                row
                for row in rows
                if row["task"] == task
                and row["metric"] == metric
                and row["inference_allowed"]
            ]
            ordered = sorted(family, key=lambda row: row["sign_flip_p_value"])
            running = 0.0
            total = len(ordered)
            for index, row in enumerate(ordered):
                adjusted = min(1.0, row["sign_flip_p_value"] * (total - index))
                running = max(running, adjusted)
                row["holm_adjusted_p_value"] = running


def run_self_test() -> None:
    text = [("EP1A1", 3.0), ("EP2A1", 2.0), ("EP3A1", 1.0)]
    structure = [("EP3A1", 0.9), ("EP2A1", 0.8), ("EP1A1", 0.1)]
    ids, text_z, structure_z, structure_order = prepare_query_scores(text, structure)
    assert [item[0] for item in fuse_weighted_scores(ids, text_z, structure_z, structure_order, 1.0)] == [
        "EP1A1",
        "EP2A1",
        "EP3A1",
    ]
    assert [item[0] for item in fuse_weighted_scores(ids, text_z, structure_z, structure_order, 0.0)] == [
        "EP3A1",
        "EP2A1",
        "EP1A1",
    ]
    middle = fuse_weighted_scores(ids, text_z, structure_z, structure_order, 0.5)
    assert len(middle) == 3
    assert math.isclose(float(text_z.mean()), 0.0, abs_tol=1e-12)
    assert math.isclose(float(text_z.std(ddof=0)), 1.0, abs_tol=1e-12)
    parsed = parse_alpha_grid("0,0.5,1")
    assert parsed == [0.0, 0.5, 1.0]
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
            f"Initial weighted-fusion experiment allows only one T2 model: {sorted(structure_models)}"
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
    selected_alphas: dict[str, float] = {}
    for text_model in sorted(text_models):
        development_queries = eligible_queries[text_model][PRIMARY_TASK]["development"]
        metric_values = {
            alpha: {metric: [] for metric in METRICS}
            for alpha in args.alpha_grid
        }
        for query_id in development_queries:
            gold = ground_truth[query_id][PRIMARY_TASK]
            try:
                ids, text_z, structure_z, structure_order = prepare_query_scores(
                    text_rankings[text_model][query_id],
                    structure_rankings[structure_model][query_id],
                )
            except ValueError as exc:
                raise SystemExit(
                    f"{exc}: source={text_model}, query={query_id}"
                ) from exc
            for alpha in args.alpha_grid:
                fused = fuse_weighted_scores(
                    ids,
                    text_z,
                    structure_z,
                    structure_order,
                    alpha,
                )
                metrics = evaluate_ranking(fused, gold)
                for metric in METRICS:
                    metric_values[alpha][metric].append(metrics[metric])
        source_rows: list[dict[str, Any]] = []
        for alpha in args.alpha_grid:
            row = {
                "text_source": text_model,
                "structure_method": structure_model,
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
        selected_alphas[text_model] = selected_alpha
        for row in source_rows:
            row["selected"] = float(row["alpha"]) == selected_alpha

    result_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    comparison_inputs: dict[
        tuple[str, str, str],
        tuple[list[float], list[float]],
    ] = {}
    ranking_written: set[tuple[str, str]] = set()

    for text_model in sorted(text_models):
        alpha = selected_alphas[text_model]
        method = f"zscore_alpha{alpha:g}__{text_model}__{structure_model}"
        for task in TASKS:
            for split_name in ("development", "test"):
                queries = eligible_queries[text_model][task][split_name]
                text_metric_values = {metric: [] for metric in METRICS}
                fusion_metric_values = {metric: [] for metric in METRICS}
                for query_id in queries:
                    gold = ground_truth[query_id][task]
                    if not gold:
                        raise SystemExit(f"Evaluation query has empty gold: {task}:{query_id}")
                    text_meta = text_per_query[text_model][task][query_id]
                    structure_meta = structure_per_query[structure_model][task][query_id]
                    if text_meta["gold_count"] != len(gold) or structure_meta["gold_count"] != len(gold):
                        raise SystemExit(f"Gold count differs from input: {task}:{query_id}")
                    if text_meta["candidate_count_ranked"] != structure_meta["candidate_count_ranked"]:
                        raise SystemExit(f"T1/T2 candidate counts differ: {task}:{query_id}")
                    try:
                        ids, text_z, structure_z, structure_order = prepare_query_scores(
                            text_rankings[text_model][query_id],
                            structure_rankings[structure_model][query_id],
                        )
                    except ValueError as exc:
                        raise SystemExit(
                            f"{exc}: source={text_model}, query={query_id}"
                        ) from exc
                    fused = fuse_weighted_scores(
                        ids,
                        text_z,
                        structure_z,
                        structure_order,
                        alpha,
                    )
                    text_original = text_rankings[text_model][query_id]
                    text_metrics = evaluate_ranking(text_original, gold)
                    fusion_metrics = evaluate_ranking(fused, gold)
                    for metric in METRICS:
                        text_metric_values[metric].append(text_metrics[metric])
                        fusion_metric_values[metric].append(fusion_metrics[metric])
                    per_query_rows.append(
                        {
                            "method": method,
                            "text_source": text_model,
                            "structure_method": structure_model,
                            "alpha": alpha,
                            "evaluation_split": split_name,
                            "query_patent_id": query_id,
                            "task": task,
                            "gold_count": len(gold),
                            "candidate_count_ranked": len(fused),
                            "ranking_rows_written": min(
                                args.ranking_output_limit,
                                len(fused),
                            ),
                            "text_best_gold_rank": best_gold_rank(text_original, gold) or "",
                            "fusion_best_gold_rank": best_gold_rank(fused, gold) or "",
                            **fusion_metrics,
                        }
                    )
                    ranking_key = (method, query_id)
                    if ranking_key not in ranking_written:
                        ranking_rows.append(
                            {
                                "method": method,
                                "text_source": text_model,
                                "structure_method": structure_model,
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
                                        fused[: args.ranking_output_limit],
                                        start=1,
                                    )
                                ],
                            }
                        )
                        ranking_written.add(ranking_key)
                result_rows.append(
                    {
                        "method": method,
                        "text_source": text_model,
                        "structure_method": structure_model,
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
        text_model = method.split("__", 2)[1]
        inference_allowed = len(text_array) >= MINIMUM_INFERENCE_QUERIES
        row: dict[str, Any] = {
            "method": method,
            "text_source": text_model,
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
    tuning_path = args.output_dir / "t3_weighted_alpha_tuning.csv"
    result_path = args.output_dir / "t3_weighted_results.csv"
    per_query_path = args.output_dir / "t3_weighted_per_query.csv"
    comparison_path = args.output_dir / "t3_weighted_text_comparison.csv"
    ranking_path = (
        args.output_dir
        / f"t3_weighted_rankings_top{args.ranking_output_limit}.jsonl"
    )
    manifest_path = args.output_dir / "t3_weighted_manifest.json"
    write_csv_atomic(
        tuning_path,
        [
            "text_source",
            "structure_method",
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
        "method": "query_normalized_weighted_score_fusion",
        "score_definition": "alpha * z(T1 score) + (1 - alpha) * z(T2 score)",
        "normalization": {
            "scope": "All common candidates within each query",
            "method": "population z-score",
            "standard_deviation_ddof": 0,
        },
        "alpha_grid": args.alpha_grid,
        "alpha_selection": {
            "split": "development",
            "source_label": args.development_label,
            "task": PRIMARY_TASK,
            "metric": PRIMARY_SELECTION_METRIC,
            "per_text_source": True,
            "tie_policy": "Ties within 1e-12 prefer alpha nearest 0.5, then larger alpha",
            "selected_alphas": selected_alphas,
            "test_labels_not_used_for_selection": True,
        },
        "evaluation_split": {
            "development_label": args.development_label,
            "test_label": args.test_label,
            "split_file_observed_query_counts": observed_split_counts,
            "queries_without_selected_split_are_excluded": True,
            "excluded_query_counts": excluded_query_counts,
        },
        "candidate_policy": (
            "Fuse identical complete T1/T2 candidate scores after self-exclusion; "
            "stop on mismatched or truncated candidate sets"
        ),
        "endpoint_reproduction_policy": (
            "Alpha 1.0 retains T1 order; alpha 0.0 retains T2 order"
        ),
        "tie_break_policy": "For intermediate alpha, equal fused scores break by ascending EP identifier",
        "primary_task": PRIMARY_TASK,
        "secondary_task": "Conflict-X",
        "minimum_inference_queries": MINIMUM_INFERENCE_QUERIES,
        "small_sample_policy": (
            "With fewer than 30 test queries, report only means and win/tie/loss, "
            "without bootstrap CI or p-values"
        ),
        "text_sources": sorted(text_models),
        "structure_method": structure_model,
        "ranking_output_limit": args.ranking_output_limit,
        "paired_inference": {
            "evaluation_split": "test only",
            "unit": "query",
            "comparison": "Selected weighted fusion versus its corresponding original T1",
            "bootstrap_iterations": args.iterations,
            "sign_flip_iterations": args.iterations,
            "seed": args.seed,
            "confidence_interval": "percentile paired bootstrap 95% CI",
            "multiple_comparison_policy": "Holm correction over T1-source comparisons per task/metric",
        },
        "ground_truth_field_mapping": {
            "Citation": "all_cited_patents",
            "Conflict-X": args.conflict_field,
        },
        "interpretation_limit": (
            "Exploratory results on the current oracle-derived closed candidate pool; "
            "not open-corpus generalization performance"
        ),
        "provenance": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
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
