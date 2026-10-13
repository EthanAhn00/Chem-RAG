#!/usr/bin/env python3
"""Analyze uncertainty, random baselines, and structure-count bias on the common B1/B2 subset. Align per-query results by query/task for paired bootstrap and sign-flip comparisons. Simulate random rankings with actual candidate/gold counts. Optionally use Top-k rankings and structure files to diagnose associations between structure counts, exposure, and performance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .evaluate_t2_structure_retrieval import METRIC_DEFINITIONS
except ImportError:
    from evaluate_t2_structure_retrieval import METRIC_DEFINITIONS


METRICS = [
    "R@1",
    "R@5",
    "R@10",
    "R@50",
    "R@100",
    "MRR@100",
    "MAP@100",
    "nDCG@10",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def normalize_patent_id(value: Any) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").strip().upper())
    match = re.fullmatch(r"EP(\d+)([A-Z]\d?)", compact)
    return f"EP{match.group(1)}{match.group(2)}" if match else None


def model_name(row: dict[str, Any], path: Path) -> str:
    source = str(row.get("source") or "").strip()
    if source:
        return source
    method = str(row.get("method") or path.stem).strip()
    maximum = str(row.get("max_representatives") or "").strip()
    if maximum and method == "representative_max_tanimoto":
        return f"{method}_k{maximum}"
    return method


def model_display_name(name: str) -> str:
    fixed = {
        "mean_morgan_cosine": "Patent-mean Morgan fingerprint cosine",
        "mean_morgan_cosine_original": (
            "Patent-mean Morgan fingerprint cosine (original canonical SMILES)"
        ),
        "mean_morgan_cosine_common_input": (
            "Patent-mean Morgan fingerprint cosine (common MoLFormer model_smiles)"
        ),
        "molformer_mean_pooling": "Patent-mean MoLFormer molecule-embedding cosine",
        "qwen_whole_claims": "Qwen whole-claims text baseline",
        "nv_document": "NV-Embed document text baseline",
    }
    if name in fixed:
        return fixed[name]
    match = re.fullmatch(r"representative_max_tanimoto_k(\d+)", name)
    if match:
        return (
            "Representative max Tanimoto "
            f"(farthest-first K={match.group(1)} approximation)"
        )
    return name


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path_value = value.partition("=")
    label = label.strip()
    path_value = path_value.strip()
    if not separator or not label or not path_value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH format")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(
            "LABEL may contain only ASCII letters, digits, dots, underscores, and hyphens"
        )
    return label, Path(path_value)


def load_per_query(
    inputs: list[tuple[Path, str | None]],
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], dict[str, Any]]:
    data: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    metadata: dict[str, Any] = {}
    for path, label_override in inputs:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "query_patent_id",
                "task",
                "gold_count",
                "candidate_count_ranked",
                *METRICS,
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"Missing required columns in query results: {path}:{sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                query_id = normalize_patent_id(row.get("query_patent_id"))
                task = str(row.get("task") or "").strip()
                if query_id is None or task not in {"Citation", "Conflict-X"}:
                    raise SystemExit(f"Invalid query-result row: {path}:{row_number}")
                name = label_override or model_name(row, path)
                if query_id in data[name][task]:
                    raise SystemExit(
                        f"Duplicate model/task/query: {path}:{row_number}:{name}:{task}:{query_id}"
                    )
                values = {
                    "gold_count": float(row["gold_count"]),
                    "candidate_count_ranked": float(row["candidate_count_ranked"]),
                    **{metric: float(row[metric]) for metric in METRICS},
                }
                if str(row.get("query_structure_count") or "").strip():
                    values["query_structure_count"] = float(row["query_structure_count"])
                data[name][task][query_id] = values
                metadata.setdefault(
                    name,
                    {
                        "paths": set(),
                        "tasks": set(),
                        "input_label_override": label_override,
                    },
                )
                if metadata[name]["input_label_override"] != label_override:
                    raise SystemExit(
                        f"Conflicting input-alias policies for one model name: {name}"
                    )
                metadata[name]["paths"].add(str(path))
                metadata[name]["tasks"].add(task)
    if len(data) < 2:
        raise SystemExit("Paired comparison requires results from at least two distinct models")
    serializable_metadata = {
        name: {
            "paths": sorted(values["paths"]),
            "tasks": sorted(values["tasks"]),
            "input_label_override": values["input_label_override"],
        }
        for name, values in sorted(metadata.items())
    }
    return data, serializable_metadata


def require_matched_queries(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, list[str]]:
    matched: dict[str, list[str]] = {}
    model_names = sorted(data)
    for task in ("Citation", "Conflict-X"):
        query_sets = {name: set(data[name].get(task, {})) for name in model_names}
        reference_name = model_names[0]
        reference = query_sets[reference_name]
        for name in model_names[1:]:
            if query_sets[name] != reference:
                raise SystemExit(
                    "Paired models have different query sets: "
                    f"task={task}, {reference_name}={len(reference)}, "
                    f"{name}={len(query_sets[name])}, "
                    f"only_reference={len(reference - query_sets[name])}, "
                    f"only_other={len(query_sets[name] - reference)}"
                )
        for query_id in reference:
            gold_counts = {
                data[name][task][query_id]["gold_count"] for name in model_names
            }
            candidate_counts = {
                data[name][task][query_id]["candidate_count_ranked"]
                for name in model_names
            }
            if len(gold_counts) != 1 or len(candidate_counts) != 1:
                raise SystemExit(
                    "Gold or candidate counts differ in paired comparisons: "
                    f"task={task}, query={query_id}"
                )
        matched[task] = sorted(reference)
    return matched


def paired_statistics(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    matched_queries: dict[str, list[str]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for task, query_ids in matched_queries.items():
        for left, right in itertools.combinations(sorted(data), 2):
            for metric in METRICS:
                left_values = np.asarray(
                    [data[left][task][query_id][metric] for query_id in query_ids],
                    dtype=np.float64,
                )
                right_values = np.asarray(
                    [data[right][task][query_id][metric] for query_id in query_ids],
                    dtype=np.float64,
                )
                differences = left_values - right_values
                sample_indices = rng.integers(
                    0,
                    len(differences),
                    size=(iterations, len(differences)),
                )
                bootstrap_means = differences[sample_indices].mean(axis=1)
                del sample_indices
                signs = rng.choice(
                    np.asarray([-1.0, 1.0]),
                    size=(iterations, len(differences)),
                )
                null_means = (signs * differences).mean(axis=1)
                observed = float(differences.mean())
                p_value = (
                    1.0 + float(np.count_nonzero(np.abs(null_means) >= abs(observed)))
                ) / (iterations + 1.0)
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "model_left": left,
                        "model_right": right,
                        "query_count": len(query_ids),
                        "left_mean": float(left_values.mean()),
                        "right_mean": float(right_values.mean()),
                        "mean_difference_left_minus_right": observed,
                        "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                        "sign_flip_p_value": p_value,
                    }
                )
    for task in ("Citation", "Conflict-X"):
        for metric in METRICS:
            selected = [
                row for row in rows if row["task"] == task and row["metric"] == metric
            ]
            ordered = sorted(selected, key=lambda row: row["sign_flip_p_value"])
            running = 0.0
            total = len(ordered)
            for index, row in enumerate(ordered):
                adjusted = min(1.0, row["sign_flip_p_value"] * (total - index))
                running = max(running, adjusted)
                row["holm_adjusted_p_value"] = running
    return rows


def random_metrics(ranks: np.ndarray, gold_count: int) -> dict[str, float]:
    result = {
        f"R@{cutoff}": float(np.count_nonzero(ranks <= cutoff) / gold_count)
        for cutoff in (1, 5, 10, 50, 100)
    }
    first = int(ranks[0])
    result["MRR@100"] = 1.0 / first if first <= 100 else 0.0
    within_100 = ranks[ranks <= 100]
    result["MAP@100"] = float(
        sum((index + 1) / rank for index, rank in enumerate(within_100))
        / gold_count
    )
    within_10 = ranks[ranks <= 10]
    dcg = sum(1.0 / math.log2(int(rank) + 1) for rank in within_10)
    idcg = sum(
        1.0 / math.log2(index + 2)
        for index in range(min(gold_count, 10))
    )
    result["nDCG@10"] = dcg / idcg if idcg else 0.0
    return result


def simulate_random_baseline(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    matched_queries: dict[str, list[str]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    reference_model = sorted(data)[0]
    rows: list[dict[str, Any]] = []
    for task, query_ids in matched_queries.items():
        aggregate = {metric: np.zeros(iterations, dtype=np.float64) for metric in METRICS}
        candidate_counts: set[int] = set()
        for query_id in query_ids:
            values = data[reference_model][task][query_id]
            gold_count = int(values["gold_count"])
            candidate_count = int(values["candidate_count_ranked"])
            candidate_counts.add(candidate_count)
            for iteration in range(iterations):
                ranks = np.sort(
                    rng.choice(candidate_count, size=gold_count, replace=False) + 1
                )
                result = random_metrics(ranks, gold_count)
                for metric in METRICS:
                    aggregate[metric][iteration] += result[metric]
        for metric in METRICS:
            distribution = aggregate[metric] / len(query_ids)
            cutoff = int(metric.split("@", 1)[1]) if metric.startswith("R@") else None
            analytic_expectation = ""
            if cutoff is not None and len(candidate_counts) == 1:
                analytic_expectation = min(cutoff, next(iter(candidate_counts))) / next(
                    iter(candidate_counts)
                )
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "query_count": len(query_ids),
                    "candidate_count_values": "|".join(map(str, sorted(candidate_counts))),
                    "analytic_expectation": analytic_expectation,
                    "simulation_mean": float(distribution.mean()),
                    "simulation_ci_low": float(np.quantile(distribution, 0.025)),
                    "simulation_ci_high": float(np.quantile(distribution, 0.975)),
                }
            )
    return rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 3:
        return None
    array_x = np.asarray(values_x, dtype=np.float64)
    array_y = np.asarray(values_y, dtype=np.float64)
    if np.all(array_x == array_x[0]) or np.all(array_y == array_y[0]):
        return None
    ranked_x = average_ranks(array_x)
    ranked_y = average_ranks(array_y)
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def population_skewness(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(array.std())
    if standard_deviation == 0.0:
        return 0.0
    centered = (array - float(array.mean())) / standard_deviation
    return float(np.mean(centered**3))


def load_structure_counts(path: Path) -> dict[str, int]:
    structures: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patent_id", "canonical_smiles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in structure file: {path}:{sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            patent_id = normalize_patent_id(row.get("patent_id"))
            smiles = str(row.get("canonical_smiles") or "").strip()
            if patent_id is None:
                raise SystemExit(f"Invalid structure patent identifier: {path}:{row_number}")
            if smiles:
                structures[patent_id].add(smiles)
    return {patent_id: len(values) for patent_id, values in structures.items()}


def analyze_exposure_bias(
    ranking_inputs: list[tuple[Path, str | None]],
    structure_counts: dict[str, int],
    analysis_query_ids: set[str],
    expected_model_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rankings_by_model: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
    for path, label_override in ranking_inputs:
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                name = label_override or model_name(row, path)
                query_id = normalize_patent_id(row.get("query_patent_id"))
                if query_id is None:
                    raise SystemExit(
                        f"Invalid ranking query identifier: {path}:{row_number}"
                    )
                if query_id in rankings_by_model[name]:
                    raise SystemExit(
                        f"Duplicate model/query ranking: "
                        f"{path}:{row_number}:{name}:{query_id}"
                    )
                ranking = row.get("ranking")
                if not isinstance(ranking, list):
                    raise SystemExit(f"Invalid ranking format: {path}:{row_number}")
                normalized_ranking: list[tuple[str, float]] = []
                for item in ranking:
                    patent_id = normalize_patent_id(item.get("patent_id"))
                    if patent_id is None:
                        raise SystemExit(f"Invalid ranking patent identifier: {path}:{row_number}")
                    normalized_ranking.append((patent_id, float(item["score"])))
                rankings_by_model[name][query_id] = normalized_ranking

    if not rankings_by_model:
        raise SystemExit("No rankings available for structure-count bias analysis")
    ranking_model_names = set(rankings_by_model)
    if ranking_model_names != expected_model_names:
        raise SystemExit(
            "Query-result and ranking model-name sets differ: "
            f"query_only={sorted(expected_model_names - ranking_model_names)}, "
            f"ranking_only={sorted(ranking_model_names - expected_model_names)}"
        )
    if not analysis_query_ids:
        raise SystemExit("No common queries for structure-count bias analysis")

    source_query_counts = {
        name: len(query_rankings)
        for name, query_rankings in sorted(rankings_by_model.items())
    }
    missing_queries = {
        name: sorted(analysis_query_ids - set(query_rankings))
        for name, query_rankings in sorted(rankings_by_model.items())
        if analysis_query_ids - set(query_rankings)
    }
    if missing_queries:
        raise SystemExit(
            "Missing common-query rankings for exposure-bias comparison: "
            + json.dumps(missing_queries, ensure_ascii=False, sort_keys=True)
        )

    exposures: dict[str, defaultdict[str, int]] = {}
    scores: dict[str, defaultdict[str, list[float]]] = {}
    analyzed_query_count = len(analysis_query_ids)
    for name, query_rankings in sorted(rankings_by_model.items()):
        exposures[name] = defaultdict(int)
        scores[name] = defaultdict(list)
        for query_id in sorted(analysis_query_ids):
            for patent_id, score in query_rankings[query_id]:
                exposures[name][patent_id] += 1
                scores[name][patent_id].append(score)

    patent_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for name in sorted(exposures):
        structure_values: list[float] = []
        exposure_values: list[float] = []
        exposed_structure_values: list[float] = []
        mean_score_values: list[float] = []
        for patent_id, structure_count in sorted(structure_counts.items()):
            exposure_count = exposures[name][patent_id]
            patent_scores = scores[name][patent_id]
            mean_score = float(np.mean(patent_scores)) if patent_scores else ""
            patent_rows.append(
                {
                    "model": name,
                    "patent_id": patent_id,
                    "structure_count": structure_count,
                    "topk_exposure_count": exposure_count,
                    "topk_exposure_fraction": exposure_count / analyzed_query_count,
                    "mean_topk_score_when_exposed": mean_score,
                }
            )
            structure_values.append(float(structure_count))
            exposure_values.append(float(exposure_count))
            if patent_scores:
                exposed_structure_values.append(float(structure_count))
                mean_score_values.append(float(mean_score))
        ordered_exposures = sorted(exposure_values, reverse=True)
        top_one_percent_count = max(1, math.ceil(len(ordered_exposures) * 0.01))
        total_exposure = float(sum(ordered_exposures))
        top_one_percent_share = (
            sum(ordered_exposures[:top_one_percent_count]) / total_exposure
            if total_exposure
            else 0.0
        )
        summary_rows.append(
            {
                "model": name,
                "patent_count": len(structure_values),
                "source_ranking_query_count": source_query_counts[name],
                "ranking_query_count": analyzed_query_count,
                "ignored_noncommon_query_count": (
                    source_query_counts[name] - analyzed_query_count
                ),
                "spearman_structure_count_vs_topk_exposure": spearman(
                    structure_values,
                    exposure_values,
                ),
                "exposed_patent_count": len(exposed_structure_values),
                "spearman_structure_count_vs_mean_topk_score_exposed": spearman(
                    exposed_structure_values,
                    mean_score_values,
                ),
                "topk_exposure_population_skewness": population_skewness(
                    exposure_values
                ),
                "zero_exposure_patent_fraction": (
                    sum(value == 0 for value in exposure_values)
                    / len(exposure_values)
                ),
                "top_1_percent_patent_count": top_one_percent_count,
                "top_1_percent_exposure_share": top_one_percent_share,
                "maximum_candidate_exposure_fraction_of_queries": (
                    max(exposure_values) / analyzed_query_count
                ),
            }
        )
    metadata = {
        "policy": (
            "Use only identical queries in each task's paired-comparison query union "
            "for Top-k exposure-bias analysis across all models"
        ),
        "analyzed_query_count": analyzed_query_count,
        "source_ranking_query_counts": source_query_counts,
        "ignored_noncommon_query_counts": {
            name: count - analyzed_query_count
            for name, count in source_query_counts.items()
        },
    }
    return patent_rows, summary_rows, metadata


def analyze_query_structure_bias(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, tasks in sorted(data.items()):
        for task, queries in sorted(tasks.items()):
            available = [
                values for values in queries.values() if "query_structure_count" in values
            ]
            if not available:
                continue
            for metric in METRICS:
                rows.append(
                    {
                        "model": name,
                        "task": task,
                        "metric": metric,
                        "query_count": len(available),
                        "spearman_query_structure_count_vs_metric": spearman(
                            [values["query_structure_count"] for values in available],
                            [values[metric] for values in available],
                        ),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=Path, action="append", default=[])
    parser.add_argument(
        "--per-query-labeled",
        type=parse_labeled_path,
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Apply explicit analysis aliases instead of model names stored in files",
    )
    parser.add_argument("--rankings", type=Path, action="append", default=[])
    parser.add_argument(
        "--rankings-labeled",
        type=parse_labeled_path,
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Apply explicit analysis aliases to ranking files",
    )
    parser.add_argument("--structures", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--random-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    per_query_inputs = [
        *((path, None) for path in args.per_query),
        *((path, label) for label, path in args.per_query_labeled),
    ]
    ranking_inputs = [
        *((path, None) for path in args.rankings),
        *((path, label) for label, path in args.rankings_labeled),
    ]
    if not per_query_inputs:
        parser.error("Specify at least one --per-query or --per-query-labeled input")
    if args.iterations < 1000 or args.random_iterations < 500:
        parser.error("Require at least 1000 paired repetitions and 500 random repetitions")
    if bool(ranking_inputs) != bool(args.structures):
        parser.error("Structure-count bias analysis requires both --rankings and --structures")
    input_paths = [
        *(path for path, _ in per_query_inputs),
        *(path for path, _ in ranking_inputs),
    ]
    if args.structures:
        input_paths.append(args.structures)
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Input file missing: {missing}")

    data, model_metadata = load_per_query(per_query_inputs)
    matched_queries = require_matched_queries(data)
    paired_rows = paired_statistics(data, matched_queries, args.iterations, args.seed)
    random_rows = simulate_random_baseline(
        data,
        matched_queries,
        args.random_iterations,
        args.seed + 1,
    )
    query_bias_rows = analyze_query_structure_bias(data)
    patent_bias_rows: list[dict[str, Any]] = []
    exposure_summary_rows: list[dict[str, Any]] = []
    exposure_metadata: dict[str, Any] | None = None
    if args.structures:
        structure_counts = load_structure_counts(args.structures)
        exposure_query_ids = set().union(*matched_queries.values())
        patent_bias_rows, exposure_summary_rows, exposure_metadata = analyze_exposure_bias(
            ranking_inputs,
            structure_counts,
            exposure_query_ids,
            set(data),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[Path, list[str], list[dict[str, Any]]]] = [
        (
            args.output_dir / "paired_model_comparisons.csv",
            [
                "task",
                "metric",
                "model_left",
                "model_right",
                "query_count",
                "left_mean",
                "right_mean",
                "mean_difference_left_minus_right",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "sign_flip_p_value",
                "holm_adjusted_p_value",
            ],
            paired_rows,
        ),
        (
            args.output_dir / "random_ranking_baseline.csv",
            [
                "task",
                "metric",
                "query_count",
                "candidate_count_values",
                "analytic_expectation",
                "simulation_mean",
                "simulation_ci_low",
                "simulation_ci_high",
            ],
            random_rows,
        ),
        (
            args.output_dir / "query_structure_count_correlations.csv",
            [
                "model",
                "task",
                "metric",
                "query_count",
                "spearman_query_structure_count_vs_metric",
            ],
            query_bias_rows,
        ),
    ]
    if patent_bias_rows:
        outputs.extend(
            [
                (
                    args.output_dir / "patent_structure_exposure.csv",
                    [
                        "model",
                        "patent_id",
                        "structure_count",
                        "topk_exposure_count",
                        "topk_exposure_fraction",
                        "mean_topk_score_when_exposed",
                    ],
                    patent_bias_rows,
                ),
                (
                    args.output_dir / "structure_exposure_correlations.csv",
                    [
                        "model",
                        "patent_count",
                        "source_ranking_query_count",
                        "ranking_query_count",
                        "ignored_noncommon_query_count",
                        "spearman_structure_count_vs_topk_exposure",
                        "exposed_patent_count",
                        "spearman_structure_count_vs_mean_topk_score_exposed",
                        "topk_exposure_population_skewness",
                        "zero_exposure_patent_fraction",
                        "top_1_percent_patent_count",
                        "top_1_percent_exposure_share",
                        "maximum_candidate_exposure_fraction_of_queries",
                    ],
                    exposure_summary_rows,
                ),
            ]
        )
    for path, fieldnames, rows in outputs:
        write_csv_atomic(path, fieldnames, rows)

    manifest = {
        "status": "completed",
        "paired_policy": (
            "Apply query-level paired bootstrap and paired sign-flip tests "
            "to model metric differences for identical task/query pairs"
        ),
        "multiple_comparison_policy": "Holm correction over model-pair p-values per task/metric",
        "paired_inference_resolution": {
            "sign_flip_tail": "two-sided absolute mean difference",
            "plus_one_correction": True,
            "permutation_iterations": args.iterations,
            "minimum_raw_p_value": 1.0 / (args.iterations + 1.0),
            "family_unit": "All model pairs within each task/metric",
            "model_count": len(data),
            "comparisons_per_family": math.comb(len(data), 2),
            "minimum_possible_first_rank_holm_adjusted_p_value": min(
                1.0,
                math.comb(len(data), 2) / (args.iterations + 1.0),
            ),
            "interpretation": (
                "Do not claim p-values below the repetition-dependent resolution; "
                "interpret the minimum as resolution-limited"
            ),
        },
        "random_baseline_policy": (
            "Simulate uniform random rankings preserving actual candidate and gold counts per query"
        ),
        "random_baseline_inference": (
            "Outputs are descriptive baselines and simulation intervals; "
            "no multiplicity-corrected model-vs-random p-value test is performed"
        ),
        "candidate_count_policy": (
            "Use actual structure-bearing common-candidate counts after excluding self"
        ),
        "model_metadata": model_metadata,
        "model_display_names": {
            name: model_display_name(name) for name in sorted(data)
        },
        "matched_query_counts": {
            task: len(query_ids) for task, query_ids in matched_queries.items()
        },
        "exposure_analysis": exposure_metadata,
        "metric_definitions": METRIC_DEFINITIONS,
        "iterations": args.iterations,
        "random_iterations": args.random_iterations,
        "seed": args.seed,
        "provenance": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "input_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        },
        "output_files": {},
    }
    for path, _, _ in outputs:
        manifest["output_files"][str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json_atomic(args.output_dir / "t2_robustness_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
