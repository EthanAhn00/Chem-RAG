#!/usr/bin/env python3
"""T1 텍스트 순위와 T2 구조 순위를 Equal RRF로 결합해 평가한다.

이 스크립트는 동일한 query와 후보 집합에서 생성한 전체 순위만
입력으로 받는다. 두 순위의 후보 집합이 다르거나 상위 일부만
저장된 경우에는 결합하지 않고 실행을 중단한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .evaluate_t2_structure_retrieval import (
        METRIC_DEFINITIONS,
        evaluate_ranking,
    )
except ImportError:
    from evaluate_t2_structure_retrieval import (
        METRIC_DEFINITIONS,
        evaluate_ranking,
    )


EP_EXACT_RE = re.compile(r"^EP(\d+)([A-Z]\d?)$")
TASKS = ("Citation", "Conflict-X")
METRICS = (
    "R@1",
    "R@5",
    "R@10",
    "R@50",
    "R@100",
    "MRR@100",
    "MAP@100",
    "nDCG@10",
)


def normalize_exact_ep(value: Any) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").strip().upper())
    match = EP_EXACT_RE.fullmatch(compact)
    return f"EP{match.group(1)}{match.group(2)}" if match else None


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


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_ground_truth(
    path: Path,
    conflict_field: str,
) -> dict[str, dict[str, set[str]]]:
    alternate_field = (
        "x_category_patents"
        if conflict_field == "conflict_patents"
        else "conflict_patents"
    )
    output: dict[str, dict[str, set[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = normalize_exact_ep(row.get("query_patent_id"))
            if query_id is None or query_id in output:
                raise SystemExit(f"ground truth query가 잘못됐습니다: {path}:{row_number}")
            if conflict_field not in row:
                raise SystemExit(
                    f"지정한 X-category 필드가 없습니다: "
                    f"{path}:{row_number}:{conflict_field}"
                )

            def normalized(values: Any, label: str) -> set[str]:
                if not isinstance(values, list):
                    raise SystemExit(f"{label} 형식이 잘못됐습니다: {path}:{row_number}")
                patents = {normalize_exact_ep(value) for value in values}
                if None in patents:
                    raise SystemExit(f"{label} 특허번호가 잘못됐습니다: {path}:{row_number}")
                return {value for value in patents if value is not None}

            citation = normalized(row.get("all_cited_patents") or [], "Citation gold")
            conflict = normalized(row.get(conflict_field) or [], "Conflict-X gold")
            if alternate_field in row:
                alternate = normalized(row.get(alternate_field) or [], "호환 X gold")
                if conflict != alternate:
                    raise SystemExit(
                        "X-category 호환 필드가 서로 다릅니다: "
                        f"{path}:{row_number}"
                    )
            if not conflict.issubset(citation):
                raise SystemExit(f"Conflict-X gold가 Citation gold의 부분집합이 아닙니다: {query_id}")
            output[query_id] = {"Citation": citation, "Conflict-X": conflict}
    return output


def load_per_query_index(
    path: Path,
    model_column: str,
) -> tuple[dict[str, dict[str, dict[str, dict[str, Any]]]], set[str]]:
    output: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    models: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            model_column,
            "query_patent_id",
            "task",
            "gold_count",
            "candidate_count_ranked",
            "ranking_rows_written",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"query 결과 필수 열이 없습니다: {path}:{sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            model = str(row.get(model_column) or "").strip()
            query_id = normalize_exact_ep(row.get("query_patent_id"))
            task = str(row.get("task") or "").strip()
            if not model or query_id is None or task not in TASKS:
                raise SystemExit(f"query 결과 행이 잘못됐습니다: {path}:{row_number}")
            if query_id in output[model][task]:
                raise SystemExit(
                    f"모델·task·query가 중복됩니다: "
                    f"{path}:{row_number}:{model}:{task}:{query_id}"
                )
            output[model][task][query_id] = {
                "gold_count": int(row["gold_count"]),
                "candidate_count_ranked": int(row["candidate_count_ranked"]),
                "ranking_rows_written": int(row["ranking_rows_written"]),
            }
            models.add(model)
    if not models:
        raise SystemExit(f"query 결과가 없습니다: {path}")
    return output, models


def load_rankings(
    path: Path,
    model_column: str,
) -> tuple[dict[str, dict[str, list[str]]], set[str]]:
    output: dict[str, dict[str, list[str]]] = defaultdict(dict)
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
                raise SystemExit(f"ranking 행이 잘못됐습니다: {path}:{row_number}")
            if query_id in output[model]:
                raise SystemExit(f"모델·query ranking이 중복됩니다: {path}:{row_number}")
            patent_ids: list[str] = []
            for expected_rank, item in enumerate(ranking, start=1):
                if not isinstance(item, dict) or item.get("rank") != expected_rank:
                    raise SystemExit(
                        f"ranking 순위가 연속적이지 않습니다: "
                        f"{path}:{row_number}:{expected_rank}"
                    )
                patent_id = normalize_exact_ep(item.get("patent_id"))
                if patent_id is None or patent_id == query_id:
                    raise SystemExit(f"ranking 특허번호가 잘못됐습니다: {path}:{row_number}")
                patent_ids.append(patent_id)
            if len(patent_ids) != len(set(patent_ids)):
                raise SystemExit(f"ranking에 중복 후보가 있습니다: {path}:{row_number}")
            output[model][query_id] = patent_ids
            models.add(model)
    if not models:
        raise SystemExit(f"ranking이 없습니다: {path}")
    return output, models


def require_full_rankings(
    per_query: dict[str, dict[str, dict[str, dict[str, Any]]]],
    rankings: dict[str, dict[str, list[str]]],
    models: set[str],
    label: str,
) -> None:
    for model in sorted(models):
        expected_queries = set().union(
            *(set(per_query[model].get(task, {})) for task in TASKS)
        )
        available_queries = set(rankings.get(model, {}))
        missing_queries = expected_queries - available_queries
        if missing_queries:
            raise SystemExit(
                f"{label} 평가 query의 ranking이 누락됐습니다: "
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
                raise SystemExit(f"{label} 후보 수가 task별로 다릅니다: {model}:{query_id}")
            expected_count = next(iter(expected_counts))
            written_count = next(iter(written_counts))
            actual_count = len(rankings[model][query_id])
            if written_count != expected_count or actual_count != expected_count:
                raise SystemExit(
                    f"{label} ranking이 전체 후보를 포함하지 않습니다: "
                    f"{model}:{query_id}:expected={expected_count}, "
                    f"declared={written_count}, actual={actual_count}"
                )


def fuse_rrf(
    text_ranking: list[str],
    structure_ranking: list[str],
    rrf_k: float,
) -> list[tuple[str, float]]:
    if set(text_ranking) != set(structure_ranking):
        raise ValueError("T1과 T2 ranking의 후보 집합이 다릅니다")
    text_rank = {patent_id: rank for rank, patent_id in enumerate(text_ranking, start=1)}
    structure_rank = {
        patent_id: rank for rank, patent_id in enumerate(structure_ranking, start=1)
    }
    fused = [
        (
            patent_id,
            1.0 / (rrf_k + text_rank[patent_id])
            + 1.0 / (rrf_k + structure_rank[patent_id]),
        )
        for patent_id in text_ranking
    ]
    return sorted(fused, key=lambda item: (-item[1], item[0]))


def best_gold_rank(ranking: list[tuple[str, float]], gold: set[str]) -> int | None:
    return next(
        (rank for rank, (patent_id, _) in enumerate(ranking, start=1) if patent_id in gold),
        None,
    )


def paired_summary(
    text_values: np.ndarray,
    fusion_values: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    differences = fusion_values - text_values
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
    tolerance = 1e-15
    return {
        "mean_difference_fusion_minus_text": observed,
        "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
        "sign_flip_p_value": (
            1.0 + float(np.count_nonzero(np.abs(null_means) >= abs(observed)))
        )
        / (iterations + 1.0),
        "fusion_wins": int(np.count_nonzero(differences > tolerance)),
        "ties": int(np.count_nonzero(np.abs(differences) <= tolerance)),
        "fusion_losses": int(np.count_nonzero(differences < -tolerance)),
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    for task in TASKS:
        for metric in METRICS:
            family = [
                row for row in rows if row["task"] == task and row["metric"] == metric
            ]
            ordered = sorted(family, key=lambda row: row["sign_flip_p_value"])
            running = 0.0
            total = len(ordered)
            for index, row in enumerate(ordered):
                adjusted = min(1.0, row["sign_flip_p_value"] * (total - index))
                running = max(running, adjusted)
                row["holm_adjusted_p_value"] = running


def run_self_test() -> None:
    text = ["EP1A1", "EP2A1", "EP3A1"]
    structure = ["EP3A1", "EP2A1", "EP1A1"]
    fused = fuse_rrf(text, structure, 60.0)
    assert [patent_id for patent_id, _ in fused] == ["EP1A1", "EP3A1", "EP2A1"]
    assert math.isclose(fused[0][1], 1 / 61 + 1 / 63)
    try:
        fuse_rrf(text, ["EP1A1", "EP2A1", "EP4A1"], 60.0)
    except ValueError:
        pass
    else:
        raise AssertionError("다른 후보 집합을 허용했습니다")
    metrics = evaluate_ranking(fused, {"EP3A1"})
    assert metrics["R@1"] == 0.0
    assert metrics["R@5"] == 1.0
    print("자체 검증을 통과했습니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--text-per-query", type=Path)
    parser.add_argument("--text-rankings", type=Path)
    parser.add_argument("--structure-per-query", type=Path)
    parser.add_argument("--structure-rankings", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rrf-k", type=float, default=60.0)
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
    if args.rrf_k <= 0:
        parser.error("--rrf-k는 0보다 커야 합니다")
    if args.ranking_output_limit < 100:
        parser.error("--ranking-output-limit은 100 이상이어야 합니다")
    if args.iterations < 1000:
        parser.error("--iterations는 1000 이상이어야 합니다")

    started_at = time.perf_counter()
    ground_truth = load_ground_truth(args.ground_truth, args.conflict_field)
    text_per_query, text_models = load_per_query_index(args.text_per_query, "source")
    structure_per_query, structure_models = load_per_query_index(
        args.structure_per_query,
        "method",
    )
    if len(structure_models) != 1:
        raise SystemExit(
            f"Equal RRF 1차 실험은 T2 모델 하나만 허용합니다: {sorted(structure_models)}"
        )
    structure_model = next(iter(structure_models))
    text_rankings, text_ranking_models = load_rankings(args.text_rankings, "source")
    structure_rankings, structure_ranking_models = load_rankings(
        args.structure_rankings,
        "method",
    )
    if text_models != text_ranking_models:
        raise SystemExit("T1 query 결과와 ranking의 source 집합이 다릅니다")
    if structure_models != structure_ranking_models:
        raise SystemExit("T2 query 결과와 ranking의 method 집합이 다릅니다")
    require_full_rankings(
        text_per_query,
        text_rankings,
        text_models,
        "T1",
    )
    require_full_rankings(
        structure_per_query,
        structure_rankings,
        structure_models,
        "T2",
    )

    fusion_rankings: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
    result_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    comparison_inputs: dict[
        tuple[str, str, str],
        tuple[list[float], list[float]],
    ] = {}
    query_counts: dict[str, dict[str, int]] = {}

    for text_model in sorted(text_models):
        fusion_name = f"rrf_k{args.rrf_k:g}__{text_model}__{structure_model}"
        query_counts[fusion_name] = {}
        for task in TASKS:
            text_queries = set(text_per_query[text_model].get(task, {}))
            structure_queries = set(structure_per_query[structure_model].get(task, {}))
            if text_queries != structure_queries:
                raise SystemExit(
                    f"T1과 T2의 평가 query 집합이 다릅니다: "
                    f"source={text_model}, task={task}"
                )
            query_counts[fusion_name][task] = len(text_queries)
            task_rows: list[dict[str, Any]] = []
            text_metric_values = {metric: [] for metric in METRICS}
            fusion_metric_values = {metric: [] for metric in METRICS}
            for query_id in sorted(text_queries):
                if query_id not in ground_truth:
                    raise SystemExit(f"ground truth에 query가 없습니다: {query_id}")
                gold = ground_truth[query_id][task]
                if not gold:
                    raise SystemExit(f"평가 query의 gold가 비어 있습니다: {task}:{query_id}")
                text_meta = text_per_query[text_model][task][query_id]
                structure_meta = structure_per_query[structure_model][task][query_id]
                if text_meta["gold_count"] != len(gold) or structure_meta["gold_count"] != len(gold):
                    raise SystemExit(f"gold 수가 입력과 다릅니다: {task}:{query_id}")
                if text_meta["candidate_count_ranked"] != structure_meta["candidate_count_ranked"]:
                    raise SystemExit(f"T1과 T2의 후보 수가 다릅니다: {task}:{query_id}")

                text_ids = text_rankings[text_model][query_id]
                structure_ids = structure_rankings[structure_model][query_id]
                try:
                    fused = fuse_rrf(text_ids, structure_ids, args.rrf_k)
                except ValueError as exc:
                    raise SystemExit(f"{exc}: source={text_model}, query={query_id}") from exc
                fusion_rankings[fusion_name][query_id] = fused
                text_for_evaluation = [(patent_id, 0.0) for patent_id in text_ids]
                text_metrics = evaluate_ranking(text_for_evaluation, gold)
                fusion_metrics = evaluate_ranking(fused, gold)
                for metric in METRICS:
                    text_metric_values[metric].append(text_metrics[metric])
                    fusion_metric_values[metric].append(fusion_metrics[metric])
                row = {
                    "method": fusion_name,
                    "text_source": text_model,
                    "structure_method": structure_model,
                    "query_patent_id": query_id,
                    "task": task,
                    "gold_count": len(gold),
                    "candidate_count_ranked": len(fused),
                    "ranking_rows_written": min(args.ranking_output_limit, len(fused)),
                    "text_best_gold_rank": best_gold_rank(text_for_evaluation, gold) or "",
                    "fusion_best_gold_rank": best_gold_rank(fused, gold) or "",
                    **fusion_metrics,
                }
                task_rows.append(row)
                per_query_rows.append(row)

            result_rows.append(
                {
                    "method": fusion_name,
                    "text_source": text_model,
                    "structure_method": structure_model,
                    "task": task,
                    "Queries": len(task_rows),
                    **{
                        metric: float(np.mean(fusion_metric_values[metric]))
                        for metric in METRICS
                    },
                }
            )
            for metric in METRICS:
                comparison_inputs[(fusion_name, task, metric)] = (
                    text_metric_values[metric],
                    fusion_metric_values[metric],
                )

    rng = np.random.default_rng(args.seed)
    comparison_rows: list[dict[str, Any]] = []
    for (fusion_name, task, metric), (text_values, fusion_values) in sorted(
        comparison_inputs.items()
    ):
        text_array = np.asarray(text_values, dtype=np.float64)
        fusion_array = np.asarray(fusion_values, dtype=np.float64)
        text_model = fusion_name.split("__", 2)[1]
        summary = paired_summary(
            text_array,
            fusion_array,
            args.iterations,
            rng,
        )
        comparison_rows.append(
            {
                "method": fusion_name,
                "text_source": text_model,
                "task": task,
                "metric": metric,
                "query_count": len(text_array),
                "text_mean": float(text_array.mean()),
                "fusion_mean": float(fusion_array.mean()),
                **summary,
            }
        )
    holm_adjust(comparison_rows)

    ranking_rows = [
        {
            "method": fusion_name,
            "query_patent_id": query_id,
            "ranking": [
                {
                    "rank": rank,
                    "patent_id": patent_id,
                    "score": round(score, 12),
                }
                for rank, (patent_id, score) in enumerate(
                    ranking[: args.ranking_output_limit],
                    start=1,
                )
            ],
        }
        for fusion_name, by_query in sorted(fusion_rankings.items())
        for query_id, ranking in sorted(by_query.items())
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "t3_rrf_results.csv"
    per_query_path = args.output_dir / "t3_rrf_per_query.csv"
    comparison_path = args.output_dir / "t3_rrf_text_comparison.csv"
    ranking_path = args.output_dir / "t3_rrf_rankings_top100.jsonl"
    manifest_path = args.output_dir / "t3_rrf_manifest.json"
    write_csv_atomic(
        result_path,
        [
            "method",
            "text_source",
            "structure_method",
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
        args.text_per_query,
        args.text_rankings,
        args.structure_per_query,
        args.structure_rankings,
    )
    manifest = {
        "status": "completed",
        "method": "equal_reciprocal_rank_fusion",
        "rrf_k": args.rrf_k,
        "rrf_k_policy": "원 논문의 k=60을 사전 고정하며 테스트 결과로 조정하지 않음",
        "score_definition": "1/(k + text rank) + 1/(k + structure rank)",
        "candidate_policy": (
            "T1과 T2에서 query 자신을 제외한 동일 전체 후보 순위만 결합; "
            "후보 집합이 다르거나 상위 일부만 있으면 중단"
        ),
        "tie_break_policy": "RRF 점수가 같으면 EP 특허번호 오름차순",
        "metric_definitions": METRIC_DEFINITIONS,
        "text_sources": sorted(text_models),
        "structure_method": structure_model,
        "query_counts": query_counts,
        "ranking_output_limit": args.ranking_output_limit,
        "paired_inference": {
            "unit": "query",
            "comparison": "각 RRF 결합과 해당 결합에 사용한 T1 원본",
            "bootstrap_iterations": args.iterations,
            "sign_flip_iterations": args.iterations,
            "seed": args.seed,
            "confidence_interval": "percentile paired bootstrap 95% CI",
            "multiple_comparison_policy": "task·metric별 T1 source 두 개 비교에 Holm 보정",
        },
        "ground_truth_field_mapping": {
            "Citation": "all_cited_patents",
            "Conflict-X": args.conflict_field,
        },
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
    for path in (result_path, per_query_path, comparison_path, ranking_path):
        manifest["output_files"][str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
