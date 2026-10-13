#!/usr/bin/env python3
"""Exploratory learned T1/T2 fusion with a pairwise logistic ranker. Split existing train queries into fit/development by shared-gold connected components. Select regularization on development Citation MRR@100, retrain on all train queries, and evaluate the existing test set once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit("scikit-learn is required") from exc

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
        best_gold_rank,
        load_query_splits,
        load_score_rankings,
        prepare_query_scores,
        require_full_score_rankings,
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
        best_gold_rank,
        load_query_splits,
        load_score_rankings,
        prepare_query_scores,
        require_full_score_rankings,
    )


PRIMARY_TASK = "Citation"
SELECTION_METRIC = "MRR@100"
FEATURE_NAMES = (
    "text_z",
    "structure_z",
    "text_reciprocal_log_rank",
    "structure_reciprocal_log_rank",
    "z_interaction",
)


def parse_positive_grid(value: str) -> list[float]:
    try:
        values = sorted({float(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("C grid must be numeric") from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("C grid values must be finite and greater than 0")
    return values


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)]).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            if left_root < right_root:
                self.parent[right_root] = left_root
            else:
                self.parent[left_root] = right_root


def split_training_queries(
    queries: list[str],
    ground_truth: dict[str, dict[str, set[str]]],
    development_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Place queries connected by shared gold in the same fit/development partition."""
    if not 0.05 <= development_fraction <= 0.5:
        raise ValueError("Development fraction must be between 0.05 and 0.5")
    query_set = set(queries)
    union_find = UnionFind()
    for query_id in queries:
        query_node = f"q:{query_id}"
        union_find.find(query_node)
        for patent_id in sorted(ground_truth[query_id][PRIMARY_TASK]):
            union_find.union(query_node, f"g:{patent_id}")
    components: dict[str, list[str]] = defaultdict(list)
    for query_id in queries:
        components[union_find.find(f"q:{query_id}")].append(query_id)
    ordered = sorted(
        (sorted(items) for items in components.values()),
        key=lambda items: (
            stable_seed(seed, "component", "|".join(items)),
            items,
        ),
    )
    target = max(1, round(len(queries) * development_fraction))
    development: list[str] = []
    fit: list[str] = []
    for component in ordered:
        if len(development) < target:
            development.extend(component)
        else:
            fit.extend(component)
    if not fit or not development:
        raise ValueError("Cannot obtain both fit and development query sets")
    overlap = set(fit) & set(development)
    if overlap or set(fit) | set(development) != query_set:
        raise AssertionError("Invalid fit/development query partition")
    return sorted(fit), sorted(development), {
        "component_count": len(ordered),
        "largest_component_query_count": max(len(items) for items in ordered),
        "target_development_query_count": target,
        "actual_development_query_count": len(development),
        "actual_fit_query_count": len(fit),
        "shared_gold_component_policy": True,
    }


def candidate_features(
    text_ranking: list[tuple[str, float]],
    structure_ranking: list[tuple[str, float]],
) -> tuple[list[str], np.ndarray]:
    ids, text_z, structure_z, structure_order = prepare_query_scores(
        text_ranking,
        structure_ranking,
    )
    text_rank = np.arange(1, len(ids) + 1, dtype=np.float64)
    structure_position = {
        patent_id: rank
        for rank, patent_id in enumerate(structure_order, start=1)
    }
    structure_rank = np.asarray(
        [structure_position[patent_id] for patent_id in ids],
        dtype=np.float64,
    )
    features = np.column_stack(
        (
            text_z,
            structure_z,
            1.0 / np.log2(text_rank + 1.0),
            1.0 / np.log2(structure_rank + 1.0),
            text_z * structure_z,
        )
    )
    if features.shape != (len(ids), len(FEATURE_NAMES)):
        raise AssertionError("Invalid feature-matrix shape")
    if not np.isfinite(features).all():
        raise ValueError("Features contain nonfinite values")
    return ids, features


def select_negative_indices(
    ids: list[str],
    gold: set[str],
    text_ranking: list[tuple[str, float]],
    structure_ranking: list[tuple[str, float]],
    hard_count: int,
    random_count: int,
    seed: int,
) -> list[int]:
    id_to_index = {patent_id: index for index, patent_id in enumerate(ids)}
    hard_ids: list[str] = []
    seen: set[str] = set()
    for ranking in (text_ranking, structure_ranking):
        added = 0
        for patent_id, _ in ranking:
            if patent_id in gold or patent_id in seen:
                continue
            hard_ids.append(patent_id)
            seen.add(patent_id)
            added += 1
            if added >= hard_count:
                break
    remaining = [
        patent_id
        for patent_id in ids
        if patent_id not in gold and patent_id not in seen
    ]
    rng = np.random.default_rng(seed)
    if random_count and remaining:
        size = min(random_count, len(remaining))
        selected = rng.choice(len(remaining), size=size, replace=False)
        for index in sorted(int(item) for item in selected):
            hard_ids.append(remaining[index])
    return [id_to_index[patent_id] for patent_id in hard_ids]


def build_pairwise_training_data(
    queries: list[str],
    text_model: str,
    ground_truth: dict[str, dict[str, set[str]]],
    text_rankings: dict[str, dict[str, list[tuple[str, float]]]],
    structure_rankings: dict[str, dict[str, list[tuple[str, float]]]],
    structure_model: str,
    hard_count: int,
    random_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    positive_count = 0
    negative_candidate_count = 0
    for query_id in sorted(queries):
        ids, features = candidate_features(
            text_rankings[text_model][query_id],
            structure_rankings[structure_model][query_id],
        )
        id_to_index = {patent_id: index for index, patent_id in enumerate(ids)}
        gold = ground_truth[query_id][PRIMARY_TASK]
        missing_gold = gold - set(ids)
        if missing_gold:
            raise ValueError(
                f"Training-query gold absent from candidates: {query_id}:{sorted(missing_gold)}"
            )
        positive_indices = [id_to_index[item] for item in sorted(gold)]
        negative_indices = select_negative_indices(
            ids,
            gold,
            text_rankings[text_model][query_id],
            structure_rankings[structure_model][query_id],
            hard_count,
            random_count,
            stable_seed(seed, "negative", text_model, query_id),
        )
        if not positive_indices or not negative_indices:
            raise ValueError(f"Cannot construct training pairs: {query_id}")
        positive_count += len(positive_indices)
        negative_candidate_count += len(negative_indices)
        for positive_index in positive_indices:
            differences = features[positive_index] - features[negative_indices]
            feature_blocks.extend((differences, -differences))
            label_blocks.extend(
                (
                    np.ones(len(differences), dtype=np.int8),
                    np.zeros(len(differences), dtype=np.int8),
                )
            )
    return (
        np.vstack(feature_blocks).astype(np.float64, copy=False),
        np.concatenate(label_blocks),
        {
            "query_count": len(queries),
            "known_positive_count": positive_count,
            "selected_negative_candidate_count": negative_candidate_count,
            "symmetric_pair_rows": int(sum(len(block) for block in label_blocks)),
        },
    )


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    regularization_c: float,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    sklearn_seed = int(seed % (2**32))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = LogisticRegression(
        C=regularization_c,
        fit_intercept=False,
        solver="liblinear",
        max_iter=1000,
        random_state=sklearn_seed,
    )
    model.fit(scaled, labels)
    if int(model.n_iter_[0]) >= model.max_iter:
        raise RuntimeError("Pairwise logistic ranker did not converge")
    return scaler, model


def rank_candidates(
    ids: list[str],
    features: np.ndarray,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> list[tuple[str, float]]:
    scores = model.decision_function(scaler.transform(features))
    id_array = np.asarray(ids, dtype=str)
    order = np.lexsort((id_array, -scores))
    return [(ids[index], float(scores[index])) for index in order]


def choose_c(rows: list[dict[str, Any]]) -> float:
    best = max(float(row[SELECTION_METRIC]) for row in rows)
    tied = [
        float(row["regularization_c"])
        for row in rows
        if math.isclose(
            float(row[SELECTION_METRIC]),
            best,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    return min(tied)


def load_weighted_metrics(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    output = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["evaluation_split"] != "test":
                continue
            key = (row["text_source"], row["task"], row["query_patent_id"])
            if key in output:
                raise SystemExit(f"Duplicate method 2 per-query row: {key}")
            output[key] = row
    return output


def adjust_holm(rows: list[dict[str, Any]]) -> None:
    for baseline in sorted({row["baseline"] for row in rows}):
        for task in TASKS:
            for metric in METRICS:
                family = [
                    row
                    for row in rows
                    if row["baseline"] == baseline
                    and row["task"] == task
                    and row["metric"] == metric
                    and row["inference_allowed"]
                ]
                ordered = sorted(family, key=lambda row: row["sign_flip_p_value"])
                running = 0.0
                total = len(ordered)
                for index, row in enumerate(ordered):
                    adjusted = min(
                        1.0,
                        float(row["sign_flip_p_value"]) * (total - index),
                    )
                    running = max(running, adjusted)
                    row["holm_adjusted_p_value"] = running


def run_self_test() -> None:
    text = [("EP1A1", 3.0), ("EP2A1", 2.0), ("EP3A1", 1.0)]
    structure = [("EP3A1", 0.9), ("EP2A1", 0.8), ("EP1A1", 0.1)]
    ids, features = candidate_features(text, structure)
    assert ids == ["EP1A1", "EP2A1", "EP3A1"]
    assert features.shape == (3, len(FEATURE_NAMES))
    training = np.asarray([[1, 0, 1, 0, 0], [-1, 0, -1, 0, 0]], dtype=float)
    labels = np.asarray([1, 0])
    scaler, model = train_model(training, labels, 1.0, 7)
    ranking = rank_candidates(ids, features, scaler, model)
    assert len(ranking) == 3
    ground_truth = {
        "EP1A1": {PRIMARY_TASK: {"EP9A1"}},
        "EP2A1": {PRIMARY_TASK: {"EP9A1"}},
        "EP3A1": {PRIMARY_TASK: {"EP8A1"}},
    }
    fit, development, details = split_training_queries(
        sorted(ground_truth), ground_truth, 0.34, 7
    )
    assert set(fit).isdisjoint(development)
    assert not ({"EP1A1", "EP2A1"} & set(fit) and {"EP1A1", "EP2A1"} & set(development))
    assert details["component_count"] == 2
    assert parse_positive_grid("1,0.1,1") == [0.1, 1.0]
    print("Self-check passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--query-split", type=Path)
    parser.add_argument("--text-per-query", type=Path)
    parser.add_argument("--text-rankings", type=Path)
    parser.add_argument("--structure-per-query", type=Path)
    parser.add_argument("--structure-rankings", type=Path)
    parser.add_argument("--weighted-per-query", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--c-grid",
        type=parse_positive_grid,
        default=parse_positive_grid("0.01,0.1,1,10"),
    )
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--hard-negatives", type=int, default=50)
    parser.add_argument("--random-negatives", type=int, default=50)
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
        "--weighted-per-query": args.weighted_per_query,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")
    for name, path in required.items():
        if name != "--output-dir" and not path.is_file():
            raise SystemExit(f"Input file missing: {path}")
    if args.hard_negatives < 1 or args.random_negatives < 0:
        parser.error("Invalid negative-sample counts")
    if args.ranking_output_limit < 100 or args.iterations < 1000:
        parser.error("Ranking limit must be at least 100 and iterations at least 1000")

    started_at = time.perf_counter()
    ground_truth = load_ground_truth(args.ground_truth, args.conflict_field)
    query_split, observed_split_counts = load_query_splits(
        args.query_split,
        args.development_label,
        args.test_label,
    )
    text_per_query, text_models = load_per_query_index(args.text_per_query, "source")
    structure_per_query, structure_models = load_per_query_index(
        args.structure_per_query, "method"
    )
    if len(structure_models) != 1:
        raise SystemExit(f"Only one structure model is allowed: {sorted(structure_models)}")
    structure_model = next(iter(structure_models))
    text_rankings, text_ranking_models = load_score_rankings(args.text_rankings, "source")
    structure_rankings, structure_ranking_models = load_score_rankings(
        args.structure_rankings, "method"
    )
    if text_models != text_ranking_models or structure_models != structure_ranking_models:
        raise SystemExit("Per-query and ranking model sets differ")
    require_full_score_rankings(text_per_query, text_rankings, text_models, "T1")
    require_full_score_rankings(
        structure_per_query, structure_rankings, structure_models, "T2"
    )
    weighted_metrics = load_weighted_metrics(args.weighted_per_query)

    common_training_queries: set[str] | None = None
    for text_model in sorted(text_models):
        text_queries = set(text_per_query[text_model][PRIMARY_TASK])
        structure_queries = set(structure_per_query[structure_model][PRIMARY_TASK])
        if text_queries != structure_queries:
            raise SystemExit(f"T1/T2 queries differ: {text_model}")
        training_queries = {
            query_id
            for query_id in text_queries
            if query_split.get(query_id) == args.development_label
        }
        common_training_queries = (
            training_queries
            if common_training_queries is None
            else common_training_queries & training_queries
        )
    if not common_training_queries:
        raise SystemExit("No training queries")
    fit_queries, development_queries, internal_split = split_training_queries(
        sorted(common_training_queries),
        ground_truth,
        args.development_fraction,
        args.seed,
    )
    full_training_queries = sorted(common_training_queries)

    tuning_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    selected_models: dict[str, tuple[StandardScaler, LogisticRegression, float]] = {}
    training_details: dict[str, Any] = {}
    for text_model in sorted(text_models):
        fit_x, fit_y, fit_details = build_pairwise_training_data(
            fit_queries,
            text_model,
            ground_truth,
            text_rankings,
            structure_rankings,
            structure_model,
            args.hard_negatives,
            args.random_negatives,
            args.seed,
        )
        source_rows = []
        for regularization_c in args.c_grid:
            scaler, model = train_model(
                fit_x,
                fit_y,
                regularization_c,
                stable_seed(args.seed, "fit", text_model, regularization_c),
            )
            metric_values = {metric: [] for metric in METRICS}
            for query_id in development_queries:
                ids, features = candidate_features(
                    text_rankings[text_model][query_id],
                    structure_rankings[structure_model][query_id],
                )
                ranking = rank_candidates(ids, features, scaler, model)
                metrics = evaluate_ranking(
                    ranking, ground_truth[query_id][PRIMARY_TASK]
                )
                for metric in METRICS:
                    metric_values[metric].append(metrics[metric])
            row = {
                "text_source": text_model,
                "structure_method": structure_model,
                "fit_query_count": len(fit_queries),
                "development_query_count": len(development_queries),
                "regularization_c": regularization_c,
                **{
                    metric: float(np.mean(metric_values[metric]))
                    for metric in METRICS
                },
            }
            source_rows.append(row)
            tuning_rows.append(row)
        selected_c = choose_c(source_rows)
        for row in source_rows:
            row["selected"] = float(row["regularization_c"]) == selected_c

        full_x, full_y, full_details = build_pairwise_training_data(
            full_training_queries,
            text_model,
            ground_truth,
            text_rankings,
            structure_rankings,
            structure_model,
            args.hard_negatives,
            args.random_negatives,
            args.seed,
        )
        scaler, model = train_model(
            full_x,
            full_y,
            selected_c,
            stable_seed(args.seed, "refit", text_model, selected_c),
        )
        selected_models[text_model] = (scaler, model, selected_c)
        training_details[text_model] = {
            "fit": fit_details,
            "refit_full_train": full_details,
        }
        for feature_name, mean, scale, coefficient in zip(
            FEATURE_NAMES,
            scaler.mean_,
            scaler.scale_,
            model.coef_[0],
        ):
            model_rows.append(
                {
                    "text_source": text_model,
                    "structure_method": structure_model,
                    "selected_c": selected_c,
                    "feature": feature_name,
                    "scaler_mean": float(mean),
                    "scaler_scale": float(scale),
                    "coefficient": float(coefficient),
                    "solver_iterations": int(model.n_iter_[0]),
                }
            )

    result_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    comparison_inputs: dict[
        tuple[str, str, str, str], tuple[list[float], list[float]]
    ] = {}
    ranking_written: set[tuple[str, str]] = set()
    for text_model in sorted(text_models):
        scaler, model, selected_c = selected_models[text_model]
        method = f"pairwise_logistic_c{selected_c:g}__{text_model}__{structure_model}"
        for task in TASKS:
            queries = sorted(
                query_id
                for query_id in text_per_query[text_model].get(task, {})
                if query_split.get(query_id) == args.test_label
            )
            learned_values = {metric: [] for metric in METRICS}
            text_values = {metric: [] for metric in METRICS}
            weighted_values = {metric: [] for metric in METRICS}
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
                    raise SystemExit(
                        f"Gold count differs from input: {task}:{query_id}"
                    )
                if (
                    text_meta["candidate_count_ranked"]
                    != structure_meta["candidate_count_ranked"]
                ):
                    raise SystemExit(
                        f"T1/T2 candidate counts differ: {task}:{query_id}"
                    )
                ids, features = candidate_features(
                    text_rankings[text_model][query_id],
                    structure_rankings[structure_model][query_id],
                )
                if len(ids) != text_meta["candidate_count_ranked"]:
                    raise SystemExit(
                        f"Ranking candidate count differs from per-query input: "
                        f"{task}:{query_id}"
                    )
                learned = rank_candidates(ids, features, scaler, model)
                text_original = text_rankings[text_model][query_id]
                learned_metrics = evaluate_ranking(learned, gold)
                text_metrics = evaluate_ranking(text_original, gold)
                weighted_key = (text_model, task, query_id)
                if weighted_key not in weighted_metrics:
                    raise SystemExit(f"Method 2 test row missing: {weighted_key}")
                weighted_row = weighted_metrics[weighted_key]
                if (
                    int(weighted_row["gold_count"]) != len(gold)
                    or int(weighted_row["candidate_count_ranked"]) != len(learned)
                ):
                    raise SystemExit(
                        f"Method 2 and learned fusion use different evaluation units: "
                        f"{weighted_key}"
                    )
                for metric in METRICS:
                    learned_values[metric].append(learned_metrics[metric])
                    text_values[metric].append(text_metrics[metric])
                    weighted_values[metric].append(float(weighted_row[metric]))
                per_query_rows.append(
                    {
                        "method": method,
                        "text_source": text_model,
                        "structure_method": structure_model,
                        "selected_c": selected_c,
                        "evaluation_split": "test",
                        "query_patent_id": query_id,
                        "task": task,
                        "gold_count": len(gold),
                        "candidate_count_ranked": len(learned),
                        "ranking_rows_written": min(
                            args.ranking_output_limit, len(learned)
                        ),
                        "text_best_gold_rank": best_gold_rank(text_original, gold) or "",
                        "learned_best_gold_rank": best_gold_rank(learned, gold) or "",
                        **learned_metrics,
                    }
                )
                ranking_key = (method, query_id)
                if ranking_key not in ranking_written:
                    ranking_rows.append(
                        {
                            "method": method,
                            "text_source": text_model,
                            "structure_method": structure_model,
                            "selected_c": selected_c,
                            "evaluation_split": "test",
                            "query_patent_id": query_id,
                            "ranking": [
                                {
                                    "rank": rank,
                                    "patent_id": patent_id,
                                    "score": round(score, 12),
                                }
                                for rank, (patent_id, score) in enumerate(
                                    learned[: args.ranking_output_limit], start=1
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
                    "selected_c": selected_c,
                    "evaluation_split": "test",
                    "task": task,
                    "Queries": len(queries),
                    **{
                        metric: float(np.mean(learned_values[metric]))
                        if queries
                        else ""
                        for metric in METRICS
                    },
                }
            )
            for metric in METRICS:
                comparison_inputs[(text_model, task, metric, "T1_text")] = (
                    text_values[metric],
                    learned_values[metric],
                )
                comparison_inputs[(text_model, task, metric, "method2_weighted")] = (
                    weighted_values[metric],
                    learned_values[metric],
                )

    comparison_rows: list[dict[str, Any]] = []
    for (text_model, task, metric, baseline), (
        baseline_values,
        learned_values,
    ) in sorted(comparison_inputs.items()):
        baseline_array = np.asarray(baseline_values, dtype=np.float64)
        learned_array = np.asarray(learned_values, dtype=np.float64)
        inference_allowed = len(learned_array) >= MINIMUM_INFERENCE_QUERIES
        row: dict[str, Any] = {
            "text_source": text_model,
            "task": task,
            "metric": metric,
            "baseline": baseline,
            "query_count": len(learned_array),
            "baseline_mean": float(baseline_array.mean()),
            "learned_mean": float(learned_array.mean()),
            "mean_difference_learned_minus_baseline": float(
                (learned_array - baseline_array).mean()
            ),
            "inference_allowed": inference_allowed,
            "bootstrap_ci_low": "",
            "bootstrap_ci_high": "",
            "sign_flip_p_value": "",
            "holm_adjusted_p_value": "",
            "resampling_seed": "",
            "learned_wins": int(
                np.count_nonzero(learned_array > baseline_array + 1e-15)
            ),
            "ties": int(
                np.count_nonzero(np.abs(learned_array - baseline_array) <= 1e-15)
            ),
            "learned_losses": int(
                np.count_nonzero(learned_array < baseline_array - 1e-15)
            ),
        }
        if inference_allowed:
            sub_seed = stable_seed(
                args.seed, "inference", text_model, task, metric, baseline
            )
            row.update(
                paired_summary(
                    baseline_array,
                    learned_array,
                    args.iterations,
                    np.random.default_rng(sub_seed),
                )
            )
            row["resampling_seed"] = sub_seed
            row["holm_adjusted_p_value"] = ""
        comparison_rows.append(row)
    adjust_holm(comparison_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = args.output_dir / "t3_learned_c_tuning.csv"
    model_path = args.output_dir / "t3_learned_model_parameters.csv"
    result_path = args.output_dir / "t3_learned_results.csv"
    per_query_path = args.output_dir / "t3_learned_per_query.csv"
    comparison_path = args.output_dir / "t3_learned_baseline_comparison.csv"
    ranking_path = args.output_dir / f"t3_learned_rankings_top{args.ranking_output_limit}.jsonl"
    manifest_path = args.output_dir / "t3_learned_manifest.json"
    write_csv_atomic(
        tuning_path,
        [
            "text_source",
            "structure_method",
            "fit_query_count",
            "development_query_count",
            "regularization_c",
            "selected",
            *METRICS,
        ],
        tuning_rows,
    )
    write_csv_atomic(
        model_path,
        [
            "text_source",
            "structure_method",
            "selected_c",
            "feature",
            "scaler_mean",
            "scaler_scale",
            "coefficient",
            "solver_iterations",
        ],
        model_rows,
    )
    write_csv_atomic(
        result_path,
        [
            "method",
            "text_source",
            "structure_method",
            "selected_c",
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
            "selected_c",
            "evaluation_split",
            "query_patent_id",
            "task",
            "gold_count",
            "candidate_count_ranked",
            "ranking_rows_written",
            "text_best_gold_rank",
            "learned_best_gold_rank",
            *METRICS,
        ],
        per_query_rows,
    )
    write_csv_atomic(
        comparison_path,
        [
            "text_source",
            "task",
            "metric",
            "baseline",
            "query_count",
            "baseline_mean",
            "learned_mean",
            "mean_difference_learned_minus_baseline",
            "inference_allowed",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "sign_flip_p_value",
            "holm_adjusted_p_value",
            "resampling_seed",
            "learned_wins",
            "ties",
            "learned_losses",
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
        args.weighted_per_query,
    )
    manifest = {
        "status": "completed",
        "method": "pairwise_logistic_learned_fusion",
        "interpretation": "exploratory_supplementary",
        "feature_names": FEATURE_NAMES,
        "feature_policy": (
            "Per-query T1/T2 z-scores, reciprocal log ranks, and z-score interaction"
        ),
        "label_policy": (
            "Only known Citation gold is positive; non-gold may contain unverified relevant patents, "
            "so it is not guaranteed to be truly negative"
        ),
        "negative_sampling": {
            "hard_per_modality": args.hard_negatives,
            "random_per_query": args.random_negatives,
            "symmetric_pair_rows": True,
        },
        "outer_split": {
            "development_source_label": args.development_label,
            "test_label": args.test_label,
            "observed_query_counts": observed_split_counts,
            "test_labels_not_used_for_training_or_selection": True,
            "actual_family_id_split": False,
        },
        "internal_training_split": {
            "development_fraction": args.development_fraction,
            **internal_split,
            "warning": "Split by shared-gold connected components, not actual family_id",
        },
        "regularization_selection": {
            "grid": args.c_grid,
            "task": PRIMARY_TASK,
            "metric": SELECTION_METRIC,
            "tie_policy": "Choose the smallest C for ties within 1e-12",
            "selected_by_text_source": {
                source: selected_models[source][2] for source in sorted(selected_models)
            },
            "refit_on_all_outer_train_queries": True,
        },
        "training_details": training_details,
        "candidate_policy": "Use T1/T2 scores for the same complete 2,217-candidate pool",
        "primary_task": PRIMARY_TASK,
        "secondary_task": "Conflict-X",
        "minimum_inference_queries": MINIMUM_INFERENCE_QUERIES,
        "small_sample_policy": (
            "With fewer than 30 test queries, report only means and win/tie/loss"
        ),
        "text_sources": sorted(text_models),
        "structure_method": structure_model,
        "ranking_output_limit": args.ranking_output_limit,
        "paired_inference": {
            "evaluation_split": "test only",
            "unit": "query",
            "baselines": ["T1_text", "method2_weighted"],
            "bootstrap_iterations": args.iterations,
            "sign_flip_iterations": args.iterations,
            "base_seed": args.seed,
            "seed_policy": (
                "Derive fixed sub-seeds using SHA-256 of base seed and text source/task/metric/baseline "
                "identifiers"
            ),
            "multiple_comparison_policy": (
                "Holm correction across text-source comparisons per baseline/task/metric"
            ),
        },
        "ground_truth_field_mapping": {
            "Citation": "all_cited_patents",
            "Conflict-X": args.conflict_field,
        },
        "limitations": [
            "Not an actual family_id-based split",
            "Non-gold is not guaranteed to be truly negative",
            "Evaluation on an oracle-derived closed candidate pool",
        ],
        "software": {
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
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
    for path in (
        tuning_path,
        model_path,
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
