#!/usr/bin/env python3
"""Recompute M3 fold tuning and Top-N reranking independently."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from validate_t123_5fold_m2_weighted import (
    ALPHAS,
    METRICS,
    SOURCES,
    TASKS,
    compare,
    csv_rows,
    jsonl_rows,
    metrics,
    parse_rankings,
    sha256,
)


TOP_N_VALUES = (100, 500)
DEFAULT_DIR = Path("structure_retrieval_outputs/t123_5fold_v1/m3_text_first_reranking")
MANIFEST = "t123_m3_reranking_manifest.json"
TUNING = "t123_m3_alpha_tuning.csv"
CHOICES = "t123_m3_alpha_choices.csv"
PER_QUERY = "t123_m3_reranking_oof_per_query.csv"
SUMMARY = "t123_m3_reranking_oof_summary.csv"
FOLDS = "t123_m3_reranking_oof_folds.csv"
RANKINGS = "t123_m3_reranking_rankings_top100.jsonl"


def normalized_or_zero(values: np.ndarray) -> tuple[np.ndarray, bool]:
    average = float(np.mean(values))
    deviation = float(np.std(values, ddof=0))
    if not math.isfinite(average) or not math.isfinite(deviation):
        raise ValueError("nonfinite Top-N score statistics")
    if deviation <= 1e-15:
        return np.zeros(len(values), dtype=np.float64), True
    return (values - average) / deviation, False


def prepare(text: tuple, structure: tuple, top_n: int):
    text_ids, text_scores = text
    structure_ids, structure_scores = structure
    if len(text_ids) != 2217 or set(text_ids) != set(structure_ids):
        raise ValueError("full candidate IDs differ")
    structure_by_id = dict(zip(structure_ids, structure_scores))
    head = text_ids[:top_n]
    text_z, text_zero = normalized_or_zero(text_scores[:top_n])
    structure_z, structure_zero = normalized_or_zero(np.asarray(
        [structure_by_id[patent] for patent in head], dtype=np.float64))
    return text_ids, head, text_z, structure_z, text_zero, structure_zero


def rerank(prepared: tuple, alpha: float) -> tuple[list[str], dict[str, float]]:
    full_ids, head, text_z, structure_z, _, _ = prepared
    weighted = alpha * text_z + (1 - alpha) * structure_z
    scores = dict(zip(head, (float(value) for value in weighted)))
    if alpha == 1.0:
        head_order = head
    else:
        original_rank = {patent: index for index, patent in enumerate(head)}
        head_order = sorted(head, key=lambda patent: (-scores[patent], original_rank[patent], patent))
    return head_order + full_ids[len(head):], scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    base = args.analysis_dir
    manifest = json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed"
            or manifest.get("top_n_values") != list(TOP_N_VALUES)
            or manifest.get("alpha_grid") != list(ALPHAS)
            or set(manifest.get("text_sources", [])) != set(SOURCES)
            or manifest.get("structure_method") != "mean_morgan_cosine"):
        raise ValueError("M3 manifest settings differ")
    for label, record in manifest["input_files"].items():
        path = Path(record["path"])
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"input hash/size mismatch: {label}")
    for filename in (TUNING, CHOICES, PER_QUERY, SUMMARY, FOLDS, RANKINGS):
        path = base / filename
        record = manifest["output_files"][filename]
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"output hash/size mismatch: {filename}")

    assignment = {}
    groups = {}
    for row in csv_rows(Path(manifest["input_files"]["assignment"]["path"])):
        query, group, fold = row["query_id"], row["group_id"], int(row["outer_fold"])
        if query in assignment or fold not in range(5):
            raise ValueError(f"invalid fold assignment: {query}")
        if group in groups and groups[group] != fold:
            raise ValueError(f"split connected group: {group}")
        assignment[query] = (group, fold, row["conflict_x_eligible"] == "1")
        groups[group] = fold
    if len(assignment) != 1126 or len(groups) != 908:
        raise ValueError("assignment counts differ")
    truth = {}
    for row in jsonl_rows(Path(manifest["input_files"]["ground_truth"]["path"])):
        query = row["query_patent_id"]
        if query in truth:
            raise ValueError(f"duplicate ground truth: {query}")
        truth[query] = {"Citation": set(row["all_cited_patents"]),
                        "Conflict-X": set(row["conflict_patents"])}

    required = set(assignment)
    text, text_extras = parse_rankings(
        Path(manifest["input_files"]["text_rankings"]["path"]), "source", set(SOURCES), required)
    structure, structure_extras = parse_rankings(
        Path(manifest["input_files"]["structure_rankings"]["path"]), "method",
        {"mean_morgan_cosine"}, required)
    if text_extras != 0 or structure_extras != 73:
        raise ValueError("extra source ranking count differs")

    prepared = {(source, top_n, query): prepare(
                    text[source][query], structure["mean_morgan_cosine"][query], top_n)
                for source in SOURCES for top_n in TOP_N_VALUES for query in sorted(required)}
    development_mrr = {}
    for (source, top_n, query), values in prepared.items():
        gold = truth[query]["Citation"]
        if not gold or not gold.issubset(values[0]):
            raise ValueError(f"Citation gold outside candidates: {source}:{query}")
        for alpha in ALPHAS:
            ids, _ = rerank(values, alpha)
            development_mrr[(source, top_n, query, alpha)] = metrics(ids, gold)["MRR@100"]

    tuning = {}
    for row in csv_rows(base / TUNING):
        key = (row["text_source"], int(row["top_n"]), int(row["outer_fold"]), float(row["alpha"]))
        if (key in tuning or key[0] not in SOURCES or key[1] not in TOP_N_VALUES
                or key[2] not in range(5) or key[3] not in ALPHAS):
            raise ValueError(f"invalid/duplicate tuning row: {key}")
        tuning[key] = row
    choices = {}
    for row in csv_rows(base / CHOICES):
        key = (row["text_source"], int(row["top_n"]), int(row["outer_fold"]))
        if key in choices or key[0] not in SOURCES or key[1] not in TOP_N_VALUES or key[2] not in range(5):
            raise ValueError(f"invalid/duplicate alpha choice: {key}")
        choices[key] = row
    if len(tuning) != 220 or len(choices) != 20:
        raise ValueError("tuning/choice row counts differ")

    selected = {}
    max_metric_difference = 0.0
    for source in SOURCES:
        for top_n in TOP_N_VALUES:
            for fold in range(5):
                train = [query for query in sorted(required) if assignment[query][1] != fold]
                test = [query for query in sorted(required) if assignment[query][1] == fold]
                curve = {}
                for alpha in ALPHAS:
                    mean = math.fsum(development_mrr[(source, top_n, query, alpha)]
                                     for query in train) / len(train)
                    curve[alpha] = mean
                    item = tuning[(source, top_n, fold, alpha)]
                    if (int(item["training_query_count"]) != len(train)
                            or int(item["test_query_count"]) != len(test)):
                        raise ValueError(f"tuning cohort differs: {source}:{top_n}:{fold}:{alpha}")
                    max_metric_difference = max(max_metric_difference,
                        compare(item["training_Citation_MRR@100"], mean,
                                f"tuning:{source}:{top_n}:{fold}:{alpha}"))
                best = max(curve.values())
                tied = [alpha for alpha, value in curve.items() if abs(value - best) <= 1e-12]
                chosen = min(tied, key=lambda alpha: (abs(alpha - 0.5), -alpha))
                choice = choices[(source, top_n, fold)]
                if (float(choice["selected_alpha"]) != chosen
                        or int(choice["training_query_count"]) != len(train)
                        or int(choice["test_query_count"]) != len(test)):
                    raise ValueError(f"selected alpha/cohort differs: {source}:{top_n}:{fold}")
                max_metric_difference = max(max_metric_difference,
                    compare(choice["selected_training_Citation_MRR@100"], curve[chosen],
                            f"choice:{source}:{top_n}:{fold}"))
                zero_counts = {
                    "train_text_zero_variance_queries": sum(prepared[(source, top_n, query)][4] for query in train),
                    "train_structure_zero_variance_queries": sum(prepared[(source, top_n, query)][5] for query in train),
                    "test_text_zero_variance_queries": sum(prepared[(source, top_n, query)][4] for query in test),
                    "test_structure_zero_variance_queries": sum(prepared[(source, top_n, query)][5] for query in test),
                }
                for label, value in zero_counts.items():
                    if int(choice[label]) != value:
                        raise ValueError(f"zero-variance count differs: {source}:{top_n}:{fold}:{label}")
                for alpha in ALPHAS:
                    if (tuning[(source, top_n, fold, alpha)]["selected"] == "True") != (alpha == chosen):
                        raise ValueError(f"tuning selected flag differs: {source}:{top_n}:{fold}:{alpha}")
                selected[(source, top_n, fold)] = chosen

    stored_rankings = {}
    for row in jsonl_rows(base / RANKINGS):
        key = (row["text_source"], int(row["top_n"]), row["query_patent_id"])
        if (key in stored_rankings or row["method"] !=
                f"text_top{key[1]}_weighted__{key[0]}__mean_morgan_cosine"):
            raise ValueError(f"invalid/duplicate stored ranking: {key}")
        stored_rankings[key] = row
    stored_per_query = {}
    for row in csv_rows(base / PER_QUERY):
        key = (row["text_source"], int(row["top_n"]), row["task"], row["query_patent_id"])
        if (key in stored_per_query or row["method"] !=
                f"text_top{key[1]}_weighted__{key[0]}__mean_morgan_cosine"):
            raise ValueError(f"invalid/duplicate per-query row: {key}")
        stored_per_query[key] = row
    if len(stored_rankings) != 4504 or len(stored_per_query) != 4692:
        raise ValueError("stored ranking/per-query counts differ")

    aggregate = defaultdict(list)
    checked = set()
    max_score_difference = 0.0
    for source in SOURCES:
        for top_n in TOP_N_VALUES:
            for query in sorted(required):
                group, fold, conflict_eligible = assignment[query]
                alpha = selected[(source, top_n, fold)]
                order, scores = rerank(prepared[(source, top_n, query)], alpha)
                saved = stored_rankings[(source, top_n, query)]
                if (int(saved["outer_fold"]) != fold or float(saved["alpha"]) != alpha
                        or len(saved["ranking"]) != 100):
                    raise ValueError(f"stored ranking metadata differs: {source}:{top_n}:{query}")
                for rank, (patent, item) in enumerate(zip(order[:100], saved["ranking"]), 1):
                    if item["rank"] != rank or item["patent_id"] != patent:
                        raise ValueError(f"Top-100 order differs: {source}:{top_n}:{query}:{rank}")
                    max_score_difference = max(max_score_difference,
                        compare(item["score"], scores[patent],
                                f"score:{source}:{top_n}:{query}:{rank}"))
                for task in TASKS:
                    if task == "Conflict-X" and not conflict_eligible:
                        continue
                    key = (source, top_n, task, query)
                    row = stored_per_query[key]
                    gold = truth[query][task]
                    if (not gold or not gold.issubset(order)
                            or row["group_id"] != group or int(row["outer_fold"]) != fold
                            or float(row["alpha"]) != alpha
                            or int(row["gold_count"]) != len(gold)
                            or int(row["candidate_count_ranked"]) != 2217):
                        raise ValueError(f"per-query metadata differs: {key}")
                    result = metrics(order, gold)
                    for metric in METRICS:
                        max_metric_difference = max(max_metric_difference,
                            compare(row[metric], result[metric], f"per-query:{key}:{metric}"))
                    aggregate[(source, top_n, task)].append((group, fold, result))
                    checked.add(key)
    if checked != set(stored_per_query):
        raise ValueError("per-query coverage differs")

    for filename, by_fold, expected_count in ((SUMMARY, False, 8), (FOLDS, True, 40)):
        table = csv_rows(base / filename)
        if len(table) != expected_count:
            raise ValueError(f"aggregate row count differs: {filename}")
        seen = set()
        for row in table:
            source, task = row["text_source"], row["task"]
            top_n = int(row["method"].split("_", 2)[1][3:])
            fold = int(row["outer_fold"]) if by_fold else None
            key = (source, top_n, task, fold)
            if (source not in SOURCES or top_n not in TOP_N_VALUES or task not in TASKS
                    or key in seen or row["method"] !=
                    f"text_top{top_n}_weighted__{source}__mean_morgan_cosine"):
                raise ValueError(f"aggregate key differs: {filename}:{key}")
            seen.add(key)
            members = [member for member in aggregate[(source, top_n, task)]
                       if not by_fold or member[1] == fold]
            if (int(row["query_count"]) != len(members)
                    or int(row["group_count"]) != len({member[0] for member in members})
                    or int(row["candidate_count_ranked"]) != 2217):
                raise ValueError(f"aggregate counts differ: {filename}:{key}")
            for metric in METRICS:
                mean = math.fsum(member[2][metric] for member in members) / len(members)
                max_metric_difference = max(max_metric_difference,
                    compare(row[metric], mean, f"aggregate:{filename}:{key}:{metric}"))
        if len(seen) != expected_count:
            raise ValueError(f"aggregate coverage differs: {filename}")

    print("independent_m3_5fold_validation: PASS")
    print("alpha_choices_checked:", len(choices))
    print("tuning_rows_checked:", len(tuning))
    print("ranking_rows_checked:", len(stored_rankings))
    print("per_query_rows_checked:", len(stored_per_query))
    print("maximum_metric_abs_difference:", max_metric_difference)
    print("maximum_score_abs_difference:", max_score_difference)
    print("input_and_output_hashes: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
