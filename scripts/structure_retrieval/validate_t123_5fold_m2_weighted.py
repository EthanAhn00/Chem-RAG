#!/usr/bin/env python3
"""Independently reconstruct M2 fold tuning, rankings, and metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("R@1", "R@5", "R@10", "R@50", "R@100", "MRR@100", "MAP@100", "nDCG@10")
SOURCES = ("nv_document", "qwen_whole_claims")
TASKS = ("Citation", "Conflict-X")
ALPHAS = tuple(round(index / 10, 1) for index in range(11))
DEFAULT_DIR = Path("structure_retrieval_outputs/t123_5fold_v1/m2_weighted")
MANIFEST = "t123_m2_weighted_manifest.json"
TUNING = "t123_m2_alpha_tuning.csv"
CHOICES = "t123_m2_alpha_choices.csv"
PER_QUERY = "t123_m2_weighted_oof_per_query.csv"
SUMMARY = "t123_m2_weighted_oof_summary.csv"
FOLDS = "t123_m2_weighted_oof_folds.csv"
RANKINGS = "t123_m2_weighted_rankings_top100.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compare(actual: object, expected: float, label: str, tolerance: float = 1e-12) -> float:
    value = float(actual)
    difference = abs(value - expected)
    if not math.isfinite(value) or difference > tolerance:
        raise ValueError(f"numeric mismatch: {label}: {value} != {expected}")
    return difference


def metrics(ids: list[str], gold: set[str]) -> dict[str, float]:
    hits = [int(patent in gold) for patent in ids[:100]]
    result = {f"R@{k}": sum(hits[:k]) / len(gold) for k in (1, 5, 10, 50, 100)}
    result["MRR@100"] = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    found = 0
    precision_total = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            found += 1
            precision_total += found / rank
    result["MAP@100"] = precision_total / len(gold)
    dcg = math.fsum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:10], 1))
    idcg = math.fsum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
    result["nDCG@10"] = dcg / idcg
    return result


def parse_rankings(path: Path, field: str, expected_models: set[str], required: set[str]):
    output = defaultdict(dict)
    extra_count = 0
    for row in jsonl_rows(path):
        model, query = row[field], row["query_patent_id"]
        if model not in expected_models:
            raise ValueError(f"unexpected model: {model}")
        if query not in required:
            extra_count += 1
            continue
        if query in output[model]:
            raise ValueError(f"duplicate ranking: {model}:{query}")
        items = row["ranking"]
        if len(items) != 2217:
            raise ValueError(f"ranking length differs: {model}:{query}")
        ids = []
        values = []
        for rank, item in enumerate(items, 1):
            if item["rank"] != rank or item["patent_id"] == query:
                raise ValueError(f"ranking position/ID differs: {model}:{query}:{rank}")
            score = float(item["score"])
            if not math.isfinite(score):
                raise ValueError(f"nonfinite source score: {model}:{query}:{rank}")
            ids.append(item["patent_id"])
            values.append(score)
        if len(set(ids)) != 2217:
            raise ValueError(f"duplicate candidates: {model}:{query}")
        output[model][query] = (ids, np.asarray(values, dtype=np.float64))
    if any(set(output[model]) != required for model in expected_models):
        raise ValueError(f"ranking query coverage differs: {path}")
    return output, extra_count


def prepare(text: tuple, structure: tuple):
    ids, raw_text = text
    structure_ids, raw_structure = structure
    if set(ids) != set(structure_ids):
        raise ValueError("T1/T2 candidate IDs differ")
    structure_values = dict(zip(structure_ids, raw_structure))
    aligned_structure = np.asarray([structure_values[patent] for patent in ids], dtype=np.float64)
    text_std = float(np.std(raw_text, ddof=0))
    structure_std = float(np.std(aligned_structure, ddof=0))
    if not math.isfinite(text_std) or not math.isfinite(structure_std) or min(text_std, structure_std) <= 1e-15:
        raise ValueError("zero/nonfinite score variance")
    text_z = (raw_text - float(np.mean(raw_text))) / text_std
    structure_z = (aligned_structure - float(np.mean(aligned_structure))) / structure_std
    return ids, structure_ids, text_z, structure_z


def fused(prepared: tuple, alpha: float) -> tuple[list[str], dict[str, float]]:
    ids, structure_ids, text_z, structure_z = prepared
    combined = alpha * text_z + (1 - alpha) * structure_z
    scores = dict(zip(ids, (float(value) for value in combined)))
    if alpha == 1.0:
        order = ids
    elif alpha == 0.0:
        order = structure_ids
    else:
        order = sorted(ids, key=lambda patent: (-scores[patent], patent))
    return order, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    base = args.analysis_dir
    manifest = json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("alpha_grid") != list(ALPHAS)
            or set(manifest.get("text_sources", [])) != set(SOURCES)
            or manifest.get("structure_method") != "mean_morgan_cosine"):
        raise ValueError("M2 manifest settings differ")
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
            raise ValueError(f"invalid assignment: {query}")
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
        raise ValueError("ranking extra-query count differs")
    prepared = {(source, query): prepare(text[source][query], structure["mean_morgan_cosine"][query])
                for source in SOURCES for query in sorted(required)}

    development_mrr = {}
    for (source, query), values in prepared.items():
        gold = truth[query]["Citation"]
        if not gold or not gold.issubset(values[0]):
            raise ValueError(f"Citation gold outside candidates: {source}:{query}")
        for alpha in ALPHAS:
            ids, _ = fused(values, alpha)
            development_mrr[(source, query, alpha)] = metrics(ids, gold)["MRR@100"]

    tuning = {}
    for row in csv_rows(base / TUNING):
        key = (row["text_source"], int(row["outer_fold"]), float(row["alpha"]))
        if key in tuning or key[0] not in SOURCES or key[1] not in range(5) or key[2] not in ALPHAS:
            raise ValueError(f"invalid/duplicate tuning row: {key}")
        tuning[key] = row
    if len(tuning) != 110:
        raise ValueError("tuning row count differs")
    choices = {}
    for row in csv_rows(base / CHOICES):
        key = (row["text_source"], int(row["outer_fold"]))
        if key in choices or key[0] not in SOURCES or key[1] not in range(5):
            raise ValueError(f"invalid/duplicate choice row: {key}")
        choices[key] = row
    if len(choices) != 10:
        raise ValueError("choice row count differs")
    max_difference = 0.0
    selected = {}
    for source in SOURCES:
        for fold in range(5):
            train = [query for query in sorted(required) if assignment[query][1] != fold]
            test = [query for query in sorted(required) if assignment[query][1] == fold]
            curve = {}
            for alpha in ALPHAS:
                mean = math.fsum(development_mrr[(source, query, alpha)] for query in train) / len(train)
                curve[alpha] = mean
                row = tuning[(source, fold, alpha)]
                if int(row["training_query_count"]) != len(train) or int(row["test_query_count"]) != len(test):
                    raise ValueError(f"tuning split count differs: {source}:{fold}:{alpha}")
                max_difference = max(max_difference,
                    compare(row["training_Citation_MRR@100"], mean, f"tuning:{source}:{fold}:{alpha}"))
            best = max(curve.values())
            tied = [alpha for alpha, value in curve.items() if abs(value - best) <= 1e-12]
            chosen = min(tied, key=lambda alpha: (abs(alpha - 0.5), -alpha))
            choice = choices[(source, fold)]
            if (float(choice["selected_alpha"]) != chosen
                    or int(choice["training_query_count"]) != len(train)
                    or int(choice["test_query_count"]) != len(test)):
                raise ValueError(f"fold alpha choice differs: {source}:{fold}")
            max_difference = max(max_difference,
                compare(choice["selected_training_Citation_MRR@100"], curve[chosen],
                        f"choice:{source}:{fold}"))
            for alpha in ALPHAS:
                if (tuning[(source, fold, alpha)]["selected"] == "True") != (alpha == chosen):
                    raise ValueError(f"selected tuning flag differs: {source}:{fold}:{alpha}")
            selected[(source, fold)] = chosen

    stored_rankings = {}
    for row in jsonl_rows(base / RANKINGS):
        key = (row["text_source"], row["query_patent_id"])
        if key in stored_rankings or row["method"] != f"zscore_weighted__{key[0]}__mean_morgan_cosine":
            raise ValueError(f"invalid/duplicate fused ranking: {key}")
        stored_rankings[key] = row
    if len(stored_rankings) != 2252:
        raise ValueError("fused ranking count differs")
    stored_per_query = {}
    for row in csv_rows(base / PER_QUERY):
        key = (row["text_source"], row["task"], row["query_patent_id"])
        if key in stored_per_query or row["method"] != f"zscore_weighted__{key[0]}__mean_morgan_cosine":
            raise ValueError(f"invalid/duplicate per-query row: {key}")
        stored_per_query[key] = row
    if len(stored_per_query) != 2346:
        raise ValueError("per-query row count differs")

    aggregate = defaultdict(list)
    checked = set()
    max_score_difference = 0.0
    for source in SOURCES:
        for query in sorted(required):
            group, fold, conflict_eligible = assignment[query]
            alpha = selected[(source, fold)]
            order, scores = fused(prepared[(source, query)], alpha)
            saved = stored_rankings[(source, query)]
            if (int(saved["outer_fold"]) != fold or float(saved["alpha"]) != alpha
                    or len(saved["ranking"]) != 100):
                raise ValueError(f"ranking metadata differs: {source}:{query}")
            for rank, (patent, item) in enumerate(zip(order[:100], saved["ranking"]), 1):
                if item["rank"] != rank or item["patent_id"] != patent:
                    raise ValueError(f"Top-100 ranking differs: {source}:{query}:{rank}")
                max_score_difference = max(max_score_difference,
                    compare(item["score"], scores[patent], f"score:{source}:{query}:{rank}"))
            for task in TASKS:
                if task == "Conflict-X" and not conflict_eligible:
                    continue
                key = (source, task, query)
                row = stored_per_query[key]
                gold = truth[query][task]
                if (not gold or not gold.issubset(scores)
                        or row["group_id"] != group or int(row["outer_fold"]) != fold
                        or float(row["alpha"]) != alpha
                        or int(row["gold_count"]) != len(gold)
                        or int(row["candidate_count_ranked"]) != 2217):
                    raise ValueError(f"per-query metadata differs: {key}")
                result = metrics(order, gold)
                for name in METRICS:
                    max_difference = max(max_difference,
                        compare(row[name], result[name], f"per-query:{key}:{name}"))
                aggregate[(source, task)].append((group, fold, result))
                checked.add(key)
    if checked != set(stored_per_query):
        raise ValueError("per-query coverage differs")

    for filename, by_fold, expected_count in ((SUMMARY, False, 4), (FOLDS, True, 20)):
        table = csv_rows(base / filename)
        if len(table) != expected_count:
            raise ValueError(f"aggregate row count differs: {filename}")
        seen = set()
        for row in table:
            source, task = row["text_source"], row["task"]
            fold = int(row["outer_fold"]) if by_fold else None
            key = (source, task, fold)
            if (source not in SOURCES or task not in TASKS or key in seen
                    or row["method"] != f"zscore_weighted__{source}__mean_morgan_cosine"):
                raise ValueError(f"aggregate key differs: {filename}:{key}")
            seen.add(key)
            members = [member for member in aggregate[(source, task)]
                       if not by_fold or member[1] == fold]
            if (int(row["query_count"]) != len(members)
                    or int(row["group_count"]) != len({member[0] for member in members})
                    or int(row["candidate_count_ranked"]) != 2217):
                raise ValueError(f"aggregate counts differ: {filename}:{key}")
            for name in METRICS:
                mean = math.fsum(member[2][name] for member in members) / len(members)
                max_difference = max(max_difference,
                    compare(row[name], mean, f"aggregate:{filename}:{key}:{name}"))
        if len(seen) != expected_count:
            raise ValueError(f"aggregate coverage differs: {filename}")

    print("independent_m2_5fold_validation: PASS")
    print("alpha_choices_checked:", len(choices))
    print("tuning_rows_checked:", len(tuning))
    print("ranking_rows_checked:", len(stored_rankings))
    print("per_query_rows_checked:", len(stored_per_query))
    print("maximum_metric_abs_difference:", max_difference)
    print("maximum_score_abs_difference:", max_score_difference)
    print("input_and_output_hashes: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
