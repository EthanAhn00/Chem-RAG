#!/usr/bin/env python3
"""Recompute frozen-fold M1 RRF outputs from the original full rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("R@1", "R@5", "R@10", "R@50", "R@100", "MRR@100", "MAP@100", "nDCG@10")
SOURCES = ("nv_document", "qwen_whole_claims")
TASKS = ("Citation", "Conflict-X")
DEFAULT_DIR = Path("structure_retrieval_outputs/t123_5fold_v1/m1_equal_rrf")
PER_QUERY = "t123_m1_rrf_oof_per_query.csv"
SUMMARY = "t123_m1_rrf_oof_summary.csv"
FOLDS = "t123_m1_rrf_oof_folds.csv"
RANKINGS = "t123_m1_rrf_rankings_top100.jsonl"
MANIFEST = "t123_m1_rrf_manifest.json"


def sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def evaluate(ids: list[str], gold: set[str]) -> dict[str, float]:
    hits = [int(patent in gold) for patent in ids[:100]]
    output = {f"R@{k}": sum(hits[:k]) / len(gold) for k in (1, 5, 10, 50, 100)}
    output["MRR@100"] = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    hit_count = 0
    precision_total = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            hit_count += 1
            precision_total += hit_count / rank
    output["MAP@100"] = precision_total / len(gold)
    dcg = math.fsum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:10], 1))
    idcg = math.fsum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
    output["nDCG@10"] = dcg / idcg
    return output


def check_number(value: str, expected: float, label: str) -> float:
    actual = float(value)
    difference = abs(actual - expected)
    if not math.isfinite(actual) or difference > 1e-12:
        raise ValueError(f"metric mismatch: {label}: actual={actual}, expected={expected}")
    return difference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    base = args.analysis_dir
    manifest = json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("rrf_k") != 60.0
            or set(manifest.get("text_sources", [])) != set(SOURCES)
            or manifest.get("structure_method") != "mean_morgan_cosine"):
        raise ValueError("M1 manifest settings differ")
    for label, record in manifest["input_files"].items():
        path = Path(record["path"])
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"input hash/size mismatch: {label}")
    for filename in (PER_QUERY, SUMMARY, FOLDS, RANKINGS):
        record = manifest["output_files"][filename]
        path = base / filename
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"output hash/size mismatch: {filename}")

    assignment = {}
    group_folds = {}
    for row in csv_rows(Path(manifest["input_files"]["assignment"]["path"])):
        query, group, fold = row["query_id"], row["group_id"], int(row["outer_fold"])
        if query in assignment or fold not in range(5):
            raise ValueError(f"invalid assignment: {query}")
        if group in group_folds and group_folds[group] != fold:
            raise ValueError(f"split connected group: {group}")
        group_folds[group] = fold
        assignment[query] = (group, fold, row["conflict_x_eligible"] == "1")
    if len(assignment) != 1126 or len(group_folds) != 908:
        raise ValueError("assignment counts differ")

    truth = {}
    for row in jsonl_rows(Path(manifest["input_files"]["ground_truth"]["path"])):
        query = row["query_patent_id"]
        if query in truth:
            raise ValueError(f"duplicate gold query: {query}")
        truth[query] = {"Citation": set(row["all_cited_patents"]),
                        "Conflict-X": set(row["conflict_patents"])}

    text_rankings = defaultdict(dict)
    for row in jsonl_rows(Path(manifest["input_files"]["text_rankings"]["path"])):
        source, query = row["source"], row["query_patent_id"]
        if source not in SOURCES or query not in assignment or query in text_rankings[source]:
            raise ValueError(f"unexpected/duplicate text ranking: {source}:{query}")
        items = row["ranking"]
        if len(items) != 2217 or any(item["rank"] != rank for rank, item in enumerate(items, 1)):
            raise ValueError(f"invalid text ranking length/order: {source}:{query}")
        ids = [item["patent_id"] for item in items]
        if len(set(ids)) != 2217 or query in ids:
            raise ValueError(f"invalid text candidates: {source}:{query}")
        text_rankings[source][query] = ids
    if any(set(text_rankings[source]) != set(assignment) for source in SOURCES):
        raise ValueError("text query coverage differs")

    structure_rankings = {}
    for row in jsonl_rows(Path(manifest["input_files"]["structure_rankings"]["path"])):
        if row["method"] != "mean_morgan_cosine":
            raise ValueError(f"unexpected structure method: {row['method']}")
        query = row["query_patent_id"]
        if query not in assignment:
            continue
        if query in structure_rankings:
            raise ValueError(f"duplicate structure ranking: {query}")
        items = row["ranking"]
        if len(items) != 2217 or any(item["rank"] != rank for rank, item in enumerate(items, 1)):
            raise ValueError(f"invalid structure ranking length/order: {query}")
        ids = [item["patent_id"] for item in items]
        if len(set(ids)) != 2217 or query in ids:
            raise ValueError(f"invalid structure candidates: {query}")
        structure_rankings[query] = ids
    if set(structure_rankings) != set(assignment):
        raise ValueError("structure query coverage differs")

    stored_rankings = {}
    for row in jsonl_rows(base / RANKINGS):
        key = (row["text_source"], row["query_patent_id"])
        if key in stored_rankings or row["method"] != f"rrf_k60__{key[0]}__mean_morgan_cosine":
            raise ValueError(f"unexpected/duplicate fused ranking: {key}")
        stored_rankings[key] = row["ranking"]
    if len(stored_rankings) != 2252:
        raise ValueError("stored ranking count differs")

    stored_per_query = {}
    for row in csv_rows(base / PER_QUERY):
        key = (row["text_source"], row["task"], row["query_patent_id"])
        if key in stored_per_query or row["method"] != f"rrf_k60__{key[0]}__mean_morgan_cosine":
            raise ValueError(f"unexpected/duplicate per-query row: {key}")
        stored_per_query[key] = row
    if len(stored_per_query) != 2346:
        raise ValueError("stored per-query count differs")

    recomputed = defaultdict(list)
    checked = set()
    max_metric_difference = 0.0
    max_score_difference = 0.0
    for source in SOURCES:
        for query, text_ids in text_rankings[source].items():
            structure_ids = structure_rankings[query]
            if set(text_ids) != set(structure_ids):
                raise ValueError(f"candidate IDs differ: {source}:{query}")
            rank_t = {patent: rank for rank, patent in enumerate(text_ids, 1)}
            rank_s = {patent: rank for rank, patent in enumerate(structure_ids, 1)}
            scores = {patent: 1 / (60 + rank_t[patent]) + 1 / (60 + rank_s[patent])
                      for patent in text_ids}
            fused = sorted(text_ids, key=lambda patent: (-scores[patent], patent))
            stored = stored_rankings[(source, query)]
            if len(stored) != 100:
                raise ValueError(f"stored Top-100 length differs: {source}:{query}")
            for rank, (patent, item) in enumerate(zip(fused[:100], stored), 1):
                if item["rank"] != rank or item["patent_id"] != patent:
                    raise ValueError(f"M1 ranking mismatch: {source}:{query}:{rank}")
                difference = abs(float(item["score"]) - scores[patent])
                if not math.isfinite(float(item["score"])) or difference > 1e-12:
                    raise ValueError(f"M1 score mismatch: {source}:{query}:{rank}")
                max_score_difference = max(max_score_difference, difference)
            group, fold, conflict_eligible = assignment[query]
            for task in TASKS:
                if task == "Conflict-X" and not conflict_eligible:
                    continue
                key = (source, task, query)
                row = stored_per_query[key]
                gold = truth[query][task]
                if not gold or not gold.issubset(scores):
                    raise ValueError(f"gold outside candidates: {key}")
                if (row["group_id"] != group or int(row["outer_fold"]) != fold
                        or int(row["gold_count"]) != len(gold)
                        or int(row["candidate_count_ranked"]) != 2217):
                    raise ValueError(f"per-query metadata differs: {key}")
                metrics = evaluate(fused, gold)
                for name in METRICS:
                    max_metric_difference = max(max_metric_difference,
                        check_number(row[name], metrics[name], f"{key}:{name}"))
                recomputed[(source, task)].append((query, group, fold, metrics))
                checked.add(key)
    if checked != set(stored_per_query):
        raise ValueError("per-query coverage differs")

    for filename, by_fold, expected_count in ((SUMMARY, False, 4), (FOLDS, True, 20)):
        table = csv_rows(base / filename)
        if len(table) != expected_count:
            raise ValueError(f"aggregate count differs: {filename}")
        seen = set()
        for row in table:
            source, task = row["text_source"], row["task"]
            fold = int(row["outer_fold"]) if by_fold else None
            key = (source, task, fold)
            if (source not in SOURCES or task not in TASKS or key in seen
                    or row["method"] != f"rrf_k60__{source}__mean_morgan_cosine"):
                raise ValueError(f"aggregate key differs: {filename}:{key}")
            seen.add(key)
            members = [member for member in recomputed[(source, task)]
                       if not by_fold or member[2] == fold]
            if (int(row["query_count"]) != len(members)
                    or int(row["group_count"]) != len({member[1] for member in members})
                    or int(row["candidate_count_ranked"]) != 2217):
                raise ValueError(f"aggregate counts differ: {filename}:{key}")
            for name in METRICS:
                mean = math.fsum(member[3][name] for member in members) / len(members)
                max_metric_difference = max(max_metric_difference,
                    check_number(row[name], mean, f"{filename}:{key}:{name}"))
        if len(seen) != expected_count:
            raise ValueError(f"aggregate coverage differs: {filename}")

    print("independent_m1_5fold_validation: PASS")
    print("ranking_rows:", len(stored_rankings))
    print("per_query_rows:", len(stored_per_query))
    print("summary_rows: 4")
    print("fold_rows: 20")
    print("maximum_metric_abs_difference:", max_metric_difference)
    print("maximum_score_abs_difference:", max_score_difference)
    print("input_and_output_hashes: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
