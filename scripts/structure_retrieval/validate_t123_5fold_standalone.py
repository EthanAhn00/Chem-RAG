#!/usr/bin/env python3
"""Independently check the frozen-fold standalone tables against source rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("R@1", "R@5", "R@10", "R@50", "R@100", "MRR@100", "MAP@100", "nDCG@10")
MODELS = {
    "nv_document": ("T1", "source"),
    "qwen_whole_claims": ("T1", "source"),
    "arctic_whole_claims": ("T1", "source"),
    "mean_morgan_cosine": ("T2", "method"),
    "molformer_mean_pooling": ("T2", "method"),
    "representative_max_tanimoto": ("T2", "method"),
}
TASKS = {"Citation": 1126, "Conflict-X": 47}
DEFAULT_DIR = Path("structure_retrieval_outputs/t123_5fold_v1/standalone")
PER_QUERY = "t123_standalone_oof_per_query.csv"
SUMMARY = "t123_standalone_oof_summary.csv"
FOLDS = "t123_standalone_oof_folds.csv"
MANIFEST = "t123_standalone_oof_manifest.json"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_close(actual: str, expected: float, label: str) -> float:
    value = float(actual)
    difference = abs(value - expected)
    if not math.isfinite(value) or difference > 1e-12:
        raise ValueError(f"numeric mismatch: {label}: {value} != {expected}")
    return difference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    base = args.analysis_dir
    manifest = json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or set(manifest.get("models", [])) != set(MODELS):
        raise ValueError("manifest status or model set differs")
    for filename in (PER_QUERY, SUMMARY, FOLDS):
        path = base / filename
        record = manifest["output_files"][filename]
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"output hash/size mismatch: {path}")

    assignment_path = Path(manifest["input_files"]["assignment"]["path"])
    if digest(assignment_path) != manifest["input_files"]["assignment"]["sha256"]:
        raise ValueError("assignment hash mismatch")
    assignment = {}
    group_folds = {}
    for row in rows(assignment_path):
        query, group, fold = row["query_id"], row["group_id"], int(row["outer_fold"])
        if query in assignment or fold not in range(5):
            raise ValueError(f"invalid assignment: {query}")
        if group in group_folds and group_folds[group] != fold:
            raise ValueError(f"split connected group: {group}")
        assignment[query] = (group, fold, row["conflict_x_eligible"] == "1")
        group_folds[group] = fold
    if len(assignment) != 1126 or len(group_folds) != 908:
        raise ValueError("assignment counts differ")

    expected = {}
    for model, (modality, field) in MODELS.items():
        source = manifest["input_files"]["sources"][model]["per_query"]
        path = Path(source["path"])
        if digest(path) != source["sha256"] or path.stat().st_size != source["bytes"]:
            raise ValueError(f"source hash/size mismatch: {model}")
        for row in rows(path):
            if row[field] != model:
                continue
            task, query = row["task"], row["query_patent_id"]
            if task not in TASKS or query not in assignment:
                raise ValueError(f"unexpected source query: {model}:{task}:{query}")
            group, fold, conflict_eligible = assignment[query]
            if task == "Conflict-X" and not conflict_eligible:
                raise ValueError(f"ineligible Conflict-X source row: {model}:{query}")
            key = (modality, model, task, query)
            if key in expected:
                raise ValueError(f"duplicate source row: {key}")
            expected[key] = (group, fold, int(row["gold_count"]),
                             int(row["candidate_count_ranked"]),
                             {metric: float(row[metric]) for metric in METRICS})
        for task, count in TASKS.items():
            observed = sum(key[1] == model and key[2] == task for key in expected)
            if observed != count:
                raise ValueError(f"source cohort mismatch: {model}:{task}: {observed}")

    observed = set()
    numeric_max = 0.0
    grouped = defaultdict(list)
    per_query_rows = rows(base / PER_QUERY)
    for row in per_query_rows:
        key = (row["modality"], row["model"], row["task"], row["query_patent_id"])
        if key not in expected or key in observed:
            raise ValueError(f"unexpected/duplicate output row: {key}")
        observed.add(key)
        group, fold, gold_count, candidate_count, metrics = expected[key]
        if (row["group_id"] != group or int(row["outer_fold"]) != fold
                or int(row["gold_count"]) != gold_count
                or int(row["candidate_count_ranked"]) != candidate_count):
            raise ValueError(f"output metadata mismatch: {key}")
        for metric in METRICS:
            numeric_max = max(numeric_max, require_close(row[metric], metrics[metric], f"{key}:{metric}"))
        grouped[key[:3]].append((group, fold, metrics))
    if observed != set(expected) or len(observed) != 7038:
        raise ValueError(f"per-query coverage mismatch: {len(observed)}")

    for filename, with_fold, expected_count in ((SUMMARY, False, 12), (FOLDS, True, 60)):
        table = rows(base / filename)
        if len(table) != expected_count:
            raise ValueError(f"aggregate row count mismatch: {filename}")
        seen = set()
        for row in table:
            key = (row["modality"], row["model"], row["task"])
            if key not in grouped:
                raise ValueError(f"unexpected aggregate row: {filename}:{key}")
            fold = int(row["outer_fold"]) if with_fold else None
            unique = (*key, fold)
            if unique in seen or (with_fold and fold not in range(5)):
                raise ValueError(f"duplicate/invalid aggregate row: {filename}:{unique}")
            seen.add(unique)
            members = [item for item in grouped[key] if not with_fold or item[1] == fold]
            if not members:
                raise ValueError(f"empty aggregate: {filename}:{unique}")
            if (int(row["query_count"]) != len(members)
                    or int(row["group_count"]) != len({item[0] for item in members})
                    or int(row["candidate_count_ranked"]) != 2217):
                raise ValueError(f"aggregate count mismatch: {filename}:{unique}")
            for metric in METRICS:
                mean = math.fsum(item[2][metric] for item in members) / len(members)
                numeric_max = max(numeric_max, require_close(row[metric], mean, f"{filename}:{unique}:{metric}"))
        if len(seen) != expected_count:
            raise ValueError(f"aggregate coverage mismatch: {filename}")

    print("independent_standalone_5fold_validation: PASS")
    print("per_query_rows:", len(per_query_rows))
    print("summary_rows: 12")
    print("fold_rows: 60")
    print("maximum_numeric_abs_difference:", numeric_max)
    print("source_and_output_hashes: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
