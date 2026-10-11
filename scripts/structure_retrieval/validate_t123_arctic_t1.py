#!/usr/bin/env python3
"""Independently validate the Arctic T1 full-ranking input for T123 folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("R@1", "R@5", "R@10", "R@50", "R@100", "MRR@100", "MAP@100", "nDCG@10")
TASK_FIELDS = {"Citation": "all_cited_patents", "Conflict-X": "conflict_patents"}
SOURCE = "arctic_whole_claims"
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def input_record(manifest: dict, basename: str) -> dict:
    matches = [record for name, record in manifest["input_files"].items()
               if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(f"expected one input named {basename}")
    return matches[0]


def verify_recorded_files(records: dict[str, dict[str, object]]) -> None:
    for name, record in records.items():
        path = Path(name)
        if not path.is_file():
            raise ValueError(f"missing recorded file: {path}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"file hash/size mismatch: {path}")


def metrics_for(ranking: list[str], gold: set[str]) -> dict[str, float]:
    if not gold:
        raise ValueError("empty gold set")
    hits = [patent in gold for patent in ranking[:100]]
    values = {f"R@{k}": sum(hits[:k]) / len(gold) for k in (1, 5, 10, 50, 100)}
    values["MRR@100"] = next((1.0 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
    values["MAP@100"] = sum(
        sum(hits[: i + 1]) / (i + 1) for i, hit in enumerate(hits) if hit
    ) / len(gold)
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits[:10]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    values["nDCG@10"] = dcg / idcg
    return values


def require_close(observed: str, expected: float, description: str) -> float:
    difference = abs(float(observed) - expected)
    if not math.isfinite(difference) or difference > TOLERANCE:
        raise ValueError(f"metric mismatch: {description}: {observed} != {expected}")
    return difference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arctic-dir", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.arctic_dir / "t1_matched_subset_manifest.json").read_text())
    reference = json.loads(args.reference_manifest.read_text())
    if reference.get("status") != "completed":
        raise ValueError("reference T1 full-ranking run is not completed")
    if manifest.get("status") != "completed" or manifest.get("ranking_limit") != 2217:
        raise ValueError("Arctic manifest is not a completed full-ranking run")
    if set(manifest.get("sources", {})) != {SOURCE}:
        raise ValueError("unexpected Arctic source")
    if manifest.get("query_counts") != {"Citation": 1126, "Conflict-X": 47}:
        raise ValueError("unexpected Arctic query counts")
    if manifest.get("candidate_count") != 2218 or manifest.get("ranked_candidates_per_query") != 2217:
        raise ValueError("unexpected Arctic candidate mask")
    source = manifest["sources"][SOURCE]
    if (source.get("collection") != "patent_text_units_arctic_l_v2"
            or source.get("unit") != "whole_claims"
            or source.get("full_vector_patents") != 2376
            or source.get("unique_vector_patents") != 2218
            or source.get("vector_dimension") != 1024
            or source.get("device") != "cpu"):
        raise ValueError("unexpected Arctic vector provenance")

    verify_recorded_files(manifest["input_files"])
    verify_recorded_files(manifest["output_files"])
    script_path = Path(manifest["provenance"]["script_path"])
    if sha256_file(script_path) != manifest["provenance"]["script_sha256"]:
        raise ValueError("T1 evaluation script changed since the Arctic run")
    for basename in ("retrieval_ground_truth.jsonl", "text_candidate_patents_common.csv",
                     "surechembl_structures.csv"):
        if input_record(manifest, basename)["sha256"] != input_record(reference, basename)["sha256"]:
            raise ValueError(f"T1 input differs from reference: {basename}")
    if reference.get("candidate_count") != 2218 or reference.get("ranking_limit") != 2217:
        raise ValueError("reference T1 run uses a different candidate mask or depth")

    input_paths = list(manifest["input_files"])
    ground_truth_path = next(Path(p) for p in input_paths if p.endswith("retrieval_ground_truth.jsonl"))
    candidate_path = next(Path(p) for p in input_paths if p.endswith("text_candidate_patents_common.csv"))
    structure_path = next(Path(p) for p in input_paths if p.endswith("surechembl_structures.csv"))

    original_candidates = {row["patent_id"] for row in csv_rows(candidate_path)}
    eligible = set()
    with structure_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["patent_id"] in original_candidates and row["canonical_smiles"].strip():
                eligible.add(row["patent_id"])
    if len(original_candidates) != 2376 or len(eligible) != 2218:
        raise ValueError("candidate source or structure mask changed")

    ground_truth = {}
    with ground_truth_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                query = row["query_patent_id"]
                if query in ground_truth:
                    raise ValueError(f"duplicate ground-truth query: {query}")
                ground_truth[query] = {
                    task: set(row[field]) for task, field in TASK_FIELDS.items()
                }

    assignment = csv_rows(args.assignment)
    frozen_manifest = json.loads(args.assignment.with_name(
        "t123_5fold_assignment_manifest.json").read_text())
    if sha256_file(args.assignment) != frozen_manifest["assignment_file"]["sha256"]:
        raise ValueError("frozen fold assignment hash mismatch")
    assignment_queries = {row["query_id"] for row in assignment}
    if len(assignment) != 1126 or len(assignment_queries) != 1126:
        raise ValueError("frozen assignment is not the expected 1126 unique queries")

    per_query_path = args.arctic_dir / "t1_matched_subset_per_query.csv"
    result_path = args.arctic_dir / "t1_matched_subset_results.csv"
    ranking_path = args.arctic_dir / "t1_matched_subset_rankings_top2217.jsonl"
    per_query = {}
    for row in csv_rows(per_query_path):
        key = (row["task"], row["query_patent_id"])
        if key in per_query or row["source"] != SOURCE or key[0] not in TASK_FIELDS:
            raise ValueError(f"duplicate or invalid per-query row: {key}")
        per_query[key] = row
    if len(per_query) != 1173 or {q for task, q in per_query if task == "Citation"} != assignment_queries:
        raise ValueError("Arctic per-query population differs from frozen folds")

    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    seen = set()
    maximum_difference = 0.0
    with ranking_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query = row["query_patent_id"]
            if row["source"] != SOURCE or query in seen or query not in assignment_queries:
                raise ValueError(f"duplicate or unexpected ranking: {query}")
            seen.add(query)
            items = row["ranking"]
            patent_ids = [item["patent_id"] for item in items]
            if len(items) != 2217 or len(set(patent_ids)) != 2217:
                raise ValueError(f"invalid ranking length/uniqueness: {query}")
            if set(patent_ids) != eligible - {query}:
                raise ValueError(f"candidate ID mask mismatch: {query}")
            scores = [float(item["score"]) for item in items]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError(f"non-finite ranking score: {query}")
            if any(score < next_score for score, next_score in zip(scores, scores[1:])):
                raise ValueError(f"ranking scores are not descending: {query}")
            if any(int(item["rank"]) != i for i, item in enumerate(items, start=1)):
                raise ValueError(f"invalid rank sequence: {query}")
            for task in TASK_FIELDS:
                key = (task, query)
                if key not in per_query:
                    continue
                saved = per_query[key]
                gold = ground_truth[query][task]
                if not gold or int(saved["gold_count"]) != len(gold):
                    raise ValueError(f"gold count mismatch: {key}")
                if (int(saved["source_candidate_count_before_mask"]) != 2375
                        or int(saved["candidate_count_ranked"]) != 2217
                        or int(saved["ranking_rows_written"]) != 2217):
                    raise ValueError(f"per-query candidate count mismatch: {key}")
                calculated = metrics_for(patent_ids, gold)
                for metric in METRICS:
                    maximum_difference = max(
                        maximum_difference,
                        require_close(saved[metric], calculated[metric], f"{key}/{metric}"),
                    )
                    sums[task][metric] += calculated[metric]
                counts[task] += 1
    if seen != assignment_queries or counts != {"Citation": 1126, "Conflict-X": 47}:
        raise ValueError("ranking or task population mismatch")

    results = csv_rows(result_path)
    if len(results) != 2 or {row["task"] for row in results} != set(TASK_FIELDS):
        raise ValueError("unexpected result rows")
    for row in results:
        task = row["task"]
        if row["source"] != SOURCE or int(row["Queries"]) != counts[task]:
            raise ValueError(f"aggregate population mismatch: {task}")
        for metric in METRICS:
            maximum_difference = max(
                maximum_difference,
                require_close(
                    row[metric], sums[task][metric] / counts[task], f"aggregate/{task}/{metric}"
                ),
            )

    print("independent_arctic_t1_validation: PASS")
    print("citation_queries:", counts["Citation"])
    print("conflict_x_queries:", counts["Conflict-X"])
    print("ranked_candidates_per_query:", len(eligible) - 1)
    print("maximum_metric_abs_difference:", maximum_difference)
    print("input_and_output_hashes: MATCH")
    print("reference_t1_input_hashes: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
