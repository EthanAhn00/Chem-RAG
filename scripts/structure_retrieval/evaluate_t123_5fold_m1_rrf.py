#!/usr/bin/env python3
"""Evaluate fixed equal-RRF fusion on the frozen T123 five-fold cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from evaluate_t3_rrf import (
    METRICS,
    TASKS,
    evaluate_ranking,
    fuse_rrf,
    load_ground_truth,
    load_rankings,
)


ROOT = Path("structure_retrieval_outputs")
FIVEFOLD = ROOT / "t123_5fold_v1"
ASSIGNMENT = FIVEFOLD / "t123_5fold_assignment.csv"
ASSIGNMENT_MANIFEST = FIVEFOLD / "t123_5fold_assignment_manifest.json"
STANDALONE = FIVEFOLD / "standalone/t123_standalone_oof_per_query.csv"
STANDALONE_MANIFEST = FIVEFOLD / "standalone/t123_standalone_oof_manifest.json"
GROUND_TRUTH = Path("data/processed/retrieval_ground_truth.jsonl")
TEXT_RANKINGS = ROOT / "t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_rankings_top2217.jsonl"
STRUCTURE_RANKINGS = ROOT / "t3_rrf_v1/inputs/t2_morgan_full/t2_structure_rankings_top2217.jsonl"
STRUCTURE_PER_QUERY = ROOT / "t3_rrf_v1/inputs/t2_morgan_full/t2_structure_retrieval_per_query.csv"
OUTPUT = FIVEFOLD / "m1_equal_rrf"
TEXT_SOURCES = ("nv_document", "qwen_whole_claims")
STRUCTURE_METHOD = "mean_morgan_cosine"
RRF_K = 60.0
CANDIDATE_COUNT = 2217


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs() -> tuple[dict[str, dict[str, str]], dict[tuple[str, str, str], dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    frozen = json.loads(ASSIGNMENT_MANIFEST.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen" or sha256(ASSIGNMENT) != frozen["assignment_file"]["sha256"]:
        raise ValueError("frozen fold assignment mismatch")
    for label, path in (("ground_truth", GROUND_TRUTH), ("structure_per_query", STRUCTURE_PER_QUERY),
                        ("text_full_rankings", TEXT_RANKINGS),
                        ("structure_full_rankings", STRUCTURE_RANKINGS)):
        if sha256(path) != frozen["input_files"][label]["sha256"]:
            raise ValueError(f"frozen input mismatch: {label}")
    standalone_manifest = json.loads(STANDALONE_MANIFEST.read_text(encoding="utf-8"))
    if (standalone_manifest.get("status") != "completed"
            or sha256(STANDALONE) != standalone_manifest["output_files"][STANDALONE.name]["sha256"]):
        raise ValueError("validated standalone input mismatch")

    assignment = {}
    group_folds = {}
    for row in csv_rows(ASSIGNMENT):
        query, group, fold = row["query_id"], row["group_id"], int(row["outer_fold"])
        if query in assignment or fold not in range(5) or row["citation_eligible"] != "1":
            raise ValueError(f"invalid assignment: {query}")
        if group in group_folds and group_folds[group] != fold:
            raise ValueError(f"group split across folds: {group}")
        assignment[query] = row
        group_folds[group] = fold
    if len(assignment) != 1126 or len(group_folds) != 908:
        raise ValueError("assignment count mismatch")

    baseline = {}
    for row in csv_rows(STANDALONE):
        key = (row["model"], row["task"], row["query_patent_id"])
        if key in baseline:
            raise ValueError(f"duplicate standalone row: {key}")
        baseline[key] = row
    for model in TEXT_SOURCES:
        for task in TASKS:
            expected = {query for query, row in assignment.items()
                        if task == "Citation" or row["conflict_x_eligible"] == "1"}
            actual = {query for m, t, query in baseline if m == model and t == task}
            if actual != expected:
                raise ValueError(f"standalone cohort mismatch: {model}:{task}")
    structure_baseline = {}
    for row in csv_rows(STRUCTURE_PER_QUERY):
        if row["method"] != STRUCTURE_METHOD:
            raise ValueError(f"unexpected Morgan method: {row['method']}")
        key = (row["task"], row["query_patent_id"])
        if key in structure_baseline:
            raise ValueError(f"duplicate original Morgan row: {key}")
        structure_baseline[key] = row
    for task in TASKS:
        expected = {query for query, row in assignment.items()
                    if task == "Citation" or row["conflict_x_eligible"] == "1"}
        if {query for t, query in structure_baseline if t == task} != expected:
            raise ValueError(f"original Morgan cohort mismatch: {task}")
    return assignment, baseline, structure_baseline


def summarize(rows: list[dict], by_fold: bool) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["method"], row["text_source"], row["task"])
        if by_fold:
            key += (row["outer_fold"],)
        groups[key].append(row)
    output = []
    for key, members in sorted(groups.items()):
        item = {"method": key[0], "text_source": key[1], "task": key[2],
                "query_count": len(members),
                "group_count": len({member["group_id"] for member in members}),
                "candidate_count_ranked": CANDIDATE_COUNT}
        if by_fold:
            item["outer_fold"] = key[3]
        for metric in METRICS:
            item[metric] = math.fsum(member[metric] for member in members) / len(members)
        output.append(item)
    return output


def self_test() -> None:
    text = ["EP1A1", "EP2A1", "EP3A1"]
    structure = ["EP3A1", "EP2A1", "EP1A1"]
    result = fuse_rrf(text, structure, RRF_K)
    assert [patent for patent, _ in result] == ["EP1A1", "EP3A1", "EP2A1"]
    assert evaluate_ranking(result, {"EP3A1"})["R@5"] == 1.0
    print("self_test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")

    assignment, baseline, structure_baseline = verify_inputs()
    gold = load_ground_truth(GROUND_TRUTH, "conflict_patents")
    text_rankings, text_models = load_rankings(TEXT_RANKINGS, "source")
    structure_rankings, structure_models = load_rankings(STRUCTURE_RANKINGS, "method")
    if text_models != set(TEXT_SOURCES) or structure_models != {STRUCTURE_METHOD}:
        raise ValueError("ranking models differ from prespecified M1 inputs")
    if any(set(text_rankings[source]) != set(assignment) for source in TEXT_SOURCES):
        raise ValueError("text query coverage mismatch")
    if not set(assignment).issubset(structure_rankings[STRUCTURE_METHOD]):
        missing = set(assignment) - set(structure_rankings[STRUCTURE_METHOD])
        raise ValueError(f"structure query coverage mismatch: missing={len(missing)}")
    extra_structure_queries = len(set(structure_rankings[STRUCTURE_METHOD]) - set(assignment))

    per_query = []
    ranking_rows = []
    for source in TEXT_SOURCES:
        method = f"rrf_k60__{source}__{STRUCTURE_METHOD}"
        for query in sorted(assignment):
            meta = assignment[query]
            text_ids = text_rankings[source][query]
            structure_ids = structure_rankings[STRUCTURE_METHOD][query]
            if len(text_ids) != CANDIDATE_COUNT or len(structure_ids) != CANDIDATE_COUNT:
                raise ValueError(f"incomplete full ranking: {source}:{query}")
            fused = fuse_rrf(text_ids, structure_ids, RRF_K)
            if len(fused) != CANDIDATE_COUNT:
                raise ValueError(f"fused candidate count mismatch: {source}:{query}")
            ranking_rows.append({
                "method": method, "text_source": source, "query_patent_id": query,
                "ranking": [{"rank": rank, "patent_id": patent, "score": round(score, 12)}
                            for rank, (patent, score) in enumerate(fused[:100], 1)],
            })
            for task in TASKS:
                if task == "Conflict-X" and meta["conflict_x_eligible"] != "1":
                    continue
                relevant = gold[query][task]
                if not relevant or not relevant.issubset(set(text_ids) - {query}):
                    raise ValueError(f"gold outside common candidates: {task}:{query}")
                text_baseline = baseline[(source, task, query)]
                structure_reference = structure_baseline[(task, query)]
                text_metrics = evaluate_ranking([(patent, 0.0) for patent in text_ids], relevant)
                structure_metrics = evaluate_ranking(
                    [(patent, 0.0) for patent in structure_ids], relevant
                )
                if (int(text_baseline["gold_count"]) != len(relevant)
                        or int(structure_reference["gold_count"]) != len(relevant)):
                    raise ValueError(f"baseline gold count mismatch: {task}:{query}")
                for metric in METRICS:
                    if abs(text_metrics[metric] - float(text_baseline[metric])) > 1e-12:
                        raise ValueError(f"T1 ranking/standalone metric mismatch: {source}:{task}:{query}:{metric}")
                    if abs(structure_metrics[metric] - float(structure_reference[metric])) > 1e-12:
                        raise ValueError(f"T2 ranking/standalone metric mismatch: {task}:{query}:{metric}")
                per_query.append({
                    "method": method, "text_source": source, "structure_method": STRUCTURE_METHOD,
                    "query_patent_id": query, "group_id": meta["group_id"],
                    "outer_fold": int(meta["outer_fold"]), "task": task,
                    "gold_count": len(relevant), "candidate_count_ranked": CANDIDATE_COUNT,
                    **evaluate_ranking(fused, relevant),
                })
    if len(per_query) != 2 * (1126 + 47) or len(ranking_rows) != 2 * 1126:
        raise ValueError("M1 output cohort incomplete")
    per_query.sort(key=lambda row: (row["method"], row["task"], row["query_patent_id"]))
    summary = summarize(per_query, False)
    folds = summarize(per_query, True)
    if len(summary) != 4 or len(folds) != 20:
        raise ValueError("M1 summary coverage mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_query_path = args.output_dir / "t123_m1_rrf_oof_per_query.csv"
    summary_path = args.output_dir / "t123_m1_rrf_oof_summary.csv"
    folds_path = args.output_dir / "t123_m1_rrf_oof_folds.csv"
    ranking_path = args.output_dir / "t123_m1_rrf_rankings_top100.jsonl"
    columns = ["method", "text_source", "task"]
    write_csv(per_query_path, columns + ["structure_method", "query_patent_id", "group_id",
                                         "outer_fold", "gold_count", "candidate_count_ranked", *METRICS], per_query)
    write_csv(summary_path, columns + ["query_count", "group_count", "candidate_count_ranked", *METRICS], summary)
    write_csv(folds_path, columns + ["outer_fold", "query_count", "group_count",
                                     "candidate_count_ranked", *METRICS], folds)
    with ranking_path.open("x", encoding="utf-8") as handle:
        for row in ranking_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "completed", "method": "M1 equal RRF", "rrf_k": RRF_K,
        "scope": "fixed full-pool rankings on frozen folds; no fold-specific learning",
        "text_sources": list(TEXT_SOURCES), "structure_method": STRUCTURE_METHOD,
        "query_counts": {"Citation": 1126, "Conflict-X": 47},
        "candidate_count_ranked": CANDIDATE_COUNT,
        "structure_ranking_queries_outside_evaluation_cohort": extra_structure_queries,
        "historical_selection_caveat": "Benchmark models/methods inspected before this five-fold evaluation; not untouched external validation.",
        "oracle_candidate_pool_caveat": "Closed candidate pool includes benchmark-gold-derived patents; not open-corpus retrieval.",
        "structure_baseline_note": "M1 uses the original full Morgan rankings; the separate T2 standalone table labels Morgan common input and may differ.",
        "input_files": {label: file_record(path) for label, path in (
            ("assignment", ASSIGNMENT), ("assignment_manifest", ASSIGNMENT_MANIFEST),
            ("standalone", STANDALONE), ("standalone_manifest", STANDALONE_MANIFEST),
            ("ground_truth", GROUND_TRUTH), ("text_rankings", TEXT_RANKINGS),
            ("structure_rankings", STRUCTURE_RANKINGS),
            ("structure_per_query", STRUCTURE_PER_QUERY))},
        "output_files": {path.name: file_record(path) for path in
                         (per_query_path, summary_path, folds_path, ranking_path)},
        "provenance": file_record(Path(__file__)),
    }
    manifest_path = args.output_dir / "t123_m1_rrf_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "completed", "per_query_rows": len(per_query),
                      "summary_rows": len(summary), "fold_rows": len(folds),
                      "ranking_rows": len(ranking_rows),
                      "structure_ranking_queries_outside_evaluation_cohort": extra_structure_queries,
                      "manifest": str(manifest_path)},
                     ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
