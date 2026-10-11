#!/usr/bin/env python3
"""Place fixed T1/T2 standalone results on the frozen T123 five-fold cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("R@1", "R@5", "R@10", "R@50", "R@100", "MRR@100", "MAP@100", "nDCG@10")
TASKS = ("Citation", "Conflict-X")
ROOT = Path("structure_retrieval_outputs")
ASSIGNMENT = ROOT / "t123_5fold_v1/t123_5fold_assignment.csv"
GROUND_TRUTH = Path("data/processed/retrieval_ground_truth.jsonl")
CANDIDATES = ROOT / "t2_candidate_corpus_v1/text_candidate_patents_common.csv"
OUTPUT = ROOT / "t123_5fold_v1/standalone"

# Extra T1 is descriptive only; the prespecified T3 hypothesis family still uses NV/Qwen.
INPUTS = (
    ("T1", "nv_document", ROOT / "t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_per_query.csv", "source", 2217),
    ("T1", "qwen_whole_claims", ROOT / "t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_per_query.csv", "source", 2217),
    ("T1", "arctic_whole_claims", ROOT / "t123_5fold_v1/inputs/t1_arctic_whole_claims_full/t1_matched_subset_per_query.csv", "source", 2217),
    ("T2", "mean_morgan_cosine", ROOT / "t2_morgan_common_input_v1/t2_structure_retrieval_per_query.csv", "method", 100),
    ("T2", "molformer_mean_pooling", ROOT / "t2_molformer_mean_v1/t2_molformer_per_query.csv", "method", 100),
    ("T2", "representative_max_tanimoto", ROOT / "t2_representative_tanimoto_k16_v1/t2_representative_tanimoto_per_query.csv", "method", 100),
)


def file_info(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def read_assignment() -> dict[str, dict[str, str]]:
    frozen = json.loads(ASSIGNMENT.with_name("t123_5fold_assignment_manifest.json").read_text())
    if frozen["status"] != "frozen" or file_info(ASSIGNMENT)["sha256"] != frozen["assignment_file"]["sha256"]:
        raise ValueError("fold assignment differs from frozen manifest")
    for label, path in (("ground_truth", GROUND_TRUTH), ("candidate_corpus", CANDIDATES)):
        if file_info(path)["sha256"] != frozen["input_files"][label]["sha256"]:
            raise ValueError(f"frozen {label} input changed: {path}")
    rows = {}
    groups = {}
    for row in read_csv(ASSIGNMENT):
        query = row["query_id"]
        fold = int(row["outer_fold"])
        if query in rows or fold not in range(5):
            raise ValueError(f"duplicate/invalid fold assignment: {query}")
        group = row["group_id"]
        if group in groups and groups[group] != fold:
            raise ValueError(f"connected group split between folds: {group}")
        groups[group] = fold
        rows[query] = row
    if len(rows) != 1126 or len(groups) != 908:
        raise ValueError("unexpected frozen query/group counts")
    return rows


def read_gold() -> dict[str, dict[str, set[str]]]:
    output = {}
    with GROUND_TRUTH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query = row["query_patent_id"]
            if query in output:
                raise ValueError(f"duplicate ground truth: {query}")
            output[query] = {
                "Citation": set(row["all_cited_patents"]),
                "Conflict-X": set(row["conflict_patents"]),
            }
    return output


def eligible_queries(assignment: dict, gold: dict) -> dict[str, set[str]]:
    output = {"Citation": set(assignment), "Conflict-X": set()}
    for query, row in assignment.items():
        if row["citation_eligible"] != "1" or query not in gold or not gold[query]["Citation"]:
            raise ValueError(f"invalid Citation assignment: {query}")
        if row["conflict_x_eligible"] == "1":
            if not gold[query]["Conflict-X"]:
                raise ValueError(f"empty Conflict-X gold: {query}")
            output["Conflict-X"].add(query)
        elif row["conflict_x_eligible"] != "0":
            raise ValueError(f"invalid Conflict-X flag: {query}")
    if len(output["Conflict-X"]) != 47:
        raise ValueError("unexpected Conflict-X query count")
    return output


def input_manifest(path: Path) -> dict:
    name = path.name
    if name == "t1_matched_subset_per_query.csv":
        manifest_path = path.with_name("t1_matched_subset_manifest.json")
    elif name == "t2_structure_retrieval_per_query.csv":
        manifest_path = path.with_name("t2_structure_retrieval_manifest.json")
    elif name == "t2_molformer_per_query.csv":
        manifest_path = path.with_name("t2_molformer_manifest.json")
    else:
        manifest_path = path.with_name("t2_representative_tanimoto_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError(f"input run is incomplete: {manifest_path}")
    recorded = [record for key, record in manifest.get("output_files", {}).items()
                if Path(key).name == name]
    observed = file_info(path)
    if len(recorded) != 1 or recorded[0]["sha256"] != observed["sha256"]:
        raise ValueError(f"input per-query hash differs from manifest: {path}")
    for expected_path in (GROUND_TRUTH, CANDIDATES):
        matches = [record for key, record in manifest.get("input_files", {}).items()
                   if Path(key).name == expected_path.name]
        if len(matches) != 1 or matches[0]["sha256"] != file_info(expected_path)["sha256"]:
            raise ValueError(f"source input differs from frozen cohort: {path}:{expected_path}")
    return {"manifest": file_info(manifest_path), "per_query": observed}


def load_model_rows(
    path: Path, model_field: str, model: str, ranking_rows: int,
    assignment: dict, gold: dict, cohorts: dict[str, set[str]],
) -> list[dict[str, object]]:
    rows = []
    seen = set()
    for raw in read_csv(path):
        if raw[model_field] != model:
            continue
        task, query = raw["task"], raw["query_patent_id"]
        key = (task, query)
        if task not in TASKS or query not in cohorts[task] or key in seen:
            raise ValueError(f"unexpected/duplicate model row: {model}:{key}")
        seen.add(key)
        if (int(raw["gold_count"]) != len(gold[query][task])
                or int(raw["candidate_count_ranked"]) != 2217
                or int(raw["ranking_rows_written"]) != ranking_rows):
            raise ValueError(f"input grain/count mismatch: {model}:{key}")
        metrics = {}
        for name in METRICS:
            value = float(raw[name])
            if not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
                raise ValueError(f"invalid metric value: {model}:{key}:{name}")
            metrics[name] = value
        rows.append({
            "modality": "T1" if model in {"nv_document", "qwen_whole_claims", "arctic_whole_claims"} else "T2",
            "model": model,
            "task": task,
            "query_patent_id": query,
            "group_id": assignment[query]["group_id"],
            "outer_fold": int(assignment[query]["outer_fold"]),
            "gold_count": len(gold[query][task]),
            "candidate_count_ranked": 2217,
            **metrics,
        })
    expected = {(task, query) for task in TASKS for query in cohorts[task]}
    if seen != expected:
        raise ValueError(f"model cohort mismatch: {model}: missing={len(expected - seen)} extra={len(seen - expected)}")
    return rows


def aggregates(rows: list[dict[str, object]], by_fold: bool) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["modality"], row["model"], row["task"])
        if by_fold:
            key += (row["outer_fold"],)
        grouped[key].append(row)
    output = []
    for key, members in sorted(grouped.items()):
        record = {
            "modality": key[0], "model": key[1], "task": key[2],
            "query_count": len(members),
            "group_count": len({row["group_id"] for row in members}),
            "candidate_count_ranked": 2217,
        }
        if by_fold:
            record["outer_fold"] = key[3]
        for metric in METRICS:
            record[metric] = math.fsum(float(row[metric]) for row in members) / len(members)
        output.append(record)
    return output


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")

    assignment = read_assignment()
    gold = read_gold()
    cohorts = eligible_queries(assignment, gold)
    source_rows = []
    sources = {}
    for modality, model, path, model_field, ranking_rows in INPUTS:
        if model in sources:
            raise ValueError(f"duplicate model: {model}")
        sources[model] = {"modality": modality, **input_manifest(path)}
        source_rows.extend(load_model_rows(path, model_field, model, ranking_rows,
                                           assignment, gold, cohorts))
    if len(source_rows) != 6 * (1126 + 47):
        raise ValueError("standalone OOF rows are incomplete")
    source_rows.sort(key=lambda row: (row["modality"], row["model"], row["task"], row["query_patent_id"]))
    summary = aggregates(source_rows, by_fold=False)
    folds = aggregates(source_rows, by_fold=True)
    if len(summary) != 12 or len(folds) != 60:
        raise ValueError("unexpected standalone summary/fold row count")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_query_path = args.output_dir / "t123_standalone_oof_per_query.csv"
    summary_path = args.output_dir / "t123_standalone_oof_summary.csv"
    folds_path = args.output_dir / "t123_standalone_oof_folds.csv"
    common = ["modality", "model", "task"]
    write_csv(per_query_path, common + ["query_patent_id", "group_id", "outer_fold",
                                      "gold_count", "candidate_count_ranked", *METRICS], source_rows)
    write_csv(summary_path, common + ["query_count", "group_count", "candidate_count_ranked", *METRICS], summary)
    write_csv(folds_path, common + ["outer_fold", "query_count", "group_count",
                                    "candidate_count_ranked", *METRICS], folds)
    manifest = {
        "status": "completed",
        "scope": "fixed standalone T1/T2 rankings placed on frozen folds; no fold-specific refitting",
        "primary_task": "Citation",
        "primary_metric": "MRR@100",
        "historical_selection_caveat": "Arctic Whole Claims and other representation choices were inspected on this benchmark before five-fold evaluation; OOF is not an untouched external test.",
        "oracle_candidate_pool_caveat": "Candidate pool was derived from benchmark gold patents; not open-corpus retrieval.",
        "arctic_policy": "descriptive standalone T1 only; not added to prespecified T3 primary comparisons",
        "query_counts": {task: len(queries) for task, queries in cohorts.items()},
        "models": [model for _, model, _, _, _ in INPUTS],
        "input_files": {
            "assignment": file_info(ASSIGNMENT),
            "ground_truth": file_info(GROUND_TRUTH),
            "candidate_corpus": file_info(CANDIDATES),
            "sources": sources,
        },
        "output_files": {path.name: file_info(path) for path in (per_query_path, summary_path, folds_path)},
        "provenance": file_info(Path(__file__)),
    }
    manifest_path = args.output_dir / "t123_standalone_oof_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "completed", "query_counts": manifest["query_counts"],
                      "models": manifest["models"], "summary_rows": len(summary),
                      "fold_rows": len(folds), "per_query_rows": len(source_rows),
                      "manifest": str(manifest_path)}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
