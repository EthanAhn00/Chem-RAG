#!/usr/bin/env python3
"""Independently recompute saved T1 metrics and optionally export frozen-fold summaries.

Checks saved rankings, gold, populations, candidate IDs and file hashes. This
does not recompute cosine scores from Qdrant vectors or refit any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "structure_retrieval_outputs/t123_5fold_v1"
REFERENCE = ROOT / "structure_retrieval_outputs/t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_manifest.json"
METRICS = ("R@1", "R@5", "R@10", "R@50", "R@100", "MRR@100", "MAP@100", "nDCG@10")
FIELDS = {"Citation": "all_cited_patents", "Conflict-X": "conflict_patents"}
SETTINGS = {
    "nv_whole_claims": ("patent_text_units_nv_embed_v2", "whole_claims"),
    "nv_claim": ("patent_text_units_nv_embed_v2", "claim"),
    "qwen_document": ("patent_text_units_qwen3_0_6b", "document"),
    "qwen_claim": ("patent_text_units_qwen3_0_6b", "claim"),
    "arctic_document": ("patent_text_units_arctic_l_v2", "document"),
    "arctic_claim": ("patent_text_units_arctic_l_v2", "claim"),
}
TOLERANCE = 1e-12


def norm(value: str) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    if not re.fullmatch(r"EP\d+[A-Z]\d?", compact):
        raise ValueError(f"Invalid patent ID: {value!r}")
    return compact


def info(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def resolve(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def checked_records(records: dict) -> dict[str, Path]:
    paths = {}
    for name, recorded in records.items():
        path = resolve(name)
        if path.name in paths:
            raise ValueError(f"Duplicate recorded basename: {path.name}")
        actual = info(path)
        if any(actual[key] != recorded[key] for key in ("bytes", "sha256")):
            raise ValueError(f"File hash/size mismatch: {path}")
        paths[path.name] = path
    return paths


def metrics_for(ranking: list[str], gold: set[str]) -> dict[str, float]:
    if not gold:
        raise ValueError("Empty gold")
    hits = [patent in gold for patent in ranking[:100]]
    result = {f"R@{k}": sum(hits[:k]) / len(gold) for k in (1, 5, 10, 50, 100)}
    result["MRR@100"] = next((1.0 / position for position, hit in enumerate(hits, 1) if hit), 0.0)
    found = 0
    precisions = []
    for position, hit in enumerate(hits, 1):
        if hit:
            found += 1
            precisions.append(found / position)
    result["MAP@100"] = math.fsum(precisions) / len(gold)
    dcg = math.fsum(1 / math.log2(position + 1) for position, hit in enumerate(hits[:10], 1) if hit)
    ideal = math.fsum(1 / math.log2(position + 1) for position in range(1, min(10, len(gold)) + 1))
    result["nDCG@10"] = dcg / ideal
    return result


def close(actual, expected: float, context: str) -> float:
    difference = abs(float(actual) - expected)
    if not math.isfinite(difference) or difference > TOLERANCE:
        raise ValueError(f"Numeric mismatch: {context}: {actual} != {expected}")
    return difference


def aggregates(rows: list[dict], by_fold: bool) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["task"], row["outer_fold"]) if by_fold else (row["task"],)
        groups[key].append(row)
    result = []
    for key, members in sorted(groups.items()):
        record = {"modality": "T1", "model": members[0]["model"], "task": key[0],
                  "query_count": len(members), "group_count": len({row["group_id"] for row in members}),
                  "candidate_count_ranked": 2217}
        if by_fold:
            record["outer_fold"] = key[1]
        record.update({metric: math.fsum(row[metric] for row in members) / len(members) for metric in METRICS})
        result.append(record)
    return result


def csv_text(rows: list[dict]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def self_test() -> None:
    result = metrics_for(["EP1A1", "EP2A1", "EP3A1"], {"EP1A1", "EP3A1"})
    close(result["R@1"], 0.5, "self-test recall")
    close(result["MRR@100"], 1, "self-test MRR")
    close(result["MAP@100"], 5 / 6, "self-test MAP")
    close(result["nDCG@10"], 1.5 / (1 + 1 / math.log2(3)), "self-test nDCG")
    assert all(value == 0 for value in metrics_for(["EP2A1"] * 100 + ["EP1A1"], {"EP1A1"}).values())
    try:
        close(float("nan"), 0, "self-test nonfinite")
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite value accepted")
    rows = [
        {"model": "example", "task": "Citation", "outer_fold": fold,
         "group_id": str(index), **{metric: value for metric in METRICS}}
        for index, (fold, value) in enumerate(((0, 1.0), (1, 0.0), (1, 0.0)))
    ]
    close(aggregates(rows, False)[0]["MRR@100"], 1 / 3, "query-weighted summary")
    assert [row["query_count"] for row in aggregates(rows, True)] == [1, 2]
    print("self_test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SETTINGS)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--export-oof", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.source is None:
        parser.error("--source is required")
    target = BASE / "inputs" / f"t1_{args.source}_full"
    manifest_path = target / "t1_matched_subset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for name, document in (("reference", reference), ("extra", manifest)):
        if (document.get("status") != "completed" or document.get("ranking_limit") != 2217
                or document.get("query_counts") != {"Citation": 1126, "Conflict-X": 47}
                or document.get("candidate_count") != 2218
                or document.get("ranked_candidates_per_query") != 2217
                or document.get("ground_truth_field_mapping") != FIELDS):
            raise ValueError(f"Unexpected {name} manifest scope")
    if set(manifest["sources"]) != {args.source}:
        raise ValueError("Unexpected source set")
    source = manifest["sources"][args.source]
    collection, unit = SETTINGS[args.source]
    if (source.get("collection") != collection or source.get("unit") != unit
            or source.get("device") != "cpu" or source.get("full_vector_patents") != 2376
            or source.get("unique_vector_patents") != 2218
            or source.get("full_vector_rows", 0) < 2376
            or source.get("vector_rows", 0) < 2218
            or source.get("vector_dimension", 0) <= 0):
        raise ValueError("Unexpected recorded vector provenance")
    if unit != "claim" and (source["full_vector_rows"] != 2376 or source["vector_rows"] != 2218):
        raise ValueError("Document/whole-claims row counts differ from patent counts")
    for existing in reference["sources"].values():
        if existing["collection"] == collection and existing["vector_dimension"] != source["vector_dimension"]:
            raise ValueError("Vector dimension differs from reference collection")
    inputs = checked_records(manifest["input_files"])
    reference_inputs = checked_records(reference["input_files"])
    if set(inputs) != set(reference_inputs) or any(info(path) != info(reference_inputs[name]) for name, path in inputs.items()):
        raise ValueError("Extra-setting inputs differ from original T1 run")
    outputs = checked_records(manifest["output_files"])
    for path in outputs.values():
        if path.parent.resolve() != target.resolve():
            raise ValueError("Output record points outside this setting directory")
    evaluator = ROOT / "scripts/structure_retrieval/evaluate_t1_matched_subset.py"
    if (info(evaluator)["sha256"] != manifest["provenance"]["script_sha256"]
            or manifest["provenance"]["script_sha256"] != reference["provenance"]["script_sha256"]):
        raise ValueError("Evaluator provenance mismatch")

    assignment_path = BASE / "t123_5fold_assignment.csv"
    frozen_path = BASE / "t123_5fold_assignment_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen" or info(assignment_path)["sha256"] != frozen["assignment_file"]["sha256"]:
        raise ValueError("Frozen assignment mismatch")
    for label, basename in (("ground_truth", "retrieval_ground_truth.jsonl"), ("candidate_corpus", "text_candidate_patents_common.csv")):
        if info(inputs[basename])["sha256"] != frozen["input_files"][label]["sha256"]:
            raise ValueError(f"Input differs from frozen cohort: {label}")
    assignment = {}
    group_folds = {}
    for row in csv_rows(assignment_path):
        query = norm(row["query_id"])
        fold = int(row["outer_fold"])
        group = row["group_id"]
        if query in assignment or fold not in range(5) or row["citation_eligible"] != "1":
            raise ValueError("Invalid/duplicate assignment")
        if group in group_folds and group_folds[group] != fold:
            raise ValueError("Connected group split across folds")
        group_folds[group] = fold
        assignment[query] = {**row, "outer_fold": fold}
    if len(assignment) != 1126 or len(group_folds) != 908:
        raise ValueError("Unexpected frozen cohort size")

    candidates = {norm(row["patent_id"]) for row in csv_rows(inputs["text_candidate_patents_common.csv"])}
    eligible = {norm(row["patent_id"]) for row in csv_rows(inputs["surechembl_structures.csv"])
                if row["canonical_smiles"].strip() and norm(row["patent_id"]) in candidates}
    if len(candidates) != 2376 or len(eligible) != 2218 or not set(assignment) <= eligible:
        raise ValueError("Candidate mask changed")
    gold = {}
    with inputs["retrieval_ground_truth.jsonl"].open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                query = norm(row["query_patent_id"])
                if query in gold:
                    raise ValueError("Duplicate ground truth")
                gold[query] = {task: {norm(value) for value in row[field]} for task, field in FIELDS.items()}
    queries = {"Citation": set(assignment), "Conflict-X": set()}
    for query, row in assignment.items():
        if (not gold[query]["Citation"] or not gold[query]["Citation"] <= eligible - {query}
                or not gold[query]["Conflict-X"] <= gold[query]["Citation"]):
            raise ValueError(f"Incomplete or inconsistent gold: {query}")
        if row["conflict_x_eligible"] not in {"0", "1"}:
            raise ValueError("Invalid Conflict-X eligibility flag")
        if (row["conflict_x_eligible"] == "1") != bool(gold[query]["Conflict-X"]):
            raise ValueError("Conflict-X eligibility differs from gold")
        if row["conflict_x_eligible"] == "1":
            queries["Conflict-X"].add(query)
    if len(queries["Conflict-X"]) != 47:
        raise ValueError("Unexpected Conflict-X population")
    original_query_sets = defaultdict(set)
    for row in csv_rows(inputs["t2_structure_retrieval_per_query.csv"]):
        original_query_sets[row["task"]].add(norm(row["query_patent_id"]))
    if dict(original_query_sets) != queries:
        raise ValueError("Original T2 query populations differ")

    saved = {}
    for row in csv_rows(outputs["t1_matched_subset_per_query.csv"]):
        key = (row["task"], norm(row["query_patent_id"]))
        if key in saved or key[0] not in queries or key[1] not in queries[key[0]] or row["source"] != args.source:
            raise ValueError(f"Unexpected/duplicate metric row: {key}")
        if (int(row["candidate_count_ranked"]) != 2217 or int(row["ranking_rows_written"]) != 2217
                or int(row["source_candidate_count_before_mask"]) != 2375
                or int(row["gold_count"]) != len(gold[key[1]][key[0]])):
            raise ValueError(f"Saved counts differ: {key}")
        saved[key] = row
    if set(saved) != {(task, query) for task, members in queries.items() for query in members}:
        raise ValueError("Metric-row population mismatch")
    recalculated = []
    seen = set()
    maximum_difference = 0.0
    with outputs["t1_matched_subset_rankings_top2217.jsonl"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query = norm(row["query_patent_id"])
            if row["source"] != args.source or query in seen or query not in assignment:
                raise ValueError(f"Unexpected/duplicate ranking: {query}")
            seen.add(query)
            items = row["ranking"]
            ids = [norm(item["patent_id"]) for item in items]
            if len(ids) != 2217 or len(set(ids)) != 2217 or set(ids) != eligible - {query}:
                raise ValueError(f"Candidate IDs differ: {query}")
            scores = [float(item["score"]) for item in items]
            if (not all(math.isfinite(score) for score in scores)
                    or any(left < right for left, right in zip(scores, scores[1:]))
                    or any(int(item["rank"]) != position for position, item in enumerate(items, 1))):
                raise ValueError(f"Invalid scores or rank order: {query}")
            for task, members in queries.items():
                if query not in members:
                    continue
                calculated = metrics_for(ids, gold[query][task])
                for metric in METRICS:
                    maximum_difference = max(maximum_difference, close(saved[task, query][metric], calculated[metric], f"{task}/{query}/{metric}"))
                recalculated.append({"modality": "T1", "model": args.source, "task": task,
                                     "query_patent_id": query, "outer_fold": assignment[query]["outer_fold"],
                                     "group_id": assignment[query]["group_id"], "gold_count": len(gold[query][task]),
                                     "candidate_count_ranked": 2217, **calculated})
    if seen != queries["Citation"]:
        raise ValueError("Ranking query population differs")
    recalculated.sort(key=lambda row: (row["task"], row["query_patent_id"]))
    summary = aggregates(recalculated, False)
    folds = aggregates(recalculated, True)
    for records, label in ((list(csv_rows(outputs["t1_matched_subset_results.csv"])), "CSV"), (manifest["results"], "manifest")):
        if len(records) != 2 or {row["task"] for row in records} != set(FIELDS):
            raise ValueError(f"Unexpected {label} aggregate rows")
        for row in records:
            expected = next(item for item in summary if item["task"] == row["task"])
            if row["source"] != args.source or int(row["Queries"]) != expected["query_count"]:
                raise ValueError(f"Unexpected {label} aggregate counts")
            for metric in METRICS:
                maximum_difference = max(maximum_difference, close(row[metric], expected[metric], f"{label}/{row['task']}/{metric}"))

    if args.export_oof:
        texts = {
            target / "t123_extra_t1_oof_per_query.csv": csv_text(recalculated),
            target / "t123_extra_t1_oof_summary.csv": csv_text(summary),
            target / "t123_extra_t1_oof_by_fold.csv": csv_text(folds),
        }
        evidence_paths = [manifest_path, REFERENCE, assignment_path, frozen_path, *inputs.values(), *outputs.values()]
        export_manifest = {
            "status": "independently_validated", "source": args.source,
            "evaluation": "fixed T1 scores partitioned by existing frozen outer folds; no fitting/tuning",
            "input_files": {str(path): info(path) for path in evidence_paths},
            "output_files": {str(path): {"bytes": len(value.encode("utf-8")), "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()} for path, value in texts.items()},
            "provenance": {"script_path": str(Path(__file__).resolve()), "script_sha256": info(Path(__file__).resolve())["sha256"]},
            "maximum_metric_abs_difference": maximum_difference,
            "limitations": "Saved-ranking metric validation; cosine scores not independently recomputed from vectors. Additional settings are descriptive and do not alter prespecified T3 comparisons.",
        }
        texts[target / "t123_extra_t1_oof_manifest.json"] = json.dumps(export_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        for path, value in texts.items():
            if path.exists() and path.read_bytes() != value.encode("utf-8"):
                raise ValueError(f"Refusing to overwrite different OOF output: {path}")
        for path, value in texts.items():
            if not path.exists():
                with path.open("x", encoding="utf-8", newline="") as handle:
                    handle.write(value)
            if path.read_bytes() != value.encode("utf-8"):
                raise ValueError(f"Saved OOF output differs: {path}")
    print("independent_extra_t1_setting_validation: PASS")
    print("source:", args.source)
    print("ranking_rows:", len(seen))
    print("per_query_rows:", len(recalculated))
    print("summary_rows:", len(summary))
    print("fold_rows:", len(folds))
    print("maximum_metric_abs_difference:", maximum_difference)
    print("input_and_output_hashes: MATCH")
    print("oof_exported:", args.export_oof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
