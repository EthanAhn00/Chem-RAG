#!/usr/bin/env python3
"""Compare frozen T1/T2/T3 OOF results with group-paired primary inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("structure_retrieval_outputs/t123_5fold_v1")
OUTPUT = ROOT / "comparisons"
ASSIGNMENT = ROOT / "t123_5fold_assignment.csv"
ASSIGNMENT_MANIFEST = ROOT / "t123_5fold_assignment_manifest.json"
INPUTS = {
    "standalone": (ROOT / "standalone", "t123_standalone_oof_manifest.json",
                   "t123_standalone_oof_per_query.csv"),
    "M1": (ROOT / "m1_equal_rrf", "t123_m1_rrf_manifest.json",
           "t123_m1_rrf_oof_per_query.csv"),
    "M2": (ROOT / "m2_weighted", "t123_m2_weighted_manifest.json",
           "t123_m2_weighted_oof_per_query.csv"),
    "M3": (ROOT / "m3_text_first_reranking", "t123_m3_reranking_manifest.json",
           "t123_m3_reranking_oof_per_query.csv"),
    "M4": (ROOT / "m4_pairwise_logistic", "t123_m4_learned_manifest.json",
           "t123_m4_learned_oof_per_query.csv"),
}
METRICS = ("R@1", "R@5", "R@10", "R@50", "R@100", "MRR@100", "MAP@100", "nDCG@10")
SOURCES = ("nv_document", "qwen_whole_claims")
T2_MODELS = ("mean_morgan_cosine", "molformer_mean_pooling", "representative_max_tanimoto")
TASKS = ("Citation", "Conflict-X")
B = 10000
BASE_SEED = 20260926


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def seed(*parts: object) -> int:
    payload = "\x1f".join(str(value) for value in (BASE_SEED, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def verify_input(label: str) -> tuple[Path, dict]:
    directory, manifest_name, csv_name = INPUTS[label]
    manifest_path = directory / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"incomplete input: {label}")
    csv_path = directory / csv_name
    record = manifest["output_files"][csv_name]
    if csv_path.stat().st_size != record["bytes"] or digest(csv_path) != record["sha256"]:
        raise ValueError(f"input hash mismatch: {label}")
    assignment_record = manifest["input_files"]["assignment"]
    if assignment_record["sha256"] != digest(ASSIGNMENT):
        raise ValueError(f"input assignment mismatch: {label}")
    return csv_path, manifest


def load_assignment() -> dict[str, dict]:
    frozen = json.loads(ASSIGNMENT_MANIFEST.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen" or digest(ASSIGNMENT) != frozen["assignment_file"]["sha256"]:
        raise ValueError("frozen assignment mismatch")
    result, groups = {}, {}
    for row in read_csv(ASSIGNMENT):
        query, group, fold = row["query_id"], row["group_id"], int(row["outer_fold"])
        if query in result or fold not in range(5) or row["citation_eligible"] != "1":
            raise ValueError(f"invalid assignment: {query}")
        if group in groups and groups[group] != fold:
            raise ValueError(f"group crosses folds: {group}")
        result[query] = row
        groups[group] = fold
    if len(result) != 1126 or len(groups) != 908:
        raise ValueError("assignment cohort differs")
    return result


def expected_queries(assignment: dict, task: str) -> set[str]:
    return {query for query, row in assignment.items()
            if task == "Citation" or row["conflict_x_eligible"] == "1"}


def t3_method(family: str, source: str) -> str:
    suffix = f"__{source}__mean_morgan_cosine"
    return {
        "M1": "rrf_k60" + suffix,
        "M2": "zscore_weighted" + suffix,
        "M3_Top100": "text_top100_weighted" + suffix,
        "M3_Top500": "text_top500_weighted" + suffix,
        "M4": "pairwise_logistic" + suffix,
    }[family]


def load_models(assignment: dict) -> tuple[dict, dict]:
    records = {}
    inputs = {}
    for label in INPUTS:
        path, manifest = verify_input(label)
        inputs[label] = {"manifest": file_record(path.with_name(INPUTS[label][1])),
                         "per_query": file_record(path)}
        for row in read_csv(path):
            if label == "standalone":
                modality, model = row["modality"], row["model"]
                if (modality, model) not in {
                    ("T1", "nv_document"), ("T1", "qwen_whole_claims"),
                    ("T1", "arctic_whole_claims"),
                    *(("T2", name) for name in T2_MODELS),
                }:
                    raise ValueError(f"unexpected standalone model: {modality}:{model}")
                model_id = f"{modality}:{model}"
            else:
                source, method = row["text_source"], row["method"]
                family = label if label != "M3" else f"M3_Top{row['top_n']}"
                if source not in SOURCES or method != t3_method(family, source):
                    raise ValueError(f"unexpected fusion method: {method}")
                model_id = f"T3:{family}:{source}"
            task, query = row["task"], row["query_patent_id"]
            if task not in TASKS or query not in expected_queries(assignment, task):
                raise ValueError(f"unexpected task/query: {model_id}:{task}:{query}")
            metadata = assignment[query]
            if (row["group_id"] != metadata["group_id"]
                    or int(row["outer_fold"]) != int(metadata["outer_fold"])
                    or int(row["candidate_count_ranked"]) != 2217
                    or int(row["gold_count"]) < 1):
                raise ValueError(f"cohort metadata mismatch: {model_id}:{task}:{query}")
            key = (model_id, task, query)
            if key in records:
                raise ValueError(f"duplicate result: {key}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in values.values()):
                raise ValueError(f"invalid metric: {key}")
            records[key] = {"model_id": model_id, "task": task, "query_patent_id": query,
                            "group_id": metadata["group_id"],
                            "outer_fold": int(metadata["outer_fold"]),
                            "gold_count": int(row["gold_count"]), **values}
    model_ids = {key[0] for key in records}
    if len(model_ids) != 16 or len(records) != 16 * (1126 + 47):
        raise ValueError(f"model/cohort count differs: {len(model_ids)} models, {len(records)} rows")
    for model in model_ids:
        for task in TASKS:
            actual = {query for label, t, query in records if label == model and t == task}
            if actual != expected_queries(assignment, task):
                raise ValueError(f"incomplete model/task cohort: {model}:{task}")
    for task in TASKS:
        for query in expected_queries(assignment, task):
            if len({records[(model, task, query)]["gold_count"] for model in model_ids}) != 1:
                raise ValueError(f"gold count differs between models: {task}:{query}")
    return records, inputs


def infer_group_paired(differences: list[tuple[str, float]], family: str,
                       comparison_id: str) -> dict:
    groups = defaultdict(list)
    for group, value in differences:
        groups[group].append(value)
    ordered = sorted(groups)
    sums = np.asarray([math.fsum(groups[group]) for group in ordered], dtype=np.float64)
    counts = np.asarray([len(groups[group]) for group in ordered], dtype=np.int64)
    observed = math.fsum(value for _, value in differences) / len(differences)
    if len(ordered) < 30:
        return {"bootstrap_ci_low": "", "bootstrap_ci_high": "",
                "sign_flip_p_value": "", "bootstrap_seed": "", "sign_flip_seed": "",
                "inference_allowed": False}
    bootstrap = np.empty(B, dtype=np.float64)
    bootstrap_seed = seed(family, comparison_id, "bootstrap")
    rng = np.random.default_rng(bootstrap_seed)
    for start in range(0, B, 250):
        size = min(250, B - start)
        picked = rng.integers(0, len(ordered), size=(size, len(ordered)))
        bootstrap[start:start + size] = sums[picked].sum(axis=1) / counts[picked].sum(axis=1)
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
    sign_seed = seed(family, comparison_id, "sign_flip")
    rng = np.random.default_rng(sign_seed)
    exceed = 0
    threshold = abs(observed) - 1e-15
    for start in range(0, B, 250):
        size = min(250, B - start)
        signs = rng.integers(0, 2, size=(size, len(ordered)), dtype=np.int8) * 2 - 1
        null = np.abs(signs @ sums / len(differences))
        exceed += int(np.count_nonzero(null >= threshold))
    return {"bootstrap_ci_low": float(ci_low), "bootstrap_ci_high": float(ci_high),
            "sign_flip_p_value": (exceed + 1) / (B + 1),
            "bootstrap_seed": bootstrap_seed, "sign_flip_seed": sign_seed,
            "inference_allowed": True}


def holm(rows: list[dict], family: str) -> None:
    selected = sorted((row for row in rows if row["comparison_family"] == family
                       and row["task"] == "Citation" and row["metric"] == "MRR@100"),
                      key=lambda row: row["sign_flip_p_value"])
    expected = 10 if family == "primary_T3_vs_T1" else 6
    if len(selected) != expected or any(not row["inference_allowed"] for row in selected):
        raise ValueError(f"incomplete primary inference family: {family}")
    running = 0.0
    for index, row in enumerate(selected):
        running = max(running, min(1.0, (len(selected) - index) * row["sign_flip_p_value"]))
        row["holm_adjusted_p_value"] = running


def self_test() -> None:
    rows = [{"comparison_family": "primary_T3_vs_T1", "task": "Citation",
             "metric": "MRR@100", "sign_flip_p_value": value,
             "inference_allowed": True} for value in (0.01,) * 10]
    holm(rows, "primary_T3_vs_T1")
    if rows[0]["holm_adjusted_p_value"] != 0.1:
        raise AssertionError("Holm correction differs")
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
    assignment = load_assignment()
    records, inputs = load_models(assignment)
    descriptive = []
    fold_descriptive = []
    for model in sorted({key[0] for key in records}):
        for task in TASKS:
            members = [records[(model, task, query)] for query in sorted(expected_queries(assignment, task))]
            item = {"model_id": model, "task": task, "query_count": len(members),
                    "group_count": len({row["group_id"] for row in members}),
                    "candidate_count_ranked": 2217}
            item.update({metric: math.fsum(row[metric] for row in members) / len(members)
                         for metric in METRICS})
            descriptive.append(item)
            for fold in range(5):
                fold_members = [row for row in members if row["outer_fold"] == fold]
                if not fold_members:
                    raise ValueError(f"empty fold: {model}:{task}:{fold}")
                fold_item = {"model_id": model, "task": task, "outer_fold": fold,
                             "query_count": len(fold_members),
                             "group_count": len({row["group_id"] for row in fold_members}),
                             "candidate_count_ranked": 2217}
                fold_item.update({metric: math.fsum(row[metric] for row in fold_members) / len(fold_members)
                                  for metric in METRICS})
                fold_descriptive.append(fold_item)
    comparisons = []
    specs = []
    for source in SOURCES:
        for method in ("M1", "M2", "M3_Top100", "M3_Top500", "M4"):
            specs.append(("primary_T3_vs_T1", f"{method}:{source}",
                          f"T3:{method}:{source}", f"T1:{source}"))
        for t2 in T2_MODELS:
            specs.append(("secondary_T1_vs_T2", f"{source}:{t2}",
                          f"T1:{source}", f"T2:{t2}"))
    if len(specs) != 16:
        raise AssertionError("prespecified comparison list differs")
    for family, comparison_id, left, right in specs:
        for task in TASKS:
            paired = [(records[(left, task, query)], records[(right, task, query)])
                      for query in sorted(expected_queries(assignment, task))]
            for metric in METRICS:
                differences = [(a["group_id"], a[metric] - b[metric]) for a, b in paired]
                values = [value for _, value in differences]
                mean = math.fsum(values) / len(values)
                row = {"comparison_family": family, "comparison_id": comparison_id,
                       "left_model": left, "right_model": right, "task": task,
                       "metric": metric, "query_count": len(paired),
                       "group_count": len({group for group, _ in differences}),
                       "left_mean": math.fsum(a[metric] for a, _ in paired) / len(paired),
                       "right_mean": math.fsum(b[metric] for _, b in paired) / len(paired),
                       "mean_difference_left_minus_right": mean,
                       "left_wins": sum(value > 1e-12 for value in values),
                       "ties": sum(abs(value) <= 1e-12 for value in values),
                       "left_losses": sum(value < -1e-12 for value in values),
                       "bootstrap_ci_low": "", "bootstrap_ci_high": "",
                       "sign_flip_p_value": "", "holm_adjusted_p_value": "",
                       "bootstrap_seed": "", "sign_flip_seed": "",
                       "inference_allowed": False}
                if task == "Citation" and metric == "MRR@100":
                    row.update(infer_group_paired(differences, family, comparison_id))
                comparisons.append(row)
    for family in ("primary_T3_vs_T1", "secondary_T1_vs_T2"):
        holm(comparisons, family)
    if len(descriptive) != 32 or len(fold_descriptive) != 160 or len(comparisons) != 256:
        raise ValueError("OOF comparison output incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "t123_oof_model_summary.csv"
    fold_path = args.output_dir / "t123_oof_model_folds.csv"
    comparison_path = args.output_dir / "t123_oof_paired_comparisons.csv"
    write_csv(summary_path, ["model_id", "task", "query_count", "group_count",
                             "candidate_count_ranked", *METRICS], descriptive)
    write_csv(fold_path, ["model_id", "task", "outer_fold", "query_count", "group_count",
                          "candidate_count_ranked", *METRICS], fold_descriptive)
    write_csv(comparison_path, ["comparison_family", "comparison_id", "left_model", "right_model",
                                "task", "metric", "query_count", "group_count", "left_mean",
                                "right_mean", "mean_difference_left_minus_right", "left_wins", "ties",
                                "left_losses", "bootstrap_ci_low", "bootstrap_ci_high",
                                "sign_flip_p_value", "holm_adjusted_p_value", "bootstrap_seed",
                                "sign_flip_seed",
                                "inference_allowed"], comparisons)
    manifest = {
        "status": "completed", "population": "1126 Citation; 47 Conflict-X; frozen five-fold OOF",
        "unit": "query", "candidate_count_per_query": 2217,
        "primary_metric": "Citation MRR@100",
        "primary_family": "10 paired T3-minus-corresponding-T1 comparisons; Holm within family",
        "secondary_family": "6 paired T1-minus-T2 comparisons; separate Holm family",
        "additional_metrics": "Other metrics and Conflict-X reported descriptively without p-values or CI.",
        "resampling": {"unit": "connected query group", "bootstrap": "group resampling with replacement",
                       "sign_flip": "two-sided group-coherent sign flip on query-macro mean",
                       "iterations": B, "base_seed": BASE_SEED,
                       "confidence_interval": "percentile 95%",
                       "small_sample_policy": "suppress inference below 30 independent groups"},
        "limitations": ["Benchmark and method choices were previously inspected; not untouched external validation.",
                        "Closed gold-derived candidate pool; not open-corpus retrieval.",
                        "Group inference conditions on fixed OOF predictions and does not capture overlapping-fold training variability.",
                        "Group independence and sign symmetry are assumptions, not empirically established.",
                        "T2 standalone Morgan common-input differs from original full Morgan input used by T3."],
        "inputs": {"assignment": file_record(ASSIGNMENT),
                   "assignment_manifest": file_record(ASSIGNMENT_MANIFEST), **inputs},
        "outputs": {path.name: file_record(path) for path in (summary_path, fold_path, comparison_path)},
        "provenance": file_record(Path(__file__)),
    }
    manifest_path = args.output_dir / "t123_oof_comparison_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "completed", "model_summary_rows": len(descriptive),
                      "model_fold_rows": len(fold_descriptive),
                      "paired_comparison_rows": len(comparisons),
                      "primary_tests": 10, "secondary_tests": 6,
                      "manifest": str(manifest_path)}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
