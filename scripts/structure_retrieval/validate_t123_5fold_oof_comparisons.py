#!/usr/bin/env python3
"""Recompute T123 OOF comparisons from independently validated per-query files."""

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
DEFAULT_DIR = ROOT / "comparisons"
MANIFEST = "t123_oof_comparison_manifest.json"
METRICS = ("R@1", "R@5", "R@10", "R@50", "R@100", "MRR@100", "MAP@100", "nDCG@10")
SOURCES = ("nv_document", "qwen_whole_claims")
T2 = ("mean_morgan_cosine", "molformer_mean_pooling", "representative_max_tanimoto")
TASKS = ("Citation", "Conflict-X")
REPEATS = 10000
BASE_SEED = 20260926


def sha(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            output.update(piece)
    return output.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def derive_seed(*parts: object) -> int:
    joined = "\x1f".join(map(str, (BASE_SEED, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


def check_number(actual: str, expected: float, label: str, tolerance: float = 1e-12) -> float:
    observed = float(actual)
    difference = abs(observed - expected)
    if not math.isfinite(observed) or difference > tolerance:
        raise ValueError(f"{label}: {observed} != {expected}")
    return difference


def index(rows: list[dict], columns: tuple[str, ...], expected: int) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key in result:
            raise ValueError(f"duplicate row: {key}")
        result[key] = row
    if len(result) != expected:
        raise ValueError(f"row count mismatch: {len(result)} != {expected}")
    return result


def group_inference(items: list[tuple[str, float]], family: str, name: str):
    grouped = defaultdict(list)
    for group, delta in items:
        grouped[group].append(delta)
    names = sorted(grouped)
    if len(names) < 30:
        raise ValueError(f"too few groups for planned inference: {name}")
    totals = np.asarray([math.fsum(grouped[group]) for group in names], dtype=float)
    sizes = np.asarray([len(grouped[group]) for group in names], dtype=int)
    observed = math.fsum(delta for _, delta in items) / len(items)
    bootstrap_seed = derive_seed(family, name, "bootstrap")
    rng = np.random.default_rng(bootstrap_seed)
    samples = []
    for start in range(0, REPEATS, 250):
        count = min(250, REPEATS - start)
        indices = rng.integers(len(names), size=(count, len(names)))
        samples.extend((totals[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    lower, upper = np.percentile(np.asarray(samples), [2.5, 97.5])
    sign_seed = derive_seed(family, name, "sign_flip")
    rng = np.random.default_rng(sign_seed)
    tail = 0
    for start in range(0, REPEATS, 250):
        count = min(250, REPEATS - start)
        signs = rng.integers(0, 2, size=(count, len(names)), dtype=np.int8) * 2 - 1
        simulated = np.abs(np.matmul(signs, totals) / len(items))
        tail += int(np.count_nonzero(simulated >= abs(observed) - 1e-15))
    return float(lower), float(upper), (tail + 1) / (REPEATS + 1), bootstrap_seed, sign_seed


def self_test() -> None:
    if derive_seed("a") == derive_seed("b") or len(METRICS) != 8:
        raise AssertionError("seed/metric list check failed")
    print("self_test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    base = args.analysis_dir
    manifest = json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest["resampling"]["iterations"] != REPEATS
            or manifest["resampling"]["base_seed"] != BASE_SEED):
        raise ValueError("comparison manifest settings differ")
    for label, record in manifest["inputs"].items():
        nested = record if label in ("assignment", "assignment_manifest") else record.values()
        for item in ([nested] if isinstance(nested, dict) else nested):
            path = Path(item["path"])
            if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                raise ValueError(f"input hash differs: {label}:{path}")
    for name, record in manifest["outputs"].items():
        path = base / name
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"output hash differs: {name}")
    provenance = manifest["provenance"]
    if sha(Path(provenance["path"])) != provenance["sha256"]:
        raise ValueError("comparison implementation changed")

    assignment = index(read_csv(Path(manifest["inputs"]["assignment"]["path"])), ("query_id",), 1126)
    for task in TASKS:
        count = sum(task == "Citation" or row["conflict_x_eligible"] == "1"
                    for row in assignment.values())
        if count != (1126 if task == "Citation" else 47):
            raise ValueError(f"task cohort differs: {task}")

    model_values = {}
    for source, records in manifest["inputs"].items():
        if source in ("assignment", "assignment_manifest"):
            continue
        for row in read_csv(Path(records["per_query"]["path"])):
            if source == "standalone":
                model = f"{row['modality']}:{row['model']}"
            else:
                family = source if source != "M3" else f"M3_Top{row['top_n']}"
                model = f"T3:{family}:{row['text_source']}"
            query, task = row["query_patent_id"], row["task"]
            if task not in TASKS or (query,) not in assignment:
                raise ValueError(f"unknown task/query: {model}:{task}:{query}")
            meta = assignment[(query,)]
            if row["group_id"] != meta["group_id"] or row["outer_fold"] != meta["outer_fold"]:
                raise ValueError(f"fold/group mismatch: {model}:{task}:{query}")
            key = (model, task, query)
            if key in model_values:
                raise ValueError(f"duplicate per-query result: {key}")
            model_values[key] = {metric: float(row[metric]) for metric in METRICS}
    if len(model_values) != 16 * (1126 + 47):
        raise ValueError("source per-query rows incomplete")

    summaries = index(read_csv(base / "t123_oof_model_summary.csv"), ("model_id", "task"), 32)
    folds = index(read_csv(base / "t123_oof_model_folds.csv"),
                  ("model_id", "task", "outer_fold"), 160)
    comparisons = index(read_csv(base / "t123_oof_paired_comparisons.csv"),
                        ("comparison_family", "comparison_id", "task", "metric"), 256)
    model_names = {key[0] for key in model_values}
    max_difference = 0.0
    for model in model_names:
        for task in TASKS:
            queries = sorted(query for (query,), meta in assignment.items()
                             if task == "Citation" or assignment[(query,)]["conflict_x_eligible"] == "1")
            for fold in (None, *range(5)):
                selected = [query for query in queries
                            if fold is None or int(assignment[(query,)]["outer_fold"]) == fold]
                key = (model, task) if fold is None else (model, task, str(fold))
                row = summaries[key] if fold is None else folds[key]
                if (int(row["query_count"]) != len(selected)
                        or int(row["group_count"]) != len({assignment[(query,)]["group_id"] for query in selected})
                        or int(row["candidate_count_ranked"]) != 2217):
                    raise ValueError(f"model aggregate count differs: {key}")
                for metric in METRICS:
                    average = math.fsum(model_values[(model, task, query)][metric]
                                        for query in selected) / len(selected)
                    max_difference = max(max_difference, check_number(row[metric], average,
                                                                       f"model aggregate:{key}:{metric}"))

    expected_specs = {}
    for source in SOURCES:
        for method in ("M1", "M2", "M3_Top100", "M3_Top500", "M4"):
            expected_specs[("primary_T3_vs_T1", f"{method}:{source}")] = (
                f"T3:{method}:{source}", f"T1:{source}")
        for t2 in T2:
            expected_specs[("secondary_T1_vs_T2", f"{source}:{t2}")] = (
                f"T1:{source}", f"T2:{t2}")
    expected_keys = {(family, name, task, metric)
                     for family, name in expected_specs for task in TASKS for metric in METRICS}
    if set(comparisons) != expected_keys:
        raise ValueError("planned comparison set differs")
    raw_p = defaultdict(list)
    for key, row in comparisons.items():
        family, name, task, metric = key
        left, right = expected_specs[(family, name)]
        if row["left_model"] != left or row["right_model"] != right:
            raise ValueError(f"comparison direction differs: {key}")
        queries = sorted(query for (query,), meta in assignment.items()
                         if task == "Citation" or assignment[(query,)]["conflict_x_eligible"] == "1")
        pairs = [(model_values[(left, task, query)][metric],
                  model_values[(right, task, query)][metric],
                  assignment[(query,)]["group_id"]) for query in queries]
        delta = [(group, a - b) for a, b, group in pairs]
        differences = [value for _, value in delta]
        if (int(row["query_count"]) != len(queries)
                or int(row["group_count"]) != len({group for group, _ in delta})
                or int(row["left_wins"]) != sum(value > 1e-12 for value in differences)
                or int(row["ties"]) != sum(abs(value) <= 1e-12 for value in differences)
                or int(row["left_losses"]) != sum(value < -1e-12 for value in differences)):
            raise ValueError(f"paired counts differ: {key}")
        expected_numeric = {
            "left_mean": math.fsum(a for a, _, _ in pairs) / len(pairs),
            "right_mean": math.fsum(b for _, b, _ in pairs) / len(pairs),
            "mean_difference_left_minus_right": math.fsum(differences) / len(pairs),
        }
        for field, expected in expected_numeric.items():
            max_difference = max(max_difference, check_number(row[field], expected, f"{key}:{field}"))
        is_inference = task == "Citation" and metric == "MRR@100"
        if (row["inference_allowed"] == "True") != is_inference:
            raise ValueError(f"inference flag differs: {key}")
        if is_inference:
            lower, upper, p, boot_seed, sign_seed = group_inference(delta, family, name)
            for field, expected in (("bootstrap_ci_low", lower), ("bootstrap_ci_high", upper),
                                    ("sign_flip_p_value", p)):
                max_difference = max(max_difference, check_number(row[field], expected, f"{key}:{field}"))
            if int(row["bootstrap_seed"]) != boot_seed or int(row["sign_flip_seed"]) != sign_seed:
                raise ValueError(f"resampling seed differs: {key}")
            raw_p[family].append((key, p))
        elif any(row[field] for field in ("bootstrap_ci_low", "bootstrap_ci_high",
                                           "sign_flip_p_value", "holm_adjusted_p_value",
                                           "bootstrap_seed", "sign_flip_seed")):
            raise ValueError(f"unexpected exploratory inference: {key}")
    for family, pairs in raw_p.items():
        if len(pairs) != (10 if family == "primary_T3_vs_T1" else 6):
            raise ValueError(f"inference family size differs: {family}")
        running = 0.0
        for index_number, (key, p) in enumerate(sorted(pairs, key=lambda item: item[1])):
            running = max(running, min(1.0, (len(pairs) - index_number) * p))
            max_difference = max(max_difference, check_number(
                comparisons[key]["holm_adjusted_p_value"], running, f"Holm:{key}"))
    print("independent_t123_oof_comparison_validation: PASS")
    print("model_summary_rows: 32")
    print("model_fold_rows: 160")
    print("paired_comparison_rows: 256")
    print("primary_inference_tests: 10")
    print("secondary_inference_tests: 6")
    print("maximum_numeric_abs_difference:", max_difference)
    print("input_and_output_hashes: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
