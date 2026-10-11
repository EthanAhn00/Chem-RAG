#!/usr/bin/env python3
"""Independently replay M4 splits, pairwise fits, tuning, rankings and metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from validate_t123_5fold_m2_weighted import (
    METRICS, SOURCES, TASKS, compare, csv_rows, jsonl_rows, metrics,
    parse_rankings, sha256,
)


DEFAULT_DIR = Path("structure_retrieval_outputs/t123_5fold_v1/m4_pairwise_logistic")
FILES = {
    "manifest": "t123_m4_learned_manifest.json",
    "tuning": "t123_m4_c_tuning.csv",
    "choices": "t123_m4_c_choices.csv",
    "inner": "t123_m4_inner_splits.csv",
    "models": "t123_m4_models.json",
    "per_query": "t123_m4_learned_oof_per_query.csv",
    "summary": "t123_m4_learned_oof_summary.csv",
    "folds": "t123_m4_learned_oof_folds.csv",
    "rankings": "t123_m4_learned_rankings_top100.jsonl",
}
CS = (0.01, 0.1, 1.0, 10.0)
BASE_SEED = 20260924


def stable(base: int, *parts: object) -> int:
    raw = "\x1f".join(str(part) for part in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def seed(*parts: object) -> int:
    return stable(BASE_SEED, *parts)


def split_inner(train: list[str], truth: dict, fold: int) -> tuple[list[str], list[str]]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for query in train:
        find(f"q:{query}")
        for gold in sorted(truth[query]["Citation"]):
            union(f"q:{query}", f"g:{gold}")
    components = defaultdict(list)
    for query in train:
        components[find(f"q:{query}")].append(query)
    split_seed = seed("outer", fold, "m4_inner")
    ordered = sorted((sorted(value) for value in components.values()),
                     key=lambda value: (stable(split_seed, "component", "|".join(value)), value))
    target = max(1, round(len(train) * 0.2))
    fit, dev = [], []
    for component in ordered:
        (dev if len(dev) < target else fit).extend(component)
    return sorted(fit), sorted(dev)


def features(text: tuple, structure: tuple) -> tuple[list[str], np.ndarray]:
    ids, raw_text = text
    structure_ids, raw_structure = structure
    if len(ids) != 2217 or set(ids) != set(structure_ids):
        raise ValueError("T1/T2 candidate mismatch")
    structure_score = dict(zip(structure_ids, raw_structure))
    raw_structure_aligned = np.asarray([structure_score[patent] for patent in ids])
    text_sd = float(np.std(raw_text, ddof=0))
    structure_sd = float(np.std(raw_structure_aligned, ddof=0))
    if min(text_sd, structure_sd) <= 1e-15:
        raise ValueError("zero score variance")
    text_z = (raw_text - float(np.mean(raw_text))) / text_sd
    structure_z = (raw_structure_aligned - float(np.mean(raw_structure_aligned))) / structure_sd
    positions = {patent: rank for rank, patent in enumerate(structure_ids, 1)}
    text_rank_feature = 1 / np.log2(np.arange(1, len(ids) + 1, dtype=float) + 1)
    structure_rank_feature = 1 / np.log2(np.asarray([positions[item] for item in ids], dtype=float) + 1)
    matrix = np.column_stack((text_z, structure_z, text_rank_feature,
                              structure_rank_feature, text_z * structure_z))
    if matrix.shape != (2217, 5) or not np.isfinite(matrix).all():
        raise ValueError("nonfinite/invalid feature matrix")
    return ids, matrix


def pair_data(queries: list[str], source: str, truth: dict, text: dict,
              structure: dict, sampling_seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    blocks, label_blocks = [], []
    positives = negatives = 0
    for query in sorted(queries):
        text_ranking = text[source][query]
        structure_ranking = structure["mean_morgan_cosine"][query]
        ids, matrix = features(text_ranking, structure_ranking)
        id_index = {patent: index for index, patent in enumerate(ids)}
        gold = truth[query]["Citation"]
        if not gold or not gold.issubset(id_index):
            raise ValueError(f"training gold outside candidate pool: {query}")
        selected, seen = [], set()
        for order in (text_ranking[0], structure_ranking[0]):
            count = 0
            for patent in order:
                if patent in gold or patent in seen:
                    continue
                selected.append(patent)
                seen.add(patent)
                count += 1
                if count == 50:
                    break
        remaining = [patent for patent in ids if patent not in gold and patent not in seen]
        random_seed = stable(sampling_seed, "negative", source, query)
        rng = np.random.default_rng(random_seed)
        drawn = rng.choice(len(remaining), size=min(50, len(remaining)), replace=False)
        selected.extend(remaining[int(index)] for index in sorted(int(item) for item in drawn))
        negative_indices = [id_index[patent] for patent in selected]
        positives += len(gold)
        negatives += len(selected)
        for patent in sorted(gold):
            diff = matrix[id_index[patent]] - matrix[negative_indices]
            blocks.extend((diff, -diff))
            label_blocks.extend((np.ones(len(diff), dtype=np.int8),
                                 np.zeros(len(diff), dtype=np.int8)))
    x = np.vstack(blocks).astype(np.float64, copy=False)
    y = np.concatenate(label_blocks)
    return x, y, {"query_count": len(queries), "known_positive_count": positives,
                  "selected_negative_candidate_count": negatives,
                  "symmetric_pair_rows": len(y)}


def fit_model(x: np.ndarray, y: np.ndarray, c: float, model_seed: int):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    model = LogisticRegression(C=c, fit_intercept=False, solver="liblinear",
                               max_iter=1000, random_state=int(model_seed % (2**32)))
    model.fit(scaled, y)
    if model.n_iter_[0] >= model.max_iter:
        raise ValueError("independent logistic replay did not converge")
    return scaler, model


def rank(ids: list[str], matrix: np.ndarray, scaler, model):
    values = model.decision_function(scaler.transform(matrix))
    order = np.lexsort((np.asarray(ids, dtype=str), -values))
    return [ids[index] for index in order], values


def require_index(rows: list[dict], fields: tuple, expected: int) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate output: {key}")
        result[key] = row
    if len(result) != expected:
        raise ValueError(f"output row count {len(result)} != {expected}")
    return result


def self_test() -> None:
    if seed("a") == seed("b") or len(CS) != 4:
        raise AssertionError("seed/grid check failed")
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
    manifest = json.loads((base / FILES["manifest"]).read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("c_grid") != list(CS)
            or manifest.get("base_seed") != BASE_SEED
            or set(manifest.get("text_sources", [])) != set(SOURCES)
            or manifest.get("structure_method") != "mean_morgan_cosine"):
        raise ValueError("M4 manifest configuration differs")
    for label, record in manifest["input_files"].items():
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"input file differs: {label}")
    for label, record in manifest["provenance"].items():
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"implementation file differs: {label}")
    for label, filename in FILES.items():
        if label == "manifest":
            continue
        path = base / filename
        record = manifest["output_files"][filename]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"output file differs: {filename}")

    assignment = require_index(csv_rows(Path(manifest["input_files"]["assignment"]["path"])),
                               ("query_id",), 1126)
    assigned_groups = {}
    for (query,), row in assignment.items():
        fold = int(row["outer_fold"])
        group = row["group_id"]
        if (fold not in range(5) or row["citation_eligible"] != "1"
                or group in assigned_groups and assigned_groups[group] != fold):
            raise ValueError(f"invalid frozen group assignment: {query}")
        assigned_groups[group] = fold
    if len(assigned_groups) != 908:
        raise ValueError("frozen group count differs")
    truth = {}
    for row in jsonl_rows(Path(manifest["input_files"]["ground_truth"]["path"])):
        query = row["query_patent_id"]
        if query in truth:
            raise ValueError(f"duplicate truth query: {query}")
        truth[query] = {"Citation": set(row["all_cited_patents"]),
                        "Conflict-X": set(row["conflict_patents"])}
    required = {key[0] for key in assignment}
    text, text_extras = parse_rankings(Path(manifest["input_files"]["text_rankings"]["path"]),
                                       "source", set(SOURCES), required)
    structure, structure_extras = parse_rankings(
        Path(manifest["input_files"]["structure_rankings"]["path"]),
        "method", {"mean_morgan_cosine"}, required)
    if text_extras != 0 or structure_extras != 73:
        raise ValueError("source query coverage differs")

    tuning = require_index(csv_rows(base / FILES["tuning"]),
                           ("text_source", "outer_fold", "regularization_c"), 40)
    choices = require_index(csv_rows(base / FILES["choices"]),
                            ("text_source", "outer_fold"), 10)
    inner = require_index(csv_rows(base / FILES["inner"]),
                          ("outer_fold", "query_patent_id"), 4504)
    models = require_index(json.loads((base / FILES["models"]).read_text(encoding="utf-8")),
                           ("text_source", "outer_fold"), 10)
    stored_rankings = require_index(list(jsonl_rows(base / FILES["rankings"])),
                                    ("text_source", "query_patent_id"), 2252)
    stored_per_query = require_index(csv_rows(base / FILES["per_query"]),
                                    ("text_source", "task", "query_patent_id"), 2346)
    stored_summary = require_index(csv_rows(base / FILES["summary"]),
                                   ("text_source", "task"), 4)
    stored_folds = require_index(csv_rows(base / FILES["folds"]),
                                 ("text_source", "task", "outer_fold"), 20)
    max_numeric = 0.0
    max_score = 0.0
    checked_per_query = []

    for fold in range(5):
        train = sorted(query for query in required
                       if int(assignment[(query,)]["outer_fold"]) != fold)
        test = sorted(query for query in required
                      if int(assignment[(query,)]["outer_fold"]) == fold)
        fit, development = split_inner(train, truth, fold)
        fit_set = set(fit)
        fit_groups = {assignment[(q,)]["group_id"] for q in fit}
        dev_groups = {assignment[(q,)]["group_id"] for q in development}
        if (set(fit) & set(development) or set(fit) | set(development) != set(train)
                or fit_groups & dev_groups):
            raise ValueError(f"inner fold group leakage: {fold}")
        for query in train:
            row = inner[(str(fold), query)]
            expected_role = "fit" if query in fit_set else "development"
            if (row["inner_role"] != expected_role
                    or row["group_id"] != assignment[(query,)]["group_id"]):
                raise ValueError(f"inner role/group differs: {fold}:{query}")
        if any(key[0] == str(fold) and key[1] in test for key in inner):
            raise ValueError(f"outer test appears in inner split: {fold}")
        for source in SOURCES:
            choice = choices[(source, str(fold))]
            fit_seed = seed("outer", fold, "M4", source, "fit")
            x, y, fit_counts = pair_data(fit, source, truth, text, structure, fit_seed)
            if (int(choice["fit_query_count"]) != len(fit)
                    or int(choice["development_query_count"]) != len(development)
                    or int(choice["outer_train_query_count"]) != len(train)
                    or int(choice["outer_test_query_count"]) != len(test)
                    or int(choice["fit_group_count"]) != len(fit_groups)
                    or int(choice["development_group_count"]) != len(dev_groups)
                    or int(choice["fit_pair_rows"]) != len(y)
                    or int(choice["fit_sampling_seed"]) != fit_seed
                    or int(choice["inner_split_seed"]) != seed("outer", fold, "m4_inner")):
                raise ValueError(f"M4 fit metadata differs: {source}:{fold}")
            dev_features = {query: features(text[source][query], structure["mean_morgan_cosine"][query])
                            for query in development}
            curve = {}
            for c in CS:
                scaler, model = fit_model(x, y, c, seed("outer", fold, "M4", source, "fit", c))
                mean = math.fsum(metrics(rank(*dev_features[query], scaler, model)[0],
                                         truth[query]["Citation"])["MRR@100"]
                                 for query in development) / len(development)
                curve[c] = mean
                row = tuning[(source, str(fold), str(c))]
                max_numeric = max(max_numeric, compare(row["development_Citation_MRR@100"],
                                                       mean, f"tuning:{source}:{fold}:{c}"))
            best = max(curve.values())
            chosen = min(c for c, value in curve.items() if abs(value - best) <= 1e-12)
            if float(choice["selected_c"]) != chosen:
                raise ValueError(f"C choice differs: {source}:{fold}")
            max_numeric = max(max_numeric, compare(choice["selected_development_Citation_MRR@100"],
                                                   curve[chosen], f"choice:{source}:{fold}"))
            for c in CS:
                if (tuning[(source, str(fold), str(c))]["selected"] == "True") != (c == chosen):
                    raise ValueError(f"tuning selected flag differs: {source}:{fold}:{c}")
            del x, y, dev_features

            refit_seed = seed("outer", fold, "M4", source, "refit")
            x, y, refit_counts = pair_data(train, source, truth, text, structure, refit_seed)
            model_seed = seed("outer", fold, "M4", source, "refit", chosen)
            scaler, model = fit_model(x, y, chosen, model_seed)
            stored_model = models[(source, fold)]
            if (stored_model["selected_c"] != chosen
                    or stored_model["fit_sampling_seed"] != fit_seed
                    or stored_model["refit_seed"] != refit_seed
                    or stored_model["model_seed"] != model_seed
                    or stored_model["n_iter"] != int(model.n_iter_[0])):
                raise ValueError(f"refit metadata differs: {source}:{fold}")
            for key, value in refit_counts.items():
                if stored_model[key] != value:
                    raise ValueError(f"refit count differs: {source}:{fold}:{key}")
            for label, expected in (("scaler_mean", scaler.mean_),
                                    ("scaler_scale", scaler.scale_),
                                    ("coefficients", model.coef_[0])):
                if not np.allclose(stored_model[label], expected, atol=1e-10, rtol=0):
                    raise ValueError(f"refit {label} differs: {source}:{fold}")
            del x, y
            method = f"pairwise_logistic__{source}__mean_morgan_cosine"
            for query in test:
                ids, matrix = features(text[source][query], structure["mean_morgan_cosine"][query])
                ordered, score_by_index = rank(ids, matrix, scaler, model)
                score_lookup = dict(zip(ids, score_by_index))
                ranking_row = stored_rankings[(source, query)]
                if (ranking_row["method"] != method or ranking_row["outer_fold"] != fold
                        or ranking_row["selected_c"] != chosen
                        or len(ranking_row["ranking"]) != 100):
                    raise ValueError(f"stored ranking metadata differs: {source}:{query}")
                for rank_number, item in enumerate(ranking_row["ranking"], 1):
                    if item["rank"] != rank_number or item["patent_id"] != ordered[rank_number - 1]:
                        raise ValueError(f"ranking order differs: {source}:{query}:{rank_number}")
                    diff = compare(item["score"], float(score_lookup[item["patent_id"]]),
                                   f"rank score:{source}:{query}:{rank_number}", 1e-11)
                    max_score = max(max_score, diff)
                for task in TASKS:
                    if task == "Conflict-X" and assignment[(query,)]["conflict_x_eligible"] != "1":
                        continue
                    gold = truth[query][task]
                    row = stored_per_query[(source, task, query)]
                    if (row["method"] != method or row["group_id"] != assignment[(query,)]["group_id"]
                            or int(row["outer_fold"]) != fold or float(row["selected_c"]) != chosen
                            or int(row["gold_count"]) != len(gold)
                            or int(row["candidate_count_ranked"]) != 2217):
                        raise ValueError(f"per-query metadata differs: {source}:{task}:{query}")
                    computed = metrics(ordered, gold)
                    for metric in METRICS:
                        max_numeric = max(max_numeric, compare(row[metric], computed[metric],
                                                               f"per-query:{source}:{task}:{query}:{metric}"))
                    checked_per_query.append((source, task, fold, assignment[(query,)]["group_id"], computed))
            print(f"validated M4 outer fold {fold}/4, {source}", flush=True)

    if len(checked_per_query) != 2346:
        raise ValueError("M4 per-query coverage differs")
    for by_fold, records in ((False, stored_summary), (True, stored_folds)):
        grouped = defaultdict(list)
        for source, task, fold, group, values in checked_per_query:
            key = (source, task, fold) if by_fold else (source, task)
            grouped[key].append((group, values))
        if len(grouped) != (20 if by_fold else 4):
            raise ValueError("M4 aggregate coverage differs")
        for key, members in grouped.items():
            stored_key = tuple(str(item) for item in key)
            row = records[stored_key]
            if (int(row["query_count"]) != len(members)
                    or int(row["group_count"]) != len({group for group, _ in members})
                    or int(row["candidate_count_ranked"]) != 2217):
                raise ValueError(f"aggregate count differs: {stored_key}")
            for metric in METRICS:
                mean = math.fsum(values[metric] for _, values in members) / len(members)
                max_numeric = max(max_numeric, compare(row[metric], mean,
                                                       f"aggregate:{stored_key}:{metric}"))
    print("independent_m4_5fold_validation: PASS")
    print("C_choices_checked: 10")
    print("tuning_rows_checked: 40")
    print("ranking_rows_checked: 2252")
    print("per_query_rows_checked: 2346")
    print("maximum_metric_abs_difference:", max_numeric)
    print("maximum_score_abs_difference:", max_score)
    print("input_and_output_hashes: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
