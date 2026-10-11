#!/usr/bin/env python3
"""Fit pairwise logistic fusion within frozen outer folds and emit OOF results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluate_t123_5fold_m1_rrf import (
    ASSIGNMENT, ASSIGNMENT_MANIFEST, CANDIDATE_COUNT, FIVEFOLD, GROUND_TRUTH,
    STANDALONE, STANDALONE_MANIFEST, STRUCTURE_METHOD, STRUCTURE_PER_QUERY,
    STRUCTURE_RANKINGS, TEXT_RANKINGS, TEXT_SOURCES, file_record, summarize,
    verify_inputs, write_csv,
)
from evaluate_t3_learned_fusion import (
    FEATURE_NAMES, build_pairwise_training_data, candidate_features, choose_c,
    rank_candidates, split_training_queries, stable_seed, train_model,
)
from evaluate_t3_rrf import METRICS, TASKS, evaluate_ranking, load_ground_truth
from evaluate_t3_weighted_fusion import load_score_rankings


OUTPUT = FIVEFOLD / "m4_pairwise_logistic"
BASE_SEED = 20260924
C_GRID = (0.01, 0.1, 1.0, 10.0)
DEVELOPMENT_FRACTION = 0.2
HARD_NEGATIVES = 50
RANDOM_NEGATIVES = 50


def partition(assignment: dict, outer_fold: int) -> tuple[list[str], list[str]]:
    train = sorted(query for query, row in assignment.items()
                   if int(row["outer_fold"]) != outer_fold)
    test = sorted(query for query, row in assignment.items()
                  if int(row["outer_fold"]) == outer_fold)
    if len(train) + len(test) != 1126 or not train or not test:
        raise ValueError(f"invalid outer fold: {outer_fold}")
    if {assignment[q]["group_id"] for q in train} & {assignment[q]["group_id"] for q in test}:
        raise ValueError(f"group crosses outer fold: {outer_fold}")
    return train, test


def inner_partition(train: list[str], truth: dict, assignment: dict,
                    outer_fold: int) -> tuple[list[str], list[str], dict]:
    seed = stable_seed(BASE_SEED, "outer", outer_fold, "m4_inner")
    fit, development, details = split_training_queries(
        train, truth, DEVELOPMENT_FRACTION, seed)
    fit_groups = {assignment[q]["group_id"] for q in fit}
    development_groups = {assignment[q]["group_id"] for q in development}
    if fit_groups & development_groups:
        raise ValueError(f"frozen connected group crosses inner split: {outer_fold}")
    if set(fit) | set(development) != set(train):
        raise ValueError(f"inner split incomplete: {outer_fold}")
    return fit, development, {
        **details, "split_seed": seed, "fit_group_count": len(fit_groups),
        "development_group_count": len(development_groups),
    }


def self_test() -> None:
    rows = [{"regularization_c": 0.01, "MRR@100": 0.5},
            {"regularization_c": 1.0, "MRR@100": 0.5}]
    if choose_c(rows) != 0.01 or len(FEATURE_NAMES) != 5:
        raise AssertionError("M4 selection or features changed")
    if stable_seed(BASE_SEED, "a") == stable_seed(BASE_SEED, "b"):
        raise AssertionError("stable seed collision")
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
    truth = load_ground_truth(GROUND_TRUTH, "conflict_patents")
    text_rankings, text_models = load_score_rankings(TEXT_RANKINGS, "source")
    structure_rankings, structure_models = load_score_rankings(STRUCTURE_RANKINGS, "method")
    if text_models != set(TEXT_SOURCES) or structure_models != {STRUCTURE_METHOD}:
        raise ValueError("ranking models differ from frozen M4 inputs")
    if any(set(text_rankings[source]) != set(assignment) for source in TEXT_SOURCES):
        raise ValueError("text query coverage mismatch")
    if not set(assignment).issubset(structure_rankings[STRUCTURE_METHOD]):
        raise ValueError("structure query coverage mismatch")

    tuning: list[dict] = []
    choices: list[dict] = []
    inner_rows: list[dict] = []
    model_rows: list[dict] = []
    per_query: list[dict] = []
    rankings: list[dict] = []
    for outer_fold in range(5):
        train, test = partition(assignment, outer_fold)
        fit, development, split_details = inner_partition(train, truth, assignment, outer_fold)
        for query in fit:
            inner_rows.append({"outer_fold": outer_fold, "query_patent_id": query,
                               "group_id": assignment[query]["group_id"], "inner_role": "fit"})
        for query in development:
            inner_rows.append({"outer_fold": outer_fold, "query_patent_id": query,
                               "group_id": assignment[query]["group_id"], "inner_role": "development"})
        for source in TEXT_SOURCES:
            method = f"pairwise_logistic__{source}__{STRUCTURE_METHOD}"
            fit_seed = stable_seed(BASE_SEED, "outer", outer_fold, "M4", source, "fit")
            fit_features, fit_labels, fit_counts = build_pairwise_training_data(
                fit, source, truth, text_rankings, structure_rankings,
                STRUCTURE_METHOD, HARD_NEGATIVES, RANDOM_NEGATIVES, fit_seed)
            development_features = {}
            for query in development:
                ids, features = candidate_features(text_rankings[source][query],
                                                   structure_rankings[STRUCTURE_METHOD][query])
                if len(ids) != CANDIDATE_COUNT or not truth[query]["Citation"].issubset(ids):
                    raise ValueError(f"development candidate/gold mismatch: {source}:{query}")
                development_features[query] = (ids, features)
            curve = []
            for c in C_GRID:
                model_seed = stable_seed(BASE_SEED, "outer", outer_fold, "M4", source, "fit", c)
                scaler, model = train_model(fit_features, fit_labels, c, model_seed)
                mrr = math.fsum(evaluate_ranking(
                    rank_candidates(*development_features[query], scaler, model),
                    truth[query]["Citation"])["MRR@100"] for query in development) / len(development)
                curve.append({"regularization_c": c, "MRR@100": mrr})
            selected_c = choose_c(curve)
            for point in curve:
                tuning.append({"text_source": source, "outer_fold": outer_fold,
                               "fit_query_count": len(fit), "development_query_count": len(development),
                               "regularization_c": point["regularization_c"],
                               "development_Citation_MRR@100": point["MRR@100"],
                               "selected": point["regularization_c"] == selected_c})
            choices.append({"text_source": source, "outer_fold": outer_fold,
                            "outer_train_query_count": len(train), "outer_test_query_count": len(test),
                            "fit_query_count": len(fit), "development_query_count": len(development),
                            "fit_group_count": split_details["fit_group_count"],
                            "development_group_count": split_details["development_group_count"],
                            "inner_split_seed": split_details["split_seed"],
                            "fit_sampling_seed": fit_seed,
                            "fit_pair_rows": fit_counts["symmetric_pair_rows"],
                            "selected_c": selected_c,
                            "selected_development_Citation_MRR@100": next(
                                point["MRR@100"] for point in curve
                                if point["regularization_c"] == selected_c)})
            del fit_features, fit_labels, development_features

            refit_seed = stable_seed(BASE_SEED, "outer", outer_fold, "M4", source, "refit")
            refit_features, refit_labels, refit_counts = build_pairwise_training_data(
                train, source, truth, text_rankings, structure_rankings,
                STRUCTURE_METHOD, HARD_NEGATIVES, RANDOM_NEGATIVES, refit_seed)
            model_seed = stable_seed(BASE_SEED, "outer", outer_fold, "M4", source,
                                     "refit", selected_c)
            scaler, model = train_model(refit_features, refit_labels, selected_c, model_seed)
            model_rows.append({"text_source": source, "outer_fold": outer_fold,
                               "selected_c": selected_c, "refit_seed": refit_seed,
                               "fit_sampling_seed": fit_seed,
                               "model_seed": model_seed, **refit_counts,
                               "scaler_mean": [float(value) for value in scaler.mean_],
                               "scaler_scale": [float(value) for value in scaler.scale_],
                               "coefficients": [float(value) for value in model.coef_[0]],
                               "n_iter": int(model.n_iter_[0])})
            del refit_features, refit_labels
            for query in test:
                ids, features = candidate_features(text_rankings[source][query],
                                                   structure_rankings[STRUCTURE_METHOD][query])
                if len(ids) != CANDIDATE_COUNT:
                    raise ValueError(f"test candidate count mismatch: {source}:{query}")
                ranked = rank_candidates(ids, features, scaler, model)
                rankings.append({"method": method, "text_source": source,
                                 "query_patent_id": query, "outer_fold": outer_fold,
                                 "selected_c": selected_c,
                                 "ranking": [{"rank": rank, "patent_id": patent,
                                              "score": round(score, 12)}
                                             for rank, (patent, score) in enumerate(ranked[:100], 1)]})
                for task in TASKS:
                    if task == "Conflict-X" and assignment[query]["conflict_x_eligible"] != "1":
                        continue
                    gold = truth[query][task]
                    if (not gold or not gold.issubset(ids)
                            or int(baseline[(source, task, query)]["gold_count"]) != len(gold)
                            or int(structure_baseline[(task, query)]["gold_count"]) != len(gold)):
                        raise ValueError(f"test gold/cohort mismatch: {source}:{task}:{query}")
                    per_query.append({"method": method, "text_source": source,
                                      "structure_method": STRUCTURE_METHOD,
                                      "query_patent_id": query, "group_id": assignment[query]["group_id"],
                                      "outer_fold": outer_fold, "selected_c": selected_c,
                                      "task": task, "gold_count": len(gold),
                                      "candidate_count_ranked": CANDIDATE_COUNT,
                                      **evaluate_ranking(ranked, gold)})
            print(f"M4 outer fold {outer_fold}/4, {source}: selected C={selected_c}", flush=True)

    per_query.sort(key=lambda row: (row["method"], row["task"], row["query_patent_id"]))
    rankings.sort(key=lambda row: (row["method"], row["query_patent_id"]))
    inner_rows.sort(key=lambda row: (row["outer_fold"], row["query_patent_id"]))
    summary = summarize(per_query, False)
    folds = summarize(per_query, True)
    if (len(tuning) != 40 or len(choices) != 10 or len(model_rows) != 10
            or len(inner_rows) != 4504 or len(rankings) != 2252
            or len(per_query) != 2346 or len(summary) != 4 or len(folds) != 20):
        raise ValueError("M4 output cohort incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tuning_path = args.output_dir / "t123_m4_c_tuning.csv"
    choices_path = args.output_dir / "t123_m4_c_choices.csv"
    inner_path = args.output_dir / "t123_m4_inner_splits.csv"
    model_path = args.output_dir / "t123_m4_models.json"
    per_query_path = args.output_dir / "t123_m4_learned_oof_per_query.csv"
    summary_path = args.output_dir / "t123_m4_learned_oof_summary.csv"
    folds_path = args.output_dir / "t123_m4_learned_oof_folds.csv"
    rankings_path = args.output_dir / "t123_m4_learned_rankings_top100.jsonl"
    write_csv(tuning_path, ["text_source", "outer_fold", "fit_query_count",
                            "development_query_count", "regularization_c",
                            "development_Citation_MRR@100", "selected"], tuning)
    write_csv(choices_path, ["text_source", "outer_fold", "outer_train_query_count",
                             "outer_test_query_count", "fit_query_count", "development_query_count",
                             "fit_group_count", "development_group_count", "inner_split_seed",
                             "fit_sampling_seed", "fit_pair_rows", "selected_c",
                             "selected_development_Citation_MRR@100"], choices)
    write_csv(inner_path, ["outer_fold", "query_patent_id", "group_id", "inner_role"], inner_rows)
    with model_path.open("x", encoding="utf-8") as handle:
        json.dump(model_rows, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    common = ["method", "text_source", "task"]
    write_csv(per_query_path, common + ["structure_method", "query_patent_id", "group_id",
                                         "outer_fold", "selected_c", "gold_count",
                                         "candidate_count_ranked", *METRICS], per_query)
    write_csv(summary_path, common + ["query_count", "group_count", "candidate_count_ranked", *METRICS], summary)
    write_csv(folds_path, common + ["outer_fold", "query_count", "group_count",
                                     "candidate_count_ranked", *METRICS], folds)
    with rankings_path.open("x", encoding="utf-8") as handle:
        for row in rankings:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "completed", "method": "M4 pairwise logistic learned fusion",
        "text_sources": list(TEXT_SOURCES), "structure_method": STRUCTURE_METHOD,
        "feature_names": list(FEATURE_NAMES), "base_seed": BASE_SEED,
        "classifier": {"type": "LogisticRegression", "solver": "liblinear",
                       "fit_intercept": False, "max_iter": 1000,
                       "feature_scaling": "StandardScaler fit on training pairs only"},
        "c_grid": list(C_GRID), "development_fraction": DEVELOPMENT_FRACTION,
        "negative_sampling": {"hard_per_modality": HARD_NEGATIVES,
                              "random_after_hard": RANDOM_NEGATIVES,
                              "note": "Non-gold candidates are sampled as negatives; unlabeled relevance is possible."},
        "selection_rule": "Maximize inner development Citation MRR@100; within 1e-12 tie choose smallest C.",
        "outer_test_policy": "Fit and tune using other four folds only; refit scaler/model on all outer training queries; evaluate held-out queries once.",
        "inner_split_policy": "Shared Citation gold connected components, checked against frozen group IDs.",
        "query_counts": {"Citation": 1126, "Conflict-X": 47},
        "candidate_count_ranked": CANDIDATE_COUNT,
        "historical_selection_caveat": "Methods and benchmark inspected before five-fold evaluation; not untouched external validation.",
        "oracle_candidate_pool_caveat": "Closed gold-derived candidate pool, not open-corpus retrieval.",
        "input_files": {label: file_record(path) for label, path in (
            ("assignment", ASSIGNMENT), ("assignment_manifest", ASSIGNMENT_MANIFEST),
            ("standalone", STANDALONE), ("standalone_manifest", STANDALONE_MANIFEST),
            ("ground_truth", GROUND_TRUTH), ("text_rankings", TEXT_RANKINGS),
            ("structure_rankings", STRUCTURE_RANKINGS),
            ("structure_per_query", STRUCTURE_PER_QUERY))},
        "output_files": {path.name: file_record(path) for path in (
            tuning_path, choices_path, inner_path, model_path, per_query_path,
            summary_path, folds_path, rankings_path)},
        "provenance": {"runner": file_record(Path(__file__)),
                       "training_helper": file_record(Path(__file__).with_name("evaluate_t3_learned_fusion.py"))},
    }
    manifest_path = args.output_dir / "t123_m4_learned_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "completed", "tuning_rows": len(tuning),
                      "choice_rows": len(choices), "ranking_rows": len(rankings),
                      "per_query_rows": len(per_query), "summary_rows": len(summary),
                      "fold_rows": len(folds), "manifest": str(manifest_path)},
                     ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
