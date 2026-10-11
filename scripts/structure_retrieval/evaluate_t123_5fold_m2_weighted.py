#!/usr/bin/env python3
"""Select M2 alpha on four frozen folds, then evaluate each held-out fold."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluate_t123_5fold_m1_rrf import (
    ASSIGNMENT,
    ASSIGNMENT_MANIFEST,
    CANDIDATE_COUNT,
    FIVEFOLD,
    GROUND_TRUTH,
    STANDALONE,
    STANDALONE_MANIFEST,
    STRUCTURE_METHOD,
    STRUCTURE_PER_QUERY,
    STRUCTURE_RANKINGS,
    TEXT_RANKINGS,
    TEXT_SOURCES,
    file_record,
    summarize,
    verify_inputs,
    write_csv,
)
from evaluate_t3_rrf import METRICS, TASKS, evaluate_ranking, load_ground_truth
from evaluate_t3_weighted_fusion import (
    fuse_weighted_scores,
    load_score_rankings,
    prepare_query_scores,
    select_alpha,
)


OUTPUT = FIVEFOLD / "m2_weighted"
ALPHAS = tuple(round(index / 10, 1) for index in range(11))


def choose_alpha(rows: list[dict[str, float]]) -> float:
    return select_alpha(rows)[0]


def self_test() -> None:
    selected = choose_alpha([
        {"alpha": 0.4, "MRR@100": 0.5},
        {"alpha": 0.6, "MRR@100": 0.5},
    ])
    if selected != 0.6:
        raise AssertionError("alpha tie rule differs")
    text = [("EP1A1", 1.0), ("EP2A1", 0.0), ("EP3A1", -1.0)]
    structure = [("EP3A1", 1.0), ("EP2A1", 0.0), ("EP1A1", -1.0)]
    ids, text_z, structure_z, structure_ids = prepare_query_scores(text, structure)
    if [item[0] for item in fuse_weighted_scores(ids, text_z, structure_z, structure_ids, 1.0)] != ids:
        raise AssertionError("alpha=1 endpoint differs")
    if [item[0] for item in fuse_weighted_scores(ids, text_z, structure_z, structure_ids, 0.0)] != structure_ids:
        raise AssertionError("alpha=0 endpoint differs")
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
        raise ValueError("ranking models differ from frozen M2 inputs")
    if any(set(text_rankings[source]) != set(assignment) for source in TEXT_SOURCES):
        raise ValueError("text query coverage mismatch")
    if not set(assignment).issubset(structure_rankings[STRUCTURE_METHOD]):
        raise ValueError("structure query coverage mismatch")

    prepared = {}
    development_mrr = {}
    for source in TEXT_SOURCES:
        for query in sorted(assignment):
            scores = prepare_query_scores(text_rankings[source][query],
                                          structure_rankings[STRUCTURE_METHOD][query])
            if len(scores[0]) != CANDIDATE_COUNT:
                raise ValueError(f"candidate count mismatch: {source}:{query}")
            gold = truth[query]["Citation"]
            if not gold or not gold.issubset(scores[0]):
                raise ValueError(f"Citation gold outside candidates: {source}:{query}")
            prepared[(source, query)] = scores
            for alpha in ALPHAS:
                fused = fuse_weighted_scores(*scores, alpha)
                development_mrr[(source, query, alpha)] = evaluate_ranking(fused, gold)["MRR@100"]

    tuning = []
    choices = []
    selected = {}
    for source in TEXT_SOURCES:
        for outer_fold in range(5):
            train = [query for query in sorted(assignment)
                     if int(assignment[query]["outer_fold"]) != outer_fold]
            test = [query for query in sorted(assignment)
                    if int(assignment[query]["outer_fold"]) == outer_fold]
            if len(train) + len(test) != 1126 or not train or not test:
                raise ValueError(f"invalid outer fold: {outer_fold}")
            curve = []
            for alpha in ALPHAS:
                mean = math.fsum(development_mrr[(source, query, alpha)] for query in train) / len(train)
                curve.append({"alpha": alpha, "MRR@100": mean})
            chosen = choose_alpha(curve)
            selected[(source, outer_fold)] = chosen
            for point in curve:
                tuning.append({"text_source": source, "outer_fold": outer_fold,
                               "training_query_count": len(train), "test_query_count": len(test),
                               "alpha": point["alpha"], "training_Citation_MRR@100": point["MRR@100"],
                               "selected": point["alpha"] == chosen})
            choices.append({"text_source": source, "outer_fold": outer_fold,
                            "training_query_count": len(train), "test_query_count": len(test),
                            "selected_alpha": chosen,
                            "selected_training_Citation_MRR@100": next(
                                point["MRR@100"] for point in curve if point["alpha"] == chosen)})

    per_query = []
    rankings = []
    for source in TEXT_SOURCES:
        method = f"zscore_weighted__{source}__{STRUCTURE_METHOD}"
        for query in sorted(assignment):
            meta = assignment[query]
            outer_fold = int(meta["outer_fold"])
            alpha = selected[(source, outer_fold)]
            fused = fuse_weighted_scores(*prepared[(source, query)], alpha)
            rankings.append({
                "method": method, "text_source": source, "query_patent_id": query,
                "outer_fold": outer_fold, "alpha": alpha,
                "ranking": [{"rank": rank, "patent_id": patent, "score": round(score, 12)}
                            for rank, (patent, score) in enumerate(fused[:100], 1)],
            })
            for task in TASKS:
                if task == "Conflict-X" and meta["conflict_x_eligible"] != "1":
                    continue
                gold = truth[query][task]
                if (not gold or not gold.issubset(prepared[(source, query)][0])
                        or int(baseline[(source, task, query)]["gold_count"]) != len(gold)
                        or int(structure_baseline[(task, query)]["gold_count"]) != len(gold)):
                    raise ValueError(f"gold/cohort mismatch: {source}:{task}:{query}")
                per_query.append({
                    "method": method, "text_source": source, "structure_method": STRUCTURE_METHOD,
                    "query_patent_id": query, "group_id": meta["group_id"],
                    "outer_fold": outer_fold, "alpha": alpha, "task": task,
                    "gold_count": len(gold), "candidate_count_ranked": CANDIDATE_COUNT,
                    **evaluate_ranking(fused, gold),
                })
    per_query.sort(key=lambda row: (row["method"], row["task"], row["query_patent_id"]))
    summary = summarize(per_query, False)
    folds = summarize(per_query, True)
    if (len(tuning) != 110 or len(choices) != 10 or len(rankings) != 2252
            or len(per_query) != 2346 or len(summary) != 4 or len(folds) != 20):
        raise ValueError("M2 output cohort incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tuning_path = args.output_dir / "t123_m2_alpha_tuning.csv"
    choices_path = args.output_dir / "t123_m2_alpha_choices.csv"
    per_query_path = args.output_dir / "t123_m2_weighted_oof_per_query.csv"
    summary_path = args.output_dir / "t123_m2_weighted_oof_summary.csv"
    folds_path = args.output_dir / "t123_m2_weighted_oof_folds.csv"
    rankings_path = args.output_dir / "t123_m2_weighted_rankings_top100.jsonl"
    write_csv(tuning_path, ["text_source", "outer_fold", "training_query_count", "test_query_count",
                            "alpha", "training_Citation_MRR@100", "selected"], tuning)
    write_csv(choices_path, ["text_source", "outer_fold", "training_query_count", "test_query_count",
                             "selected_alpha", "selected_training_Citation_MRR@100"], choices)
    common = ["method", "text_source", "task"]
    write_csv(per_query_path, common + ["structure_method", "query_patent_id", "group_id",
                                         "outer_fold", "alpha", "gold_count", "candidate_count_ranked",
                                         *METRICS], per_query)
    write_csv(summary_path, common + ["query_count", "group_count", "candidate_count_ranked", *METRICS], summary)
    write_csv(folds_path, common + ["outer_fold", "query_count", "group_count",
                                     "candidate_count_ranked", *METRICS], folds)
    with rankings_path.open("x", encoding="utf-8") as handle:
        for row in rankings:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "completed", "method": "M2 query-zscore weighted fusion",
        "selection_rule": "maximize Citation training-fold MRR@100; within 1e-12 tie choose alpha nearest 0.5, then larger alpha",
        "alpha_grid": list(ALPHAS), "alpha_choices": choices,
        "held_out_fold_policy": "For each outer fold, select alpha on all other four folds; never use held-out gold for selection.",
        "scoring_rule": "alpha * population_zscore(text) + (1-alpha) * population_zscore(original_full_Morgan)",
        "text_sources": list(TEXT_SOURCES), "structure_method": STRUCTURE_METHOD,
        "query_counts": {"Citation": 1126, "Conflict-X": 47},
        "candidate_count_ranked": CANDIDATE_COUNT,
        "historical_selection_caveat": "The benchmark and methods were inspected before five-fold evaluation; this is not untouched external validation.",
        "oracle_candidate_pool_caveat": "Closed gold-derived candidate pool, not open-corpus retrieval.",
        "input_files": {label: file_record(path) for label, path in (
            ("assignment", ASSIGNMENT), ("assignment_manifest", ASSIGNMENT_MANIFEST),
            ("standalone", STANDALONE), ("standalone_manifest", STANDALONE_MANIFEST),
            ("ground_truth", GROUND_TRUTH), ("text_rankings", TEXT_RANKINGS),
            ("structure_rankings", STRUCTURE_RANKINGS),
            ("structure_per_query", STRUCTURE_PER_QUERY))},
        "output_files": {path.name: file_record(path) for path in
                         (tuning_path, choices_path, per_query_path, summary_path, folds_path, rankings_path)},
        "provenance": file_record(Path(__file__)),
    }
    manifest_path = args.output_dir / "t123_m2_weighted_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "completed", "alpha_choices": choices,
                      "tuning_rows": len(tuning), "ranking_rows": len(rankings),
                      "per_query_rows": len(per_query), "summary_rows": len(summary),
                      "fold_rows": len(folds), "manifest": str(manifest_path)},
                     ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
