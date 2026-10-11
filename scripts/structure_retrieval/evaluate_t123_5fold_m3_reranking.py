#!/usr/bin/env python3
"""Select M3 alpha within four frozen folds for each fixed text Top-N."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

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
from evaluate_t3_text_first_reranking import zscore_or_zero
from evaluate_t3_weighted_fusion import load_score_rankings, select_alpha


OUTPUT = FIVEFOLD / "m3_text_first_reranking"
TOP_N_VALUES = (100, 500)
ALPHAS = tuple(round(index / 10, 1) for index in range(11))


def prepare_query(text: list[tuple[str, float]], structure: list[tuple[str, float]], top_n: int):
    text_ids = [patent for patent, _ in text]
    structure_ids = [patent for patent, _ in structure]
    if len(text_ids) != CANDIDATE_COUNT or set(text_ids) != set(structure_ids):
        raise ValueError("T1/T2 full candidate IDs differ")
    head = text[:top_n]
    head_ids = [patent for patent, _ in head]
    structure_scores = dict(structure)
    text_z, text_zero = zscore_or_zero(np.asarray([score for _, score in head], dtype=np.float64))
    structure_z, structure_zero = zscore_or_zero(np.asarray(
        [structure_scores[patent] for patent in head_ids], dtype=np.float64))
    return text, head_ids, text_z, structure_z, text_zero, structure_zero


def rerank(prepared: tuple, alpha: float) -> list[tuple[str, float]]:
    text, head_ids, text_z, structure_z, _, _ = prepared
    top_n = len(head_ids)
    fused_scores = alpha * text_z + (1.0 - alpha) * structure_z
    if math.isclose(alpha, 1.0, abs_tol=1e-15):
        order = np.arange(top_n)
    else:
        order = np.lexsort((np.asarray(head_ids, dtype=str), np.arange(top_n), -fused_scores))
    return ([(head_ids[index], float(fused_scores[index])) for index in order]
            + [(patent, float(score)) for patent, score in text[top_n:]])


def self_test() -> None:
    if ALPHAS[0] != 0.0 or ALPHAS[-1] != 1.0 or TOP_N_VALUES != (100, 500):
        raise AssertionError("fixed grid differs")
    zero, flag = zscore_or_zero(np.asarray([1.0, 1.0]))
    if not flag or not np.array_equal(zero, np.zeros(2)):
        raise AssertionError("zero-variance policy differs")
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
        raise ValueError("ranking models differ from frozen M3 inputs")
    if any(set(text_rankings[source]) != set(assignment) for source in TEXT_SOURCES):
        raise ValueError("text query coverage mismatch")
    if not set(assignment).issubset(structure_rankings[STRUCTURE_METHOD]):
        raise ValueError("structure query coverage mismatch")

    prepared = {}
    development_mrr = {}
    for source in TEXT_SOURCES:
        for query in sorted(assignment):
            gold = truth[query]["Citation"]
            if not gold:
                raise ValueError(f"empty Citation gold: {query}")
            for top_n in TOP_N_VALUES:
                record = prepare_query(text_rankings[source][query],
                                       structure_rankings[STRUCTURE_METHOD][query], top_n)
                if not gold.issubset({patent for patent, _ in record[0]}):
                    raise ValueError(f"Citation gold outside candidates: {source}:{query}")
                prepared[(source, top_n, query)] = record
                for alpha in ALPHAS:
                    ranking = rerank(record, alpha)
                    development_mrr[(source, top_n, query, alpha)] = evaluate_ranking(
                        ranking, gold)["MRR@100"]

    tuning = []
    choices = []
    selected = {}
    for source in TEXT_SOURCES:
        for top_n in TOP_N_VALUES:
            for outer_fold in range(5):
                train = [query for query in sorted(assignment)
                         if int(assignment[query]["outer_fold"]) != outer_fold]
                test = [query for query in sorted(assignment)
                        if int(assignment[query]["outer_fold"]) == outer_fold]
                curve = []
                for alpha in ALPHAS:
                    mean = math.fsum(development_mrr[(source, top_n, query, alpha)]
                                     for query in train) / len(train)
                    curve.append({"alpha": alpha, "MRR@100": mean})
                chosen = select_alpha(curve)[0]
                selected[(source, top_n, outer_fold)] = chosen
                train_text_zero = sum(prepared[(source, top_n, query)][4] for query in train)
                train_structure_zero = sum(prepared[(source, top_n, query)][5] for query in train)
                test_text_zero = sum(prepared[(source, top_n, query)][4] for query in test)
                test_structure_zero = sum(prepared[(source, top_n, query)][5] for query in test)
                for point in curve:
                    tuning.append({"text_source": source, "top_n": top_n,
                                   "outer_fold": outer_fold, "training_query_count": len(train),
                                   "test_query_count": len(test), "alpha": point["alpha"],
                                   "training_Citation_MRR@100": point["MRR@100"],
                                   "selected": point["alpha"] == chosen})
                choices.append({"text_source": source, "top_n": top_n,
                                "outer_fold": outer_fold, "training_query_count": len(train),
                                "test_query_count": len(test), "selected_alpha": chosen,
                                "selected_training_Citation_MRR@100": next(
                                    point["MRR@100"] for point in curve if point["alpha"] == chosen),
                                "train_text_zero_variance_queries": train_text_zero,
                                "train_structure_zero_variance_queries": train_structure_zero,
                                "test_text_zero_variance_queries": test_text_zero,
                                "test_structure_zero_variance_queries": test_structure_zero})

    per_query = []
    rankings = []
    for source in TEXT_SOURCES:
        for top_n in TOP_N_VALUES:
            method = f"text_top{top_n}_weighted__{source}__{STRUCTURE_METHOD}"
            for query in sorted(assignment):
                meta = assignment[query]
                fold = int(meta["outer_fold"])
                alpha = selected[(source, top_n, fold)]
                result = rerank(prepared[(source, top_n, query)], alpha)
                rankings.append({"method": method, "text_source": source,
                                 "top_n": top_n, "query_patent_id": query,
                                 "outer_fold": fold, "alpha": alpha,
                                 "ranking": [{"rank": rank, "patent_id": patent,
                                              "score": round(score, 12)}
                                             for rank, (patent, score) in enumerate(result[:100], 1)]})
                for task in TASKS:
                    if task == "Conflict-X" and meta["conflict_x_eligible"] != "1":
                        continue
                    gold = truth[query][task]
                    if (not gold or not gold.issubset({patent for patent, _ in result})
                            or int(baseline[(source, task, query)]["gold_count"]) != len(gold)
                            or int(structure_baseline[(task, query)]["gold_count"]) != len(gold)):
                        raise ValueError(f"gold/cohort mismatch: {source}:{task}:{query}")
                    per_query.append({
                        "method": method, "text_source": source, "structure_method": STRUCTURE_METHOD,
                        "top_n": top_n, "query_patent_id": query, "group_id": meta["group_id"],
                        "outer_fold": fold, "alpha": alpha, "task": task,
                        "gold_count": len(gold), "candidate_count_ranked": CANDIDATE_COUNT,
                        **evaluate_ranking(result, gold),
                    })
    per_query.sort(key=lambda row: (row["method"], row["task"], row["query_patent_id"]))
    summary = summarize(per_query, False)
    folds = summarize(per_query, True)
    if (len(tuning) != 220 or len(choices) != 20 or len(rankings) != 4504
            or len(per_query) != 4692 or len(summary) != 8 or len(folds) != 40):
        raise ValueError("M3 output cohort incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tuning_path = args.output_dir / "t123_m3_alpha_tuning.csv"
    choices_path = args.output_dir / "t123_m3_alpha_choices.csv"
    per_query_path = args.output_dir / "t123_m3_reranking_oof_per_query.csv"
    summary_path = args.output_dir / "t123_m3_reranking_oof_summary.csv"
    folds_path = args.output_dir / "t123_m3_reranking_oof_folds.csv"
    rankings_path = args.output_dir / "t123_m3_reranking_rankings_top100.jsonl"
    write_csv(tuning_path, ["text_source", "top_n", "outer_fold", "training_query_count",
                            "test_query_count", "alpha", "training_Citation_MRR@100", "selected"], tuning)
    write_csv(choices_path, ["text_source", "top_n", "outer_fold", "training_query_count",
                             "test_query_count", "selected_alpha", "selected_training_Citation_MRR@100",
                             "train_text_zero_variance_queries", "train_structure_zero_variance_queries",
                             "test_text_zero_variance_queries", "test_structure_zero_variance_queries"], choices)
    common = ["method", "text_source", "task"]
    write_csv(per_query_path, common + ["structure_method", "top_n", "query_patent_id",
                                         "group_id", "outer_fold", "alpha", "gold_count",
                                         "candidate_count_ranked", *METRICS], per_query)
    write_csv(summary_path, common + ["query_count", "group_count", "candidate_count_ranked", *METRICS], summary)
    write_csv(folds_path, common + ["outer_fold", "query_count", "group_count",
                                     "candidate_count_ranked", *METRICS], folds)
    with rankings_path.open("x", encoding="utf-8") as handle:
        for row in rankings:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "completed", "method": "M3 text-first Top-N reranking",
        "top_n_values": list(TOP_N_VALUES), "alpha_grid": list(ALPHAS),
        "top_n_policy": "Report Top-100 and Top-500 separately; never choose Top-N using held-out results.",
        "selection_rule": "For each source and Top-N, maximize Citation MRR@100 on the other four folds; 1e-12 tie nearest 0.5 then larger alpha.",
        "alpha_choices": choices,
        "zero_variance_policy": "Within Top-N, use a zero vector for a constant modality score; record query counts.",
        "text_sources": list(TEXT_SOURCES), "structure_method": STRUCTURE_METHOD,
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
        "output_files": {path.name: file_record(path) for path in
                         (tuning_path, choices_path, per_query_path, summary_path, folds_path, rankings_path)},
        "provenance": file_record(Path(__file__)),
    }
    manifest_path = args.output_dir / "t123_m3_reranking_manifest.json"
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
