#!/usr/bin/env python3
"""Read-only audit of extra T1/T2 candidate masks and M4 split feasibility."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

try:
    from .audit_t123_5fold_preflight import load_ground_truth, make_groups, require_id
    from .evaluate_t3_learned_fusion import split_training_queries, stable_seed
except ImportError:
    from audit_t123_5fold_preflight import load_ground_truth, make_groups, require_id
    from evaluate_t3_learned_fusion import split_training_queries, stable_seed


ROOT = Path("structure_retrieval_outputs")
CANDIDATE_PATH = ROOT / "t2_candidate_corpus_v1/text_candidate_patents_common.csv"
ORIGINAL_PATH = ROOT / "t2_surechembl/surechembl_structures.csv"
PREPARED_PATH = ROOT / "t2_molformer_input_v1/molformer_eligible_structures.csv"
TRUTH_PATH = Path("data/processed/retrieval_ground_truth.jsonl")
CORE_PATH = ROOT / "t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_per_query.csv"
CORE_RANKING_PATH = (
    ROOT / "t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_rankings_top2217.jsonl"
)

SPECS = (
    ("T1", "t1_matched_subset_v1", "t1_matched_subset_per_query.csv",
     "t1_matched_subset_rankings_top100.jsonl", "t1_matched_subset_manifest.json",
     "source", {"nv_document", "qwen_whole_claims"}, ORIGINAL_PATH),
    ("Morgan common", "t2_morgan_common_input_v1",
     "t2_structure_retrieval_per_query.csv", "t2_structure_rankings_top100.jsonl",
     "t2_structure_retrieval_manifest.json", "method", {"mean_morgan_cosine"},
     PREPARED_PATH),
    ("MoLFormer", "t2_molformer_mean_v1", "t2_molformer_per_query.csv",
     "t2_molformer_rankings_top100.jsonl", "t2_molformer_manifest.json",
     "method", {"molformer_mean_pooling"}, PREPARED_PATH),
    ("Tanimoto K16", "t2_representative_tanimoto_k16_v1",
     "t2_representative_tanimoto_per_query.csv",
     "t2_representative_tanimoto_rankings_top100.jsonl",
     "t2_representative_tanimoto_manifest.json", "method",
     {"representative_max_tanimoto"}, ORIGINAL_PATH),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path, required: set[str]):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}: {path}")
        yield from reader


def candidate_ids() -> set[str]:
    result = set()
    for row in csv_rows(CANDIDATE_PATH, {"patent_id"}):
        patent = require_id(row["patent_id"], str(CANDIDATE_PATH))
        if patent in result:
            raise ValueError(f"duplicate candidate: {patent}")
        result.add(patent)
    if not result:
        raise ValueError("empty candidate corpus")
    return result


def structure_ids(path: Path, field: str, candidates: set[str]) -> set[str]:
    result = set()
    for row in csv_rows(path, {"patent_id", field}):
        patent = require_id(row["patent_id"], str(path))
        if patent in candidates and str(row[field] or "").strip():
            result.add(patent)
    if not result:
        raise ValueError(f"empty structure-eligible set: {path}")
    return result


def reference_queries(truth: dict) -> dict[str, set[str]]:
    result = {"Citation": set(), "Conflict-X": set()}
    for row in csv_rows(CORE_PATH, {"source", "task", "query_patent_id", "gold_count"}):
        if row["source"] != "nv_document":
            continue
        task = row["task"]
        query = require_id(row["query_patent_id"], str(CORE_PATH))
        if task not in result or query in result[task]:
            raise ValueError(f"invalid/duplicate core query: {task}:{query}")
        if int(row["gold_count"]) != len(truth[query][task]):
            raise ValueError(f"core gold count mismatch: {task}:{query}")
        result[task].add(query)
    if len(result["Citation"]) != 1126 or len(result["Conflict-X"]) != 47:
        raise ValueError(f"unexpected reference query counts: { {k: len(v) for k, v in result.items()} }")
    return result


def check_manifest(path: Path, structure_path: Path, hashes: dict[Path, str]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"manifest not completed: {path}")
    inputs = manifest.get("input_files")
    if not isinstance(inputs, dict):
        raise ValueError(f"manifest input hashes missing: {path}")
    for expected_path in (CANDIDATE_PATH, structure_path, TRUTH_PATH):
        matches = [value for name, value in inputs.items()
                   if Path(name).name == expected_path.name]
        if len(matches) != 1 or not isinstance(matches[0], dict):
            raise ValueError(f"manifest input missing/ambiguous: {path}:{expected_path}")
        if matches[0].get("sha256") != hashes[expected_path]:
            raise ValueError(f"manifest input hash differs: {path}:{expected_path}")


def check_model_files(base: Path, per_query_name: str, ranking_name: str,
                      model_field: str, models: set[str], truth: dict,
                      reference: dict[str, set[str]], mask: set[str]) -> dict[str, int]:
    per_query_seen = set()
    for row in csv_rows(base / per_query_name,
                        {model_field, "task", "query_patent_id", "gold_count",
                         "candidate_count_ranked", "ranking_rows_written"}):
        model = row[model_field]
        task = row["task"]
        query = require_id(row["query_patent_id"], str(base / per_query_name))
        key = (model, task, query)
        if model not in models or task not in reference or query not in reference[task]:
            raise ValueError(f"unexpected per-query row: {key}")
        if key in per_query_seen or query not in mask:
            raise ValueError(f"duplicate or unmasked per-query row: {key}")
        if (int(row["gold_count"]) != len(truth[query][task])
                or int(row["candidate_count_ranked"]) != len(mask) - 1
                or int(row["ranking_rows_written"]) != 100):
            raise ValueError(f"per-query count mismatch: {key}")
        if not truth[query][task].issubset(mask - {query}):
            raise ValueError(f"gold outside candidate mask: {key}")
        per_query_seen.add(key)
    expected_rows = {(model, task, query) for model in models
                     for task, queries in reference.items() for query in queries}
    if per_query_seen != expected_rows:
        raise ValueError(f"per-query coverage mismatch: {base}: missing={len(expected_rows - per_query_seen)}")

    ranking_seen = set()
    with (base / ranking_name).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            model = row.get(model_field)
            query = require_id(row.get("query_patent_id"), f"{base}:{line_no}")
            key = (model, query)
            if model not in models or query not in truth or query not in mask:
                raise ValueError(f"ranking query outside ground truth or candidate mask: {key}")
            if key in ranking_seen:
                raise ValueError(f"duplicate ranking: {key}")
            items = row.get("ranking")
            if not isinstance(items, list) or len(items) != 100:
                raise ValueError(f"ranking length not 100: {key}")
            ids = []
            for rank, item in enumerate(items, 1):
                if not isinstance(item, dict) or item.get("rank") != rank:
                    raise ValueError(f"invalid ranking position: {key}:{rank}")
                ids.append(require_id(item.get("patent_id"), f"{key}:{rank}"))
                try:
                    score = float(item.get("score"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid ranking score: {key}:{rank}") from exc
                if not math.isfinite(score):
                    raise ValueError(f"nonfinite ranking score: {key}:{rank}")
            if len(ids) != len(set(ids)) or not set(ids).issubset(mask - {query}):
                raise ValueError(f"ranking IDs outside common candidate mask: {key}")
            ranking_seen.add(key)
    expected_rankings = {(model, query) for model in models for query in reference["Citation"]}
    if not expected_rankings.issubset(ranking_seen):
        raise ValueError(f"ranking coverage mismatch: {base}: missing={len(expected_rankings - ranking_seen)}")
    return {"models": len(models), "per_query_rows": len(per_query_seen),
            "eligible_ranking_rows": len(expected_rankings),
            "additional_non_evaluable_ranking_rows": len(ranking_seen - expected_rankings),
            "candidate_mask_size": len(mask)}


def check_core_full_candidate_ids(mask: set[str], queries: set[str],
                                  truth_queries: set[str]) -> dict[str, int]:
    seen = set()
    additional = set()
    with CORE_RANKING_PATH.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "nv_document":
                continue
            query = require_id(row.get("query_patent_id"),
                               f"{CORE_RANKING_PATH}:{line_no}")
            if query not in truth_queries or query not in mask:
                raise ValueError(f"core ranking query outside ground truth or mask: {query}")
            if query in seen or query in additional:
                raise ValueError(f"duplicate core ranking: {query}")
            if query not in queries:
                additional.add(query)
                continue
            items = row.get("ranking")
            if not isinstance(items, list) or len(items) != len(mask) - 1:
                raise ValueError(f"core ranking candidate count differs: {query}")
            ids = []
            for rank, item in enumerate(items, 1):
                if not isinstance(item, dict) or item.get("rank") != rank:
                    raise ValueError(f"invalid core ranking position: {query}:{rank}")
                ids.append(require_id(item.get("patent_id"), f"{query}:{rank}"))
            if len(ids) != len(set(ids)) or set(ids) != mask - {query}:
                raise ValueError(f"core full candidate IDs differ from structure mask: {query}")
            seen.add(query)
    if seen != queries:
        raise ValueError(f"core full ranking misses queries: {len(queries - seen)}")
    return {"eligible_queries_checked": len(seen),
            "additional_non_evaluable_queries": len(additional)}


def check_m4_splits(truth: dict, reference: dict[str, set[str]],
                    candidate_count: int, seed: int) -> list[dict]:
    queries = sorted(reference["Citation"])
    groups = make_groups(set(queries), truth)
    group_by_query = {query: group for group, members in groups.items() for query in members}
    labels = [int(query in reference["Conflict-X"]) for query in queries]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    report = []
    seen_test = set()
    for fold, (train_indices, test_indices) in enumerate(
            splitter.split(queries, labels, [group_by_query[query] for query in queries])):
        train = {queries[index] for index in train_indices}
        test = {queries[index] for index in test_indices}
        if not train or not test or seen_test & test:
            raise ValueError(f"invalid outer fold: {fold}")
        seen_test.update(test)
        fit, development, details = split_training_queries(
            sorted(train), truth, 0.2, stable_seed(seed, "outer", fold, "m4_inner"),
        )
        fit_groups = {group_by_query[query] for query in fit}
        development_groups = {group_by_query[query] for query in development}
        if fit_groups & development_groups or set(fit) | set(development) != train:
            raise ValueError(f"M4 inner group leakage/incomplete split: {fold}")
        for part, members in (("fit", fit), ("development", development),
                              ("outer_train", train)):
            if not members or any(not 0 < len(truth[query]["Citation"]) < candidate_count
                                  for query in members):
                raise ValueError(f"M4 positive/negative availability fails: {fold}:{part}")
        report.append({
            "outer_fold": fold,
            "outer_test_queries": len(test),
            "outer_train_queries": len(train),
            "fit_queries": len(fit),
            "development_queries": len(development),
            "fit_groups": len(fit_groups),
            "development_groups": len(development_groups),
            "inner_component_count": details["component_count"],
        })
    if seen_test != set(queries):
        raise ValueError(f"outer folds omit queries: {len(set(queries) - seen_test)}")
    expected_test_sizes = (228, 224, 224, 226, 224)
    observed_test_sizes = tuple(row["outer_test_queries"] for row in report)
    if observed_test_sizes != expected_test_sizes:
        raise ValueError(f"outer fold assignment differs from core preflight: {observed_test_sizes}")
    return report


def main() -> int:
    try:
        paths = {CANDIDATE_PATH, ORIGINAL_PATH, PREPARED_PATH, TRUTH_PATH}
        hashes = {path: sha256(path) for path in paths}
        truth = load_ground_truth(TRUTH_PATH)
        reference = reference_queries(truth)
        candidates = candidate_ids()
        masks = {
            ORIGINAL_PATH: structure_ids(ORIGINAL_PATH, "canonical_smiles", candidates),
            PREPARED_PATH: structure_ids(PREPARED_PATH, "model_smiles", candidates),
        }
        if masks[ORIGINAL_PATH] != masks[PREPARED_PATH]:
            raise ValueError("original and prepared structure candidate masks differ: "
                             f"original_only={len(masks[ORIGINAL_PATH] - masks[PREPARED_PATH])}, "
                             f"prepared_only={len(masks[PREPARED_PATH] - masks[ORIGINAL_PATH])}")
        if len(masks[ORIGINAL_PATH]) != 2218:
            raise ValueError(f"unexpected candidate mask size: {len(masks[ORIGINAL_PATH])}")
        checked_core_queries = check_core_full_candidate_ids(
            masks[ORIGINAL_PATH], reference["Citation"], set(truth),
        )
        model_report = {}
        for label, directory, per_query, ranking, manifest, model_field, models, structure in SPECS:
            base = ROOT / directory
            check_manifest(base / manifest, structure, hashes)
            model_report[label] = check_model_files(
                base, per_query, ranking, model_field, models, truth, reference,
                masks[structure],
            )
        m4_report = check_m4_splits(truth, reference, len(masks[ORIGINAL_PATH]) - 1,
                                    20260924)
        print(json.dumps({
            "status": "additional_model_masks_and_m4_split_preflight_pass",
            "candidate_corpus_count": len(candidates),
            "common_usable_candidate_count": len(masks[ORIGINAL_PATH]),
            "candidates_per_query": len(masks[ORIGINAL_PATH]) - 1,
            "core_full_candidate_ids": checked_core_queries,
            "models": model_report,
            "m4_inner_splits": m4_report,
            "limitations": [
                "Top-100 ranking ID membership is verified, not full ranking equality for extra models",
                "M4 fit/development label availability is verified, not logistic fit or convergence",
                "third T1 source and historical model-selection provenance are not audited",
            ],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "preflight_blocked", "reason": str(exc)},
                         ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
