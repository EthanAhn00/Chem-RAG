#!/usr/bin/env python3
"""Freeze the pre-registered T1/T2/T3 outer-fold assignment once."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import sklearn
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .audit_t123_5fold_preflight import load_ground_truth, make_groups
    from .audit_t123_5fold_remaining import reference_queries, sha256
except ImportError:
    from audit_t123_5fold_preflight import load_ground_truth, make_groups
    from audit_t123_5fold_remaining import reference_queries, sha256


SEED = 20260924
ROOT = Path("structure_retrieval_outputs")
GROUND_TRUTH = Path("data/processed/retrieval_ground_truth.jsonl")
CORE_PER_QUERY = (
    ROOT / "t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_per_query.csv"
)
INPUTS = {
    "ground_truth": (GROUND_TRUTH,
                     "7c15cc59563cf298e8281c1dd8401e92f4b1167c5c9a65a90c042a59c050223f"),
    "text_per_query": (CORE_PER_QUERY,
                       "456ab711820d34ffc9857aa87e51c9df1ecb8dd663064411bd97c35ccd761df0"),
    "structure_per_query": (
        ROOT / "t3_rrf_v1/inputs/t2_morgan_full/t2_structure_retrieval_per_query.csv",
        "567f15ffded8a5ade2b18d618fcba7bb402455a2ca7a75a2fc2ba60e54b47398",
    ),
    "legacy_split": (
        Path("data/db_export/split.csv"),
        "76a37f0114083b942514e6a650a7825b3fc575c28b668b1195908370a2711949",
    ),
    "candidate_corpus": (
        ROOT / "t2_candidate_corpus_v1/text_candidate_patents_common.csv",
        "2a15983a13ffcf884a14584fc1f4d60925963f7821f913574e9f383c50e604bb",
    ),
    "original_structures": (
        ROOT / "t2_surechembl/surechembl_structures.csv",
        "cfc986386ddecc4cf0075b365893093e5161b16c5ca6bbae261d2c47af977a3b",
    ),
    "prepared_structures": (
        ROOT / "t2_molformer_input_v1/molformer_eligible_structures.csv",
        "b7b221d05ee2677570bdc006fb15271aa2a07c0d6f8d8a604dfe5f74fd2efef8",
    ),
    "text_full_rankings": (
        ROOT / "t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_rankings_top2217.jsonl",
        "c92b26a00ffcadc77ae804854f1d0e375d18dcf65217ab008443ebf4a85e77ec",
    ),
    "structure_full_rankings": (
        ROOT / "t3_rrf_v1/inputs/t2_morgan_full/t2_structure_rankings_top2217.jsonl",
        "fb22041d5de3aa21e72dda53b49ee572ccc5ef3e1440d0f13116c71ee8919b1f",
    ),
}
OUTPUT_DIR = ROOT / "t123_5fold_v1"
ASSIGNMENT_NAME = "t123_5fold_assignment.csv"
MANIFEST_NAME = "t123_5fold_assignment_manifest.json"


def input_info() -> dict[str, dict[str, str | int]]:
    result = {}
    for label, (path, expected_hash) in INPUTS.items():
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"input changed since preflight: {label}: {path}: {actual_hash}")
        result[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
        }
    return result


def make_assignment(truth: dict, eligible: dict[str, set[str]]) -> tuple[bytes, list[dict]]:
    citation_queries = sorted(eligible["Citation"])
    if not eligible["Conflict-X"].issubset(eligible["Citation"]):
        raise ValueError("Conflict-X queries are not a subset of Citation queries")
    groups = make_groups(set(citation_queries), truth)
    if len(groups) != 908:
        raise ValueError(f"connected group count changed: {len(groups)}")
    group_by_query = {
        query: group_id
        for group_id, members in groups.items()
        for query in members
    }
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED,
    )
    labels = [int(query in eligible["Conflict-X"]) for query in citation_queries]
    group_labels = [group_by_query[query] for query in citation_queries]
    assignment = {}
    folds = []
    for fold, (_, test_indices) in enumerate(
            splitter.split(citation_queries, labels, group_labels)):
        test = {citation_queries[index] for index in test_indices}
        if not test or set(assignment) & test:
            raise ValueError(f"empty or overlapping outer fold: {fold}")
        for query in test:
            assignment[query] = fold
        folds.append({
            "outer_fold": fold,
            "citation_queries": len(test),
            "conflict_x_queries": len(test & eligible["Conflict-X"]),
            "groups": len({group_by_query[query] for query in test}),
        })
    if set(assignment) != set(citation_queries):
        raise ValueError(f"outer fold coverage mismatch: {len(assignment)}")
    if [item["citation_queries"] for item in folds] != [228, 224, 224, 226, 224]:
        raise ValueError(f"Citation fold sizes changed: {folds}")
    if [item["conflict_x_queries"] for item in folds] != [11, 9, 9, 9, 9]:
        raise ValueError(f"Conflict-X fold sizes changed: {folds}")
    if [item["groups"] for item in folds] != [179, 179, 181, 185, 184]:
        raise ValueError(f"group fold sizes changed: {folds}")
    if any(len({assignment[query] for query in members}) != 1
           for members in groups.values()):
        raise ValueError("connected group was split across outer folds")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("query_id", "group_id", "outer_fold", "citation_eligible",
                     "conflict_x_eligible", "seed"))
    for query in citation_queries:
        writer.writerow((query, group_by_query[query], assignment[query], 1,
                         int(query in eligible["Conflict-X"]), SEED))
    return output.getvalue().encode("utf-8"), folds


def freeze(assignment_bytes: bytes, manifest_bytes: bytes) -> str:
    assignment_path = OUTPUT_DIR / ASSIGNMENT_NAME
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    if OUTPUT_DIR.exists():
        if not OUTPUT_DIR.is_dir() or not assignment_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"existing fold directory is incomplete: {OUTPUT_DIR}")
        if (assignment_path.read_bytes() != assignment_bytes
                or manifest_path.read_bytes() != manifest_bytes):
            raise ValueError(f"existing frozen folds differ; refusing overwrite: {OUTPUT_DIR}")
        return "already_frozen_identical"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    with assignment_path.open("xb") as handle:
        handle.write(assignment_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return "frozen_new"


def main() -> int:
    try:
        inputs = input_info()
        truth = load_ground_truth(GROUND_TRUTH)
        eligible = reference_queries(truth)
        assignment_bytes, folds = make_assignment(truth, eligible)
        assignment_sha = hashlib.sha256(assignment_bytes).hexdigest()
        manifest = {
            "schema_version": 1,
            "status": "frozen",
            "seed": SEED,
            "splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True)",
            "sklearn_version": sklearn.__version__,
            "group_policy": "connected components of eligible queries sharing Citation or Conflict-X gold patent IDs; no query-to-gold edges",
            "stratification_label": "Conflict-X eligibility",
            "eligible_queries": {task: len(queries) for task, queries in eligible.items()},
            "connected_groups": 908,
            "input_files": inputs,
            "assignment_file": {
                "path": str(OUTPUT_DIR / ASSIGNMENT_NAME),
                "rows": len(eligible["Citation"]),
                "bytes": len(assignment_bytes),
                "sha256": assignment_sha,
            },
            "folds": folds,
            "interpretation_limit": "This freezes a split, not model training, OOF metrics, or external validation.",
        }
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n").encode("utf-8")
        status = freeze(assignment_bytes, manifest_bytes)
        print(json.dumps({
            "status": status,
            "assignment_path": str(OUTPUT_DIR / ASSIGNMENT_NAME),
            "assignment_sha256": assignment_sha,
            "manifest_path": str(OUTPUT_DIR / MANIFEST_NAME),
            "folds": folds,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "fold_freeze_blocked", "reason": str(exc)},
                         ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
