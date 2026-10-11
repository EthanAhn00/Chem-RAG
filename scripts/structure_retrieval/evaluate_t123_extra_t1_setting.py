#!/usr/bin/env python3
"""Evaluate one additional fixed T1 setting without replacing frozen results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "structure_retrieval_outputs/t123_5fold_v1"
REFERENCE = ROOT / "structure_retrieval_outputs/t3_rrf_v1/inputs/t1_text_full/t1_matched_subset_manifest.json"
SETTINGS = {
    "nv_whole_claims": ("patent_text_units_nv_embed_v2", "whole_claims"),
    "nv_claim": ("patent_text_units_nv_embed_v2", "claim"),
    "qwen_document": ("patent_text_units_qwen3_0_6b", "document"),
    "qwen_claim": ("patent_text_units_qwen3_0_6b", "claim"),
    "arctic_document": ("patent_text_units_arctic_l_v2", "document"),
    "arctic_claim": ("patent_text_units_arctic_l_v2", "claim"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SETTINGS, required=True)
    parser.add_argument("--run", action="store_true", help="Otherwise perform only file/cohort preflight")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--qdrant-path", type=Path, default=ROOT / "data/qdrant_text_units",
        help="Existing text-unit vector store; the frozen evaluator reads its vectors.",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if (reference.get("status") != "completed"
            or reference.get("query_counts") != {"Citation": 1126, "Conflict-X": 47}
            or reference.get("candidate_count") != 2218
            or reference.get("ranking_limit") != 2217):
        raise ValueError("Unexpected reference population, candidates, or ranking depth")
    if reference.get("ground_truth_field_mapping", {}).get("Conflict-X") != "conflict_patents":
        raise ValueError("Unexpected reference Conflict-X ground-truth field")
    inputs = {}
    for name, record in reference["input_files"].items():
        path = resolve(name)
        if path.name in inputs:
            raise ValueError(f"Ambiguous reference input basename: {path.name}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Reference input changed: {path}")
        inputs[path.name] = path
    gt_path = inputs["retrieval_ground_truth.jsonl"]
    candidates_path = inputs["text_candidate_patents_common.csv"]
    structures_path = inputs["surechembl_structures.csv"]
    t2_path = inputs["t2_structure_retrieval_per_query.csv"]

    assignment_path = BASE / "t123_5fold_assignment.csv"
    frozen = json.loads((BASE / "t123_5fold_assignment_manifest.json").read_text(encoding="utf-8"))
    if sha256(assignment_path) != frozen["assignment_file"]["sha256"]:
        raise ValueError("Frozen assignment changed")
    assignment = csv_rows(assignment_path)
    citation = {row["query_id"] for row in assignment}
    if len(assignment) != 1126 or len(citation) != 1126:
        raise ValueError("Invalid frozen cohort")

    gt = {}
    with gt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                query = row["query_patent_id"]
                if query in gt:
                    raise ValueError(f"Duplicate ground truth: {query}")
                gt[query] = row
    conflict = {query for query in citation if gt[query]["conflict_patents"]}
    if len(conflict) != 47:
        raise ValueError("Unexpected Conflict-X frozen subset")
    from evaluate_t1_matched_subset import (
        load_candidates, load_structure_eligible_candidates, load_t2_query_sets,
    )
    task_queries = load_t2_query_sets(t2_path)
    if task_queries != {"Citation": citation, "Conflict-X": conflict}:
        raise ValueError("Reference T2 query IDs differ from frozen cohort")
    original_candidates = load_candidates(candidates_path, "patent_id")
    eligible = set(load_structure_eligible_candidates(structures_path, set(original_candidates)))
    if len(original_candidates) != 2376 or len(eligible) != 2218 or not citation <= eligible:
        raise ValueError("Unexpected original/eligible candidate mask")
    for query in citation:
        gold = set(gt[query]["all_cited_patents"])
        if not gold or query in gold or not gold <= eligible - {query}:
            raise ValueError(f"Incomplete/self gold: {query}")

    evaluator = ROOT / "scripts/structure_retrieval/evaluate_t1_matched_subset.py"
    recorded_script = reference["provenance"]["script_sha256"]
    if sha256(evaluator) != recorded_script:
        raise ValueError("Original T1 evaluator changed since reference run; review before proceeding")
    target = BASE / "inputs" / f"t1_{args.source}_full"
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Refusing to overwrite existing files: {target}")
    collection, unit = SETTINGS[args.source]
    command = [
        sys.executable, str(evaluator),
        "--qdrant-path", str(args.qdrant_path),
        "--ground-truth", str(gt_path),
        "--candidate-patents", str(candidates_path),
        "--structures", str(structures_path),
        "--t2-per-query", str(t2_path),
        "--source", f"{args.source}={collection}:{unit}",
        "--output-dir", str(target), "--ranking-limit", "2217", "--device", "cpu",
    ]
    print(json.dumps({
        "status": "extra_t1_file_cohort_preflight_pass",
        "source": args.source, "unit": unit, "collection": collection,
        "citation_queries": len(citation), "conflict_x_queries": len(conflict),
        "candidates_per_query": len(eligible) - 1,
        "assignment_sha256": sha256(assignment_path),
        "evaluator_sha256": recorded_script,
        "device": "cpu", "thread_limit": args.threads,
        "output_dir": str(target.relative_to(ROOT)),
        "run_requested": args.run,
        "limitations": "Vector coverage is checked by the evaluator during --run; this is not independent output validation.",
    }, ensure_ascii=False, indent=2), flush=True)
    if not args.run:
        return 0
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(args.threads)
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
