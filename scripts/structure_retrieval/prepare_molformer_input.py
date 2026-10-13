#!/usr/bin/env python3
"""Prepare MoLFormer SMILES to match model training conditions. Remove stereochemical notation from SureChEMBL canonical SMILES; exclude inputs exceeding the fixed tokenizer length limit. Deduplicate structures per patent and record candidate/query coverage. Use the tokenizer only, without downloading model weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
DEFAULT_REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def summarize_lengths(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        position = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[position]

    middle = len(ordered) // 2
    if len(ordered) % 2:
        median: int | float = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def normalize_nonisomeric_smiles(smiles: str) -> str | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def load_and_normalize_structures(
    path: Path,
    candidate_set: set[str],
) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    normalized_links: Counter[tuple[str, str]] = Counter()
    counts: Counter[str] = Counter()
    source_smiles_strings: set[str] = set()
    source_links: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patent_id", "canonical_smiles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in structure file: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            counts["input_rows"] += 1
            patent_id = str(row.get("patent_id") or "").strip()
            smiles = str(row.get("canonical_smiles") or "").strip()
            if patent_id not in candidate_set:
                counts["outside_candidate_rows"] += 1
                continue
            if not smiles:
                counts["empty_smiles_rows"] += 1
                continue
            normalized = normalize_nonisomeric_smiles(smiles)
            if normalized is None:
                counts["rdkit_parse_failure_rows"] += 1
                continue
            source_smiles_strings.add(smiles)
            source_links.add((patent_id, smiles))
            normalized_links[(patent_id, normalized)] += 1
            counts["accepted_source_rows"] += 1
            if row_number % 50000 == 0:
                print(f"  SMILES normalized: {row_number - 1} rows processed", flush=True)
    normalized_unique_smiles = {smiles for _, smiles in normalized_links}
    counts["source_unique_smiles_strings"] = len(source_smiles_strings)
    counts["source_unique_patent_smiles_links"] = len(source_links)
    counts["normalized_unique_nonisomeric_smiles"] = len(normalized_unique_smiles)
    counts["normalized_patent_smiles_links"] = len(normalized_links)
    counts["raw_duplicate_source_rows"] = (
        counts["accepted_source_rows"] - len(source_links)
    )
    counts["nonisomeric_unique_smiles_collapses"] = (
        len(source_smiles_strings) - len(normalized_unique_smiles)
    )
    counts["nonisomeric_patent_smiles_link_collapses"] = (
        len(source_links) - len(normalized_links)
    )
    # Keep legacy aggregate keys for compatibility with existing manifest consumers.
    counts["stereochemistry_or_duplicate_collapses"] = (
        counts["accepted_source_rows"] - len(normalized_links)
    )
    return dict(normalized_links), dict(sorted(counts.items()))


def tokenize_lengths(
    tokenizer: Any,
    smiles_values: list[str],
    batch_size: int,
) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for start in range(0, len(smiles_values), batch_size):
        batch = smiles_values[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        if len(input_ids) != len(batch):
            raise SystemExit("Tokenizer output row count differs from input")
        for smiles, token_ids in zip(batch, input_ids, strict=True):
            lengths[smiles] = len(token_ids)
        completed = min(start + batch_size, len(smiles_values))
        if completed % 10000 < batch_size or completed == len(smiles_values):
            print(
                f"  Tokenizer lengths: {completed}/{len(smiles_values)} structures processed",
                flush=True,
            )
    return lengths


def run_self_test() -> None:
    from rdkit import Chem

    first = normalize_nonisomeric_smiles("C[C@H](O)Cl")
    second = normalize_nonisomeric_smiles("C[C@@H](O)Cl")
    assert first == second
    assert first == Chem.MolToSmiles(
        Chem.MolFromSmiles("CC(O)Cl"),
        canonical=True,
        isomericSmiles=False,
    )
    assert summarize_lengths([1, 2, 3, 4, 5])["p95"] == 5
    print("Self-check passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path)
    parser.add_argument("--candidate-patents", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--maximum-tokens", type=int, default=202)
    parser.add_argument("--tokenizer-batch-size", type=int, default=1024)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    required = [
        args.structures,
        args.candidate_patents,
        args.ground_truth,
        args.output_dir,
    ]
    if any(path is None for path in required):
        parser.error("Require structures, candidates, ground truth, and output arguments")
    for path in (args.structures, args.candidate_patents, args.ground_truth):
        if not path.is_file():
            raise SystemExit(f"Input file missing: {path}")
    if args.maximum_tokens < 1 or args.tokenizer_batch_size < 1:
        parser.error("Token limit and batch size must be at least 1")

    try:
        from huggingface_hub import HfApi
        from rdkit import RDLogger, rdBase
        from transformers import AutoTokenizer
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing MoLFormer input-preparation dependency: {exc}") from exc

    try:
        from .evaluate_t2_structure_retrieval import (
            load_candidates,
            load_ground_truth,
            select_complete_gold_queries,
        )
    except ImportError:
        from evaluate_t2_structure_retrieval import (
            load_candidates,
            load_ground_truth,
            select_complete_gold_queries,
        )

    RDLogger.DisableLog("rdApp.*")
    candidate_ids = load_candidates(args.candidate_patents, "patent_id")
    candidate_set = set(candidate_ids)
    ground_truth, ground_truth_field_mapping = load_ground_truth(
        args.ground_truth,
        "conflict_patents",
    )

    model_info = HfApi().model_info(args.model_id, revision=args.revision)
    resolved_revision = str(model_info.sha or "")
    if not resolved_revision:
        raise SystemExit("Could not verify the checkpoint revision on Hugging Face")
    if len(args.revision) == 40 and resolved_revision != args.revision:
        raise SystemExit(
            f"Checkpoint revision mismatch: {args.revision} != {resolved_revision}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=resolved_revision,
        trust_remote_code=True,
    )

    normalized_links, normalization_counts = load_and_normalize_structures(
        args.structures,
        candidate_set,
    )
    unique_smiles = sorted({smiles for _, smiles in normalized_links})
    token_lengths = tokenize_lengths(
        tokenizer,
        unique_smiles,
        args.tokenizer_batch_size,
    )
    eligible_smiles = {
        smiles for smiles, length in token_lengths.items() if length <= args.maximum_tokens
    }

    eligible_links = [
        {
            "patent_id": patent_id,
            "model_smiles": smiles,
            "source_canonical_smiles_count": source_count,
            "token_count": token_lengths[smiles],
        }
        for (patent_id, smiles), source_count in sorted(normalized_links.items())
        if smiles in eligible_smiles
    ]
    eligible_patents = {row["patent_id"] for row in eligible_links}
    excluded_patents = sorted(
        patent_id
        for patent_id in candidate_ids
        if patent_id not in eligible_patents
    )
    excluded_structure_rows = [
        {
            "patent_id": patent_id,
            "model_smiles": smiles,
            "source_canonical_smiles_count": source_count,
            "token_count": token_lengths[smiles],
            "reason": "token_limit_exceeded",
        }
        for (patent_id, smiles), source_count in sorted(normalized_links.items())
        if smiles not in eligible_smiles
    ]

    eligibility: dict[str, dict[str, int]] = {}
    for task in ("Citation", "Conflict-X"):
        selected, skipped = select_complete_gold_queries(
            ground_truth,
            task,
            eligible_patents,
            eligible_patents,
        )
        eligibility[task] = {
            "evaluated_queries": len(selected),
            **dict(sorted(skipped.items())),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = args.output_dir / "molformer_eligible_structures.csv"
    issue_path = args.output_dir / "molformer_structure_issues.csv"
    manifest_path = args.output_dir / "molformer_input_manifest.json"
    write_csv_atomic(
        prepared_path,
        [
            "patent_id",
            "model_smiles",
            "source_canonical_smiles_count",
            "token_count",
        ],
        eligible_links,
    )
    write_csv_atomic(
        issue_path,
        [
            "patent_id",
            "model_smiles",
            "source_canonical_smiles_count",
            "token_count",
            "reason",
        ],
        excluded_structure_rows,
    )
    manifest = {
        "status": "completed",
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "model_weights_downloaded": False,
        "smiles_policy": (
            "RDKit canonical non-isomeric SMILES; deduplicate per patent; "
            "exclude structures exceeding the tokenizer limit including special tokens; never truncate"
        ),
        "maximum_tokens": args.maximum_tokens,
        "candidate_count": len(candidate_ids),
        "eligible_patent_count": len(eligible_patents),
        "excluded_patent_count": len(excluded_patents),
        "excluded_patent_ids": excluded_patents,
        "eligible_patent_smiles_links": len(eligible_links),
        "normalized_unique_smiles": len(unique_smiles),
        "eligible_unique_smiles": len(eligible_smiles),
        "excluded_unique_smiles": len(unique_smiles) - len(eligible_smiles),
        "token_length_summary": summarize_lengths(list(token_lengths.values())),
        "normalization_counts": normalization_counts,
        "normalization_reconciliation": {
            "accepted_source_rows": normalization_counts.get(
                "accepted_source_rows", 0
            ),
            "source_unique_smiles_strings": normalization_counts.get(
                "source_unique_smiles_strings", 0
            ),
            "source_unique_patent_smiles_links": normalization_counts.get(
                "source_unique_patent_smiles_links", 0
            ),
            "normalized_unique_nonisomeric_smiles": normalization_counts.get(
                "normalized_unique_nonisomeric_smiles", 0
            ),
            "normalized_patent_smiles_links": normalization_counts.get(
                "normalized_patent_smiles_links", 0
            ),
            "raw_duplicate_source_rows": normalization_counts.get(
                "raw_duplicate_source_rows", 0
            ),
            "nonisomeric_unique_smiles_collapses": normalization_counts.get(
                "nonisomeric_unique_smiles_collapses", 0
            ),
            "nonisomeric_patent_smiles_link_collapses": normalization_counts.get(
                "nonisomeric_patent_smiles_link_collapses", 0
            ),
            "interpretation": (
                "Count source-row duplicates separately from structure/patent-link merging after stereochemistry removal"
            ),
        },
        "eligibility": eligibility,
        "ground_truth_field_mapping": ground_truth_field_mapping,
        "versions": {
            "rdkit": rdBase.rdkitVersion,
            "transformers": transformers.__version__,
        },
        "provenance": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "input_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (args.structures, args.candidate_patents, args.ground_truth)
        },
        "output_files": {},
    }
    for path in (prepared_path, issue_path):
        manifest["output_files"][str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
