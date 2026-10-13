#!/usr/bin/env python3
"""Reevaluate B1 text retrieval under the B2 complete-gold conditions. Read existing Qdrant embeddings without modification and use maximum query-unit/candidate-unit cosine per patent. Match B2 structure-bearing candidates, query subset, Top-100 cutoff, gold, and metric definitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .evaluate_t2_structure_retrieval import (
        METRIC_DEFINITIONS,
        evaluate_ranking,
    )
except ImportError:
    from evaluate_t2_structure_retrieval import (
        METRIC_DEFINITIONS,
        evaluate_ranking,
    )


EP_EXACT_RE = re.compile(r"^EP(\d+)([A-Z]\d?)$")
METRIC_NAMES = [
    "R@1",
    "R@5",
    "R@10",
    "R@50",
    "R@100",
    "MRR@100",
    "MAP@100",
    "nDCG@10",
]


def normalize_exact_ep(value: Any) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").strip().upper())
    match = EP_EXACT_RE.fullmatch(compact)
    return f"EP{match.group(1)}{match.group(2)}" if match else None


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


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_source(value: str) -> tuple[str, str, str]:
    label, equal_sign, remainder = value.partition("=")
    collection, colon, unit = remainder.partition(":")
    if not equal_sign or not colon:
        raise argparse.ArgumentTypeError(
            "--source must use label=collection:unit format"
        )
    if not label or not collection or not unit:
        raise argparse.ArgumentTypeError("--source contains an empty component")
    return label, collection, unit


def load_candidates(path: Path, id_column: str) -> list[str]:
    output: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if id_column not in set(reader.fieldnames or []):
            raise SystemExit(f"Candidate file lacks {id_column!r} column")
        for row_number, row in enumerate(reader, start=2):
            patent_id = normalize_exact_ep(row.get(id_column))
            if patent_id is None:
                raise SystemExit(f"Invalid patent identifier: {path}:{row_number}")
            output.add(patent_id)
    return sorted(output)


def load_ground_truth(
    path: Path,
    conflict_field: str = "conflict_patents",
) -> tuple[dict[str, dict[str, set[str]]], dict[str, str]]:
    if conflict_field not in {"conflict_patents", "x_category_patents"}:
        raise ValueError(f"Unsupported X-category field: {conflict_field}")
    alternate_field = (
        "x_category_patents"
        if conflict_field == "conflict_patents"
        else "conflict_patents"
    )
    output: dict[str, dict[str, set[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = normalize_exact_ep(row.get("query_patent_id"))
            if query_id is None:
                raise SystemExit(f"Invalid query patent identifier: {path}:{row_number}")
            if conflict_field not in row:
                raise SystemExit(
                    f"Requested X-category field missing: "
                    f"{path}:{row_number}:{conflict_field}"
                )
            x_raw = row.get(conflict_field) or []
            alternate_raw = row.get(alternate_field) or [] if alternate_field in row else None
            all_raw = row.get("all_cited_patents") or []
            if not isinstance(all_raw, list) or not isinstance(x_raw, list):
                raise SystemExit(f"Invalid gold-list format: {path}:{row_number}")
            if alternate_raw is not None and not isinstance(alternate_raw, list):
                raise SystemExit(f"Invalid compatibility gold-list format: {path}:{row_number}")

            def normalized(values: list[Any]) -> set[str]:
                patents = {normalize_exact_ep(value) for value in values}
                if None in patents:
                    raise SystemExit(f"Invalid gold patent identifier: {path}:{row_number}")
                return {value for value in patents if value is not None}

            if query_id in output:
                raise SystemExit(f"Duplicate query: {path}:{row_number}:{query_id}")
            all_gold = normalized(all_raw)
            x_gold = normalized(x_raw)
            if alternate_raw is not None and x_gold != normalized(alternate_raw):
                raise SystemExit(
                    "X-category compatibility fields disagree: "
                    f"{path}:{row_number}:{conflict_field}!={alternate_field}"
                )
            if not x_gold.issubset(all_gold):
                raise SystemExit(f"X gold is not a subset of all gold: {query_id}")
            output[query_id] = {
                "Citation": all_gold,
                "Conflict-X": x_gold,
            }
    return output, {
        "Citation": "all_cited_patents",
        "Conflict-X": conflict_field,
    }


def load_t2_query_sets(path: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_patent_id", "task"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise SystemExit(f"Missing required columns in T2 query file: {path}")
        for row in reader:
            query_id = normalize_exact_ep(row.get("query_patent_id"))
            task = str(row.get("task") or "")
            if query_id is None or task not in {"Citation", "Conflict-X"}:
                raise SystemExit(f"Invalid T2 query row: {row}")
            output[task].add(query_id)
    return dict(output)


def load_structure_eligible_candidates(
    path: Path,
    candidate_set: set[str],
) -> list[str]:
    """Read structure-bearing patents to build the common B1/B2 candidate mask."""
    eligible: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patent_id", "canonical_smiles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Structure file lacks required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            patent_id = normalize_exact_ep(row.get("patent_id"))
            if patent_id is None:
                raise SystemExit(f"Invalid patent identifier in structure file: {path}:{row_number}")
            if patent_id in candidate_set and str(row.get("canonical_smiles") or "").strip():
                eligible.add(patent_id)
    if not eligible:
        raise SystemExit(f"No structure-bearing candidates found: {path}")
    return sorted(eligible)


def restrict_vectors_to_candidates(
    patent_ids: list[str],
    vectors: Any,
    eligible_candidates: set[str],
) -> tuple[list[str], Any]:
    """Restrict read-only Qdrant vectors to structure-bearing common candidates."""
    import numpy as np

    positions = [
        position
        for position, patent_id in enumerate(patent_ids)
        if patent_id in eligible_candidates
    ]
    restricted_ids = [patent_ids[position] for position in positions]
    missing = eligible_candidates - set(restricted_ids)
    if missing:
        raise SystemExit(
            "Structure-bearing candidates lack text vectors: "
            f"count={len(missing)}, examples={sorted(missing)[:5]}"
        )
    return restricted_ids, np.asarray(vectors)[positions]


def load_vectors(client: Any, collection: str, unit: str) -> tuple[list[str], Any]:
    import numpy as np
    from qdrant_client import models

    condition = models.Filter(
        must=[
            models.FieldCondition(
                key="unit_type",
                match=models.MatchValue(value=unit),
            )
        ]
    )
    patents: list[str] = []
    vectors: list[Any] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=condition,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            patent_id = normalize_exact_ep(point.payload.get("patent_id"))
            if patent_id is None:
                raise SystemExit(f"Invalid patent identifier in Qdrant: {point.id}")
            vector = point.vector
            if isinstance(vector, dict):
                if len(vector) != 1:
                    raise SystemExit(f"Cannot automatically choose among multiple named vectors: {collection}")
                vector = next(iter(vector.values()))
            patents.append(patent_id)
            vectors.append(vector)
        if offset is None:
            break
    if not vectors:
        raise SystemExit(f"Requested unit vector missing: {collection}:{unit}")
    return patents, np.asarray(vectors, dtype=np.float32)


def make_rankings(
    patent_ids: list[str],
    vectors: Any,
    candidate_ids: list[str],
    query_ids: list[str],
    device: str,
    limit: int,
) -> tuple[dict[str, list[tuple[str, float]]], str]:
    import numpy as np
    import torch
    import torch.nn.functional as functional

    candidate_set = set(candidate_ids)
    vector_patents = set(patent_ids)
    if vector_patents != candidate_set:
        raise SystemExit(
            "Qdrant patent set differs from T2 common candidates: "
            f"vector_only={len(vector_patents - candidate_set)}, "
            f"candidate_only={len(candidate_set - vector_patents)}"
        )
    unit_positions: dict[str, list[int]] = defaultdict(list)
    for position, patent_id in enumerate(patent_ids):
        unit_positions[patent_id].append(position)

    selected_device = device
    if device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = functional.normalize(torch.from_numpy(vectors).to(selected_device), dim=1)
    candidate_index = {patent_id: index for index, patent_id in enumerate(candidate_ids)}
    unit_candidate_index = torch.tensor(
        [candidate_index[patent_id] for patent_id in patent_ids],
        device=selected_device,
    )
    rankings: dict[str, list[tuple[str, float]]] = {}
    candidate_array = np.asarray(candidate_ids, dtype=object)
    for number, query_id in enumerate(query_ids, start=1):
        query_positions = unit_positions.get(query_id)
        if not query_positions:
            raise SystemExit(f"Query vector missing: {query_id}")
        unit_scores = (tensor[query_positions] @ tensor.T).amax(dim=0)
        patent_scores = torch.full(
            (len(candidate_ids),),
            -torch.inf,
            dtype=unit_scores.dtype,
            device=selected_device,
        )
        patent_scores.scatter_reduce_(
            0,
            unit_candidate_index,
            unit_scores,
            reduce="amax",
            include_self=True,
        )
        scores = patent_scores.cpu().numpy()
        valid = np.isfinite(scores) & (candidate_array != query_id)
        valid_positions = np.flatnonzero(valid)
        valid_ids = candidate_array[valid_positions]
        valid_scores = scores[valid_positions]
        order = np.lexsort((valid_ids, -valid_scores))[:limit]
        rankings[query_id] = [
            (str(valid_ids[index]), float(valid_scores[index]))
            for index in order
        ]
        if number % 100 == 0:
            print(f"  {number}/{len(query_ids)} queries processed", flush=True)
    return rankings, selected_device


def run_self_test() -> None:
    assert parse_source("model=collection:unit") == ("model", "collection", "unit")
    assert normalize_exact_ep("EP-123-A1") == "EP123A1"
    metrics = evaluate_ranking([("P1", 0.9), ("P2", 0.8)], {"P2"})
    assert metrics["MAP@100"] == 0.5
    patent_ids = ["EP1A1", "EP2A1", "EP1A1"]
    vectors = __import__("numpy").arange(6).reshape(3, 2)
    restricted_ids, restricted_vectors = restrict_vectors_to_candidates(
        patent_ids,
        vectors,
        {"EP1A1"},
    )
    assert restricted_ids == ["EP1A1", "EP1A1"]
    assert restricted_vectors.shape == (2, 2)
    ranking, selected_device = make_rankings(
        ["EP1A1", "EP2A1", "EP3A1"],
        __import__("numpy").asarray(
            [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]],
            dtype=__import__("numpy").float32,
        ),
        ["EP1A1", "EP2A1", "EP3A1"],
        ["EP1A1"],
        "cpu",
        100,
    )
    assert selected_device == "cpu"
    assert [patent_id for patent_id, _ in ranking["EP1A1"]] == [
        "EP2A1",
        "EP3A1",
    ]
    print("Self-check passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-path", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--candidate-patents", type=Path)
    parser.add_argument("--candidate-id-column", default="patent_id")
    parser.add_argument("--structures", type=Path)
    parser.add_argument("--t2-per-query", type=Path)
    parser.add_argument("--source", action="append", type=parse_source)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ranking-limit", type=int, default=100)
    parser.add_argument(
        "--conflict-field",
        choices=("conflict_patents", "x_category_patents"),
        default="conflict_patents",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    required = [
        args.qdrant_path,
        args.ground_truth,
        args.candidate_patents,
        args.structures,
        args.t2_per_query,
        args.output_dir,
    ]
    if any(path is None for path in required) or not args.source:
        parser.error("Require qdrant, ground truth, candidates, structures, B2 queries, source, and output arguments")
    if not args.qdrant_path.is_dir():
        raise SystemExit(f"Qdrant directory missing: {args.qdrant_path}")
    for path in (args.ground_truth, args.candidate_patents, args.structures, args.t2_per_query):
        if not path.is_file():
            raise SystemExit(f"Input file missing: {path}")
    if args.ranking_limit < 100:
        parser.error("--ranking-limit must be at least 100")

    from qdrant_client import QdrantClient

    started_at = time.perf_counter()
    candidate_ids = load_candidates(args.candidate_patents, args.candidate_id_column)
    eligible_candidate_ids = load_structure_eligible_candidates(
        args.structures,
        set(candidate_ids),
    )
    eligible_candidate_set = set(eligible_candidate_ids)
    ground_truth, ground_truth_field_mapping = load_ground_truth(
        args.ground_truth,
        args.conflict_field,
    )
    t2_query_sets = load_t2_query_sets(args.t2_per_query)
    query_ids = sorted(set().union(*t2_query_sets.values()))
    missing_query_candidates = set(query_ids) - eligible_candidate_set
    if missing_query_candidates:
        raise SystemExit(
            "B2 query patents are absent from structure-bearing common candidates: "
            f"count={len(missing_query_candidates)}, "
            f"examples={sorted(missing_query_candidates)[:5]}"
        )
    result_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}

    client = QdrantClient(path=str(args.qdrant_path))
    try:
        available_collections = {
            item.name for item in client.get_collections().collections
        }
        for label, collection, unit in args.source:
            if collection not in available_collections:
                raise SystemExit(f"Qdrant collection missing: {collection}")
            print(f"{label}: {collection}:{unit} vectors will be loaded.", flush=True)
            patent_ids, vectors = load_vectors(client, collection, unit)
            full_vector_row_count = len(patent_ids)
            full_vector_patent_count = len(set(patent_ids))
            if set(patent_ids) != set(candidate_ids):
                raise SystemExit(
                    "Qdrant patent set differs from the original common candidate set: "
                    f"vector_only={len(set(patent_ids) - set(candidate_ids))}, "
                    f"candidate_only={len(set(candidate_ids) - set(patent_ids))}"
                )
            patent_ids, vectors = restrict_vectors_to_candidates(
                patent_ids,
                vectors,
                eligible_candidate_set,
            )
            rankings, selected_device = make_rankings(
                patent_ids,
                vectors,
                eligible_candidate_ids,
                query_ids,
                args.device,
                args.ranking_limit,
            )
            source_stats[label] = {
                "collection": collection,
                "unit": unit,
                "full_vector_rows": full_vector_row_count,
                "full_vector_patents": full_vector_patent_count,
                "vector_rows": len(patent_ids),
                "vector_dimension": int(vectors.shape[1]),
                "unique_vector_patents": len(set(patent_ids)),
                "device": selected_device,
            }
            for query_id in query_ids:
                ranking_rows.append(
                    {
                        "source": label,
                        "query_patent_id": query_id,
                        "ranking": [
                            {
                                "rank": index + 1,
                                "patent_id": patent_id,
                                "score": round(score, 8),
                            }
                            for index, (patent_id, score) in enumerate(rankings[query_id])
                        ],
                    }
                )
            for task in ("Citation", "Conflict-X"):
                task_rows: list[dict[str, Any]] = []
                for query_id in sorted(t2_query_sets.get(task, set())):
                    gold = ground_truth[query_id][task]
                    metrics = evaluate_ranking(rankings[query_id], gold)
                    row = {
                        "source": label,
                        "query_patent_id": query_id,
                        "task": task,
                        "gold_count": len(gold),
                        "source_candidate_count_before_mask": len(candidate_ids) - 1,
                        "candidate_count_ranked": len(eligible_candidate_ids) - 1,
                        "ranking_rows_written": len(rankings[query_id]),
                        **metrics,
                    }
                    task_rows.append(row)
                    per_query_rows.append(row)
                result_rows.append(
                    {
                        "source": label,
                        "task": task,
                        "Queries": len(task_rows),
                        **{
                            metric: sum(row[metric] for row in task_rows) / len(task_rows)
                            for metric in METRIC_NAMES
                        },
                    }
                )
    finally:
        client.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "t1_matched_subset_results.csv"
    per_query_path = args.output_dir / "t1_matched_subset_per_query.csv"
    ranking_path = (
        args.output_dir
        / f"t1_matched_subset_rankings_top{args.ranking_limit}.jsonl"
    )
    manifest_path = args.output_dir / "t1_matched_subset_manifest.json"
    write_csv_atomic(result_path, ["source", "task", "Queries", *METRIC_NAMES], result_rows)
    write_csv_atomic(
        per_query_path,
        [
            "source",
            "query_patent_id",
            "task",
            "gold_count",
            "source_candidate_count_before_mask",
            "candidate_count_ranked",
            "ranking_rows_written",
            *METRIC_NAMES,
        ],
        per_query_rows,
    )
    write_jsonl_atomic(ranking_path, ranking_rows)
    manifest = {
        "status": "completed",
        "method_group": "text_baseline_matched_subset",
        "candidate_pool_policy": (
            "Rank only the B2 structure-bearing subset of the original GT-derived common candidates, "
            "excluding the query itself"
        ),
        "candidate_pool_scope": "oracle-derived closed pool; not open-corpus performance",
        "score_definition": "Maximum cosine between query and candidate units",
        "query_subset_policy": "Evaluate only complete-gold queries in the B2 per-query results",
        "source_candidate_count": len(candidate_ids),
        "candidate_count": len(eligible_candidate_ids),
        "evaluation_candidate_count": len(eligible_candidate_ids),
        "ranked_candidates_per_query": len(eligible_candidate_ids) - 1,
        "ground_truth_field_mapping": ground_truth_field_mapping,
        "query_counts": {task: len(values) for task, values in t2_query_sets.items()},
        "ranking_limit": args.ranking_limit,
        "metric_definitions": METRIC_DEFINITIONS,
        "sources": source_stats,
        "results": result_rows,
        "provenance": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "input_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                args.ground_truth,
                args.candidate_patents,
                args.structures,
                args.t2_per_query,
            )
        },
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "output_files": {},
    }
    for path in (result_path, per_query_path, ranking_path):
        manifest["output_files"][str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
