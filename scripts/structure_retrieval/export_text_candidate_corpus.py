#!/usr/bin/env python3
"""Read-only export of candidate patent IDs from T1 Qdrant by evaluation unit. T2 must use the same candidate corpus. Save each collection/unit_type patent set and record common candidates/set differences in the manifest; never modify collections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_patents(path: Path, patent_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patent_id"])
        writer.writeheader()
        for patent_id in sorted(patent_ids):
            writer.writerow({"patent_id": patent_id})


def parse_source(value: str) -> tuple[str, str, str]:
    """Validate label=collection:unit_type format."""
    try:
        label, remainder = value.split("=", 1)
        collection, unit_type = remainder.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--source must use label=collection:unit_type format"
        ) from exc
    if not label.strip() or not collection.strip() or not unit_type.strip():
        raise argparse.ArgumentTypeError("--source label, collection, and unit_type cannot be empty")
    return label.strip(), collection.strip(), unit_type.strip()


def extract_patents(
    client: QdrantClient,
    collection: str,
    unit_type: str,
    batch_size: int,
) -> tuple[set[str], dict[str, int]]:
    patent_ids: set[str] = set()
    point_count = 0
    missing_patent_id = 0
    offset = None
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="unit_type",
                match=models.MatchValue(value=unit_type),
            )
        ]
    )
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=["patent_id", "unit_type"],
            with_vectors=False,
        )
        for point in points:
            point_count += 1
            patent_id = str((point.payload or {}).get("patent_id") or "").strip()
            if patent_id:
                patent_ids.add(patent_id)
            else:
                missing_patent_id += 1
        if offset is None:
            break
    return patent_ids, {
        "point_count": point_count,
        "unique_patent_count": len(patent_ids),
        "points_without_patent_id": missing_patent_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-path", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=parse_source,
        action="append",
        required=True,
        help="label=collection:unit_type format; repeatable",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    if not args.qdrant_path.is_dir():
        raise SystemExit(f"Qdrant path missing: {args.qdrant_path}")
    labels = [label for label, _, _ in args.source]
    if len(labels) != len(set(labels)):
        parser.error("--source labels must be unique")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(args.qdrant_path))
    available_collections = {item.name for item in client.get_collections().collections}

    patent_sets: dict[str, set[str]] = {}
    source_stats: dict[str, dict[str, Any]] = {}
    for label, collection, unit_type in args.source:
        if collection not in available_collections:
            raise SystemExit(f"Qdrant collection missing: {collection}")
        patents, stats = extract_patents(client, collection, unit_type, args.batch_size)
        patent_sets[label] = patents
        source_stats[label] = {
            "collection": collection,
            "unit_type": unit_type,
            **stats,
        }
        write_patents(args.output_dir / f"text_candidate_patents_{label}.csv", patents)

    common = set.intersection(*patent_sets.values()) if patent_sets else set()
    union = set.union(*patent_sets.values()) if patent_sets else set()
    write_patents(args.output_dir / "text_candidate_patents_common.csv", common)
    write_patents(args.output_dir / "text_candidate_patents_union.csv", union)

    membership_path = args.output_dir / "text_candidate_patent_membership.csv"
    with membership_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["patent_id", *[f"in_{label}" for label in labels], "in_all_sources"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for patent_id in sorted(union):
            row = {"patent_id": patent_id}
            for label in labels:
                row[f"in_{label}"] = patent_id in patent_sets[label]
            row["in_all_sources"] = patent_id in common
            writer.writerow(row)

    pairwise_differences: dict[str, dict[str, int]] = {}
    for left in labels:
        pairwise_differences[left] = {}
        for right in labels:
            pairwise_differences[left][f"not_in_{right}"] = len(
                patent_sets[left] - patent_sets[right]
            )

    manifest = {
        "status": "candidate_sets_identical" if len(common) == len(union) else "candidate_sets_differ",
        "qdrant_path": str(args.qdrant_path),
        "qdrant_path_fingerprint": hashlib.sha256(
            str(args.qdrant_path.resolve()).encode("utf-8")
        ).hexdigest(),
        "sources": source_stats,
        "common_patent_count": len(common),
        "union_patent_count": len(union),
        "pairwise_differences": pairwise_differences,
        "output_policy": (
            "T2 uses the same source-specific candidate file as the compared T1 baseline. "
            "Use the common candidate file only when all source sets are identical."
        ),
    }
    write_json(args.output_dir / "text_candidate_corpus_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
