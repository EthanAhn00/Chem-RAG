#!/usr/bin/env python3
"""Evaluate T2 using SMILES-based Morgan patent vectors. Convert valid canonical SMILES to binary Morgan fingerprints, average them per patent and L2-normalize. Rank by patent-vector cosine, not maximum molecule-pair Tanimoto; a reproducible initial baseline for large structure sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EP_EXACT_RE = re.compile(r"^EP(\d+)([A-Z]\d?)$")
DEFAULT_KS = (1, 5, 10, 50, 100)
METRIC_DEFINITIONS = {
    "R@k": (
        "Per-query retrieved gold in Top-k divided by all query gold, then "
        "averaged over queries"
    ),
    "MRR@100": (
        "Reciprocal rank of the first gold in Top-100, or zero when absent, "
        "averaged over queries"
    ),
    "MAP@100": (
        "Sum precision at gold ranks within Top-100 and divide by all query gold; "
        "average AP@100 over queries, including beyond-cutoff gold in the denominator"
    ),
    "nDCG@10": (
        "Normalize binary DCG@10 by ideal DCG from total gold and cutoff 10, "
        "then average over queries"
    ),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_exact_ep(value: Any) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").strip().upper())
    match = EP_EXACT_RE.fullmatch(compact)
    if not match:
        return None
    return f"EP{match.group(1)}{match.group(2)}"


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


def load_candidates(path: Path, id_column: str) -> list[str]:
    candidates: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise SystemExit(f"Candidate file lacks {id_column!r} column missing: {path}")
        for row_number, row in enumerate(reader, start=2):
            patent_id = normalize_exact_ep(row.get(id_column))
            if patent_id is None:
                raise SystemExit(
                    f"Not an EP publication identifier with a kind code: "
                    f"{path}:{row_number}:{row.get(id_column)!r}"
                )
            candidates.add(patent_id)
    if not candidates:
        raise SystemExit(f"No candidate patents: {path}")
    return sorted(candidates)


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
    ground_truth: dict[str, dict[str, set[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = normalize_exact_ep(row.get("query_patent_id"))
            if query_id is None:
                raise SystemExit(
                    f"Invalid query patent identifier: {path}:{row_number}"
                )
            all_raw = row.get("all_cited_patents") or []
            if conflict_field not in row:
                raise SystemExit(
                    f"Requested X-category field missing: "
                    f"{path}:{row_number}:{conflict_field}"
                )
            x_raw = row.get(conflict_field) or []
            alternate_raw = row.get(alternate_field) or [] if alternate_field in row else None
            if not isinstance(all_raw, list) or not isinstance(x_raw, list):
                raise SystemExit(f"Invalid gold-list format: {path}:{row_number}")
            if alternate_raw is not None and not isinstance(alternate_raw, list):
                raise SystemExit(f"Invalid compatibility gold-list format: {path}:{row_number}")

            def normalize_list(values: list[Any], label: str) -> set[str]:
                output: set[str] = set()
                for value in values:
                    patent_id = normalize_exact_ep(value)
                    if patent_id is None:
                        raise SystemExit(
                            f"Invalid {label} patent identifier: "
                            f"{path}:{row_number}:{value!r}"
                        )
                    output.add(patent_id)
                return output

            all_gold = normalize_list(all_raw, "all gold")
            x_gold = normalize_list(x_raw, "X gold")
            if alternate_raw is not None:
                alternate_gold = normalize_list(alternate_raw, "compatibility X gold")
                if x_gold != alternate_gold:
                    raise SystemExit(
                        "X-category compatibility fields disagree: "
                        f"{path}:{row_number}:{conflict_field}!={alternate_field}"
                    )
            if not x_gold.issubset(all_gold):
                raise SystemExit(f"X gold is not a subset of all gold: {query_id}")
            if query_id in ground_truth:
                raise SystemExit(f"Duplicate query: {query_id}")
            ground_truth[query_id] = {"Citation": all_gold, "Conflict-X": x_gold}
    return ground_truth, {
        "Citation": "all_cited_patents",
        "Conflict-X": conflict_field,
    }


def load_structure_sets(
    path: Path,
    candidates: set[str],
    smiles_column: str = "canonical_smiles",
) -> tuple[dict[str, set[str]], dict[str, int]]:
    structures: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    interned_smiles: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patent_id", smiles_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Structure file lacks required columns: {sorted(missing)}")
        for row in reader:
            counts["input_rows"] += 1
            patent_id = normalize_exact_ep(row.get("patent_id"))
            if patent_id is None:
                counts["invalid_patent_id_rows"] += 1
                continue
            if patent_id not in candidates:
                counts["outside_candidate_rows"] += 1
                continue
            smiles = str(row.get(smiles_column) or "").strip()
            if not smiles:
                counts["empty_smiles_rows"] += 1
                continue
            smiles = interned_smiles.setdefault(smiles, smiles)
            before = len(structures[patent_id])
            structures[patent_id].add(smiles)
            if len(structures[patent_id]) == before:
                counts["duplicate_patent_smiles_rows"] += 1
    counts["candidate_patents_with_structures"] = len(structures)
    counts["unique_smiles"] = len(interned_smiles)
    counts["unique_patent_smiles_links"] = sum(map(len, structures.values()))
    return dict(structures), dict(sorted(counts.items()))


def build_patent_vectors(
    candidate_ids: list[str],
    structures: dict[str, set[str]],
    radius: int,
    bits: int,
) -> tuple[Any, Any, dict[str, int]]:
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=bits,
    )
    patent_index = {patent_id: index for index, patent_id in enumerate(candidate_ids)}
    smiles_patents: dict[str, list[int]] = defaultdict(list)
    for patent_id, smiles_values in structures.items():
        patent_position = patent_index[patent_id]
        for smiles in smiles_values:
            smiles_patents[smiles].append(patent_position)

    sums = np.zeros((len(candidate_ids), bits), dtype=np.float32)
    structure_counts = np.zeros(len(candidate_ids), dtype=np.int32)
    invalid_smiles = 0
    fingerprint = np.zeros(bits, dtype=np.uint8)
    for number, (smiles, patent_positions) in enumerate(smiles_patents.items(), start=1):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid_smiles += 1
            continue
        fingerprint.fill(0)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), fingerprint)
        positions = np.asarray(patent_positions, dtype=np.int64)
        sums[positions] += fingerprint
        structure_counts[positions] += 1
        if number % 10000 == 0:
            print(
                f"  Morgan fingerprint {number}/{len(smiles_patents)} processed",
                flush=True,
            )

    nonempty = structure_counts > 0
    sums[nonempty] /= structure_counts[nonempty, None]
    norms = np.linalg.norm(sums, axis=1)
    usable = norms > 0
    sums[usable] /= norms[usable, None]
    metadata = {
        "invalid_canonical_smiles": invalid_smiles,
        "rdkit_version": rdBase.rdkitVersion,
        "patents_with_nonzero_vector": int(usable.sum()),
    }
    return sums, structure_counts, metadata


def score_matrix(vectors: Any, query_positions: list[int], device: str) -> tuple[Any, str]:
    import numpy as np

    selected_device = device
    if device == "auto":
        try:
            import torch

            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            selected_device = "cpu"

    if selected_device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("--device cuda requires PyTorch") from exc
        query_tensor = torch.from_numpy(vectors[query_positions]).to("cuda")
        candidate_tensor = torch.from_numpy(vectors).to("cuda")
        scores = (query_tensor @ candidate_tensor.T).cpu().numpy()
        del query_tensor, candidate_tensor
        return scores, "cuda"
    if selected_device != "cpu":
        raise SystemExit("--device must be auto, cpu, or cuda")
    return np.asarray(vectors[query_positions] @ vectors.T), "cpu"


def rank_candidates(
    candidate_ids: list[str],
    scores: Any,
    query_id: str,
    usable_mask: Any,
    limit: int,
) -> list[tuple[str, float]]:
    import numpy as np

    valid = np.asarray(usable_mask, dtype=bool).copy()
    candidate_array = np.asarray(candidate_ids, dtype=object)
    valid[candidate_array == query_id] = False
    valid_positions = np.flatnonzero(valid)
    valid_ids = candidate_array[valid_positions]
    score_array = np.asarray(scores)
    valid_scores = np.asarray(score_array[valid_positions], dtype=np.float64)
    order = np.lexsort((valid_ids, -valid_scores))
    selected = order[:limit]
    return [
        (str(valid_ids[index]), float(valid_scores[index]))
        for index in selected
    ]


def evaluate_ranking(
    ranking: list[tuple[str, float]],
    relevant: set[str],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, float]:
    hits = [int(patent_id in relevant) for patent_id, _ in ranking]
    result = {
        f"R@{k}": sum(hits[:k]) / len(relevant)
        for k in ks
    }
    result["MRR@100"] = next(
        (1.0 / (index + 1) for index, hit in enumerate(hits[:100]) if hit),
        0.0,
    )
    hit_count = 0
    precision_sum = 0.0
    for index, hit in enumerate(hits[:100]):
        if hit:
            hit_count += 1
            precision_sum += hit_count / (index + 1)
    result["MAP@100"] = precision_sum / len(relevant)

    dcg = sum(
        hit / math.log2(index + 2)
        for index, hit in enumerate(hits[:10])
    )
    idcg = sum(
        1.0 / math.log2(index + 2)
        for index in range(min(len(relevant), 10))
    )
    result["nDCG@10"] = dcg / idcg if idcg else 0.0
    return result


def summarize_counts(values: Any) -> dict[str, float | int | None]:
    import numpy as np

    values = np.asarray(values)
    values = values[values > 0]
    if not len(values):
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "min": int(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(values.max()),
    }


def select_complete_gold_queries(
    ground_truth: dict[str, dict[str, set[str]]],
    task: str,
    usable_patents: set[str],
    ranking_queries: set[str],
) -> tuple[list[tuple[str, set[str]]], Counter[str]]:
    selected: list[tuple[str, set[str]]] = []
    skip_counts: Counter[str] = Counter()
    for query_id, gold_by_task in ground_truth.items():
        task_gold = gold_by_task[task]
        if not task_gold:
            skip_counts["no_gold"] += 1
            continue
        if query_id not in usable_patents:
            skip_counts["query_without_structure"] += 1
            continue
        if not task_gold.issubset(usable_patents):
            skip_counts["incomplete_gold_structure"] += 1
            continue
        if query_id not in ranking_queries:
            skip_counts["query_outside_candidate_corpus"] += 1
            continue
        selected.append((query_id, task_gold))
    return selected, skip_counts


def run_self_test() -> None:
    ranking = [("P1", 0.9), ("P2", 0.8), ("P3", 0.7), ("P4", 0.6), ("P5", 0.5)]
    metrics = evaluate_ranking(ranking, {"P2", "P5"})
    assert abs(metrics["R@1"] - 0.0) < 1e-12
    assert abs(metrics["R@5"] - 1.0) < 1e-12
    assert abs(metrics["MRR@100"] - 0.5) < 1e-12
    assert abs(metrics["MAP@100"] - 0.45) < 1e-12
    ground_truth = {
        "Q1": {"Citation": {"P1"}, "Conflict-X": set()},
        "Q2": {"Citation": {"P2", "P3"}, "Conflict-X": {"P3"}},
        "Q3": {"Citation": {"P4"}, "Conflict-X": {"P4"}},
    }
    selected, skipped = select_complete_gold_queries(
        ground_truth,
        "Citation",
        {"Q1", "Q2", "P1", "P2", "P4"},
        {"Q1", "Q2"},
    )
    assert selected == [("Q1", {"P1"})]
    assert skipped == Counter(
        {"incomplete_gold_structure": 1, "query_without_structure": 1}
    )
    tied = rank_candidates(
        ["EP3A1", "EP1A1", "EP2A1", "EP4A1"],
        [1.0, 0.5, 0.5, 0.5],
        "EP3A1",
        [True, True, True, False],
        100,
    )
    assert [patent_id for patent_id, _ in tied] == ["EP1A1", "EP2A1"]
    print("Self-check passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--candidate-patents", type=Path)
    parser.add_argument("--candidate-id-column", default="patent_id")
    parser.add_argument("--structures", type=Path)
    parser.add_argument(
        "--smiles-column",
        default="canonical_smiles",
        help="SMILES column used for fingerprint inputs in the structure file",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
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
    required = {
        "--ground-truth": args.ground_truth,
        "--candidate-patents": args.candidate_patents,
        "--structures": args.structures,
        "--output-dir": args.output_dir,
    }
    missing_arguments = [name for name, value in required.items() if value is None]
    if missing_arguments:
        parser.error(f"Missing required arguments: {', '.join(missing_arguments)}")
    missing_files = [str(path) for path in required.values() if path != args.output_dir and not path.is_file()]
    if missing_files:
        raise SystemExit(f"Input file missing: {missing_files}")
    if args.fingerprint_radius < 1 or args.fingerprint_bits < 128:
        parser.error("Check fingerprint settings")
    if args.ranking_limit < max(DEFAULT_KS):
        parser.error(f"--ranking-limit must be {max(DEFAULT_KS)} or greater")

    started_at = time.perf_counter()
    candidate_ids = load_candidates(args.candidate_patents, args.candidate_id_column)
    candidate_set = set(candidate_ids)
    ground_truth, ground_truth_field_mapping = load_ground_truth(
        args.ground_truth,
        args.conflict_field,
    )
    structures, structure_load_stats = load_structure_sets(
        args.structures,
        candidate_set,
        args.smiles_column,
    )
    print(
        f"Candidates: {len(candidate_ids)}; structure-bearing patents {len(structures)} loaded.",
        flush=True,
    )

    vectors, structure_counts, vector_metadata = build_patent_vectors(
        candidate_ids,
        structures,
        radius=args.fingerprint_radius,
        bits=args.fingerprint_bits,
    )
    candidate_index = {patent_id: index for index, patent_id in enumerate(candidate_ids)}
    usable_patents = {
        patent_id
        for patent_id, index in candidate_index.items()
        if structure_counts[index] > 0
    }
    query_ids = sorted(
        query_id
        for query_id in ground_truth
        if query_id in usable_patents and query_id in candidate_index
    )
    query_positions = [candidate_index[query_id] for query_id in query_ids]
    scores, selected_device = score_matrix(vectors, query_positions, args.device)
    usable_mask = structure_counts > 0

    rankings: dict[str, list[tuple[str, float]]] = {}
    for row_index, query_id in enumerate(query_ids):
        rankings[query_id] = rank_candidates(
            candidate_ids,
            scores[row_index],
            query_id,
            usable_mask,
            args.ranking_limit,
        )

    metric_names = [f"R@{k}" for k in DEFAULT_KS] + ["MRR@100", "MAP@100", "nDCG@10"]
    per_query_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    eligibility_counts: dict[str, dict[str, int]] = {}
    for task in ("Citation", "Conflict-X"):
        rows_for_task: list[dict[str, Any]] = []
        selected_queries, skip_counts = select_complete_gold_queries(
            ground_truth,
            task,
            usable_patents,
            set(rankings),
        )
        for query_id, task_gold in selected_queries:
            metrics = evaluate_ranking(rankings[query_id], task_gold)
            best_rank = next(
                (
                    index + 1
                    for index, (patent_id, _) in enumerate(rankings[query_id])
                    if patent_id in task_gold
                ),
                None,
            )
            row = {
                "method": "mean_morgan_cosine",
                "query_patent_id": query_id,
                "task": task,
                "gold_count": len(task_gold),
                "source_candidate_count_before_mask": len(candidate_ids) - 1,
                "candidate_count_ranked": len(usable_patents) - 1,
                "ranking_rows_written": len(rankings[query_id]),
                "query_structure_count": int(structure_counts[candidate_index[query_id]]),
                "best_gold_rank": best_rank or "",
                **metrics,
            }
            rows_for_task.append(row)
            per_query_rows.append(row)

        eligibility_counts[task] = {
            "evaluated_queries": len(rows_for_task),
            **dict(sorted(skip_counts.items())),
        }
        task_rows.append(
            {
                "method": "mean_morgan_cosine",
                "task": task,
                "Queries": len(rows_for_task),
                **{
                    metric: (
                        sum(row[metric] for row in rows_for_task) / len(rows_for_task)
                        if rows_for_task
                        else ""
                    )
                    for metric in metric_names
                },
            }
        )

    ranking_rows = [
        {
            "method": "mean_morgan_cosine",
            "query_patent_id": query_id,
            "ranking": [
                {"rank": index + 1, "patent_id": patent_id, "score": round(score, 8)}
                for index, (patent_id, score) in enumerate(rankings[query_id])
            ],
        }
        for query_id in query_ids
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "t2_structure_retrieval_results.csv"
    per_query_path = args.output_dir / "t2_structure_retrieval_per_query.csv"
    ranking_path = (
        args.output_dir
        / f"t2_structure_rankings_top{args.ranking_limit}.jsonl"
    )
    manifest_path = args.output_dir / "t2_structure_retrieval_manifest.json"
    write_csv_atomic(
        result_path,
        ["method", "task", "Queries", *metric_names],
        task_rows,
    )
    write_csv_atomic(
        per_query_path,
        [
            "method",
            "query_patent_id",
            "task",
            "gold_count",
            "source_candidate_count_before_mask",
            "candidate_count_ranked",
            "ranking_rows_written",
            "query_structure_count",
            "best_gold_rank",
            *metric_names,
        ],
        per_query_rows,
    )
    write_jsonl_atomic(ranking_path, ranking_rows)

    manifest = {
        "status": "completed",
        "method": "mean_morgan_cosine",
        "method_key": "mean_morgan_cosine",
        "display_name": "Patent-mean Morgan fingerprint cosine",
        "score_definition": (
            f"{args.smiles_column} column SMILES -> Morgan binary fingerprints, "
            "averaged per patent and L2-normalized; score by vector cosine similarity"
        ),
        "not_molecular_pair_max_tanimoto": True,
        "candidate_pool_policy": (
            "Structure-bearing patents in original GT-derived common candidates; self excluded; "
            "exclude missing-structure patents rather than assigning zero scores"
        ),
        "candidate_pool_scope": "oracle-derived closed pool; not open-corpus performance",
        "evaluation_subset_policy": (
            "Evaluate only complete-gold queries with structures for the query and all task-specific gold"
        ),
        "metric_compatibility": (
            "Same R@1/5/10/50/100, MRR@100, MAP@100, binary nDCG@10 formulas as T1; "
            "reaggregate T1 on the identical complete-gold subset rather than comparing all-query summaries"
        ),
        "metric_definitions": METRIC_DEFINITIONS,
        "source_candidate_count": len(candidate_ids),
        "candidate_count": len(usable_patents),
        "ranked_candidates_per_query": len(usable_patents) - 1,
        "candidate_patents_with_structures": int((structure_counts > 0).sum()),
        "queries_with_structure_and_in_candidate_corpus": len(query_ids),
        "ground_truth_field_mapping": ground_truth_field_mapping,
        "fingerprint": {
            "type": "Morgan binary fingerprint",
            "radius": args.fingerprint_radius,
            "bits": args.fingerprint_bits,
        },
        "smiles_column": args.smiles_column,
        "provenance": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "device": selected_device,
        "ranking_limit": args.ranking_limit,
        "eligibility": eligibility_counts,
        "structure_load_stats": structure_load_stats,
        "structure_count_by_patent": summarize_counts(structure_counts),
        "vector_metadata": vector_metadata,
        "input_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (args.ground_truth, args.candidate_patents, args.structures)
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
    print(json.dumps({"results": task_rows, "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
