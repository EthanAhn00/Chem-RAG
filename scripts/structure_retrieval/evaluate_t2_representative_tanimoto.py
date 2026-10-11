#!/usr/bin/env python3
"""다양성 대표 구조의 최대 Tanimoto로 T2 structure-only 검색을 평가한다.

특허별 구조 수 차이를 제한하기 위해 Morgan fingerprint 공간에서 결정적
farthest-first 방식으로 최대 K개의 다양성 대표를 선택한다. 틴 특허와 후보 특허
대표 분자쌍의 RDKit exact Tanimoto 최댓값을 특허 점수로 사용한다. 전체
구조쌍 최댓값이 아니므로 대표 제한 K를 반드시 함께 보고한다.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .evaluate_t2_structure_retrieval import (
        DEFAULT_KS,
        METRIC_DEFINITIONS,
        evaluate_ranking,
        load_candidates,
        load_ground_truth,
        load_structure_sets,
        select_complete_gold_queries,
        sha256_file,
        summarize_counts,
        write_csv_atomic,
        write_json_atomic,
        write_jsonl_atomic,
    )
except ImportError:
    from evaluate_t2_structure_retrieval import (
        DEFAULT_KS,
        METRIC_DEFINITIONS,
        evaluate_ranking,
        load_candidates,
        load_ground_truth,
        load_structure_sets,
        select_complete_gold_queries,
        sha256_file,
        summarize_counts,
        write_csv_atomic,
        write_json_atomic,
        write_jsonl_atomic,
    )


def build_fingerprints(
    structures: dict[str, set[str]],
    radius: int,
    bits: int,
) -> tuple[dict[str, Any], str]:
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    unique_smiles = sorted({smiles for values in structures.values() for smiles in values})
    fingerprints: dict[str, Any] = {}
    for number, smiles in enumerate(unique_smiles, start=1):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise SystemExit(f"이미 검증된 canonical SMILES 파싱에 실패했습니다: {smiles}")
        fingerprints[smiles] = generator.GetFingerprint(molecule)
        if number % 10000 == 0:
            print(f"  Morgan fingerprint {number}/{len(unique_smiles)}개 처리", flush=True)
    return fingerprints, rdBase.rdkitVersion


def choose_diverse_representatives(
    smiles_values: set[str],
    fingerprints: dict[str, Any],
    maximum: int,
) -> list[str]:
    import numpy as np
    from rdkit import DataStructs

    ordered = sorted(smiles_values)
    if len(ordered) <= maximum:
        return ordered
    fps = [fingerprints[smiles] for smiles in ordered]
    bit_counts = np.asarray([fp.GetNumOnBits() for fp in fps], dtype=np.float64)
    median = float(np.median(bit_counts))
    first = min(
        range(len(ordered)),
        key=lambda index: (abs(bit_counts[index] - median), ordered[index]),
    )
    selected = [first]
    available = np.ones(len(ordered), dtype=bool)
    available[first] = False
    minimum_distances = np.ones(len(ordered), dtype=np.float64)
    while len(selected) < maximum:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fps[selected[-1]], fps),
            dtype=np.float64,
        )
        minimum_distances = np.minimum(minimum_distances, 1.0 - similarities)
        minimum_distances[~available] = -1.0
        next_index = int(np.argmax(minimum_distances))
        selected.append(next_index)
        available[next_index] = False
    return [ordered[index] for index in selected]


def build_representatives(
    candidate_ids: list[str],
    structures: dict[str, set[str]],
    fingerprints: dict[str, Any],
    maximum: int,
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    representatives: dict[str, list[Any]] = {}
    counts: dict[str, int] = {}
    for number, patent_id in enumerate(candidate_ids, start=1):
        smiles_values = structures.get(patent_id, set())
        if not smiles_values:
            continue
        selected = choose_diverse_representatives(smiles_values, fingerprints, maximum)
        representatives[patent_id] = [fingerprints[smiles] for smiles in selected]
        counts[patent_id] = len(selected)
        if number % 250 == 0:
            print(f"  대표 구조 {number}/{len(candidate_ids)}개 특허 처리", flush=True)
    return representatives, counts


def make_rankings(
    candidate_ids: list[str],
    representatives: dict[str, list[Any]],
    query_ids: list[str],
    limit: int,
) -> dict[str, list[tuple[str, float]]]:
    import numpy as np
    from rdkit import DataStructs

    flat_fingerprints: list[Any] = []
    flat_patent_positions: list[int] = []
    candidate_index = {patent_id: index for index, patent_id in enumerate(candidate_ids)}
    for patent_id in candidate_ids:
        for fingerprint in representatives.get(patent_id, []):
            flat_fingerprints.append(fingerprint)
            flat_patent_positions.append(candidate_index[patent_id])
    positions = np.asarray(flat_patent_positions, dtype=np.int32)
    candidate_array = np.asarray(candidate_ids, dtype=object)
    usable = np.asarray([patent_id in representatives for patent_id in candidate_ids])
    rankings: dict[str, list[tuple[str, float]]] = {}
    for number, query_id in enumerate(query_ids, start=1):
        patent_scores = np.full(len(candidate_ids), -np.inf, dtype=np.float32)
        for query_fingerprint in representatives[query_id]:
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(
                    query_fingerprint,
                    flat_fingerprints,
                ),
                dtype=np.float32,
            )
            np.maximum.at(patent_scores, positions, similarities)
        valid = usable & (candidate_array != query_id)
        valid_positions = np.flatnonzero(valid)
        valid_ids = candidate_array[valid_positions]
        valid_scores = patent_scores[valid_positions]
        order = np.lexsort((valid_ids, -valid_scores))[:limit]
        rankings[query_id] = [
            (str(valid_ids[index]), float(valid_scores[index]))
            for index in order
        ]
        if number % 50 == 0:
            print(f"  Tanimoto ranking {number}/{len(query_ids)} query 처리", flush=True)
    return rankings


def run_self_test() -> None:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    metrics = evaluate_ranking([("P1", 0.9), ("P2", 0.8)], {"P2"})
    assert metrics["MAP@100"] == 0.5
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    smiles = ["CCO", "CCN", "c1ccccc1", "CCCl"]
    fingerprints = {
        value: generator.GetFingerprint(Chem.MolFromSmiles(value))
        for value in smiles
    }
    selected_once = choose_diverse_representatives(
        set(smiles),
        fingerprints,
        2,
    )
    selected_twice = choose_diverse_representatives(
        set(reversed(smiles)),
        fingerprints,
        2,
    )
    assert selected_once == selected_twice
    assert len(selected_once) == 2
    assert choose_diverse_representatives(set(smiles[:2]), fingerprints, 4) == sorted(
        smiles[:2]
    )

    structures = {
        "EP1A1": {"CCO"},
        "EP2A1": {"CCO"},
        "EP3A1": {"CCO"},
        "EP4A1": {"c1ccccc1"},
    }
    representatives, counts = build_representatives(
        ["EP1A1", "EP2A1", "EP3A1", "EP4A1"],
        structures,
        fingerprints,
        1,
    )
    assert counts == {"EP1A1": 1, "EP2A1": 1, "EP3A1": 1, "EP4A1": 1}
    rankings = make_rankings(
        ["EP1A1", "EP2A1", "EP3A1", "EP4A1"],
        representatives,
        ["EP1A1"],
        3,
    )
    assert [patent_id for patent_id, _ in rankings["EP1A1"][:2]] == [
        "EP2A1",
        "EP3A1",
    ]
    assert all(patent_id != "EP1A1" for patent_id, _ in rankings["EP1A1"])
    print("자체 검증을 통과했습니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--candidate-patents", type=Path)
    parser.add_argument("--candidate-id-column", default="patent_id")
    parser.add_argument("--structures", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--max-representatives", type=int, default=16)
    parser.add_argument("--ranking-limit", type=int, default=100)
    parser.add_argument(
        "--conflict-field",
        choices=("conflict_patents", "x_category_patents"),
        default="conflict_patents",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    required = [args.ground_truth, args.candidate_patents, args.structures, args.output_dir]
    if any(path is None for path in required):
        parser.error("ground truth, 후보, 구조, 출력 인자가 필요합니다")
    for path in (args.ground_truth, args.candidate_patents, args.structures):
        if not path.is_file():
            raise SystemExit(f"입력 파일이 없습니다: {path}")
    if args.max_representatives < 1:
        parser.error("--max-representatives는 1 이상이어야 합니다")
    if args.ranking_limit < 100:
        parser.error("--ranking-limit은 100 이상이어야 합니다")

    started_at = time.perf_counter()
    candidate_ids = load_candidates(args.candidate_patents, args.candidate_id_column)
    candidate_set = set(candidate_ids)
    ground_truth, ground_truth_field_mapping = load_ground_truth(
        args.ground_truth,
        args.conflict_field,
    )
    structures, structure_stats = load_structure_sets(args.structures, candidate_set)
    fingerprints, rdkit_version = build_fingerprints(
        structures,
        args.fingerprint_radius,
        args.fingerprint_bits,
    )
    representatives, representative_counts = build_representatives(
        candidate_ids,
        structures,
        fingerprints,
        args.max_representatives,
    )
    usable_patents = set(representatives)
    query_ids = sorted(query_id for query_id in ground_truth if query_id in usable_patents)
    rankings = make_rankings(candidate_ids, representatives, query_ids, args.ranking_limit)

    metric_names = [f"R@{k}" for k in DEFAULT_KS] + ["MRR@100", "MAP@100", "nDCG@10"]
    per_query_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    eligibility: dict[str, dict[str, int]] = {}
    for task in ("Citation", "Conflict-X"):
        selected, skipped = select_complete_gold_queries(
            ground_truth,
            task,
            usable_patents,
            set(rankings),
        )
        task_rows: list[dict[str, Any]] = []
        for query_id, gold in selected:
            metrics = evaluate_ranking(rankings[query_id], gold)
            row = {
                "method": "representative_max_tanimoto",
                "max_representatives": args.max_representatives,
                "query_patent_id": query_id,
                "task": task,
                "gold_count": len(gold),
                "source_candidate_count_before_mask": len(candidate_ids) - 1,
                "candidate_count_ranked": len(usable_patents) - 1,
                "ranking_rows_written": len(rankings[query_id]),
                "query_structure_count": len(structures[query_id]),
                "query_representative_count": representative_counts[query_id],
                **metrics,
            }
            task_rows.append(row)
            per_query_rows.append(row)
        eligibility[task] = {"evaluated_queries": len(task_rows), **dict(sorted(skipped.items()))}
        result_rows.append(
            {
                "method": "representative_max_tanimoto",
                "max_representatives": args.max_representatives,
                "task": task,
                "Queries": len(task_rows),
                **{
                    metric: sum(row[metric] for row in task_rows) / len(task_rows)
                    for metric in metric_names
                },
            }
        )

    ranking_rows = [
        {
            "method": "representative_max_tanimoto",
            "max_representatives": args.max_representatives,
            "query_patent_id": query_id,
            "ranking": [
                {"rank": rank, "patent_id": patent_id, "score": round(score, 8)}
                for rank, (patent_id, score) in enumerate(rankings[query_id], start=1)
            ],
        }
        for query_id in query_ids
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "t2_representative_tanimoto_results.csv"
    per_query_path = args.output_dir / "t2_representative_tanimoto_per_query.csv"
    ranking_path = args.output_dir / "t2_representative_tanimoto_rankings_top100.jsonl"
    manifest_path = args.output_dir / "t2_representative_tanimoto_manifest.json"
    write_csv_atomic(
        result_path,
        ["method", "max_representatives", "task", "Queries", *metric_names],
        result_rows,
    )
    write_csv_atomic(
        per_query_path,
        [
            "method",
            "max_representatives",
            "query_patent_id",
            "task",
            "gold_count",
            "source_candidate_count_before_mask",
            "candidate_count_ranked",
            "ranking_rows_written",
            "query_structure_count",
            "query_representative_count",
            *metric_names,
        ],
        per_query_rows,
    )
    write_jsonl_atomic(ranking_path, ranking_rows)
    manifest = {
        "status": "completed",
        "method": "representative_max_tanimoto",
        "method_key": "representative_max_tanimoto",
        "display_name": (
            "Representative max Tanimoto "
            f"(farthest-first K={args.max_representatives} approximation)"
        ),
        "score_definition": "query/candidate 다양성 대표 분자쌍의 RDKit exact Tanimoto 최댓값",
        "exact_full_structure_set_max_tanimoto": False,
        "approximation_scope": (
            "특허당 farthest-first 대표 구조를 최대 K개만 사용한 근사치이며, "
            "전체 구조 집합의 분자쌍 최대 Tanimoto로 일반화하지 않음"
        ),
        "representative_selection": (
            "Morgan fingerprint 비트 수의 중앙값에 가까운 구조에서 시작하는 "
            "결정적 farthest-first diversity selection"
        ),
        "max_representatives_per_patent": args.max_representatives,
        "candidate_pool_policy": (
            "GT 등장 특허로 제한된 원 공통 후보 중 구조 보유 특허만 랭킹하고 "
            "자기 특허 제외"
        ),
        "candidate_pool_scope": "oracle-derived closed pool; open-corpus 성능이 아님",
        "evaluation_subset_policy": "query와 해당 task의 모든 gold가 구조를 보유한 query",
        "source_candidate_count": len(candidate_ids),
        "candidate_count": len(usable_patents),
        "ranked_candidates_per_query": len(usable_patents) - 1,
        "candidate_patents_with_structures": len(usable_patents),
        "ground_truth_field_mapping": ground_truth_field_mapping,
        "fingerprint": {
            "type": "Morgan binary fingerprint",
            "radius": args.fingerprint_radius,
            "bits": args.fingerprint_bits,
        },
        "rdkit_version": rdkit_version,
        "ranking_limit": args.ranking_limit,
        "metric_definitions": METRIC_DEFINITIONS,
        "eligibility": eligibility,
        "structure_load_stats": structure_stats,
        "original_structure_count_by_patent": summarize_counts(
            [len(values) for values in structures.values()]
        ),
        "representative_count_by_patent": summarize_counts(
            list(representative_counts.values())
        ),
        "results": result_rows,
        "provenance": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
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
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
