#!/usr/bin/env python3
"""Evaluate T2 using patent-mean MoLFormer molecule embeddings. Use only prechecked non-isomeric canonical SMILES, represent molecules with the fixed checkpoint's pooler output, average unique molecular embeddings per patent, L2-normalize, and rank by patent cosine similarity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .evaluate_t2_structure_retrieval import (
        DEFAULT_KS,
        METRIC_DEFINITIONS,
        evaluate_ranking,
        load_candidates,
        load_ground_truth,
        rank_candidates,
        score_matrix,
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
        rank_candidates,
        score_matrix,
        select_complete_gold_queries,
        sha256_file,
        summarize_counts,
        write_csv_atomic,
        write_json_atomic,
        write_jsonl_atomic,
    )


DEFAULT_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
DEFAULT_REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"


def load_prepared_structures(
    path: Path,
    candidates: set[str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    structures: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patent_id", "model_smiles", "token_count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required MoLFormer input columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            counts["input_rows"] += 1
            patent_id = str(row.get("patent_id") or "").strip()
            smiles = str(row.get("model_smiles") or "").strip()
            if patent_id not in candidates:
                counts["outside_candidate_rows"] += 1
                continue
            if not smiles:
                raise SystemExit(f"Empty MoLFormer SMILES: {path}:{row_number}")
            try:
                token_count = int(row["token_count"])
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid token count: {path}:{row_number}") from exc
            if token_count < 1:
                raise SystemExit(f"Token count below 1: {path}:{row_number}")
            before = len(structures[patent_id])
            structures[patent_id].add(smiles)
            if len(structures[patent_id]) == before:
                counts["duplicate_patent_smiles_rows"] += 1
    counts["candidate_patents_with_structures"] = len(structures)
    counts["unique_smiles"] = len(
        {smiles for values in structures.values() for smiles in values}
    )
    counts["unique_patent_smiles_links"] = sum(map(len, structures.values()))
    return dict(structures), dict(sorted(counts.items()))


def verify_input_manifest(
    manifest_path: Path,
    structures_path: Path,
    model_id: str,
    revision: str,
    maximum_tokens: int,
) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "completed":
        raise SystemExit("MoLFormer input manifest is not completed")
    checks = {
        "model_id": (manifest.get("model_id"), model_id),
        "resolved_revision": (manifest.get("resolved_revision"), revision),
        "maximum_tokens": (manifest.get("maximum_tokens"), maximum_tokens),
    }
    mismatches = {
        name: {"manifest": actual, "requested": expected}
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    output_files = manifest.get("output_files", {})
    output_record = output_files.get(str(structures_path), {})
    if not output_record:
        matching_records = [
            record
            for output_path, record in output_files.items()
            if Path(output_path).name == structures_path.name
        ]
        if len(matching_records) == 1:
            output_record = matching_records[0]
    expected_hash = str(output_record.get("sha256") or "")
    actual_hash = sha256_file(structures_path)
    if expected_hash != actual_hash:
        mismatches["prepared_structure_sha256"] = {
            "manifest": expected_hash,
            "actual": actual_hash,
        }
    if mismatches:
        raise SystemExit(
            "MoLFormer precheck conditions differ from execution settings: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return manifest


def resolve_and_hash_model_files(
    model_id: str,
    revision: str,
) -> tuple[str, dict[str, dict[str, Any]]]:
    from huggingface_hub import HfApi, hf_hub_download

    resolved_revision = str(HfApi().model_info(model_id, revision=revision).sha or "")
    if not resolved_revision:
        raise SystemExit("Could not verify the Hugging Face checkpoint revision")
    if len(revision) == 40 and resolved_revision != revision:
        raise SystemExit(f"Checkpoint revision mismatch: {revision} != {resolved_revision}")
    files: dict[str, dict[str, Any]] = {}
    for filename in (
        "config.json",
        "configuration_molformer.py",
        "modeling_molformer.py",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        cached_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=resolved_revision,
            )
        )
        files[filename] = {
            "bytes": cached_path.stat().st_size,
            "sha256": sha256_file(cached_path),
        }
    return resolved_revision, files


def encode_molecules(
    smiles_values: list[str],
    model_id: str,
    revision: str,
    maximum_tokens: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if device not in {"cpu", "cuda"}:
        raise SystemExit("--device must be cpu or cuda")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        deterministic_eval=True,
        dtype=torch.float32,
    )
    model.eval()
    model.to(device)

    chunks: list[np.ndarray] = []
    observed_max_tokens = 0
    pooler_verification: dict[str, Any] | None = None
    with torch.inference_mode():
        for start in range(0, len(smiles_values), batch_size):
            batch = smiles_values[start : start + batch_size]
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            lengths = encoded["attention_mask"].sum(dim=1)
            batch_max_tokens = int(lengths.max().item())
            observed_max_tokens = max(observed_max_tokens, batch_max_tokens)
            if batch_max_tokens > maximum_tokens:
                raise SystemExit(
                    f"Input exceeds the prechecked limit: {batch_max_tokens}"
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None or pooled.ndim != 2:
                raise SystemExit("MoLFormer pooler_output is unavailable")
            if pooler_verification is None:
                last_hidden_state = getattr(outputs, "last_hidden_state", None)
                if last_hidden_state is None or last_hidden_state.ndim != 3:
                    raise SystemExit(
                        "MoLFormer last_hidden_state is unavailable; pooling cannot be verified"
                    )
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    raise SystemExit(
                        "attention_mask is unavailable; MoLFormer pooling cannot be verified"
                    )
                expanded_mask = attention_mask.unsqueeze(-1).to(
                    dtype=last_hidden_state.dtype
                )
                manual_mean = (last_hidden_state * expanded_mask).sum(dim=1)
                manual_mean /= expanded_mask.sum(dim=1).clamp_min(1.0)
                absolute_difference = (pooled - manual_mean).abs()
                maximum_absolute_difference = float(
                    absolute_difference.max().detach().cpu().item()
                )
                absolute_tolerance = 1e-6
                relative_tolerance = 1e-5
                matches = bool(
                    torch.allclose(
                        pooled,
                        manual_mean,
                        atol=absolute_tolerance,
                        rtol=relative_tolerance,
                    )
                )
                pooler_verification = {
                    "checked_on_first_batch": True,
                    "batch_size": len(batch),
                    "comparison": (
                        "Direct comparison of pooler_output with the attention-mask mean of last_hidden_state"
                    ),
                    "allclose": matches,
                    "absolute_tolerance": absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                    "maximum_absolute_difference": maximum_absolute_difference,
                }
                if not matches:
                    raise SystemExit(
                        "MoLFormer pooler_output differs from the attention-mask mean: "
                        f"max_abs_diff={maximum_absolute_difference}"
                    )
            chunk = pooled.detach().to("cpu", dtype=torch.float32).numpy()
            if not np.isfinite(chunk).all():
                raise SystemExit("MoLFormer embeddings contain nonfinite values")
            chunks.append(chunk)
            completed = min(start + batch_size, len(smiles_values))
            if completed % 5000 < batch_size or completed == len(smiles_values):
                print(
                    f"  MoLFormer embedding {completed}/{len(smiles_values)} processed",
                    flush=True,
                )
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    metadata = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "embedding_dimension": int(embeddings.shape[1]),
        "encoded_molecule_count": int(embeddings.shape[0]),
        "observed_max_tokens": observed_max_tokens,
        "dtype": "float32",
        "deterministic_eval": True,
        "pooler_runtime_verification": pooler_verification,
    }
    if device == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
    return embeddings, metadata


def aggregate_patent_vectors(
    candidate_ids: list[str],
    structures: dict[str, set[str]],
    smiles_values: list[str],
    molecule_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    patent_index = {patent_id: index for index, patent_id in enumerate(candidate_ids)}
    smiles_index = {smiles: index for index, smiles in enumerate(smiles_values)}
    dimension = molecule_embeddings.shape[1]
    sums = np.zeros((len(candidate_ids), dimension), dtype=np.float32)
    counts = np.zeros(len(candidate_ids), dtype=np.int32)
    for patent_id, patent_smiles in structures.items():
        position = patent_index[patent_id]
        indices = [smiles_index[smiles] for smiles in sorted(patent_smiles)]
        sums[position] = molecule_embeddings[indices].mean(axis=0, dtype=np.float32)
        counts[position] = len(indices)
    norms = np.linalg.norm(sums, axis=1)
    usable = norms > 0
    sums[usable] /= norms[usable, None]
    if not np.isfinite(sums[usable]).all():
        raise SystemExit("Patent embeddings contain nonfinite values")
    return sums, counts


def run_self_test() -> None:
    candidate_ids = ["EP1A1", "EP2A1", "EP3A1"]
    structures = {
        "EP1A1": {"A", "B"},
        "EP2A1": {"B"},
        "EP3A1": {"C"},
    }
    smiles_values = ["A", "B", "C"]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    vectors, counts = aggregate_patent_vectors(
        candidate_ids,
        structures,
        smiles_values,
        embeddings,
    )
    assert counts.tolist() == [2, 1, 1]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    ranking = rank_candidates(
        candidate_ids,
        vectors @ vectors[0],
        "EP1A1",
        counts > 0,
        2,
    )
    assert ranking[0][0] == "EP3A1"
    assert all(patent_id != "EP1A1" for patent_id, _ in ranking)
    print("Self-check passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--candidate-patents", type=Path)
    parser.add_argument("--prepared-structures", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--maximum-tokens", type=int, default=202)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ranking-limit", type=int, default=100)
    parser.add_argument(
        "--conflict-field",
        choices=("conflict_patents", "x_category_patents"),
        default="conflict_patents",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    required = {
        "--ground-truth": args.ground_truth,
        "--candidate-patents": args.candidate_patents,
        "--prepared-structures": args.prepared_structures,
        "--input-manifest": args.input_manifest,
        "--output-dir": args.output_dir,
    }
    missing_arguments = [name for name, value in required.items() if value is None]
    if missing_arguments:
        parser.error(f"Missing required arguments: {', '.join(missing_arguments)}")
    input_paths = [
        args.ground_truth,
        args.candidate_patents,
        args.prepared_structures,
        args.input_manifest,
    ]
    missing_files = [str(path) for path in input_paths if not path.is_file()]
    if missing_files:
        raise SystemExit(f"Input file missing: {missing_files}")
    if args.maximum_tokens < 1 or args.batch_size < 1:
        parser.error("Token limit and batch size must be at least 1")
    if args.ranking_limit < max(DEFAULT_KS):
        parser.error(f"--ranking-limit must be {max(DEFAULT_KS)} or greater")

    started_at = time.perf_counter()
    input_manifest = verify_input_manifest(
        args.input_manifest,
        args.prepared_structures,
        args.model_id,
        args.revision,
        args.maximum_tokens,
    )
    resolved_revision, model_files = resolve_and_hash_model_files(
        args.model_id,
        args.revision,
    )
    candidate_ids = load_candidates(args.candidate_patents, "patent_id")
    candidate_set = set(candidate_ids)
    ground_truth, ground_truth_field_mapping = load_ground_truth(
        args.ground_truth,
        args.conflict_field,
    )
    structures, structure_load_stats = load_prepared_structures(
        args.prepared_structures,
        candidate_set,
    )
    smiles_values = sorted(
        {smiles for patent_smiles in structures.values() for smiles in patent_smiles}
    )
    print(
        f"Candidates: {len(candidate_ids)}; structure-bearing patents {len(structures)}; "
        f"unique SMILES {len(smiles_values)} loaded.",
        flush=True,
    )

    molecule_embeddings, encoder_metadata = encode_molecules(
        smiles_values,
        args.model_id,
        resolved_revision,
        args.maximum_tokens,
        args.batch_size,
        args.device,
    )
    patent_vectors, structure_counts = aggregate_patent_vectors(
        candidate_ids,
        structures,
        smiles_values,
        molecule_embeddings,
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
    scores, score_device = score_matrix(patent_vectors, query_positions, args.device)
    usable_mask = structure_counts > 0
    rankings = {
        query_id: rank_candidates(
            candidate_ids,
            scores[row_index],
            query_id,
            usable_mask,
            args.ranking_limit,
        )
        for row_index, query_id in enumerate(query_ids)
    }

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
            best_rank = next(
                (
                    rank
                    for rank, (patent_id, _) in enumerate(rankings[query_id], start=1)
                    if patent_id in gold
                ),
                None,
            )
            row = {
                "method": "molformer_mean_pooling",
                "query_patent_id": query_id,
                "task": task,
                "gold_count": len(gold),
                "source_candidate_count_before_mask": len(candidate_ids) - 1,
                "candidate_count_ranked": len(usable_patents) - 1,
                "ranking_rows_written": len(rankings[query_id]),
                "query_structure_count": int(structure_counts[candidate_index[query_id]]),
                "best_gold_rank": best_rank or "",
                **metrics,
            }
            task_rows.append(row)
            per_query_rows.append(row)
        eligibility[task] = {"evaluated_queries": len(task_rows), **dict(sorted(skipped.items()))}
        result_rows.append(
            {
                "method": "molformer_mean_pooling",
                "task": task,
                "Queries": len(task_rows),
                **{
                    metric: (
                        sum(row[metric] for row in task_rows) / len(task_rows)
                        if task_rows
                        else ""
                    )
                    for metric in metric_names
                },
            }
        )

    ranking_rows = [
        {
            "method": "molformer_mean_pooling",
            "query_patent_id": query_id,
            "ranking": [
                {"rank": rank, "patent_id": patent_id, "score": round(score, 8)}
                for rank, (patent_id, score) in enumerate(rankings[query_id], start=1)
            ],
        }
        for query_id in query_ids
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "t2_molformer_results.csv"
    per_query_path = args.output_dir / "t2_molformer_per_query.csv"
    ranking_path = args.output_dir / "t2_molformer_rankings_top100.jsonl"
    manifest_path = args.output_dir / "t2_molformer_manifest.json"
    write_csv_atomic(
        result_path,
        ["method", "task", "Queries", *metric_names],
        result_rows,
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
        "method": "molformer_mean_pooling",
        "method_key": "molformer_mean_pooling",
        "display_name": "Patent-mean MoLFormer molecule-embedding cosine",
        "score_definition": (
            "Use pooler_output from the fixed MoLFormer revision for molecule embeddings. "
            "In the first batch, verify it against the attention-mask mean of last_hidden_state. "
            "Average unique molecular embeddings per patent, then "
            "L2-normalize and compare by cosine similarity."
        ),
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "model_files": model_files,
        "model_input_policy": input_manifest.get("smiles_policy"),
        "maximum_tokens": args.maximum_tokens,
        "patent_aggregation": "Mean of unique molecule embeddings per patent, followed by L2 normalization",
        "candidate_pool_policy": (
            "Within original GT-derived common candidates, rank only patents "
            "with eligible MoLFormer structures and exclude self"
        ),
        "candidate_pool_scope": "oracle-derived closed pool; not open-corpus performance",
        "evaluation_subset_policy": (
            "Queries whose own patent and every task-specific gold patent have eligible MoLFormer structures"
        ),
        "source_candidate_count": len(candidate_ids),
        "candidate_count": len(usable_patents),
        "ranked_candidates_per_query": len(usable_patents) - 1,
        "queries_with_structure_and_in_candidate_corpus": len(query_ids),
        "ranking_limit": args.ranking_limit,
        "metric_definitions": METRIC_DEFINITIONS,
        "ground_truth_field_mapping": ground_truth_field_mapping,
        "eligibility": eligibility,
        "structure_load_stats": structure_load_stats,
        "structure_count_by_patent": summarize_counts(structure_counts),
        "encoder": encoder_metadata,
        "score_device": score_device,
        "provenance": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "results": result_rows,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "input_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        },
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
