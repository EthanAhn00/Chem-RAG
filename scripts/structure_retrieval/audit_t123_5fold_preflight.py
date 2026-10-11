#!/usr/bin/env python3
"""Read-only input and fold-feasibility audit for the proposed T1/T2/T3 study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .evaluate_t3_rrf import normalize_exact_ep
except ImportError:
    from evaluate_t3_rrf import normalize_exact_ep


TASKS = ("Citation", "Conflict-X")
TEXT_SOURCES = {"nv_document", "qwen_whole_claims"}


class AuditError(ValueError):
    pass


def require_id(value: Any, label: str) -> str:
    patent_id = normalize_exact_ep(value)
    if patent_id is None:
        raise AuditError(f"invalid patent ID: {label}: {value!r}")
    return patent_id


def file_info(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_ground_truth(path: Path) -> dict[str, dict[str, set[str]]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query = require_id(row.get("query_patent_id"), f"{path}:{line_no}")
            if query in result:
                raise AuditError(f"duplicate ground-truth query: {query}")
            tasks = {}
            for task, field in (("Citation", "all_cited_patents"), ("Conflict-X", "conflict_patents")):
                values = row.get(field)
                if not isinstance(values, list):
                    raise AuditError(f"invalid {field}: {query}")
                ids = [require_id(value, f"{query}:{field}") for value in values]
                if len(ids) != len(set(ids)):
                    raise AuditError(f"duplicate {field}: {query}")
                tasks[task] = set(ids)
            if not tasks["Conflict-X"].issubset(tasks["Citation"]):
                raise AuditError(f"Conflict-X not subset of Citation: {query}")
            result[query] = tasks
    if not result:
        raise AuditError(f"empty ground truth: {path}")
    return result


def load_legacy_split(path: Path) -> dict[str, tuple[str, str]]:
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "role", "split"}.issubset(reader.fieldnames or []):
            raise AuditError(f"missing split columns: {path}")
        for line_no, row in enumerate(reader, 2):
            query = require_id(row.get("id"), f"{path}:{line_no}")
            value = (str(row.get("role") or "").strip(), str(row.get("split") or "").strip())
            if query in result and result[query] != value:
                raise AuditError(f"conflicting split rows: {query}")
            result[query] = value
    return result


def load_per_query(path: Path, model_field: str) -> dict[str, dict[str, dict[str, tuple[int, int]]]]:
    result: dict[str, dict[str, dict[str, tuple[int, int]]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {model_field, "query_patent_id", "task", "gold_count", "candidate_count_ranked", "ranking_rows_written"}
        if not required.issubset(reader.fieldnames or []):
            raise AuditError(f"missing per-query columns: {path}: {sorted(required - set(reader.fieldnames or []))}")
        for line_no, row in enumerate(reader, 2):
            model = str(row.get(model_field) or "").strip()
            task = str(row.get("task") or "").strip()
            query = require_id(row.get("query_patent_id"), f"{path}:{line_no}")
            if not model or task not in TASKS or query in result[model][task]:
                raise AuditError(f"invalid or duplicate per-query row: {path}:{line_no}")
            try:
                gold = int(row["gold_count"])
                candidates = int(row["candidate_count_ranked"])
                written = int(row["ranking_rows_written"])
            except (TypeError, ValueError) as exc:
                raise AuditError(f"invalid per-query counts: {path}:{line_no}") from exc
            if gold < 1 or candidates < 1 or written != candidates:
                raise AuditError(f"incomplete per-query ranking: {path}:{line_no}")
            result[model][task][query] = (gold, candidates)
    if not result:
        raise AuditError(f"empty per-query file: {path}")
    return result


def load_ranking_fingerprints(
    path: Path,
    model_field: str,
    truth: dict[str, dict[str, set[str]]],
    target_ids_by_query: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, tuple[Any, ...]]]:
    result: dict[str, dict[str, tuple[Any, ...]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            model = str(row.get(model_field) or "").strip()
            query = require_id(row.get("query_patent_id"), f"{path}:{line_no}")
            items = row.get("ranking")
            if not model or query in result[model] or not isinstance(items, list):
                raise AuditError(f"invalid or duplicate ranking row: {path}:{line_no}")
            ids = []
            scores = []
            for rank, item in enumerate(items, 1):
                if not isinstance(item, dict) or item.get("rank") != rank:
                    raise AuditError(f"nonconsecutive rank: {path}:{line_no}:{rank}")
                candidate = require_id(item.get("patent_id"), f"{path}:{line_no}:{rank}")
                if candidate == query:
                    raise AuditError(f"self candidate: {path}:{line_no}:{rank}")
                try:
                    score = float(item.get("score"))
                except (TypeError, ValueError) as exc:
                    raise AuditError(f"invalid score: {path}:{line_no}:{rank}") from exc
                if not math.isfinite(score):
                    raise AuditError(f"nonfinite score: {path}:{line_no}:{rank}")
                ids.append(candidate)
                scores.append(score)
            if len(ids) != len(set(ids)):
                raise AuditError(f"duplicate candidates: {path}:{line_no}")
            if not ids:
                raise AuditError(f"empty ranking: {path}:{line_no}")
            digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
            relevant = truth.get(query, {}).get("Citation", set())
            found_gold = frozenset(set(ids) & relevant)
            targets = (target_ids_by_query or {}).get(query, set())
            selected_scores = {
                candidate: score
                for candidate, score in zip(ids, scores)
                if candidate in targets
            }
            result[model][query] = (
                len(ids), digest, statistics.pstdev(scores), found_gold,
                tuple(ids[:500]), tuple(scores[:500]), selected_scores,
            )
    if not result:
        raise AuditError(f"empty ranking file: {path}")
    return result


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def make_groups(queries: set[str], truth: dict[str, dict[str, set[str]]]) -> dict[str, list[str]]:
    graph = UnionFind()
    for query in sorted(queries):
        graph.find(f"q:{query}")
        for gold in sorted(truth[query]["Citation"] | truth[query]["Conflict-X"]):
            graph.union(f"q:{query}", f"g:{gold}")
    components: dict[str, list[str]] = defaultdict(list)
    for query in sorted(queries):
        components[graph.find(f"q:{query}")].append(query)
    return {hashlib.sha256("|".join(members).encode()).hexdigest(): members for members in components.values()}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "ground_truth": args.ground_truth,
        "legacy_split": args.legacy_split,
        "text_per_query": args.text_per_query,
        "text_rankings": args.text_rankings,
        "structure_per_query": args.structure_per_query,
        "structure_rankings": args.structure_rankings,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise AuditError(f"missing {label}: {path}")
    truth = load_ground_truth(args.ground_truth)
    legacy = load_legacy_split(args.legacy_split)
    text_meta = load_per_query(args.text_per_query, "source")
    structure_meta = load_per_query(args.structure_per_query, "method")
    text_rank = load_ranking_fingerprints(args.text_rankings, "source", truth)
    topn_targets: dict[str, set[str]] = defaultdict(set)
    for query_rankings in text_rank.values():
        for query, record in query_rankings.items():
            topn_targets[query].update(record[4])
    structure_rank = load_ranking_fingerprints(
        args.structure_rankings, "method", truth, topn_targets,
    )
    if set(text_meta) != TEXT_SOURCES or set(text_rank) != TEXT_SOURCES:
        raise AuditError(f"unexpected text models: metadata={sorted(text_meta)}, rankings={sorted(text_rank)}")
    if len(structure_meta) != 1 or set(structure_meta) != set(structure_rank):
        raise AuditError(f"unexpected structure methods: metadata={sorted(structure_meta)}, rankings={sorted(structure_rank)}")
    if set(structure_meta) & TEXT_SOURCES:
        raise AuditError("text and structure model names collide")
    structure_model = next(iter(structure_meta))
    for task in TASKS:
        query_sets = {model: set(text_meta[model].get(task, {})) for model in sorted(TEXT_SOURCES)}
        query_sets[structure_model] = set(structure_meta[structure_model].get(task, {}))
        if len({frozenset(ids) for ids in query_sets.values()}) != 1:
            raise AuditError(f"query sets differ for {task}: { {model: len(ids) for model, ids in query_sets.items()} }")
    eligible = {task: set(text_meta["nv_document"].get(task, {})) for task in TASKS}
    expected = {"Citation": args.expected_citation, "Conflict-X": args.expected_conflict}
    for task in TASKS:
        if len(eligible[task]) != expected[task]:
            raise AuditError(f"unexpected eligible {task} count: observed={len(eligible[task])}, expected={expected[task]}")
    union = eligible["Citation"] | eligible["Conflict-X"]
    if not union.issubset(truth):
        raise AuditError(f"eligible queries absent from ground truth: {sorted(union - set(truth))[:5]}")
    ranking_rows = {**text_rank, **structure_rank}
    metadata = {**text_meta, **structure_meta}
    zero_variance = Counter()
    candidate_counts = Counter()
    for query in sorted(union):
        reference = None
        for model in sorted(ranking_rows):
            ranking = ranking_rows[model].get(query)
            if ranking is None:
                raise AuditError(f"missing full ranking: {model}:{query}")
            if reference is None:
                reference = ranking[:2]
            if ranking[:2] != reference:
                raise AuditError(f"candidate set differs: {model}:{query}")
            if ranking[2] <= 1e-15:
                zero_variance[model] += 1
            candidate_counts[ranking[0]] += 1
            for task in TASKS:
                row = metadata[model].get(task, {}).get(query)
                if query not in eligible[task]:
                    if row is not None:
                        raise AuditError(f"unexpected task row: {model}:{task}:{query}")
                    continue
                if row is None or row != (len(truth[query][task]), ranking[0]):
                    raise AuditError(f"gold or candidate count differs: {model}:{task}:{query}")
                if not truth[query][task].issubset(ranking[3]):
                    raise AuditError(f"gold absent from candidates: {model}:{task}:{query}")
        if reference[0] != args.expected_candidates_per_query:
            raise AuditError(f"unexpected candidate count: {query}:{reference[0]}")
    if zero_variance:
        raise AuditError(f"zero-variance ranking scores: {dict(zero_variance)}")
    m3_zero_variance = Counter()
    for query in sorted(union):
        structure_scores = structure_rank[structure_model][query][6]
        for source in sorted(TEXT_SOURCES):
            text_record = text_rank[source][query]
            for n in (100, 500):
                top_ids = text_record[4][:n]
                if len(top_ids) != n:
                    raise AuditError(f"M3 text Top-{n} incomplete: {source}:{query}")
                if statistics.pstdev(text_record[5][:n]) <= 1e-15:
                    m3_zero_variance[f"{source}/Top-{n}/text"] += 1
                try:
                    top_structure_scores = [structure_scores[candidate] for candidate in top_ids]
                except KeyError as exc:
                    raise AuditError(f"M3 structure score missing: {source}:{query}:{exc}") from exc
                if statistics.pstdev(top_structure_scores) <= 1e-15:
                    m3_zero_variance[f"{source}/Top-{n}/structure"] += 1
    groups = make_groups(union, truth)
    if len(groups) < 5:
        raise AuditError(f"fewer than five independent groups: {len(groups)}")
    group_by_query = {query: group for group, members in groups.items() for query in members}
    ordered = sorted(union)
    labels = [int(query in eligible["Conflict-X"]) for query in ordered]
    group_ids = [group_by_query[query] for query in ordered]
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise AuditError("scikit-learn is required for proposed fold assignment") from exc
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = []
    test_seen = set()
    for fold_index, (_, test_indices) in enumerate(splitter.split(ordered, labels, group_ids)):
        members = {ordered[index] for index in test_indices}
        if not members or test_seen & members or not members & eligible["Citation"]:
            raise AuditError(f"invalid outer fold: {fold_index}")
        training_citation = eligible["Citation"] - members
        if len({group_by_query[query] for query in training_citation}) < 2:
            raise AuditError(f"M4 inner group split impossible: outer fold {fold_index}")
        test_seen |= members
        folds.append({
            "fold": fold_index,
            "queries": len(members),
            "groups": len({group_by_query[query] for query in members}),
            "citation_queries": len(members & eligible["Citation"]),
            "citation_groups": len({group_by_query[query] for query in members & eligible["Citation"]}),
            "conflict_queries": len(members & eligible["Conflict-X"]),
            "conflict_groups": len({group_by_query[query] for query in members & eligible["Conflict-X"]}),
        })
    if test_seen != union:
        raise AuditError(f"OOF assignment misses queries: {len(union - test_seen)}")
    role_counts = {task: dict(Counter(f"{legacy.get(query, ('missing', 'missing'))[0]}/{legacy.get(query, ('missing', 'missing'))[1]}" for query in eligible[task])) for task in TASKS}
    expected_corpus = {
        "Citation": {"corpus/train": 65, "corpus/test": 5},
        "Conflict-X": {"corpus/train": 16, "corpus/test": 1},
    }
    for task, counts in expected_corpus.items():
        for role_split, expected_count in counts.items():
            if role_counts[task].get(role_split, 0) != expected_count:
                raise AuditError(
                    f"legacy exclusion count changed: {task}:{role_split}: "
                    f"observed={role_counts[task].get(role_split, 0)}, expected={expected_count}"
                )
    group_sizes = sorted((len(members) for members in groups.values()), reverse=True)
    return {
        "status": "core_input_preflight_pass",
        "scope": "existing two text sources and Morgan full rankings only; additional T1/T2 models not audited",
        "remaining_before_evaluation": [
            "audit additional standalone T1/T2 models on the identical eligible queries and candidate mask",
            "verify M4 inner fit/development viability for each outer training set",
            "freeze fold assignment and inspect historical model/benchmark selection provenance",
        ],
        "input_files": {label: file_info(path) for label, path in paths.items()},
        "ground_truth_query_count": len(truth),
        "eligible_query_count": {task: len(eligible[task]) for task in TASKS},
        "eligible_union_query_count": len(union),
        "conflict_query_subset_of_citation": eligible["Conflict-X"].issubset(eligible["Citation"]),
        "legacy_role_split_counts": role_counts,
        "candidate_count_per_query": args.expected_candidates_per_query,
        "candidate_record_count_by_length": dict(candidate_counts),
        "m2_full_variance_checked": True,
        "m3_topn_zero_variance_counts": dict(m3_zero_variance),
        "connected_group_count": len(groups),
        "singleton_group_count": sum(size == 1 for size in group_sizes),
        "largest_group_sizes": group_sizes[:10],
        "conflict_group_count": len({group_by_query[query] for query in eligible["Conflict-X"]}),
        "folds": folds,
        "seed": args.seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--legacy-split", type=Path, required=True)
    parser.add_argument("--text-per-query", type=Path, required=True)
    parser.add_argument("--text-rankings", type=Path, required=True)
    parser.add_argument("--structure-per-query", type=Path, required=True)
    parser.add_argument("--structure-rankings", type=Path, required=True)
    parser.add_argument("--expected-citation", type=int, default=1126)
    parser.add_argument("--expected-conflict", type=int, default=47)
    parser.add_argument("--expected-candidates-per-query", type=int, default=2217)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    try:
        report = audit(args)
    except (AuditError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "preflight_blocked", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
