#!/usr/bin/env python3
"""T2 구조 검색 실험 전에 특허별 SMILES 커버리지와 입력 품질을 감사한다.

이 스크립트는 검색 점수를 계산하거나 Qdrant를 수정하지 않는다. 평가 ground truth,
후보 특허 목록, 하나 이상의 구조원을 특허번호 단위로 결합해 다음을 확인한다.

- query, 전체 인용 gold, X 인용 gold, 후보 코퍼스의 구조 보유율
- RDKit 파싱 실패, 빈 SMILES, 잘못된 특허번호, 중복 구조 연결
- query별 전체 gold/X gold 구조 가용성과 부분집합 평가 가능 여부
- 국가·공개번호·kind code 정규화 충돌과 kind code 미상 입력

직접 특허번호와 SMILES가 함께 있는 CSV/JSONL은 ``--structure-file``로 받는다.
``structure_id``와 특허 출현 정보가 분리된 PatCID 산출물은 ``--occurrences``를 함께
지정한다. 기존 ``molecule.csv``/``patent_molecule.csv`` 쌍도 별도 옵션으로 받을 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


EP_NUMBER_RE = re.compile(r"^(?:EP)?(\d+)([A-Z]\d?)?$", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """CSV 또는 JSONL을 읽고 1부터 시작하는 데이터 행 번호와 레코드를 반환한다."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                yield row_number, dict(row)
        return
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{row_number} JSON 객체가 아닙니다")
                yield row_number, value
        return
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path} (CSV 또는 JSONL 필요)")


def normalize_patent_id(value: Any) -> str | None:
    """EP 공개번호를 정규화하되 kind code는 보존한다."""
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").strip().upper())
    match = EP_NUMBER_RE.fullmatch(compact)
    if not match:
        return None
    kind_code = match.group(2) or ""
    return f"EP{match.group(1)}{kind_code}"


def split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in re.split(r"[|;,]", str(value)) if item.strip()]


def first_nonempty(row: dict[str, Any], columns: tuple[str, ...]) -> str | None:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


@dataclass(frozen=True)
class StructureRecord:
    patent_key: str
    patent_id_raw: str
    canonical_smiles: str
    source: str
    structure_key: str


class Canonicalizer:
    """SMILES 표준화 결과를 캐시하고 RDKit 오류를 외부로 전파하지 않는다."""

    def __init__(self) -> None:
        try:
            from rdkit import Chem, RDLogger
        except ImportError as exc:
            raise SystemExit("RDKit이 필요합니다. chemrag_property 환경에서 실행하세요.") from exc
        RDLogger.DisableLog("rdApp.*")
        self._chem = Chem
        self._cache: dict[str, str | None] = {}

    def canonicalize(self, smiles: str) -> str | None:
        if smiles in self._cache:
            return self._cache[smiles]
        canonical = None
        try:
            molecule = self._chem.MolFromSmiles(smiles)
            if molecule is not None:
                canonical = self._chem.MolToSmiles(molecule, canonical=True)
        except Exception:
            canonical = None
        self._cache[smiles] = canonical
        return canonical


def issue(
    issues: list[dict[str, Any]],
    *,
    source_file: Path,
    row_number: int | str,
    issue_type: str,
    severity: str,
    patent_id: Any = "",
    structure_id: Any = "",
    detail: str = "",
) -> None:
    issues.append(
        {
            "source_file": str(source_file),
            "row_number": row_number,
            "issue_type": issue_type,
            "severity": severity,
            "patent_id": str(patent_id or ""),
            "structure_id": str(structure_id or ""),
            "detail": detail,
        }
    )


def load_ground_truth(
    path: Path,
    issues: list[dict[str, Any]],
    conflict_field: str = "conflict_patents",
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, str],
]:
    if conflict_field not in {"conflict_patents", "x_category_patents"}:
        raise ValueError(f"지원하지 않는 X-category 필드입니다: {conflict_field}")
    alternate_field = (
        "x_category_patents"
        if conflict_field == "conflict_patents"
        else "conflict_patents"
    )
    queries: dict[str, dict[str, Any]] = {}
    raw_ids: dict[str, set[str]] = defaultdict(set)
    roles: dict[str, set[str]] = defaultdict(set)

    for row_number, row in read_records(path):
        query_raw = first_nonempty(row, ("query_patent_id", "query_id"))
        query_key = normalize_patent_id(query_raw)
        if not query_key:
            issue(
                issues,
                source_file=path,
                row_number=row_number,
                issue_type="invalid_query_patent_id",
                severity="critical",
                patent_id=query_raw,
            )
            continue
        if query_key in queries:
            issue(
                issues,
                source_file=path,
                row_number=row_number,
                issue_type="duplicate_normalized_query",
                severity="critical",
                patent_id=query_raw,
                detail=f"normalized={query_key}",
            )
            continue

        all_gold_raw = row.get("all_cited_patents") or []
        if conflict_field not in row:
            issue(
                issues,
                source_file=path,
                row_number=row_number,
                issue_type="missing_conflict_field",
                severity="critical",
                patent_id=query_raw,
                detail=f"required={conflict_field}",
            )
            continue
        x_gold_raw = row.get(conflict_field) or []
        if alternate_field in row:
            alternate_raw = row.get(alternate_field) or []
            if not isinstance(alternate_raw, list):
                issue(
                    issues,
                    source_file=path,
                    row_number=row_number,
                    issue_type="invalid_alternate_ground_truth_list",
                    severity="critical",
                    patent_id=query_raw,
                    detail=f"field={alternate_field}",
                )
                continue
            normalized_primary = {
                normalize_patent_id(value) for value in x_gold_raw
            }
            normalized_alternate = {
                normalize_patent_id(value) for value in alternate_raw
            }
            if normalized_primary != normalized_alternate:
                issue(
                    issues,
                    source_file=path,
                    row_number=row_number,
                    issue_type="conflicting_conflict_fields",
                    severity="critical",
                    patent_id=query_raw,
                    detail=f"{conflict_field}!={alternate_field}",
                )
                continue
        if not isinstance(all_gold_raw, list) or not isinstance(x_gold_raw, list):
            issue(
                issues,
                source_file=path,
                row_number=row_number,
                issue_type="invalid_ground_truth_list",
                severity="critical",
                patent_id=query_raw,
            )
            continue

        all_gold: list[str] = []
        x_gold: list[str] = []
        for role_name, raw_values, output in (
            ("gold_all", all_gold_raw, all_gold),
            ("gold_x", x_gold_raw, x_gold),
        ):
            for patent_raw in raw_values:
                patent_key = normalize_patent_id(patent_raw)
                if not patent_key:
                    issue(
                        issues,
                        source_file=path,
                        row_number=row_number,
                        issue_type="invalid_gold_patent_id",
                        severity="critical",
                        patent_id=patent_raw,
                        detail=f"role={role_name};query={query_raw}",
                    )
                    continue
                output.append(patent_key)
                raw_ids[patent_key].add(str(patent_raw))
                roles[patent_key].add(role_name)

        if not set(x_gold).issubset(set(all_gold)):
            issue(
                issues,
                source_file=path,
                row_number=row_number,
                issue_type="x_gold_not_subset_of_all_gold",
                severity="critical",
                patent_id=query_raw,
            )

        queries[query_key] = {
            "query_patent_id": str(query_raw),
            "query_key": query_key,
            "all_gold": sorted(set(all_gold)),
            "x_gold": sorted(set(x_gold)),
        }
        raw_ids[query_key].add(str(query_raw))
        roles[query_key].add("query")

    return queries, raw_ids, roles, {
        "Citation": "all_cited_patents",
        "Conflict-X": conflict_field,
    }


def load_candidate_patents(
    path: Path,
    id_column: str,
    issues: list[dict[str, Any]],
) -> tuple[set[str], dict[str, set[str]]]:
    candidates: set[str] = set()
    raw_ids: dict[str, set[str]] = defaultdict(set)
    seen_raw: Counter[str] = Counter()
    for row_number, row in read_records(path):
        raw = row.get(id_column)
        key = normalize_patent_id(raw)
        if not key:
            issue(
                issues,
                source_file=path,
                row_number=row_number,
                issue_type="invalid_candidate_patent_id",
                severity="high",
                patent_id=raw,
                detail=f"column={id_column}",
            )
            continue
        candidates.add(key)
        raw_ids[key].add(str(raw))
        seen_raw[str(raw)] += 1
    for raw, count in seen_raw.items():
        if count > 1:
            issue(
                issues,
                source_file=path,
                row_number="",
                issue_type="duplicate_candidate_row",
                severity="medium",
                patent_id=raw,
                detail=f"rows={count}",
            )
    return candidates, raw_ids


def load_occurrences(
    paths: list[Path],
    issues: list[dict[str, Any]],
) -> dict[str, set[tuple[str, str]]]:
    """structure_id -> {(정규화 특허번호, 원본 특허번호)} 매핑을 만든다."""
    mapping: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for path in paths:
        for row_number, row in read_records(path):
            structure_id = first_nonempty(row, ("structure_id", "mol_id"))
            patent_raw = first_nonempty(row, ("patent_id", "publication_number"))
            patent_key = normalize_patent_id(patent_raw)
            if not structure_id:
                issue(
                    issues,
                    source_file=path,
                    row_number=row_number,
                    issue_type="missing_occurrence_structure_id",
                    severity="high",
                    patent_id=patent_raw,
                )
                continue
            if not patent_key:
                issue(
                    issues,
                    source_file=path,
                    row_number=row_number,
                    issue_type="invalid_occurrence_patent_id",
                    severity="high",
                    patent_id=patent_raw,
                    structure_id=structure_id,
                )
                continue
            mapping[structure_id].add((patent_key, str(patent_raw)))
    return mapping


def patents_from_structure_row(
    row: dict[str, Any],
    structure_id: str | None,
    occurrence_index: dict[str, set[tuple[str, str]]],
) -> list[tuple[str, str]]:
    raw_values: list[str] = []
    direct = first_nonempty(row, ("patent_id", "publication_number"))
    if direct:
        raw_values.append(direct)
    raw_values.extend(split_values(row.get("patent_ids")))

    patents: set[tuple[str, str]] = set()
    for raw in raw_values:
        key = normalize_patent_id(raw)
        if key:
            patents.add((key, raw))
    if structure_id:
        patents.update(occurrence_index.get(structure_id, set()))
    return sorted(patents)


def load_flat_structure_files(
    paths: list[Path],
    occurrence_index: dict[str, set[tuple[str, str]]],
    canonicalizer: Canonicalizer,
    issues: list[dict[str, Any]],
) -> tuple[list[StructureRecord], dict[str, dict[str, int]]]:
    records: list[StructureRecord] = []
    stats: dict[str, dict[str, int]] = {}
    for path in paths:
        counts: Counter[str] = Counter()
        source_label = str(path)
        for row_number, row in read_records(path):
            counts["input_rows"] += 1
            structure_id = first_nonempty(row, ("structure_id", "mol_id"))
            smiles = first_nonempty(
                row,
                (
                    "canonical_smiles",
                    "canonical_isomeric_smiles",
                    "parent_smiles",
                    "smiles",
                    "raw_smiles",
                ),
            )
            if not smiles:
                counts["missing_smiles_rows"] += 1
                issue(
                    issues,
                    source_file=path,
                    row_number=row_number,
                    issue_type="missing_smiles",
                    severity="high",
                    structure_id=structure_id,
                )
                continue
            canonical = canonicalizer.canonicalize(smiles)
            if canonical is None:
                counts["unparseable_smiles_rows"] += 1
                issue(
                    issues,
                    source_file=path,
                    row_number=row_number,
                    issue_type="unparseable_smiles",
                    severity="high",
                    structure_id=structure_id,
                    detail=smiles[:200],
                )
                continue

            patents = patents_from_structure_row(row, structure_id, occurrence_index)
            if not patents:
                counts["unlinked_structure_rows"] += 1
                issue(
                    issues,
                    source_file=path,
                    row_number=row_number,
                    issue_type="structure_without_patent_link",
                    severity="high",
                    structure_id=structure_id,
                )
                continue

            if canonical != smiles:
                counts["canonicalized_rows"] += 1
            structure_key = structure_id or hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
            for patent_key, patent_raw in patents:
                records.append(
                    StructureRecord(
                        patent_key=patent_key,
                        patent_id_raw=patent_raw,
                        canonical_smiles=canonical,
                        source=source_label,
                        structure_key=structure_key,
                    )
                )
                counts["valid_patent_structure_links"] += 1
        stats[source_label] = dict(sorted(counts.items()))
    return records, stats


def load_legacy_structure_pair(
    molecule_path: Path,
    patent_molecule_path: Path,
    canonicalizer: Canonicalizer,
    issues: list[dict[str, Any]],
) -> tuple[list[StructureRecord], dict[str, int]]:
    molecules: dict[str, tuple[str, str]] = {}
    invalid_molecules: dict[str, str] = {}
    seen_molecule_ids: set[str] = set()
    stats: Counter[str] = Counter()
    for row_number, row in read_records(molecule_path):
        stats["molecule_rows"] += 1
        molecule_id = first_nonempty(row, ("mol_id", "structure_id"))
        smiles = first_nonempty(row, ("canonical_smiles", "smiles"))
        if not molecule_id:
            issue(
                issues,
                source_file=molecule_path,
                row_number=row_number,
                issue_type="missing_molecule_id",
                severity="high",
            )
            continue
        if molecule_id in seen_molecule_ids:
            stats["duplicate_molecule_ids"] += 1
            issue(
                issues,
                source_file=molecule_path,
                row_number=row_number,
                issue_type="duplicate_molecule_id",
                severity="critical",
                structure_id=molecule_id,
            )
            continue
        seen_molecule_ids.add(molecule_id)
        if not smiles:
            stats["missing_smiles_rows"] += 1
            invalid_molecules[molecule_id] = "missing_smiles"
            issue(
                issues,
                source_file=molecule_path,
                row_number=row_number,
                issue_type="missing_smiles",
                severity="high",
                structure_id=molecule_id,
            )
            continue
        canonical = canonicalizer.canonicalize(smiles)
        if canonical is None:
            stats["unparseable_smiles_rows"] += 1
            invalid_molecules[molecule_id] = "unparseable_smiles"
            issue(
                issues,
                source_file=molecule_path,
                row_number=row_number,
                issue_type="unparseable_smiles",
                severity="high",
                structure_id=molecule_id,
                detail=smiles[:200],
            )
            continue
        molecules[molecule_id] = (canonical, str(row.get("source") or molecule_path))

    records: list[StructureRecord] = []
    for row_number, row in read_records(patent_molecule_path):
        stats["patent_molecule_rows"] += 1
        molecule_id = first_nonempty(row, ("mol_id", "structure_id"))
        patent_raw = first_nonempty(row, ("patent_id", "publication_number"))
        patent_key = normalize_patent_id(patent_raw)
        if not patent_key:
            stats["invalid_patent_links"] += 1
            issue(
                issues,
                source_file=patent_molecule_path,
                row_number=row_number,
                issue_type="invalid_structure_patent_id",
                severity="high",
                patent_id=patent_raw,
                structure_id=molecule_id,
            )
            continue
        molecule = molecules.get(str(molecule_id or ""))
        if molecule is None:
            invalid_reason = invalid_molecules.get(str(molecule_id or ""))
            if invalid_reason:
                stats["links_to_invalid_molecules"] += 1
                issue(
                    issues,
                    source_file=patent_molecule_path,
                    row_number=row_number,
                    issue_type="link_to_invalid_molecule_structure",
                    severity="high",
                    patent_id=patent_raw,
                    structure_id=molecule_id,
                    detail=f"molecule_issue={invalid_reason}",
                )
            else:
                stats["orphan_molecule_links"] += 1
                issue(
                    issues,
                    source_file=patent_molecule_path,
                    row_number=row_number,
                    issue_type="orphan_molecule_link",
                    severity="critical",
                    patent_id=patent_raw,
                    structure_id=molecule_id,
                )
            continue
        canonical, source = molecule
        records.append(
            StructureRecord(
                patent_key=patent_key,
                patent_id_raw=str(patent_raw),
                canonical_smiles=canonical,
                source=source,
                structure_key=str(molecule_id),
            )
        )
        stats["valid_patent_structure_links"] += 1
    return records, dict(sorted(stats.items()))


def pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 6) if denominator else None


def role_coverage(keys: set[str], structures_by_patent: dict[str, set[str]]) -> dict[str, Any]:
    covered = sum(bool(structures_by_patent.get(key)) for key in keys)
    return {
        "patent_count": len(keys),
        "patents_with_structure": covered,
        "patents_without_structure": len(keys) - covered,
        "coverage_percent": pct(covered, len(keys)),
    }


def build_query_rows(
    queries: dict[str, dict[str, Any]],
    structures_by_patent: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_key in sorted(queries):
        query = queries[query_key]
        all_gold = set(query["all_gold"])
        x_gold = set(query["x_gold"])
        all_covered = {key for key in all_gold if structures_by_patent.get(key)}
        x_covered = {key for key in x_gold if structures_by_patent.get(key)}
        query_has_structure = bool(structures_by_patent.get(query_key))
        rows.append(
            {
                "query_patent_id": query["query_patent_id"],
                "query_key": query_key,
                "query_has_structure": query_has_structure,
                "query_structure_count": len(structures_by_patent.get(query_key, set())),
                "all_gold_count": len(all_gold),
                "all_gold_with_structure": len(all_covered),
                "all_gold_coverage_percent": pct(len(all_covered), len(all_gold)),
                "all_gold_complete": bool(all_gold) and all_covered == all_gold,
                "all_task_any_gold_evaluable": query_has_structure and bool(all_covered),
                "all_task_complete_gold_evaluable": (
                    query_has_structure and bool(all_gold) and all_covered == all_gold
                ),
                "x_gold_count": len(x_gold),
                "x_gold_with_structure": len(x_covered),
                "x_gold_coverage_percent": pct(len(x_covered), len(x_gold)),
                "x_gold_complete": bool(x_gold) and x_covered == x_gold,
                "x_task_any_gold_evaluable": query_has_structure and bool(x_covered),
                "x_task_complete_gold_evaluable": (
                    query_has_structure and bool(x_gold) and x_covered == x_gold
                ),
                "missing_all_gold_keys": "|".join(sorted(all_gold - all_covered)),
                "missing_x_gold_keys": "|".join(sorted(x_gold - x_covered)),
            }
        )
    return rows


def boolean_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(row[field]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--candidate-patents", type=Path, required=True)
    parser.add_argument("--candidate-id-column", default="patent_id")
    parser.add_argument(
        "--structure-file",
        type=Path,
        action="append",
        default=[],
        help="patent_id/patent_ids와 SMILES가 있는 CSV 또는 JSONL, 반복 지정 가능",
    )
    parser.add_argument(
        "--occurrences",
        type=Path,
        action="append",
        default=[],
        help="structure_id와 patent_id 연결 CSV/JSONL, 반복 지정 가능",
    )
    parser.add_argument("--legacy-molecules", type=Path)
    parser.add_argument("--legacy-patent-molecules", type=Path)
    parser.add_argument(
        "--minimum-evaluable-queries",
        type=int,
        default=30,
        help="부분집합 성능지표 산출을 허용할 최소 query 수",
    )
    parser.add_argument(
        "--conflict-field",
        choices=("conflict_patents", "x_category_patents"),
        default="conflict_patents",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.structure_file and not args.legacy_molecules:
        parser.error("--structure-file 또는 legacy 구조 파일 쌍이 필요합니다")
    if bool(args.legacy_molecules) != bool(args.legacy_patent_molecules):
        parser.error("--legacy-molecules와 --legacy-patent-molecules는 함께 지정해야 합니다")
    if args.minimum_evaluable_queries < 1:
        parser.error("--minimum-evaluable-queries는 1 이상이어야 합니다")

    input_paths = [args.ground_truth, args.candidate_patents]
    input_paths.extend(args.structure_file)
    input_paths.extend(args.occurrences)
    if args.legacy_molecules:
        input_paths.extend([args.legacy_molecules, args.legacy_patent_molecules])
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"입력 파일이 없습니다: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    queries, benchmark_raw_ids, roles, ground_truth_field_mapping = load_ground_truth(
        args.ground_truth,
        issues,
        args.conflict_field,
    )
    candidates, candidate_raw_ids = load_candidate_patents(
        args.candidate_patents, args.candidate_id_column, issues
    )
    for key, values in candidate_raw_ids.items():
        benchmark_raw_ids[key].update(values)
        roles[key].add("candidate")

    occurrence_index = load_occurrences(args.occurrences, issues)
    canonicalizer = Canonicalizer()
    structure_records, flat_stats = load_flat_structure_files(
        args.structure_file, occurrence_index, canonicalizer, issues
    )
    legacy_stats: dict[str, int] | None = None
    if args.legacy_molecules and args.legacy_patent_molecules:
        legacy_records, legacy_stats = load_legacy_structure_pair(
            args.legacy_molecules,
            args.legacy_patent_molecules,
            canonicalizer,
            issues,
        )
        structure_records.extend(legacy_records)

    structures_by_patent: dict[str, set[str]] = defaultdict(set)
    sources_by_patent: dict[str, set[str]] = defaultdict(set)
    raw_structure_links: Counter[tuple[str, str]] = Counter()
    raw_ids_from_structures: dict[str, set[str]] = defaultdict(set)
    for record in structure_records:
        structures_by_patent[record.patent_key].add(record.canonical_smiles)
        sources_by_patent[record.patent_key].add(record.source)
        raw_structure_links[(record.patent_key, record.canonical_smiles)] += 1
        raw_ids_from_structures[record.patent_key].add(record.patent_id_raw)

    duplicate_link_count = sum(count - 1 for count in raw_structure_links.values() if count > 1)
    benchmark_keys = set(roles)
    query_keys = {key for key, value in roles.items() if "query" in value}
    all_gold_keys = {key for key, value in roles.items() if "gold_all" in value}
    x_gold_keys = {key for key, value in roles.items() if "gold_x" in value}

    for key, raw_values in sorted(benchmark_raw_ids.items()):
        if len(raw_values) > 1:
            issue(
                issues,
                source_file=args.ground_truth,
                row_number="",
                issue_type="normalized_patent_id_collision",
                severity="medium",
                patent_id="|".join(sorted(raw_values)),
                detail=f"normalized={key}",
            )

    patent_rows: list[dict[str, Any]] = []
    for patent_key in sorted(benchmark_keys | candidates):
        canonical_smiles = structures_by_patent.get(patent_key, set())
        patent_rows.append(
            {
                "patent_key": patent_key,
                "benchmark_raw_ids": "|".join(sorted(benchmark_raw_ids.get(patent_key, set()))),
                "structure_raw_ids": "|".join(sorted(raw_ids_from_structures.get(patent_key, set()))),
                "roles": "|".join(sorted(roles.get(patent_key, set()))),
                "has_structure": bool(canonical_smiles),
                "unique_canonical_smiles": len(canonical_smiles),
                "structure_sources": "|".join(sorted(sources_by_patent.get(patent_key, set()))),
            }
        )

    query_rows = build_query_rows(queries, structures_by_patent)
    coverage = {
        "query": role_coverage(query_keys, structures_by_patent),
        "all_gold_unique": role_coverage(all_gold_keys, structures_by_patent),
        "x_gold_unique": role_coverage(x_gold_keys, structures_by_patent),
        "candidate_corpus": role_coverage(candidates, structures_by_patent),
    }
    query_eligibility = {
        "query_count": len(query_rows),
        "all_task_any_gold_evaluable_queries": boolean_count(
            query_rows, "all_task_any_gold_evaluable"
        ),
        "all_task_complete_gold_evaluable_queries": boolean_count(
            query_rows, "all_task_complete_gold_evaluable"
        ),
        "x_queries_with_gold": sum(row["x_gold_count"] > 0 for row in query_rows),
        "x_task_any_gold_evaluable_queries": boolean_count(
            query_rows, "x_task_any_gold_evaluable"
        ),
        "x_task_complete_gold_evaluable_queries": boolean_count(
            query_rows, "x_task_complete_gold_evaluable"
        ),
    }
    query_eligibility.update(
        {
            "all_task_any_gold_evaluable_percent": pct(
                query_eligibility["all_task_any_gold_evaluable_queries"], len(query_rows)
            ),
            "all_task_complete_gold_evaluable_percent": pct(
                query_eligibility["all_task_complete_gold_evaluable_queries"], len(query_rows)
            ),
            "x_task_any_gold_evaluable_percent_of_x_queries": pct(
                query_eligibility["x_task_any_gold_evaluable_queries"],
                query_eligibility["x_queries_with_gold"],
            ),
        }
    )

    critical_issues = sum(item["severity"] == "critical" for item in issues)
    full_candidate_coverage = (
        coverage["candidate_corpus"]["patent_count"] > 0
        and coverage["candidate_corpus"]["patents_without_structure"] == 0
    )
    full_query_coverage = (
        coverage["query"]["patent_count"] > 0
        and coverage["query"]["patents_without_structure"] == 0
    )
    all_gold_complete = (
        coverage["all_gold_unique"]["patent_count"] > 0
        and coverage["all_gold_unique"]["patents_without_structure"] == 0
    )
    if critical_issues:
        feasibility = "blocked_critical_data_quality_issues"
    elif full_candidate_coverage and full_query_coverage and all_gold_complete:
        feasibility = "full_corpus_structure_only_ready"
    elif (
        query_eligibility["all_task_any_gold_evaluable_queries"]
        >= args.minimum_evaluable_queries
    ):
        feasibility = "subset_only_structure_evaluation_possible"
    else:
        feasibility = "not_benchmark_evaluable_insufficient_query_gold_structure_overlap"

    manifest = {
        "status": "completed" if not critical_issues else "completed_with_critical_issues",
        "t2_feasibility": feasibility,
        "normalization": {
            "patent_id": (
                "EP 국가·공개번호·kind code를 보존하고 구분자만 제거한 키; "
                "kind code 미상 키는 exact 문서 키와 결합하지 않음"
            ),
            "smiles": "RDKit canonical SMILES",
            "missing_structure_policy": "유사도 0으로 대체하지 않고 결측 유지",
        },
        "ground_truth": {
            "query_count": len(queries),
            "all_citation_links": sum(len(row["all_gold"]) for row in queries.values()),
            "x_citation_links": sum(len(row["x_gold"]) for row in queries.values()),
            "field_mapping": ground_truth_field_mapping,
        },
        "coverage": coverage,
        "query_eligibility": query_eligibility,
        "minimum_evaluable_queries": args.minimum_evaluable_queries,
        "structure_quality": {
            "raw_valid_patent_structure_links": len(structure_records),
            "unique_patents_with_valid_structure": len(structures_by_patent),
            "unique_patent_canonical_smiles_links": len(raw_structure_links),
            "duplicate_patent_canonical_smiles_links": duplicate_link_count,
            "unique_canonical_smiles": len(
                {smiles for values in structures_by_patent.values() for smiles in values}
            ),
            "issue_count": len(issues),
            "issue_counts_by_severity": dict(sorted(Counter(i["severity"] for i in issues).items())),
            "issue_counts_by_type": dict(sorted(Counter(i["issue_type"] for i in issues).items())),
        },
        "source_stats": {"flat_structure_files": flat_stats, "legacy_pair": legacy_stats},
        "input_files": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in input_paths
        },
        "decision_rules": {
            "full_corpus_structure_only_ready": (
                "critical issue가 없고 query, 전체 gold, 명시적 후보 코퍼스의 모든 특허가 "
                "최소 하나의 유효 구조를 보유"
            ),
            "subset_only_structure_evaluation_possible": (
                "구조가 있는 query와 전체 인용 gold가 최소 설정 query 수 이상 겹치지만 "
                "전체 코퍼스 조건은 충족하지 않음"
            ),
            "not_benchmark_evaluable_insufficient_query_gold_structure_overlap": (
                "구조가 있는 query와 전체 인용 gold의 겹침이 최소 설정 query 수보다 적어 "
                "검색 성능지표를 벤치마크 결과로 산출하지 않음"
            ),
        },
    }

    write_csv(
        args.output_dir / "t2_patent_structure_coverage.csv",
        [
            "patent_key",
            "benchmark_raw_ids",
            "structure_raw_ids",
            "roles",
            "has_structure",
            "unique_canonical_smiles",
            "structure_sources",
        ],
        patent_rows,
    )
    write_csv(
        args.output_dir / "t2_query_eligibility.csv",
        [
            "query_patent_id",
            "query_key",
            "query_has_structure",
            "query_structure_count",
            "all_gold_count",
            "all_gold_with_structure",
            "all_gold_coverage_percent",
            "all_gold_complete",
            "all_task_any_gold_evaluable",
            "all_task_complete_gold_evaluable",
            "x_gold_count",
            "x_gold_with_structure",
            "x_gold_coverage_percent",
            "x_gold_complete",
            "x_task_any_gold_evaluable",
            "x_task_complete_gold_evaluable",
            "missing_all_gold_keys",
            "missing_x_gold_keys",
        ],
        query_rows,
    )
    write_csv(
        args.output_dir / "t2_structure_quality_issues.csv",
        [
            "source_file",
            "row_number",
            "issue_type",
            "severity",
            "patent_id",
            "structure_id",
            "detail",
        ],
        issues,
    )
    write_json(args.output_dir / "t2_structure_coverage_manifest.json", manifest)

    report_lines = [
        "# T2 구조 검색 입력 커버리지 감사",
        "",
        f"- 판정: `{feasibility}`",
        f"- query: {coverage['query']['patents_with_structure']}/{coverage['query']['patent_count']} 구조 보유",
        (
            f"- 전체 인용 gold 고유 특허: {coverage['all_gold_unique']['patents_with_structure']}"
            f"/{coverage['all_gold_unique']['patent_count']} 구조 보유"
        ),
        (
            f"- X 인용 gold 고유 특허: {coverage['x_gold_unique']['patents_with_structure']}"
            f"/{coverage['x_gold_unique']['patent_count']} 구조 보유"
        ),
        (
            f"- 후보 코퍼스: {coverage['candidate_corpus']['patents_with_structure']}"
            f"/{coverage['candidate_corpus']['patent_count']} 구조 보유"
        ),
        (
            f"- 전체 인용 태스크 부분 평가 가능 query: "
            f"{query_eligibility['all_task_any_gold_evaluable_queries']}/{len(query_rows)}"
        ),
        f"- 부분집합 성능지표 최소 query gate: {args.minimum_evaluable_queries}",
        f"- critical 품질 문제: {critical_issues}건",
        "",
        "## 해석 원칙",
        "",
        "- 구조가 없는 특허를 유사도 0으로 대체하지 않는다.",
        "- 부분집합 성능은 전체 benchmark 성능과 별도로 보고한다.",
        "- 후보 코퍼스 구조가 불완전하면 full-corpus structure-only 결과로 표현하지 않는다.",
        "- 평가 가능 query가 최소 gate 미만이면 검색 성능지표를 산출하지 않는다.",
    ]
    (args.output_dir / "t2_structure_coverage_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if critical_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
