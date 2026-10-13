#!/usr/bin/env python3
"""Extract exact-document structures for T2 candidates from SureChEMBL 2.0. Require matching country/publication number/kind code. Other kind codes for the same number are diagnostic only. Read a reproducible date-pinned remote URL or predownloaded server Parquet snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SNAPSHOT_BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/"
    "SureChEMBL/bulk_data/2026-09-08"
)
EXACT_EP_RE = re.compile(r"^EP(\d+)([A-Z]\d?)$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compact_patent_id(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def parse_exact_ep_patent(value: Any) -> tuple[str, str] | None:
    exact_key = compact_patent_id(value)
    match = EXACT_EP_RE.fullmatch(exact_key)
    if not match:
        return None
    return exact_key, f"EP{match.group(1)}"


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_targets(path: Path, id_column: str) -> list[dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    invalid: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise SystemExit(f"Candidate file lacks {id_column!r} column missing: {path}")
        for row_number, row in enumerate(reader, start=2):
            raw = str(row.get(id_column) or "").strip()
            parsed = parse_exact_ep_patent(raw)
            if parsed is None:
                invalid.append(f"{row_number}:{raw}")
                continue
            exact_key, base_key = parsed
            if exact_key in targets and targets[exact_key]["raw_patent_id"] != raw:
                raise SystemExit(
                    "Distinct values normalize to the same exact patent ID: "
                    f"{targets[exact_key]['raw_patent_id']!r}, {raw!r}"
                )
            targets[exact_key] = {
                "raw_patent_id": raw,
                "exact_key": exact_key,
                "base_key": base_key,
            }
    if invalid:
        preview = ", ".join(invalid[:10])
        raise SystemExit(f"Row is not an EP patent identifier with a kind code: {preview}")
    if not targets:
        raise SystemExit(f"No valid candidate patent identifiers: {path}")
    return [targets[key] for key in sorted(targets)]


def configure_duckdb(
    connection: Any,
    threads: int,
    remote_input: bool,
) -> dict[str, Any]:
    statements = {"threads": str(threads)}
    if remote_input:
        connection.execute("LOAD httpfs")
        statements.update(
            {
                "http_keep_alive": "false",
                "http_retries": "10",
                "http_retry_wait_ms": "1000",
                "http_retry_backoff": "2",
                "http_timeout": "120",
                "enable_http_metadata_cache": "true",
            }
        )
    for name, value in statements.items():
        connection.execute(f"SET {name} = {value}")
    return {
        name: connection.execute(
            f"SELECT current_setting({sql_string(name)})"
        ).fetchone()[0]
        for name in statements
    }


def report_stage(message: str, started_at: float) -> float:
    now = time.perf_counter()
    print(f"{message} (elapsed {now - started_at:.1f} seconds)", flush=True)
    return now


def stable_structure_id(exact_key: str, canonical_smiles: str) -> str:
    digest = hashlib.sha256(
        f"{exact_key}\0{canonical_smiles}".encode("utf-8")
    ).hexdigest()[:20]
    return f"schembl_{digest}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-patents", type=Path, required=True)
    parser.add_argument("--candidate-id-column", default="patent_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--snapshot-base-url",
        default=DEFAULT_SNAPSHOT_BASE_URL,
        help="Date-pinned SureChEMBL 2.0 bulk snapshot URL",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Predownloaded date-pinned Parquet directory on the server",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="DuckDB worker threads (1 recommended for remote inputs)",
    )
    args = parser.parse_args()

    if not args.candidate_patents.is_file():
        raise SystemExit(f"Candidate patent file missing: {args.candidate_patents}")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.snapshot_base_url.rstrip("/").endswith("/latest"):
        parser.error("Use a date-pinned snapshot URL, not latest, for reproducibility")

    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("duckdb is required: pip install duckdb==1.5.5") from exc
    try:
        from rdkit import Chem, RDLogger, rdBase
    except ImportError as exc:
        raise SystemExit("RDKit is required") from exc

    RDLogger.DisableLog("rdApp.*")
    targets = load_targets(args.candidate_patents, args.candidate_id_column)
    base_url = args.snapshot_base_url.rstrip("/")
    file_names = {
        "patents": "patents.parquet",
        "patent_compound_map": "patent_compound_map.parquet",
        "compounds": "compounds.parquet",
        "fields": "fields.parquet",
    }
    if args.snapshot_dir is None:
        snapshot_files = {
            label: f"{base_url}/{file_name}"
            for label, file_name in file_names.items()
        }
        input_mode = "remote_http"
    else:
        snapshot_dir = args.snapshot_dir.resolve()
        missing = [
            str(snapshot_dir / file_name)
            for file_name in file_names.values()
            if not (snapshot_dir / file_name).is_file()
        ]
        if missing:
            raise SystemExit(
                "Required SureChEMBL Parquet files missing: " + ", ".join(missing)
            )
        snapshot_files = {
            label: str(snapshot_dir / file_name)
            for label, file_name in file_names.items()
        }
        input_mode = "server_local_files"

    print(f"Target patents: {len(targets)} loaded.", flush=True)
    connection = duckdb.connect()
    settings = configure_duckdb(
        connection,
        args.threads,
        remote_input=input_mode == "remote_http",
    )
    connection.execute(
        """
        CREATE TEMP TABLE targets (
            raw_patent_id VARCHAR,
            exact_key VARCHAR,
            base_key VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO targets VALUES (?, ?, ?)",
        [
            (row["raw_patent_id"], row["exact_key"], row["base_key"])
            for row in targets
        ],
    )

    stage_started_at = time.perf_counter()
    patents_url = sql_string(snapshot_files["patents"])
    print("Looking up SureChEMBL patent documents.", flush=True)
    connection.execute(
        f"""
        CREATE TEMP TABLE matched_patents AS
        WITH remote_patents AS (
            SELECT
                id,
                patent_number,
                regexp_replace(upper(patent_number), '[^A-Z0-9]', '', 'g')
                    AS remote_exact_key,
                publication_date,
                family_id
            FROM read_parquet({patents_url})
            WHERE upper(country) = 'EP'
        ), normalized AS (
            SELECT
                *,
                'EP' || regexp_extract(remote_exact_key, '^EP([0-9]+)', 1)
                    AS remote_base_key
            FROM remote_patents
        )
        SELECT
            t.raw_patent_id,
            t.exact_key AS target_exact_key,
            t.base_key,
            p.id AS surechembl_patent_id,
            p.patent_number,
            p.remote_exact_key,
            p.publication_date,
            p.family_id,
            t.exact_key = p.remote_exact_key AS is_exact_document
        FROM normalized AS p
        INNER JOIN targets AS t
          ON p.remote_base_key = t.base_key
        """
    )
    stage_started_at = report_stage("Patent document lookup complete.", stage_started_at)
    connection.execute(
        """
        CREATE TEMP TABLE exact_patents AS
        SELECT *
        FROM matched_patents
        WHERE is_exact_document
        """
    )

    map_url = sql_string(snapshot_files["patent_compound_map"])
    print("Looking up compound links for exact documents.", flush=True)
    connection.execute(
        f"""
        CREATE TEMP TABLE matched_mappings AS
        SELECT
            p.*,
            m.compound_id,
            m.field_id
        FROM read_parquet({map_url}) AS m
        INNER JOIN exact_patents AS p
          ON m.patent_id = p.surechembl_patent_id
        """
    )
    stage_started_at = report_stage("Compound-link lookup complete.", stage_started_at)

    compounds_url = sql_string(snapshot_files["compounds"])
    print("Looking up linked compound structures.", flush=True)
    connection.execute(
        f"""
        CREATE TEMP TABLE matched_compounds AS
        SELECT c.*
        FROM read_parquet({compounds_url}) AS c
        INNER JOIN (
            SELECT DISTINCT compound_id
            FROM matched_mappings
        ) AS m
          ON c.id = m.compound_id
        """
    )
    stage_started_at = report_stage("Compound-structure lookup complete.", stage_started_at)

    fields_url = sql_string(snapshot_files["fields"])
    field_rows = connection.execute(
        f"SELECT * FROM read_parquet({fields_url}) ORDER BY id"
    ).fetchall()
    field_names = {int(row[0]): str(row[1]) for row in field_rows}

    connection.execute(
        """
        CREATE TEMP TABLE raw_structure_rows AS
        SELECT
            m.raw_patent_id,
            m.target_exact_key,
            m.base_key,
            m.surechembl_patent_id,
            m.patent_number,
            m.publication_date,
            m.family_id,
            m.compound_id,
            c.smiles,
            c.inchi,
            c.inchi_key AS inchikey,
            c.mol_weight,
            m.field_id
        FROM matched_mappings AS m
        INNER JOIN matched_compounds AS c
          ON m.compound_id = c.id
        """
    )
    raw_rows = connection.execute(
        """
        SELECT *
        FROM raw_structure_rows
        ORDER BY target_exact_key, compound_id, field_id
        """
    ).fetchall()

    matched_document_rows = connection.execute(
        "SELECT count(*) FROM matched_patents"
    ).fetchone()[0]
    matched_base_count = connection.execute(
        "SELECT count(DISTINCT base_key) FROM matched_patents"
    ).fetchone()[0]
    exact_document_rows = connection.execute(
        "SELECT count(*) FROM exact_patents"
    ).fetchone()[0]
    exact_candidate_count = connection.execute(
        "SELECT count(DISTINCT target_exact_key) FROM exact_patents"
    ).fetchone()[0]
    base_only_count = connection.execute(
        """
        SELECT count(*)
        FROM targets AS t
        WHERE EXISTS (
            SELECT 1 FROM matched_patents AS m WHERE m.base_key = t.base_key
        )
          AND NOT EXISTS (
            SELECT 1
            FROM exact_patents AS e
            WHERE e.target_exact_key = t.exact_key
        )
        """
    ).fetchone()[0]

    match_rows = connection.execute(
        """
        WITH base_counts AS (
            SELECT
                base_key,
                count(DISTINCT surechembl_patent_id) AS base_document_count
            FROM matched_patents
            GROUP BY base_key
        ), exact_counts AS (
            SELECT
                target_exact_key,
                count(DISTINCT surechembl_patent_id) AS exact_document_count
            FROM exact_patents
            GROUP BY target_exact_key
        ), compound_counts AS (
            SELECT
                target_exact_key,
                count(DISTINCT compound_id) AS exact_compound_count
            FROM matched_mappings
            GROUP BY target_exact_key
        )
        SELECT
            t.raw_patent_id,
            t.exact_key,
            t.base_key,
            coalesce(b.base_document_count, 0) AS base_document_count,
            coalesce(e.exact_document_count, 0) AS exact_document_count,
            coalesce(c.exact_compound_count, 0) AS exact_compound_count
        FROM targets AS t
        LEFT JOIN base_counts AS b ON b.base_key = t.base_key
        LEFT JOIN exact_counts AS e ON e.target_exact_key = t.exact_key
        LEFT JOIN compound_counts AS c ON c.target_exact_key = t.exact_key
        ORDER BY t.exact_key
        """
    ).fetchall()
    connection.close()

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    for row in raw_rows:
        (
            patent_id,
            exact_key,
            base_key,
            surechembl_patent_id,
            surechembl_patent_number,
            publication_date,
            family_id,
            compound_id,
            input_smiles,
            inchi,
            inchikey,
            mol_weight,
            field_id,
        ) = row
        molecule = Chem.MolFromSmiles(str(input_smiles or ""))
        if molecule is None:
            issues.append(
                {
                    "patent_id": patent_id,
                    "surechembl_patent_number": surechembl_patent_number,
                    "compound_id": compound_id,
                    "issue_type": "unparseable_smiles",
                    "input_smiles": input_smiles,
                }
            )
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        key = (exact_key, canonical)
        item = grouped.setdefault(
            key,
            {
                "structure_id": stable_structure_id(exact_key, canonical),
                "patent_id": patent_id,
                "patent_key": base_key,
                "exact_patent_key": exact_key,
                "surechembl_patent_ids": set(),
                "surechembl_patent_numbers": set(),
                "publication_dates": set(),
                "family_ids": set(),
                "compound_ids": set(),
                "canonical_smiles": canonical,
                "input_smiles_examples": set(),
                "inchis": set(),
                "inchikeys": set(),
                "mol_weights": set(),
                "field_ids": set(),
                "field_names": set(),
            },
        )
        item["surechembl_patent_ids"].add(str(surechembl_patent_id))
        item["surechembl_patent_numbers"].add(str(surechembl_patent_number))
        if publication_date is not None:
            item["publication_dates"].add(str(publication_date))
        if family_id is not None:
            item["family_ids"].add(str(family_id))
        item["compound_ids"].add(str(compound_id))
        item["input_smiles_examples"].add(str(input_smiles))
        if inchi:
            item["inchis"].add(str(inchi))
        if inchikey:
            item["inchikeys"].add(str(inchikey))
        if mol_weight is not None:
            item["mol_weights"].add(str(mol_weight))
        item["field_ids"].add(str(field_id))
        item["field_names"].add(field_names.get(int(field_id), f"unknown_{field_id}"))

    structure_rows: list[dict[str, Any]] = []
    structure_count_by_patent: Counter[str] = Counter()
    for (exact_key, _), item in sorted(grouped.items()):
        row = {
            "structure_id": item["structure_id"],
            "patent_id": item["patent_id"],
            "patent_key": item["patent_key"],
            "exact_patent_key": exact_key,
            "canonical_smiles": item["canonical_smiles"],
            "surechembl_patent_ids": "|".join(sorted(item["surechembl_patent_ids"])),
            "surechembl_patent_numbers": "|".join(
                sorted(item["surechembl_patent_numbers"])
            ),
            "publication_dates": "|".join(sorted(item["publication_dates"])),
            "family_ids": "|".join(sorted(item["family_ids"])),
            "compound_ids": "|".join(sorted(item["compound_ids"], key=int)),
            "input_smiles_examples": "|".join(sorted(item["input_smiles_examples"])),
            "inchis": "|".join(sorted(item["inchis"])),
            "inchikeys": "|".join(sorted(item["inchikeys"])),
            "mol_weights": "|".join(sorted(item["mol_weights"])),
            "field_ids": "|".join(sorted(item["field_ids"], key=int)),
            "field_names": "|".join(sorted(item["field_names"])),
            "source": "surechembl_2_0",
            "source_snapshot": base_url.rsplit("/", 1)[-1],
            "match_type": "exact_document",
        }
        structure_rows.append(row)
        structure_count_by_patent[exact_key] += 1

    status_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for raw_id, exact_key, base_key, base_docs, exact_docs, exact_compounds in match_rows:
        valid_structures = structure_count_by_patent[exact_key]
        if valid_structures:
            status = "exact_document_with_valid_structure"
        elif exact_docs:
            status = "exact_document_without_valid_structure"
        elif base_docs:
            status = "base_number_only_different_kind"
        else:
            status = "not_found"
        status_counts[status] += 1
        status_rows.append(
            {
                "patent_id": raw_id,
                "exact_patent_key": exact_key,
                "patent_key": base_key,
                "base_document_count": base_docs,
                "exact_document_count": exact_docs,
                "exact_compound_count": exact_compounds,
                "valid_structure_count": valid_structures,
                "status": status,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    structure_path = args.output_dir / "surechembl_structures.csv"
    status_path = args.output_dir / "surechembl_patent_status.csv"
    issue_path = args.output_dir / "surechembl_structure_issues.csv"
    manifest_path = args.output_dir / "surechembl_extraction_manifest.json"

    write_csv_atomic(
        structure_path,
        [
            "structure_id",
            "patent_id",
            "patent_key",
            "exact_patent_key",
            "canonical_smiles",
            "surechembl_patent_ids",
            "surechembl_patent_numbers",
            "publication_dates",
            "family_ids",
            "compound_ids",
            "input_smiles_examples",
            "inchis",
            "inchikeys",
            "mol_weights",
            "field_ids",
            "field_names",
            "source",
            "source_snapshot",
            "match_type",
        ],
        structure_rows,
    )
    write_csv_atomic(
        status_path,
        [
            "patent_id",
            "exact_patent_key",
            "patent_key",
            "base_document_count",
            "exact_document_count",
            "exact_compound_count",
            "valid_structure_count",
            "status",
        ],
        status_rows,
    )
    write_csv_atomic(
        issue_path,
        [
            "patent_id",
            "surechembl_patent_number",
            "compound_id",
            "issue_type",
            "input_smiles",
        ],
        issues,
    )

    manifest = {
        "status": "completed",
        "candidate_file": str(args.candidate_patents),
        "candidate_file_sha256": sha256_file(args.candidate_patents),
        "candidate_count": len(targets),
        "snapshot_base_url": base_url,
        "snapshot_input_mode": input_mode,
        "snapshot_files": snapshot_files,
        "snapshot_file_sizes": {
            label: Path(path).stat().st_size
            for label, path in snapshot_files.items()
            if input_mode == "server_local_files"
        },
        "match_policy": {
            "primary": "Use only exact documents matching country/publication number/kind code",
            "base_only": "Record other kind codes diagnostically but exclude them from structure inputs",
            "family": "Patent-family structures are not used at this stage",
        },
        "duckdb_version": duckdb.__version__,
        "rdkit_version": rdBase.rdkitVersion,
        "duckdb_settings": settings,
        "field_names": dict(sorted(field_names.items())),
        "matched_document_rows_by_base": matched_document_rows,
        "matched_base_candidate_count": matched_base_count,
        "exact_document_rows": exact_document_rows,
        "exact_document_candidate_count": exact_candidate_count,
        "base_only_different_kind_candidate_count": base_only_count,
        "raw_structure_field_rows": len(raw_rows),
        "valid_structure_rows": len(structure_rows),
        "patents_with_valid_structure": len(structure_count_by_patent),
        "invalid_smiles_rows": len(issues),
        "patent_status_counts": dict(sorted(status_counts.items())),
        "output_files": {},
    }
    for path in (structure_path, status_path, issue_path):
        manifest["output_files"][str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
