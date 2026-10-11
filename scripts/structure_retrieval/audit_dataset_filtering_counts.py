#!/usr/bin/env python3
"""Read existing source files and reconcile dataset counts, without opening a DB.

Run on the research server. No embeddings, rankings, or source files are changed.
An upstream original-text count remains unresolved until its source and selection
history are supplied; epo_txt is not silently treated as the original collection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patent_id(value):
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    if not re.fullmatch(r"EP\d+[A-Z]\d?", compact):
        raise ValueError(f"invalid exact EP publication ID: {value!r}")
    return compact


def csv_rows(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"required columns missing: {path}: {fields}")
        yield from reader


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def filter_stage(stages, excluded, branch, label, before, after, unit):
    if not after.issubset(before):
        raise ValueError(f"not a subset: {branch}: {label}")
    stages.append({"branch": branch, "stage": label, "unit": unit,
                   "before": len(before), "excluded": len(before - after),
                   "remaining": len(after)})
    excluded.extend({"branch": branch, "stage": label, "id": key}
                    for key in sorted(before - after))


def read_structures(path, field, candidates):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    links = defaultdict(set)
    validity = {}
    counts = Counter()
    for row in csv_rows(path, ["patent_id", field]):
        counts["rows"] += 1
        pid = patent_id(row["patent_id"])
        if pid not in candidates:
            counts["outside_candidate_rows"] += 1
            continue
        smiles = str(row[field] or "").strip()
        if not smiles:
            raise ValueError(f"empty SMILES in finalized structure input: {path}: {pid}")
        if smiles not in validity:
            molecule = Chem.MolFromSmiles(smiles)
            validity[smiles] = molecule is not None and molecule.GetNumAtoms() > 0
        if not validity[smiles]:
            raise ValueError(f"invalid SMILES in finalized structure input: {path}: {pid}")
        if smiles in links[pid]:
            counts["duplicate_patent_smiles_rows"] += 1
        links[pid].add(smiles)
    counts["patents"] = len(links)
    counts["unique_smiles_strings"] = len(validity)
    counts["unique_patent_smiles_links"] = sum(map(len, links.values()))
    return dict(links), dict(counts)


def audit(root, text_dir):
    sources = {}
    stages, excluded = [], []

    def source(path):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(path) not in sources:
            sources[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        return path

    outputs = root / "structure_retrieval_outputs"
    fold_dir = outputs / "t123_5fold_v1"
    frozen_path = source(fold_dir / "t123_5fold_assignment_manifest.json")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen":
        raise ValueError("assignment manifest is not frozen")

    def frozen_input(label):
        record = frozen["input_files"][label]
        recorded_path = Path(record["path"])
        path = source(recorded_path if recorded_path.is_absolute() else root / recorded_path)
        if sources[str(path)]["sha256"] != record["sha256"]:
            raise ValueError(f"frozen input hash mismatch: {label}")
        return path

    gt_path = frozen_input("ground_truth")
    candidates_path = frozen_input("candidate_corpus")
    original_path = frozen_input("original_structures")
    prepared_path = frozen_input("prepared_structures")
    assignment_path = source(fold_dir / "t123_5fold_assignment.csv")
    if sources[str(assignment_path)]["sha256"] != frozen["assignment_file"]["sha256"]:
        raise ValueError("assignment hash mismatch")

    truth, relevance = {}, {}
    with gt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            q = patent_id(row["query_patent_id"])
            if q in truth:
                raise ValueError(f"duplicate GT query: {q}")
            all_ids = [patent_id(g) for g in row["all_cited_patents"]]
            x_ids = [patent_id(g) for g in row["conflict_patents"]]
            if len(all_ids) != len(set(all_ids)) or len(x_ids) != len(set(x_ids)):
                raise ValueError(f"duplicate GT gold: {q}")
            if not all_ids or not set(x_ids).issubset(all_ids) or q in all_ids:
                raise ValueError(f"invalid GT gold: {q}")
            truth[q] = {"Citation": set(all_ids), "Conflict-X": set(x_ids)}
            relevance[q] = {patent_id(g): set(v["categories"])
                            for g, v in row["relevance"].items()}
            if set(relevance[q]) != set(all_ids):
                raise ValueError(f"GT relevance coverage mismatch: {q}")
            if {g for g, cats in relevance[q].items() if "X" in cats} != set(x_ids):
                raise ValueError(f"GT X category mismatch: {q}")

    candidates = set()
    for row in csv_rows(candidates_path, ["patent_id"]):
        pid = patent_id(row["patent_id"])
        if pid in candidates:
            raise ValueError(f"duplicate candidate: {pid}")
        candidates.add(pid)
    gt_patents = set(truth) | set().union(*(r["Citation"] for r in truth.values()))
    if candidates != gt_patents:
        raise ValueError("candidate pool differs from GT query/gold union")

    # Count the current text collection, not an unknown pre-filter source.
    files = sorted(text_dir.glob("epo_txt_EP*.json"))
    if not files:
        raise FileNotFoundError(f"no epo_txt_EP*.json: {text_dir}")
    texts, text_counts, families = {}, Counter(), defaultdict(set)
    for path in files:
        source(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            raise ValueError(f"text record must be an object: {path}")
        pid = patent_id(row.get("patent_id"))
        if pid in texts:
            raise ValueError(f"duplicate text patent: {pid}")
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        claims = row.get("claims", [])
        if not isinstance(claims, list):
            raise ValueError(f"claims not a list: {path}")
        valid_claims = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError(f"claim not an object: {path}")
            if claim.get("claim_no") is not None and str(claim.get("text", "")).strip():
                valid_claims.append(int(claim["claim_no"]))
        if len(valid_claims) != len(set(valid_claims)):
            raise ValueError(f"duplicate claim unit IDs: {pid}")
        texts[pid] = row
        text_counts["document_units_title_or_abstract"] += int(bool(title or abstract))
        text_counts["whole_claims_units"] += int(bool(valid_claims))
        text_counts["individual_claim_units"] += len(valid_claims)
        family = str(row.get("family_id") or "").strip()
        if row.get("family_id") is not None and not isinstance(row["family_id"], (str, int)):
            raise ValueError(f"unexpected family_id type: {path}")
        if family:
            families[family].add(pid)
        else:
            text_counts["missing_family_id_patents"] += 1
    if not candidates.issubset(texts):
        raise ValueError("candidate patents missing from epo_txt")
    filter_stage(stages, excluded, "patent", "current_epo_txt_to_frozen_candidate_pool",
                 set(texts), candidates, "unique_exact_publication_patents")

    # Independently rebuild pair/category unions from the same raw SR pattern.
    sr_files = sorted((root / "data").glob("**/epo_srpt_label_*.json"))
    if not sr_files:
        raise FileNotFoundError("raw epo_srpt_label_*.json files missing")
    sr_counts, pair_categories = Counter(), defaultdict(set)
    raw_queries, raw_patents, raw_pairs = set(), set(), set()
    for path in sr_files:
        source(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            sr_counts["rows"] += 1
            q, g = str(row.get("querypatent_id", "")).strip(), str(row.get("cited_patent_id", "")).strip()
            if not q or not g:
                sr_counts["missing_id_rows"] += 1
                continue
            q, g = patent_id(q), patent_id(g)
            raw_queries.add(q)
            raw_patents.update((q, g))
            raw_pairs.add((q, g))
            if q not in texts or g not in texts:
                sr_counts["outside_current_text_collection_rows"] += 1
                continue
            sr_counts["usable_rows"] += 1
            pair_categories[(q, g)].update(x.strip().upper()
                                         for x in str(row.get("epo_label", "")).split(",") if x.strip())
    gt_pairs = {(q, g) for q, value in truth.items() for g in value["Citation"]}
    if set(pair_categories) != gt_pairs:
        raise ValueError("raw SR pair union differs from processed ground truth")
    for (q, g), cats in pair_categories.items():
        if cats != relevance[q][g]:
            raise ValueError(f"raw SR category union differs from GT: {q}:{g}")
    filter_stage(stages, excluded, "citation_pair", "both_patents_in_current_text_collection",
                 {f"{q}|{g}" for q, g in raw_pairs},
                 {f"{q}|{g}" for q, g in pair_categories}, "unique_query_cited_pairs")

    original, original_counts = read_structures(original_path, "canonical_smiles", candidates)
    prepared, prepared_counts = read_structures(prepared_path, "model_smiles", candidates)
    usable = set(original)
    if set(prepared) != usable:
        raise ValueError("prepared and original structure patent masks differ")
    filter_stage(stages, excluded, "patent", "exact_document_valid_structure_available",
                 candidates, usable, "unique_exact_publication_patents")
    extraction_path = source(outputs / "t2_surechembl/surechembl_extraction_manifest.json")
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if (extraction.get("status") != "completed"
            or extraction["candidate_file_sha256"] != sources[str(candidates_path)]["sha256"]):
        raise ValueError("extraction provenance/candidate hash mismatch")
    recorded_structures = [v for k, v in extraction["output_files"].items()
                           if Path(k).name == original_path.name]
    if len(recorded_structures) != 1 or recorded_structures[0]["sha256"] != sources[str(original_path)]["sha256"]:
        raise ValueError("extraction output hash mismatch")
    if (extraction["candidate_count"] != len(candidates)
            or extraction["valid_structure_rows"] != original_counts["rows"]
            or extraction["patents_with_valid_structure"] != len(usable)):
        raise ValueError("extraction counts differ from original structure CSV")
    status_path = source(outputs / "t2_surechembl/surechembl_patent_status.csv")
    recorded_status = [v for k, v in extraction["output_files"].items()
                       if Path(k).name == status_path.name]
    if len(recorded_status) != 1 or recorded_status[0]["sha256"] != sources[str(status_path)]["sha256"]:
        raise ValueError("extraction status CSV hash mismatch")
    status_ids, status_counts, missing_structure_reasons = set(), Counter(), Counter()
    for row in csv_rows(status_path, ["patent_id", "valid_structure_count", "status"]):
        pid = patent_id(row["patent_id"])
        if pid in status_ids or pid not in candidates:
            raise ValueError(f"duplicate/non-candidate extraction status: {pid}")
        status_ids.add(pid)
        if int(row["valid_structure_count"]) != len(original.get(pid, set())):
            raise ValueError(f"patent structure count mismatch: {pid}")
        status_counts[row["status"]] += 1
        if pid not in usable:
            missing_structure_reasons[row["status"]] += 1
    if status_ids != candidates or dict(status_counts) != extraction["patent_status_counts"]:
        raise ValueError("extraction status coverage/count mismatch")

    # These are actual sequential query predicates, not differences of mixed units.
    eligible = {}
    for task in ("Citation", "Conflict-X"):
        before = set(truth)
        for label, predicate in (
            ("nonempty_task_gold", lambda q: bool(truth[q][task])),
            ("query_in_candidate_pool", lambda q: q in candidates),
            ("query_has_valid_structure", lambda q: q in usable),
            ("all_task_gold_have_valid_structure", lambda q: truth[q][task].issubset(usable)),
        ):
            after = {q for q in before if predicate(q)}
            filter_stage(stages, excluded, task, label, before, after, "unique_queries")
            before = after
        eligible[task] = before
    if not eligible["Conflict-X"].issubset(eligible["Citation"]):
        raise ValueError("Conflict-X eligibility is not contained in Citation cohort")

    assignment, group_folds, folds = {}, {}, defaultdict(Counter)
    for row in csv_rows(assignment_path, ["query_id", "group_id", "outer_fold", "citation_eligible", "conflict_x_eligible"]):
        q, group, fold = patent_id(row["query_id"]), row["group_id"], int(row["outer_fold"])
        if q in assignment or fold not in range(5) or not group:
            raise ValueError(f"invalid assignment row: {q}")
        if row["citation_eligible"] != "1" or row["conflict_x_eligible"] not in ("0", "1"):
            raise ValueError(f"invalid task flags: {q}")
        if group in group_folds and group_folds[group] != fold:
            raise ValueError(f"group split between folds: {group}")
        group_folds[group] = fold
        assignment[q] = row
        folds[fold]["Citation"] += 1
        folds[fold]["Conflict-X"] += int(row["conflict_x_eligible"])
    assigned_x = {q for q, row in assignment.items() if row["conflict_x_eligible"] == "1"}
    if set(assignment) != eligible["Citation"] or assigned_x != eligible["Conflict-X"]:
        raise ValueError("independent query eligibility differs from frozen assignment")
    if {k: len(v) for k, v in eligible.items()} != frozen["eligible_queries"]:
        raise ValueError("query counts differ from frozen manifest")
    if len(group_folds) != frozen["connected_groups"]:
        raise ValueError("group count differs from frozen manifest")
    for recorded in frozen["folds"]:
        fold = recorded["outer_fold"]
        if (folds[fold]["Citation"] != recorded["citation_queries"]
                or folds[fold]["Conflict-X"] != recorded["conflict_x_queries"]):
            raise ValueError(f"fold counts differ: {fold}")

    inventory = []
    for parent in sorted((root / "data").iterdir()):
        if parent.is_dir():
            inventory.append({"directory": str(parent.relative_to(root)),
                              "direct_file_count": sum(p.is_file() for p in parent.iterdir())})
    if sorted(text_dir.glob("epo_txt_EP*.json")) != files:
        raise ValueError("text collection file membership changed during audit")
    if sorted((root / "data").glob("**/epo_srpt_label_*.json")) != sr_files:
        raise ValueError("search-report file membership changed during audit")
    for path, recorded in sources.items():
        if sha256(Path(path)) != recorded["sha256"]:
            raise ValueError(f"input changed during audit: {path}")
    report = {
        "status": "downstream_reconciled_upstream_provenance_pending",
        "scope": "existing file artifacts only; no Qdrant/DB opened; no evaluation rerun",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_collection": {"path": str(text_dir), "files": len(files), "unique_patents": len(texts), **text_counts},
        "raw_search_report": {"files": len(sr_files), **sr_counts,
                              "unique_pairs": len(raw_pairs), "unique_query_patents": len(raw_queries),
                              "unique_patent_union": len(raw_patents)},
        "ground_truth": {"queries": len(truth), "unique_citation_pairs": len(gt_pairs),
                         "unique_patent_union": len(gt_patents),
                         "nonempty_x_queries": sum(bool(v["Conflict-X"]) for v in truth.values()),
                         "raw_sr_reconstruction": "MATCH"},
        "original_structure_input": original_counts,
        "structure_extraction": {
            "match_policy": extraction["match_policy"],
            "independently_recounted_patent_statuses": dict(status_counts),
            "excluded_patents_by_status": dict(missing_structure_reasons),
            "upstream_statistics_from_hashed_producer_manifest_not_independently_reparsed": {
                key: extraction[key] for key in ("exact_document_candidate_count",
                    "base_only_different_kind_candidate_count", "raw_structure_field_rows", "invalid_smiles_rows")
            },
        },
        "molformer_common_input": prepared_counts,
        "frozen_evaluation": {"queries": {k: len(v) for k, v in eligible.items()},
                              "candidate_patents_with_structures": len(usable),
                              "candidates_per_query_excluding_self": len(usable) - 1,
                              "groups": len(group_folds), "folds": {k: dict(v) for k, v in sorted(folds.items())}},
        "family_metadata_diagnostic_not_a_filter": {
            "family_ids_among_current_text_patents": len(families),
            "families_with_multiple_publications": sum(len(v) > 1 for v in families.values()),
            "no_family_deduplication_applied_by_this_audit": True,
            "frozen_group_policy": frozen["group_policy"],
        },
        "pending_before_paper_table": [
            "Identify the initially supplied text source and its counting unit; epo_txt may already be selected.",
            "Recover upstream selection/filter logs and retained IDs; do not infer a family-filter stage from family metadata.",
            "Confirm which final dataset the manuscript table describes; retrieval and downstream RAG may have different populations.",
        ],
        "data_directory_inventory": inventory,
        "stages": stages,
        "source_files": sources,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    return report, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--text-dir", type=Path, default=Path("data/epo_txt"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        stages, excluded = [], []
        filter_stage(stages, excluded, "test", "subset", {"a", "b"}, {"b"}, "queries")
        assert stages[0]["before"] == stages[0]["excluded"] + stages[0]["remaining"]
        assert excluded == [{"branch": "test", "stage": "subset", "id": "a"}]
        try:
            filter_stage([], [], "test", "invalid", {"a"}, {"b"}, "queries")
        except ValueError:
            pass
        else:
            raise AssertionError("non-subset not rejected")
        assert patent_id("EP 1234 A1") == "EP1234A1"
        print("self_test: PASS (bookkeeping only; not a dataset validation)")
        return 0
    root = args.project_root.resolve()
    text_dir = args.text_dir if args.text_dir.is_absolute() else root / args.text_dir
    report, excluded = audit(root, text_dir)
    destination = args.output_dir or root / "structure_retrieval_outputs/dataset_filtering_audit" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "filtering_stages.csv", ["branch", "stage", "unit", "before", "excluded", "remaining"], report["stages"])
    write_csv(destination / "excluded_ids.csv", ["branch", "stage", "id"], excluded)
    with (destination / "audit_report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    compact = {k: v for k, v in report.items() if k not in ("source_files", "script_sha256")}
    compact["output_dir"] = str(destination)
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, ImportError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "audit_blocked", "reason": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
