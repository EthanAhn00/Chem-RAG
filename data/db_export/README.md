# Legacy relational export

These tables are from the earlier relational export, not the final SureChEMBL
structure set or T123 candidate/ranking bundle.

| Table | Content |
|---|---|
| `patent_content.csv` | Bibliographic fields and available text |
| `molecule.csv`, `patent_molecule.csv` | Legacy molecules and publication–molecule links |
| `examiner_citation.csv` | Legacy citation pairs/categories |
| `split.csv` | Legacy labels; also a hash-checked provenance input to the final fold script |
| `paragraph.csv` | Paragraphs; large derived file excluded from this submission copy |
| `figure.csv`, `claim_element.csv` | Figures/captions and heuristic claim elements |
| `grounding_gold.csv` | Available citation-related grounding fields |
| `exam_case.csv`, `case_relevance.csv` | Legacy schema files, not proof of populated case labels |

Coverage/validation JSON files, if supplied, are run snapshots, not live guarantees.
`known_issues.json` records the accepted legacy duplicate-key baseline. No counts
or percentages are asserted here from an outdated report.

`scripts/db_export/db_export_1_core_tables.py` through
`db_export_6_regenerate_coverage_report.py` implement export, category/full-text
enrichment, claims backfill, and coverage reporting. `validate_db_export.py`
checks CSV integrity. External patent sources and OPS credentials are needed for
the relevant stages; this copy is not sufficient for a complete fresh export.

`build_xy_gold.py` creates a separate X/Y legacy gold view.
`build_time_family_split.py` creates a connected-record/time split; its filename
does not establish verified family-ID deduplication. Neither legacy split is the
final five-fold assignment. Do not regenerate `split.csv` to bypass a frozen hash
check, or substitute legacy molecular data for final exact-publication structures.
