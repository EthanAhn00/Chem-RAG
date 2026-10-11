# Patent-data parsers

| Module | Function |
|---|---|
| `epo_stream_parser.py` | Streams search-report JSON records, filters IPC, and joins record IDs to publication IDs through a reference map |
| `epo_xml_parser.py` | Inspects drawing tags and figure-file metadata in individual patent XML files |
| `patent_fulltext_parser.py` | Local patent-XML and OPS text-only XML parsing for paragraphs, figures/captions, and claim elements |
| `patcid_surechembl_connector.py` | Legacy SureChEMBL/PatCID lookup; selects the highest-confidence available `Clean` SMILES candidate |

Search-report parsing is not a replacement for complete patent-text acquisition.
Supply external sources through configured paths or explicit constructor inputs.

When a constructor path is omitted, both parsers derive their external-data root
from `CHEMRAG_EXTERNAL_DATA_ROOT`, defaulting to this repository's `data/external/`.
`EPOStreamParser` uses `CHEMRAG_EPO_PARSED_BASE` (the parent of `Simple/` and the
other search-report subfolders); `EPOXMLParser` uses `CHEMRAG_EPO_XML_DIR`.
Explicit constructor paths take precedence. These defaults do not download or
bundle the external sources. No workstation-specific drive path is required.

The connector's legacy patent-number normalization removes publication-kind
information. It is not family normalization and must not replace the final T2
exact-publication extraction in `scripts/structure_retrieval/extract_surechembl_structures.py`.

OPS paragraph/claim numbering is heuristic. Accepted legacy duplicate-key
exceptions are recorded in `data/db_export/known_issues.json`; this does not make
the parser an exact claim-element or legal-evidence annotation system.
