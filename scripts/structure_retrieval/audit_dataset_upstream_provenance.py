#!/usr/bin/env python3
"""Recount annual citation sources and reconcile existing derivatives, read-only.

This verifies reproducibility of the selected citation pairs, not the history
of downloading full text or family deduplication. Run on the research server.
"""
import argparse
import csv
import hashlib
import json
import io
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def string_value(node):
    # Ignore phonetic annotations; preserve the text of rich-text runs.
    return ''.join(child.text or '' for part in node
                   for child in ([part] if part.tag == NS + 't'
                                 else part.findall(NS + 't') if part.tag == NS + 'r' else []))


def first_sheet_rows(path, required_columns=None, diagnostics=None):
    """Stream stored XLSX cell values with Python's standard library only.

    Sheet dimensions are not used as record counts. Missing cell positions are
    retained. Required columns are never inferred or formula-evaluated. Cells
    outside those columns contribute to row-presence counting, not pair values.
    """
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        first = workbook.find(NS + 'sheets')
        if first is None or not len(first):
            raise ValueError('workbook has no supported worksheet')
        relationship_id = first[0].get(REL + 'id')
        relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        matched = [r for r in relationships if r.get('Id') == relationship_id]
        if len(matched) != 1 or matched[0].get('TargetMode') == 'External':
            raise ValueError('unsupported worksheet relationship')
        target = matched[0].get('Target', '')
        member = posixpath.normpath(target.lstrip('/') if target.startswith('/')
                                   else posixpath.join('xl', target))
        if not member.startswith('xl/') or '..' in member.split('/'):
            raise ValueError('invalid worksheet archive path')
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            with archive.open('xl/sharedStrings.xml') as handle:
                events = ET.iterparse(handle, events=('start', 'end'))
                shared_root = None
                for event, node in events:
                    if shared_root is None:
                        shared_root = node
                    if event == 'end' and node.tag == NS + 'si':
                        shared.append(string_value(node))
                        shared_root.clear()
        width = None
        required_indices = None
        with archive.open(member) as handle:
            sheet_data = None
            for event, node in ET.iterparse(handle, events=('start', 'end')):
                if event == 'start' and node.tag == NS + 'sheetData':
                    sheet_data = node
                if event != 'end' or node.tag != NS + 'row':
                    continue
                values = {}
                for cell in node.findall(NS + 'c'):
                    match = re.fullmatch(r'([A-Z]+)[1-9][0-9]*', cell.get('r', ''))
                    if not match:
                        raise ValueError('missing/invalid XLSX cell coordinate')
                    column = 0
                    for char in match[1]:
                        column = column * 26 + ord(char) - ord('A') + 1
                    column -= 1
                    if column in values:
                        raise ValueError('duplicate XLSX cell coordinate')
                    value_node = cell.find(NS + 'v')
                    value = value_node.text if value_node is not None else None
                    formula = cell.find(NS + 'f')
                    kind = cell.get('t', 'n')
                    if required_indices is not None and column not in required_indices:
                        if formula is not None and value is None and diagnostics is not None:
                            diagnostics['unused_formula_cells_without_cached_values'] += 1
                        if kind == 'inlineStr':
                            inline = cell.find(NS + 'is')
                            present = inline is not None and bool(string_value(inline))
                        elif kind == 's' and value is not None:
                            offset = int(value)
                            if not 0 <= offset < len(shared):
                                raise ValueError('invalid shared-string reference')
                            present = bool(shared[offset])
                        else:
                            present = value is not None and value != ''
                        # A formula is nonempty even when it has no cached value.
                        # Keep its row; never drop a record because of an unused cell.
                        values[column] = '__unused_nonempty_cell__' if present or formula is not None else None
                        continue
                    if formula is not None and value is None:
                        raise ValueError(f'required/header formula has no stored value: {cell.get("r")}')
                    if kind == 's' and value is not None:
                        offset = int(value)
                        if not 0 <= offset < len(shared):
                            raise ValueError('invalid shared-string reference')
                        value = shared[offset]
                    elif kind == 'inlineStr':
                        inline = cell.find(NS + 'is')
                        value = string_value(inline) if inline is not None else None
                    elif kind == 'b' and value is not None:
                        if value not in ('0', '1'):
                            raise ValueError('invalid boolean cell')
                        value = value == '1'
                    elif kind not in ('n', 'str', 'd', 's', 'b'):
                        raise ValueError(f'unsupported/error XLSX cell type: {kind}')
                    values[column] = None if value == '' else value
                if width is None:
                    width = max(values, default=-1) + 1
                    if required_columns is not None:
                        header = tuple(values.get(i) for i in range(width))
                        if any(header.count(name) != 1 for name in required_columns):
                            raise ValueError(f'missing/duplicate required XLSX headers: {required_columns}')
                        required_indices = {header.index(name) for name in required_columns}
                if any(k >= width and v is not None for k, v in values.items()):
                    raise ValueError('data outside XLSX header columns')
                row = tuple(values.get(i) for i in range(width))
                node.clear()
                if sheet_data is not None:
                    sheet_data.clear()
                yield row


def self_test():
    """Synthetic parser checks only; does not validate research data."""
    workbook = f'<workbook xmlns="{NS[1:-1]}" xmlns:r="{REL[1:-1]}"><sheets><sheet name="Test" sheetId="1" r:id="rId1"/></sheets></workbook>'
    relationships = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet7.xml"/></Relationships>'
    worksheet = f'<worksheet xmlns="{NS[1:-1]}"><dimension ref="A1:A1"/><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>year</t></is></c><c r="C1" t="inlineStr"><is><t>cited</t></is></c></row><row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>2017</v></c></row></sheetData></worksheet>'
    strings = f'<sst xmlns="{NS[1:-1]}"><si><t>ID</t></si><si><r><t>EP123</t></r><r><t>A1</t></r><rPh><t>ignored</t></rPh></si></sst>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, content in [('xl/workbook.xml', workbook), ('xl/_rels/workbook.xml.rels', relationships), ('xl/worksheets/sheet7.xml', worksheet), ('xl/sharedStrings.xml', strings)]:
            archive.writestr(name, content)
    buffer.seek(0)
    result = list(first_sheet_rows(buffer))
    if result != [('ID', 'year', 'cited'), ('EP123A1', '2017', None)]:
        raise AssertionError(f'XLSX reader self-test failed: {result}')
    formula_sheet = worksheet.replace('<c r="B2"><v>2017</v></c>', '<c r="B2"><f>1+1</f></c>')
    buffer2 = io.BytesIO()
    with zipfile.ZipFile(buffer2, 'w') as archive:
        for name, content in [('xl/workbook.xml', workbook), ('xl/_rels/workbook.xml.rels', relationships), ('xl/worksheets/sheet7.xml', formula_sheet), ('xl/sharedStrings.xml', strings)]:
            archive.writestr(name, content)
    diagnostic = Counter()
    buffer2.seek(0)
    projected = list(first_sheet_rows(buffer2, required_columns=('ID', 'cited'), diagnostics=diagnostic))
    if len(projected) != 2 or projected[1][0] != 'EP123A1' or diagnostic['unused_formula_cells_without_cached_values'] != 1:
        raise AssertionError('unused formula handling dropped or corrupted a row')
    buffer2.seek(0)
    try:
        list(first_sheet_rows(buffer2, required_columns=('ID', 'year')))
    except ValueError as exc:
        if 'formula has no stored value' not in str(exc):
            raise
    else:
        raise AssertionError('required formula without cached value was accepted')
    print('self_test: PASS (synthetic XLSX parser only; not research data)')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path, fields):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(fields) <= set(reader.fieldnames or []):
            raise ValueError(f'missing columns: {path}: {fields}')
        return list(reader)


def text(value):
    return '' if value is None else str(value).strip()


def year(value):
    if value is None or text(value) == '':
        raise ValueError('missing file_year')
    number = float(value)
    if not number.is_integer():
        raise ValueError(f'invalid file_year: {value!r}')
    return str(int(number))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('../epo_dataset'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = Path.cwd().resolve()
    source_dir = args.source_dir.resolve()
    hashes = {}

    def source(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[str(path)] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        return path

    pc = rows(source(root / 'data/db_export/patent_content.csv'), ['patent_id'])
    known = {text(r['patent_id']) for r in pc}
    if '' in known or len(known) != len(pc):
        raise ValueError('legacy patent_content IDs are empty or duplicated')
    ec = rows(source(root / 'data/db_export/examiner_citation.csv'), ['query_patent_id', 'cited_patent_id'])
    old = {(text(r['query_patent_id']), text(r['cited_patent_id'])) for r in ec}
    query_rows = rows(source(source_dir / 'citations_for_our_query_patents.csv'), ['ID', 'dnum', 'category', 'file_year'])
    selected_rows = rows(source(source_dir / 'in_corpus_candidate_pairs.csv'), ['query_patent_id', 'cited_patent_id', 'category'])
    expected_rows = Counter((text(r['ID']), text(r['dnum']), text(r['category']), year(r['file_year'])) for r in query_rows)
    selected = {(text(r['query_patent_id']), text(r['cited_patent_id'])) for r in selected_rows}
    if len(selected) != len(selected_rows):
        raise ValueError('selected CSV has duplicate pairs')

    annual_files = sorted(source_dir.glob('parsed_citations_*.xlsx'))
    if not annual_files:
        raise FileNotFoundError('no annual citation XLSX files')
    total = 0
    all_queries, all_cited = set(), set()
    observed_rows = Counter()
    categories = defaultdict(set)
    annual_counts = []
    for path in annual_files:
        source(path)
        diagnostic = Counter()
        iterator = first_sheet_rows(path, required_columns=('ID', 'dnum', 'category', 'file_year'), diagnostics=diagnostic)
        try:
            header = next(iterator)
            index = {v: i for i, v in enumerate(header)}
            if not {'ID', 'dnum', 'category', 'file_year'} <= set(index):
                raise ValueError(f'annual required columns missing: {path.name}')
            count = 0
            for row in iterator:
                if all(v is None for v in row):
                    continue
                count += 1
                q, g, category = (text(row[index[k]]) for k in ('ID', 'dnum', 'category'))
                if q:
                    all_queries.add(q)
                if g:
                    all_cited.add(g)
                if q in known:
                    observed_rows[(q, g, category, year(row[index['file_year']]))] += 1
                    if g:
                        categories[(q, g)].update(c.strip().upper() for c in category.split(',') if c.strip())
            annual_counts.append({'file': path.name, 'nonempty_data_rows': count,
                                 'unused_formula_cells_without_cached_values': diagnostic['unused_formula_cells_without_cached_values']})
            total += count
            print(f'{path.name}: {count} nonempty data rows', flush=True)
        finally:
            iterator.close()
    if observed_rows != expected_rows:
        raise ValueError(f'annual-to-query CSV mismatch: extra rows={sum((observed_rows-expected_rows).values())}, missing rows={sum((expected_rows-observed_rows).values())}')
    valid = set(categories)
    if not old <= valid:
        raise ValueError('legacy citation pairs are not contained in annual-derived valid pairs')
    both = {p for p in valid if p[0] in known and p[1] in known}
    new_both = both - old
    if new_both != selected:
        raise ValueError(f'selected pair rule mismatch: extra={len(new_both-selected)}, missing={len(selected-new_both)}')
    for r in selected_rows:
        key = (text(r['query_patent_id']), text(r['cited_patent_id']))
        if {c.strip().upper() for c in r['category'].split(',') if c.strip()} != categories[key]:
            raise ValueError(f'selected CSV category mismatch: {key}')

    raw = defaultdict(set)
    sr_files = sorted((root / 'data').rglob('epo_srpt_label_*.json'))
    for path in sr_files:
        source(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        for r in data if isinstance(data, list) else [data]:
            key = (text(r['querypatent_id']), text(r['cited_patent_id']))
            raw[key].update(c.strip().upper() for c in text(r['epo_label']).split(',') if c.strip())
    if dict(raw) != {key: categories[key] for key in selected}:
        raise ValueError('raw SR pairs/categories differ from annual-derived selected pairs')
    if sorted(source_dir.glob('parsed_citations_*.xlsx')) != annual_files:
        raise ValueError('annual file membership changed during audit')
    if sorted((root / 'data').rglob('epo_srpt_label_*.json')) != sr_files:
        raise ValueError('raw SR file membership changed during audit')
    for path, record in hashes.items():
        if digest(Path(path)) != record['sha256']:
            raise ValueError(f'input changed during audit: {path}')
    report = {
        'status': 'annual_citation_selection_reconciled_text_creation_history_pending',
        'annual_files': annual_counts,
        'annual_nonempty_data_rows': total,
        'annual_unique_nonempty_query_ids': len(all_queries),
        'annual_unique_nonempty_cited_ids': len(all_cited),
        'legacy_corpus_unique_patents': len(known),
        'query_csv_rows': len(query_rows),
        'query_csv_missing_cited_rows': sum(not text(r['dnum']) for r in query_rows),
        'query_csv_unique_query_ids': len({text(r['ID']) for r in query_rows}),
        'valid_unique_query_cited_pairs': len(valid),
        'legacy_existing_unique_citation_pairs': len(old),
        'both_patents_in_legacy_corpus_unique_pairs': len(both),
        'new_pairs_with_both_patents_in_legacy_corpus': len(new_both),
        'selected_unique_query_patents': len({q for q, g in selected}),
        'selected_unique_patent_union': len({pid for pair in selected for pid in pair}),
        'annual_to_query_csv_multiset': 'MATCH',
        'annual_read_columns': ['ID', 'dnum', 'category', 'file_year'],
        'nonempty_row_rule': 'Count each physical data row with a nonempty stored cell or formula; unused formulas do not remove rows.',
        'selected_csv_pairs_and_categories_to_raw_sr': 'MATCH',
        'selection_rule': 'query in current legacy patent_content; nonempty cited ID; exclude current legacy examiner_citation pairs; cited ID in legacy patent_content',
        'limitations': ['Reconciliation establishes content agreement, not proof of historical execution.', 'Annual data contains citation rows, not full-text documents.', 'Text downloading and historical family normalization are not verified.'],
        'input_files': hashes,
        'script_sha256': digest(Path(__file__).resolve()),
    }
    destination = root / 'structure_retrieval_outputs/dataset_filtering_audit' / datetime.now(timezone.utc).strftime('upstream_%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'upstream_audit_report.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({**{k: v for k, v in report.items() if k not in ('input_files', 'script_sha256')}, 'output_dir': str(destination)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, ImportError, StopIteration,
            zipfile.BadZipFile, ET.ParseError) as exc:
        print(json.dumps({'status': 'upstream_audit_blocked', 'reason': str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
