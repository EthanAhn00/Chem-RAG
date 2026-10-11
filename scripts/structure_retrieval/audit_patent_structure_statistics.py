#!/usr/bin/env python3
"""Audit structures per exact publication from frozen T123 input CSVs.

Read-only source audit. No DB, model, embedding or ranking is opened or changed.
Structure identity is the stored canonical_smiles/model_smiles string, not a
new chemistry standardization. Output is written into a new timestamped folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def exact_id(value):
    value = str(value or '').strip()
    if not re.fullmatch(r'EP\d+[A-Z]\d?', value):
        raise ValueError(f'non-exact publication ID: {value!r}')
    return value


def rows(path, required):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'missing columns in {path}: {required}')
        yield from reader


def structure_links(path, field, candidates):
    links = defaultdict(set)
    global_strings = set()
    row_count = 0
    for row in rows(path, ['patent_id', field]):
        row_count += 1
        pid = exact_id(row['patent_id'])
        if pid not in candidates:
            raise ValueError(f'structure outside frozen candidate corpus: {pid}')
        smiles = row[field]
        if not smiles or smiles != smiles.strip():
            raise ValueError(f'empty or whitespace-padded {field}: {pid}')
        if smiles in links[pid]:
            raise ValueError(f'duplicate patent-structure link: {pid}: {smiles}')
        links[pid].add(smiles)
        global_strings.add(smiles)
    if not row_count:
        raise ValueError(f'empty structure input: {path}')
    return dict(links), row_count, len(global_strings)


def quantile(values, p):
    """Linear interpolation (Hyndman-Fan type 7 / Excel PERCENTILE.INC)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = position.numerator // position.denominator
    weight = position - low
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def describe(values):
    if not values or any(not isinstance(v, int) or v < 0 for v in values):
        raise ValueError('counts must be a nonempty sequence of nonnegative integers')
    total, n = sum(values), len(values)
    return {
        'patents': n, 'patent_structure_links': total,
        'zero_structure_patents': values.count(0),
        'min': min(values), 'max': max(values),
        'median': statistics.median(values),
        'mean': total / n, 'mean_exact_fraction': str(Fraction(total, n)),
        'mean_numerator': total, 'mean_denominator': n,
        'q1': quantile(values, Fraction(1, 4)),
        'q3': quantile(values, Fraction(3, 4)),
    }


def audit(root):
    manifest_path = root / 'structure_retrieval_outputs/t123_5fold_v1/t123_5fold_assignment_manifest.json'
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'frozen':
        raise ValueError('fold assignment manifest is not frozen')
    sources = {}

    def frozen_input(label):
        record = manifest['input_files'][label]
        path = Path(record['path'])
        path = path if path.is_absolute() else root / path
        digest = sha256(path)
        if digest != record['sha256']:
            raise ValueError(f'frozen source hash mismatch: {label}')
        sources[label] = {'path': str(path.resolve()), 'sha256': digest,
                          'bytes': path.stat().st_size}
        return path

    candidate_path = frozen_input('candidate_corpus')
    candidates = set()
    for row in rows(candidate_path, ['patent_id']):
        pid = exact_id(row['patent_id'])
        if pid in candidates:
            raise ValueError(f'duplicate candidate publication: {pid}')
        candidates.add(pid)
    if not candidates:
        raise ValueError('empty candidate corpus')

    summaries, per_patent = [], []
    cohorts = {}
    global_counts = {}
    for label, field in [('original_structures', 'canonical_smiles'),
                         ('prepared_structures', 'model_smiles')]:
        path = frozen_input(label)
        links, row_count, unique_strings = structure_links(path, field, candidates)
        cohorts[label] = set(links)
        global_counts[label] = {
            'identity_field': field, 'rows': row_count,
            'unique_patent_structure_links': sum(map(len, links.values())),
            'unique_structure_strings': unique_strings,
            'patents_with_structures': len(links),
        }
        for population, ids in [('structure_bearing_patents', set(links)),
                                ('all_candidate_patents_including_zeros', candidates)]:
            counts = [len(links.get(pid, set())) for pid in sorted(ids)]
            stat = {'input': label, 'identity_field': field,
                    'population': population, **describe(counts)}
            if stat['patent_structure_links'] != row_count:
                raise ValueError(f'link total mismatch: {label}: {population}')
            summaries.append(stat)
        per_patent.extend({
            'input': label, 'patent_id': pid,
            'has_structures': bool(links.get(pid)),
            'unique_structures': len(links.get(pid, set()))
        } for pid in sorted(candidates))

    if cohorts['original_structures'] != cohorts['prepared_structures']:
        raise ValueError('original and prepared structure patent masks differ')
    for label, record in sources.items():
        if sha256(Path(record['path'])) != record['sha256']:
            raise ValueError(f'source changed during audit: {label}')
    if sha256(manifest_path) != manifest_hash:
        raise ValueError('frozen manifest changed during audit')

    original = global_counts['original_structures']
    table_checks = [
        {'statistic': 'Candidate patents', 'image_claim': 2376,
         'computed': len(candidates)},
        {'statistic': 'Patents with valid structures', 'image_claim': 2218,
         'computed': original['patents_with_structures']},
        {'statistic': 'Patent-structure links', 'image_claim': 271609,
         'computed': original['unique_patent_structure_links']},
        {'statistic': 'Unique canonical SMILES', 'image_claim': 49686,
         'computed': original['unique_structure_strings']},
    ]
    for check in table_checks:
        check['match'] = check['image_claim'] == check['computed']
    report = {
        'status': ('completed' if all(x['match'] for x in table_checks)
                   else 'completed_with_paper_table_discrepancy'),
        'scope': 'frozen structure input files only; no model/DB/evaluation execution',
        'structure_identity': 'exact stored SMILES strings per publication; no re-canonicalization or chemistry transformation',
        'quantile_method': 'linear interpolation / type 7 / PERCENTILE.INC',
        'median_method': 'middle value, or arithmetic mean of the two middle values',
        'candidate_patents': len(candidates),
        'source_files': sources,
        'fold_manifest': {'path': str(manifest_path), 'sha256': manifest_hash},
        'script_sha256': sha256(Path(__file__)),
        'input_counts': global_counts, 'statistics': summaries,
        'paper_image_checks': table_checks,
    }
    return report, per_patent


def self_test():
    s = describe([1, 2, 3, 10])
    assert (s['min'], s['max'], s['median'], s['mean']) == (1, 10, 2.5, 4.0)
    assert (s['q1'], s['q3']) == (1.75, 4.75)
    s = describe([0, 1, 2, 3, 10])
    assert s['median'] == 2 and s['mean'] == 3.2 and s['zero_structure_patents'] == 1
    assert describe([7])['median'] == 7
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'synthetic.csv'
        p.write_text('patent_id,canonical_smiles\nEP1A1,CC\nEP1A1,CO\nEP2A1,CC\n', encoding='utf-8')
        links, n, unique = structure_links(p, 'canonical_smiles', {'EP1A1', 'EP2A1'})
        assert (n, unique, len(links['EP1A1'])) == (3, 2, 2)
        for invalid, candidate_ids in [
            ('EP1A1,CC\nEP1A1,CC\n', {'EP1A1'}),
            ('EP2A1,CC\n', {'EP1A1'}),
            ('EP1A1,\n', {'EP1A1'}),
        ]:
            p.write_text('patent_id,canonical_smiles\n' + invalid, encoding='utf-8')
            try:
                structure_links(p, 'canonical_smiles', candidate_ids)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid input was not rejected')
    print('self_test: PASS (synthetic counting/statistics only; not research data)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = args.project_root.resolve()
    report, per_patent = audit(root)
    out = args.output_dir or root / 'structure_retrieval_outputs/dataset_structure_statistics' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    for filename, data in [('structure_statistics_summary.csv', report['statistics']),
                           ('structures_per_patent.csv', per_patent)]:
        with (out / filename).open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (out / 'structure_statistics_report.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({key: report[key] for key in (
        'status', 'structure_identity', 'input_counts', 'statistics', 'paper_image_checks')
    } | {'output_dir': str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'structure_statistics_blocked', 'reason': str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
