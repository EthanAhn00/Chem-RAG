"""Generate patent-structure location metadata (in_claim / in_figure / occurrences) under data/db_export/derived/ without changing source CSVs.
Never fabricate unsupported values. in_claim/in_figure use true/false/unknown; unverified is unknown, not false.
Set in_claim=true only with traceable claim location/unit_ref, RDKit-validated claim_element.element_smiles, or validated InChIKey/SMILES extracted from claims. Compare canonical SMILES or InChIKey after RDKit parsing, never mere substrings.
Set in_figure=true only with verified figure-molecule links such as figure.ocsr_smiles, not merely a figure row. Assign false only when structure annotations exist for that patent and omit the molecule under a closed-world assumption.
Use integer claim_occurrence_count only when verifiable, otherwise null plus status.
Usage: python3 scripts/qdrant/qdrant_build_structure_metadata.py [--out-dir data/db_export/derived].
"""
import os
import re
import sys
import csv
import glob
import json
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
csv.field_size_limit(10 ** 9)

from config.settings import DATA_DIR
from qdrant_engine.schemas import TRUE, FALSE, UNKNOWN
from qdrant_engine.provenance import source_version

DB_EXPORT_DIR = os.path.join(DATA_DIR, 'db_export')
DEFAULT_OUT_DIR = os.path.join(DB_EXPORT_DIR, 'derived')

# InChIKey's fixed format makes accidental claim-text matches unlikely.
INCHIKEY_RE = re.compile(r'\b([A-Z]{14}-[A-Z]{10}-[A-Z])\b')
# Candidate SMILES tokens use chemical characters and include bond/ring symbols.
SMILES_CANDIDATE_RE = re.compile(r'(?<![A-Za-z0-9])([A-Za-z0-9@\+\-\[\]\(\)=#\\/%\.]{6,})(?![A-Za-z0-9])')
# Require minimum heavy atoms; tiny molecules risk accidental text matches.
MIN_HEAVY_ATOMS_FOR_TEXT_EVIDENCE = 3


def load_csv(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def nonempty(v):
    return isinstance(v, str) and v.strip() != ''


class Canonicalizer:
    """SMILES -> (canonical SMILES, InChIKey, heavy-atom count), or None on failure."""

    def __init__(self):
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        self._Chem = Chem
        self._cache = {}

    def canon(self, smiles):
        if not nonempty(smiles):
            return None
        if smiles in self._cache:
            return self._cache[smiles]
        result = None
        try:
            mol = self._Chem.MolFromSmiles(smiles)
            if mol is not None:
                result = {
                    'canonical_smiles': self._Chem.MolToSmiles(mol),
                    'inchikey': self._Chem.MolToInchiKey(mol) or None,
                    'heavy_atoms': mol.GetNumHeavyAtoms(),
                }
        except Exception:
            result = None
        self._cache[smiles] = result
        return result


def extract_text_structure_evidence(text, canon, target_canonical, target_inchikey):
    """Extract only validated structure evidence from claims. Return (evidence list, text-scanned flag). Accept exact InChIKey matches or candidate SMILES that RDKit canonicalizes to the target structure."""
    evidence = []
    if not nonempty(text):
        return evidence, False

    if target_inchikey:
        for m in INCHIKEY_RE.finditer(text):
            if m.group(1).upper() == target_inchikey.upper():
                evidence.append({'method': 'claims_text_inchikey', 'matched': m.group(1),
                                 'char_offset': m.start()})

    for m in SMILES_CANDIDATE_RE.finditer(text):
        token = m.group(1)
        # Exclude plain words/numbers without bond/ring/branching symbols from SMILES candidates.
        if not any(ch in token for ch in '()[]=#@+.\\/123456789'):
            continue
        parsed = canon.canon(token)
        if not parsed:
            continue
        if parsed['heavy_atoms'] < MIN_HEAVY_ATOMS_FOR_TEXT_EVIDENCE:
            continue
        if parsed['canonical_smiles'] == target_canonical:
            evidence.append({'method': 'claims_text_smiles', 'matched': token,
                             'char_offset': m.start()})
    return evidence, True


def load_ocsr_cache(cache_dir, canon):
    """Group successful data/ocsr_cache/*.json results by patent number. Results identify pages, not fig_id; correspondence with figure.csv fig_no/fig_id is unverified. Retain as separate unverified evidence, not verified figure evidence."""
    entries = defaultdict(list)
    stats = {'files': 0, 'success': 0, 'not_found': 0, 'unparseable_smiles': 0}
    if not os.path.isdir(cache_dir):
        return entries, stats
    for path in sorted(glob.glob(os.path.join(cache_dir, '*.json'))):
        stats['files'] += 1
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rec = json.load(f)
        except Exception:
            continue
        if rec.get('status') != 'SUCCESS':
            stats['not_found'] += 1
            continue
        stats['success'] += 1
        parsed = canon.canon(rec.get('smiles'))
        if not parsed:
            stats['unparseable_smiles'] += 1
            continue
        key = normalize_patent_key(os.path.basename(path)[:-5])
        entries[key].append({
            'canonical': parsed['canonical_smiles'], 'inchikey': parsed['inchikey'],
            'source_page': rec.get('source_page'), 'confidence': rec.get('confidence'),
            'engine': rec.get('engine'), 'crop_method': rec.get('crop_method'),
        })
    return entries, stats


def _patcid_candidate_lines(jsonl_path, prefixes):
    """Extract relevant-patent lines from roughly 4GB JSONL. Prefer grep -F for initial filtering over a slow Python full scan; without grep, use one compiled regex with equivalent accuracy but different speed."""
    import shutil
    import tempfile
    import subprocess

    if shutil.which('grep'):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write('\n'.join(sorted(prefixes)) + '\n')
            pat_file = f.name
        try:
            proc = subprocess.run(['grep', '-F', '-f', pat_file, jsonl_path],
                                  capture_output=True, text=True, timeout=600)
            # grep exit code 1 means no matches, not an error.
            if proc.returncode in (0, 1):
                return [l for l in proc.stdout.splitlines() if l.strip()], 'grep'
        except Exception:
            pass
        finally:
            try:
                os.remove(pat_file)
            except OSError:
                pass

    rx = re.compile('|'.join(re.escape(k) for k in sorted(prefixes)))
    out = []
    with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if rx.search(line):
                out.append(line)
    return out, 'python-regex'


def load_patcid_figures(jsonl_path, wanted_keys, canon, max_lines=None):
    """Read Clean structures for requested patents from the PatCID EPO map. Page/bbox recognition is not verified one-to-one with figure.csv fig_id, so record separately, not as in_figure=true evidence."""
    entries = defaultdict(list)
    stats = {'file_present': os.path.exists(jsonl_path), 'prefilter': None,
             'candidate_lines': 0, 'patents_matched': 0,
             'clean_structures': 0, 'markush_structures': 0, 'false_positive_lines': 0}
    if not stats['file_present']:
        return entries, stats

    prefixes = set(wanted_keys)
    lines, how = _patcid_candidate_lines(jsonl_path, prefixes)
    stats['prefilter'] = how
    stats['candidate_lines'] = len(lines)

    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        ids = obj.get('patent', {}).get('identifiers', []) or []
        keys = {normalize_patent_key(x) for x in ids} & prefixes
        if not keys:
            stats['false_positive_lines'] += 1   # Accidental substring match.
            continue
        stats['patents_matched'] += 1
        for fig in obj.get('figures', []) or []:
            cls = fig.get('class')
            if cls == 'Markush':
                stats['markush_structures'] += 1
            if cls != 'Clean' or not fig.get('smiles'):
                continue
            parsed = canon.canon((fig.get('smiles') or {}).get('value'))
            if not parsed:
                continue
            stats['clean_structures'] += 1
            for k in keys:
                entries[k].append({
                    'canonical': parsed['canonical_smiles'], 'inchikey': parsed['inchikey'],
                    'page': fig.get('page'), 'bbox': fig.get('bbox'),
                    'confidence': (fig.get('smiles') or {}).get('confidence'),
                    'figure_class': cls,
                })
    return entries, stats


def normalize_patent_key(patent_id):
    """'EP3660125A1', 'EP3660125', 'EP-3660125-B1' -> 'EP3660125', ignoring kind code."""
    m = re.match(r'^([A-Za-z]{2})-?(\d+)', str(patent_id or '').strip())
    return f'{m.group(1).upper()}{m.group(2)}' if m else str(patent_id or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--ocsr-cache-dir', default=os.path.join(DATA_DIR, 'ocsr_cache'))
    ap.add_argument('--patcid-epo-jsonl', default=None,
                    help='PatCID EPO mapping JSONL path (default: data/patcid_surechembl_db/patcid/...)')
    ap.add_argument('--skip-patcid', action='store_true', help='Skip the large PatCID scan')
    ap.add_argument('--limit-patents', type=int, default=None, help='Debug limit on patents to process')
    args = ap.parse_args()

    paths = {name: os.path.join(DB_EXPORT_DIR, f'{name}.csv') for name in
             ('molecule', 'patent_molecule', 'patent_content', 'claim_element', 'figure')}
    src_version, src_detail = source_version(list(paths.values()))
    print(f'[source_version] {src_version} {src_detail}')

    print('[1/4] Loading CSVs...')
    molecules = {r['mol_id']: r for r in load_csv(paths['molecule'])}
    pm_rows = load_csv(paths['patent_molecule'])
    claims_text_of = {}
    for r in load_csv(paths['patent_content']):
        claims_text_of[r['patent_id']] = r.get('claims_text') or ''

    # Supplement empty claims_text using full claims from data/epo_txt.
    # Never overwrite existing patent_content.claims_text values.
    epo_txt_dir = os.path.join(DATA_DIR, 'epo_txt')
    epo_txt_filled = 0
    if os.path.isdir(epo_txt_dir):
        for fn in sorted(os.listdir(epo_txt_dir)):
            if not (fn.startswith('epo_txt_') and fn.endswith('.json')):
                continue
            pid = fn[len('epo_txt_'):-len('.json')]
            if nonempty(claims_text_of.get(pid)):
                continue
            try:
                with open(os.path.join(epo_txt_dir, fn), encoding='utf-8') as f:
                    doc = json.load(f)
            except Exception:
                continue
            txt = '\n'.join((cl.get('text') or '') for cl in (doc.get('claims') or []))
            if txt.strip():
                claims_text_of[pid] = txt
                epo_txt_filled += 1
    print(f'  claims_text supplemented from epo_txt: {epo_txt_filled} records')

    # Known duplicate claim_element.element_id values: append row numbers for
    # collision-free internal IDs while retaining original IDs in evidence payloads.
    claim_elements_by_patent = defaultdict(list)
    for i, r in enumerate(load_csv(paths['claim_element'])):
        claim_elements_by_patent[r['patent_id']].append({
            'internal_id': f"{r['element_id']}#row{i}",
            'element_id': r['element_id'],
            'claim_no': r.get('claim_no') or None,
            'element_no': r.get('element_no') or None,
            'element_text': r.get('element_text') or '',
            'element_smiles': r.get('element_smiles') or '',
        })

    figures_by_patent = defaultdict(list)
    for r in load_csv(paths['figure']):
        figures_by_patent[r['patent_id']].append({
            'fig_id': r['fig_id'], 'fig_no': r.get('fig_no'),
            'ocsr_smiles': r.get('ocsr_smiles') or '',
            'image_path': r.get('image_path') or '',
        })

    # Count source (patent_id, mol_id) rows, i.e. DB duplicates,
    # not actual occurrences in claims (claim_occurrence_count).
    pair_row_counts = Counter((r['patent_id'], r['mol_id']) for r in pm_rows)

    # Deduplicate (patent_id, mol_id); source rows can repeat.
    seen_pairs = set()
    pairs = []
    duplicate_rows = 0
    for r in pm_rows:
        key = (r['patent_id'], r['mol_id'])
        if key in seen_pairs:
            duplicate_rows += 1
            continue
        seen_pairs.add(key)
        pairs.append(r)
    print(f'  Source patent_molecule: {len(pm_rows)} rows -> unique (patent_id, mol_id): {len(pairs)} pairs '
          f'(duplicates: {duplicate_rows} rows removed)')

    canon = Canonicalizer()

    # First determine annotation availability per patent for the closed-world assumption.
    patent_has_claim_smiles_annotation = {
        pid: any(nonempty(e['element_smiles']) for e in els)
        for pid, els in claim_elements_by_patent.items()}
    patent_has_figure_ocsr_annotation = {
        pid: any(nonempty(f['ocsr_smiles']) for f in figs)
        for pid, figs in figures_by_patent.items()}

    print('[2/4] Building canonical claim-structure annotation index...')
    claim_struct_index = defaultdict(list)   # patent_id -> [(canonical, inchikey, element_meta)]
    for pid, els in claim_elements_by_patent.items():
        for e in els:
            if not nonempty(e['element_smiles']):
                continue
            parsed = canon.canon(e['element_smiles'])
            if parsed:
                claim_struct_index[pid].append((parsed['canonical_smiles'], parsed['inchikey'], e))

    figure_struct_index = defaultdict(list)  # patent_id -> [(canonical, inchikey, figure_meta)]
    for pid, figs in figures_by_patent.items():
        for fmeta in figs:
            if not nonempty(fmeta['ocsr_smiles']):
                continue
            parsed = canon.canon(fmeta['ocsr_smiles'])
            if parsed:
                figure_struct_index[pid].append((parsed['canonical_smiles'], parsed['inchikey'], fmeta))

    wanted_keys = {normalize_patent_key(pid) for pid in {r['patent_id'] for r in pairs}}
    print('[3/5] Inspecting additional evidence (OCSR cache / PatCID)...')
    ocsr_entries, ocsr_stats = load_ocsr_cache(args.ocsr_cache_dir, canon)
    print(f'  OCSR cache: {ocsr_stats}')
    print(f'  Patents with successful OCSR present in patent_molecule: '
          f'{len(set(ocsr_entries) & wanted_keys)} records')
    patcid_path = args.patcid_epo_jsonl or os.path.join(
        DATA_DIR, 'patcid_surechembl_db', 'patcid', 'patcid_patent_to_molecules_epo.jsonl')
    if args.skip_patcid:
        patcid_entries, patcid_stats = defaultdict(list), {'skipped': True}
    else:
        patcid_entries, patcid_stats = load_patcid_figures(patcid_path, wanted_keys, canon)
    print(f'  PatCID: {patcid_stats}')

    print('[4/5] Determining locations per (patent, molecule)...')
    records = []
    patents_seen = set()
    for r in pairs:
        pid, mid = r['patent_id'], r['mol_id']
        if args.limit_patents and pid not in patents_seen and len(patents_seen) >= args.limit_patents:
            continue
        patents_seen.add(pid)

        mol = molecules.get(mid, {})
        smiles = mol.get('smiles') or ''
        parsed = canon.canon(smiles)
        target_canonical = parsed['canonical_smiles'] if parsed else None
        target_inchikey = (parsed['inchikey'] if parsed else None) or (mol.get('inchikey') or None)

        evidence = {'claim': [], 'figure': [], 'unverified': [],
                    'methods_attempted': [], 'notes': []}
        pkey = normalize_patent_key(pid)

        # ---------------- 1) Explicit patent_molecule location links ----------------
        evidence['methods_attempted'].append('patent_molecule_location')
        loc = (r.get('location') or '').strip().lower()
        unit_ref = (r.get('unit_ref') or '').strip()
        if loc in ('claim', 'claims'):
            evidence['claim'].append({'method': 'patent_molecule_location', 'unit_ref': unit_ref or None})
        elif loc in ('figure', 'fig', 'drawing'):
            evidence['figure'].append({'method': 'patent_molecule_location', 'unit_ref': unit_ref or None})

        # ---------------- 2) Validated claim_element.element_smiles matches ----------------
        evidence['methods_attempted'].append('claim_element_smiles')
        matched_claim_nos = []
        if target_canonical:
            for cand_canon, cand_ik, meta in claim_struct_index.get(pid, []):
                same = (cand_canon == target_canonical) or (
                    target_inchikey and cand_ik and cand_ik == target_inchikey)
                if same:
                    evidence['claim'].append({
                        'method': 'claim_element_smiles',
                        'element_id': meta['element_id'],
                        'internal_id': meta['internal_id'],
                        'claim_no': meta['claim_no'],
                    })
                    if meta['claim_no']:
                        matched_claim_nos.append(meta['claim_no'])

        # ---------------- 3) Validated structures extracted from claim text ----------------
        evidence['methods_attempted'].append('claims_text_extraction')
        claims_text = claims_text_of.get(pid, '')
        text_evidence, text_scanned = ([], False)
        if target_canonical:
            text_evidence, text_scanned = extract_text_structure_evidence(
                claims_text, canon, target_canonical, target_inchikey)
        evidence['claim'].extend(text_evidence)

        # ---------------- 4) Validated figure.ocsr_smiles matches ----------------
        evidence['methods_attempted'].append('figure_ocsr_smiles')
        matched_fig_ids = []
        if target_canonical:
            for cand_canon, cand_ik, fmeta in figure_struct_index.get(pid, []):
                same = (cand_canon == target_canonical) or (
                    target_inchikey and cand_ik and cand_ik == target_inchikey)
                if same:
                    evidence['figure'].append({'method': 'figure_ocsr_smiles', 'fig_id': fmeta['fig_id'],
                                               'fig_no': fmeta['fig_no']})
                    matched_fig_ids.append(fmeta['fig_id'])

        # ------- 5) OCSR image results: page-level, fig_id unverified -------
        evidence['methods_attempted'].append('ocsr_cache')
        if target_canonical:
            for e in ocsr_entries.get(pkey, []):
                if e['canonical'] == target_canonical or (
                        target_inchikey and e['inchikey'] and e['inchikey'] == target_inchikey):
                    evidence['unverified'].append({
                        'method': 'ocsr_cache', 'source_page': e['source_page'],
                        'confidence': e['confidence'], 'engine': e['engine'],
                        'crop_method': e['crop_method'],
                        'why_unverified': 'source_page correspondence to figure.csv fig_id/fig_no is unverified',
                    })

        # ------- 6) PatCID Clean structures: page/bbox-level, fig_id unverified -------
        evidence['methods_attempted'].append('patcid_clean_figures')
        if target_canonical:
            for e in patcid_entries.get(pkey, []):
                if e['canonical'] == target_canonical or (
                        target_inchikey and e['inchikey'] and e['inchikey'] == target_inchikey):
                    evidence['unverified'].append({
                        'method': 'patcid_clean_figure', 'page': e['page'], 'bbox': e['bbox'],
                        'confidence': e['confidence'], 'figure_class': e['figure_class'],
                        'why_unverified': 'PatCID page/bbox correspondence to figure.csv fig_id is unverified',
                    })

        # ---------------- Classification ----------------
        has_claim_annotation = patent_has_claim_smiles_annotation.get(pid, False)
        has_figure_annotation = patent_has_figure_ocsr_annotation.get(pid, False)

        if evidence['claim']:
            in_claim = TRUE
        elif has_claim_annotation:
            # Claim annotations exist for this patent but omit the molecule: evidence-backed false.
            in_claim = FALSE
        else:
            in_claim = UNKNOWN

        if evidence['figure']:
            in_figure = TRUE
        elif has_figure_annotation:
            in_figure = FALSE
        else:
            in_figure = UNKNOWN

        # ---- Occurrence counts ----
        # claim_occurrence_count: matched occurrences, counting repetitions within one claim separately.
        # claim_count_with_structure: unique claims containing the structure at least once.
        # Presence-only evidence, e.g. patent_molecule.location=claim, cannot establish counts.
        # Keep null + known_presence_unknown_count instead of assuming 1.
        countable = [e for e in evidence['claim']
                     if e.get('method') in ('claim_element_smiles', 'claims_text_smiles',
                                            'claims_text_inchikey')]
        presence_only = [e for e in evidence['claim']
                         if e.get('method') == 'patent_molecule_location']
        distinct_claims = sorted({str(e['claim_no']) for e in evidence['claim']
                                  if e.get('claim_no')})

        if in_claim == TRUE:
            if countable:
                count = len(countable)
                claim_count_with_structure = len(distinct_claims) if distinct_claims else None
                occurrence_status = ('verified_occurrences' if distinct_claims
                                     else 'verified_occurrences_without_claim_no')
            else:
                # Presence is known but occurrence count is unavailable.
                count = None
                claim_count_with_structure = len(distinct_claims) or None
                occurrence_status = 'known_presence_unknown_count'
        elif in_claim == FALSE:
            count, claim_count_with_structure, occurrence_status = 0, 0, 'verified_absent'
        else:
            count, claim_count_with_structure = None, None
            occurrence_status = ('unverifiable_no_claims_text' if not nonempty(claims_text)
                                 else 'unverifiable_no_claim_structure_annotation')

        occurrence_evidence = [
            {'method': e.get('method'), 'element_id': e.get('element_id'),
             'claim_no': e.get('claim_no'), 'char_offset': e.get('char_offset'),
             'matched': e.get('matched'), 'match_basis': 'canonical_smiles_or_inchikey',
             'status': 'verified'}
            for e in countable
        ] + [
            {'method': e.get('method'), 'unit_ref': e.get('unit_ref'),
             'match_basis': 'explicit_location_annotation', 'status': 'presence_only'}
            for e in presence_only
        ]

        # Summarize status and retain reasons for unknown values.
        if in_claim == TRUE or in_figure == TRUE:
            status = 'verified_' + '_'.join(
                [x for x in (('claim' if in_claim == TRUE else None),
                             ('figure' if in_figure == TRUE else None)) if x])
        elif in_claim == FALSE or in_figure == FALSE:
            status = 'annotated_absent'
        elif not target_canonical:
            status = 'unknown_molecule_smiles_unparseable'
        elif evidence['unverified']:
            methods = sorted({e['method'] for e in evidence['unverified']})
            status = 'unknown_unverified_figure_evidence_' + '_'.join(methods)
        elif text_scanned:
            status = 'unknown_text_scanned_no_verified_match'
        elif not nonempty(claims_text):
            status = 'unknown_no_claims_text'
        else:
            status = 'unknown_no_location_annotation'

        if not has_claim_annotation:
            evidence['notes'].append('claim_element.element_smiles is empty for this patent; claim-structure comparison unavailable')
        if not has_figure_annotation:
            evidence['notes'].append('figure.ocsr_smiles is empty for this patent; figure-structure comparison unavailable')

        records.append({
            'patent_id': pid,
            'mol_id': mid,
            'smiles': smiles,
            'canonical_smiles': target_canonical,
            'inchikey': target_inchikey,
            'source': mol.get('source') or None,
            'in_claim': in_claim,
            'in_figure': in_figure,
            'claim_occurrence_count': count,
            'claim_count_with_structure': claim_count_with_structure,
            'claim_occurrence_status': occurrence_status,
            'claim_unit_refs': distinct_claims,
            'occurrence_evidence': occurrence_evidence,
            'figure_unit_refs': matched_fig_ids,
            # DB duplicate count: source patent_molecule.csv rows for the same (patent_id, mol_id).
            # This counts record duplication, not textual claim occurrences.
            'db_occurrence_count': pair_row_counts[(pid, mid)],
            # Whether figure rows exist for the patent differs from molecule-level in_figure.
            # Missing figure.csv rows cannot distinguish absent drawings from unparsed drawings: unknown.
            'figure_exist': TRUE if figures_by_patent.get(pid) else UNKNOWN,
            'figure_row_count': len(figures_by_patent.get(pid) or []),
            'location_status': status,
            'location_evidence': evidence,
            'source_version': src_version,
        })

    print('[5/5] Saving and aggregating coverage...')
    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, 'structure_location_metadata.jsonl')
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def tally(key):
        c = defaultdict(int)
        for rec in records:
            c[rec[key]] += 1
        return dict(c)

    coverage = {
        'source_version': src_version,
        'source_files': src_detail,
        'patent_molecule_rows_input': len(pm_rows),
        'duplicate_pair_rows_removed': duplicate_rows,
        'unique_patent_molecule_pairs': len(records),
        'unique_patents': len({r['patent_id'] for r in records}),
        'unique_molecules': len({r['mol_id'] for r in records}),
        'molecules_with_unparseable_smiles': sum(1 for r in records if not r['canonical_smiles']),
        'in_claim': tally('in_claim'),
        'in_figure': tally('in_figure'),
        'location_status': tally('location_status'),
        'figure_exist': tally('figure_exist'),
        'db_occurrence_count': tally('db_occurrence_count'),
        'claim_occurrence_status': tally('claim_occurrence_status'),
        'claim_occurrence_count_known': sum(1 for r in records if r['claim_occurrence_count'] is not None),
        'claim_occurrence_count_null': sum(1 for r in records if r['claim_occurrence_count'] is None),
        'claim_count_with_structure_known': sum(
            1 for r in records if r['claim_count_with_structure'] is not None),
        'records_with_unverified_figure_evidence': sum(
            1 for r in records if (r['location_evidence'] or {}).get('unverified')),
        'evidence_sources_scanned': {
            'ocsr_cache': ocsr_stats,
            'patcid_epo': patcid_stats,
            'claim_element_smiles': 'Full claim_element.csv scan',
            'figure_ocsr_smiles': 'Full figure.csv scan',
            'patent_molecule_location': 'Full patent_molecule.csv scan',
            'claims_text_extraction': 'Full patent_content.claims_text scan (InChIKey/SMILES)',
        },
        'evidence_source_availability': {
            'patents_with_claim_element_smiles_annotation': sum(1 for v in patent_has_claim_smiles_annotation.values() if v),
            'patents_with_figure_ocsr_annotation': sum(1 for v in patent_has_figure_ocsr_annotation.values() if v),
            'patent_molecule_rows_with_location': sum(1 for r in pm_rows if nonempty(r.get('location'))),
            'patent_molecule_rows_with_unit_ref': sum(1 for r in pm_rows if nonempty(r.get('unit_ref'))),
        },
    }
    out_json = os.path.join(args.out_dir, 'structure_location_coverage.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)

    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    print(f'\nSaved: {out_jsonl}\nSaved: {out_json}')


if __name__ == '__main__':
    main()
