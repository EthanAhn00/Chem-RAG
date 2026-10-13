"""Full-index verification requires actual DB points: 4,123 patents / 6,928 structures. Missing/empty DB is failure, not a skip. Any failure exits 1.
Usage: python3 scripts/qdrant/qdrant_verify_index.py [--json-out benchmark/results/qdrant_verify_report.json].
"""
import os
import sys
import csv
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
csv.field_size_limit(10 ** 9)

from qdrant_client import models

from config.settings import DATA_DIR
from qdrant_engine.config import load_settings
from qdrant_engine.client import get_client, is_local_mode
from qdrant_engine.schemas import (
    patent_point_id, structure_point_id, TEXT_FIELDS, build_field_text, TRUE, FALSE, UNKNOWN)
from qdrant_engine.filters import build_prior_art_filter
from qdrant_engine.provenance import file_digest, runtime_provenance
from qdrant_engine.patent_retriever import PatentRetriever
from embeddings.registry import build_dense_encoder
from embeddings.sparse_encoder import BM25SparseEncoder
from scripts.evaluation.evaluate_full_corpus_retrieval import build_candidate_pool

DB_EXPORT_DIR = os.path.join(DATA_DIR, 'db_export')
DERIVED_DIR = os.path.join(DB_EXPORT_DIR, 'derived')
EXPECTED_PATENTS = 4123
EXPECTED_STRUCTURES = 6928


class Checker:
    def __init__(self):
        self.results = []

    def check(self, name, condition, detail=''):
        ok = bool(condition)
        self.results.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
        print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f' - {detail}' if detail else ''))
        return ok

    def info(self, name, detail):
        self.results.append({'check': name, 'status': 'INFO', 'detail': str(detail)})
        print(f'  [INFO] {name} - {detail}')

    @property
    def failed(self):
        return [r for r in self.results if r['status'] == 'FAIL']

    @property
    def passed(self):
        return [r for r in self.results if r['status'] == 'PASS']


def load_rows(name):
    with open(os.path.join(DB_EXPORT_DIR, f'{name}.csv'), 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def scroll_all(client, collection, with_vectors=False, batch=1000):
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=batch, offset=offset,
                                       with_payload=True, with_vectors=with_vectors)
        for p in points:
            yield p
        if offset is None:
            break


def scroll_ids_with_filter(client, collection, qfilter, batch=1000):
    """All patent_ids passing the filter; verify filtering independently of ranking."""
    out, offset = set(), None
    while True:
        points, offset = client.scroll(collection, scroll_filter=qfilter, limit=batch,
                                       offset=offset, with_payload=['patent_id'], with_vectors=False)
        for p in points:
            out.add((p.payload or {}).get('patent_id'))
        if offset is None:
            break
    return out


def pick_boundary_queries(patents, citations):
    """Build deterministic verification samples including boundary cases."""
    by_id = {r['patent_id']: r for r in patents}
    with_gold = sorted(q for q in citations if q in by_id)

    pub_dates = defaultdict(list)
    for r in patents:
        pub_dates[r['pub_date']].append(r['patent_id'])

    picked, reasons = [], {}

    def add(pid, why):
        if pid and pid in by_id and pid not in picked:
            picked.append(pid)
            reasons[pid] = why

    # (1) Equality boundary: candidate.pub_date == query.app_date must pass <=.
    for q in with_gold:
        if pub_dates.get(by_id[q]['app_date']):
            add(q, 'boundary_equal_date(candidate.pub_date == query.app_date)')
            break
    # (2) Queries with missing app_date skip time filtering.
    for q in with_gold:
        if not (by_id[q].get('app_date') or '').strip():
            add(q, 'missing_app_date')
            break
    # (3) Queries with missing family_id skip family filtering.
    for q in with_gold:
        if not (by_id[q].get('family_id') or '').strip():
            add(q, 'missing_family_id')
            break
    # (4) Queries with multiple actual same-family patents.
    fam = defaultdict(list)
    for r in patents:
        fam[r['family_id']].append(r['patent_id'])
    for q in with_gold:
        if len(fam.get(by_id[q]['family_id'], [])) > 1:
            add(q, 'has_same_family_siblings')
            break
    # (5) Earliest/latest filing dates.
    for q in (min(with_gold, key=lambda x: by_id[x]['app_date']),
              max(with_gold, key=lambda x: by_id[x]['app_date'])):
        add(q, 'extreme_app_date')
    # (6) Uniform sorted samples for reproducibility.
    step = max(len(with_gold) // 10, 1)
    for q in with_gold[::step][:10]:
        add(q, 'deterministic_sample')
    return picked, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-out', default=os.path.join(
        DATA_DIR, '..', 'benchmark', 'results', 'qdrant_verify_report.json'))
    args = ap.parse_args()

    settings = load_settings()
    client = get_client(settings)
    c = Checker()
    violations_summary = {}

    print('\n=== 0. Execution environment ===')
    prov = runtime_provenance(settings, ' '.join(sys.argv))
    c.info('Qdrant mode/location', f"{prov['qdrant']['mode']} @ {prov['qdrant']['url_or_path']}")
    c.info('python / qdrant-client', f"{prov['python_version']} / {prov['packages']['qdrant-client']}")

    print('\n=== 1. Source CSV integrity ===')
    checksum_path = os.path.join(DERIVED_DIR, 'original_csv_checksums.json')
    if os.path.exists(checksum_path):
        with open(checksum_path, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        changed = [fname for fname, meta in baseline.items()
                   if not os.path.exists(os.path.join(DB_EXPORT_DIR, fname))
                   or file_digest(os.path.join(DB_EXPORT_DIR, fname)) != meta['sha256']]
        c.check('Original db_export CSVs unchanged', not changed, changed or f'{len(baseline)} checked')
    else:
        c.check('Source CSV checksum reference exists', False, f'{checksum_path} missing')

    print('\n=== 2. Patent collection ===')
    try:
        pc = client.count(settings.patents_collection).count
    except Exception as exc:
        c.check(f'Patent collection ({settings.patents_collection}) exists', False, f'{type(exc).__name__}: {exc}')
        pc = -1
    c.check(f'Patent point count == {EXPECTED_PATENTS}', pc == EXPECTED_PATENTS, f'actual {pc}')

    info = client.get_collection(settings.patents_collection)
    vectors_cfg = info.config.params.vectors or {}
    sparse_cfg = info.config.params.sparse_vectors or {}
    encoders = {}
    for field in TEXT_FIELDS:
        state = os.path.join(DERIVED_DIR, f'bm25_state_{field}.json')
        encoders[field] = {
            'dense': build_dense_encoder(settings, idf_state_path=state),
            'sparse': BM25SparseEncoder.load(state) if os.path.exists(state) else None,
        }
    expected_dim = encoders['title_claims']['dense'].dim

    for field, spec in TEXT_FIELDS.items():
        c.check(f'Dense vectors exist with matching dimensions ({spec["dense"]})',
                spec['dense'] in vectors_cfg and vectors_cfg[spec['dense']].size == expected_dim,
                f'collection={vectors_cfg.get(spec["dense"]).size if spec["dense"] in vectors_cfg else None}, '
                f'encoder={expected_dim}')
        c.check(f'Sparse vectors exist ({spec["sparse"]})', spec['sparse'] in sparse_cfg)

    split_rows = {r['id']: r for r in load_rows('split')}
    patents = load_rows('patent_content')
    missing_pid = dim_mismatch = role_mismatch = id_mismatch = 0
    sig_by_field = defaultdict(set)
    seen_ids = set()
    for p in scroll_all(client, settings.patents_collection, with_vectors=True):
        pl = p.payload or {}
        pid = pl.get('patent_id')
        if not pid:
            missing_pid += 1
            continue
        seen_ids.add(pid)
        vecs = p.vector or {}
        for field, spec in TEXT_FIELDS.items():
            v = vecs.get(spec['dense'])
            if v is not None and len(v) != expected_dim:
                dim_mismatch += 1
        sp = split_rows.get(pid)
        if sp and (pl.get('role') != sp['role'] or pl.get('split') != sp['split']
                   or str(pl.get('time_cutoff')) != str(sp['time_cutoff'])):
            role_mismatch += 1
        if str(p.id) != patent_point_id(pid):
            id_mismatch += 1
        for f, s in (pl.get('embedding_signature_by_field') or {}).items():
            sig_by_field[f].add(s)

    c.check('All patent points contain patent_id', missing_pid == 0, f'missing {missing_pid}')
    c.check('All dense dimensions match the schema', dim_mismatch == 0, f'mismatches {dim_mismatch}')
    c.check('split.csv role/split/time_cutoff joined correctly', role_mismatch == 0, f'mismatches {role_mismatch}')
    c.check('Point IDs follow UUID5(patent_id)', id_mismatch == 0, f'mismatches {id_mismatch}')
    c.check('Patent-ID set matches patent_content.csv',
            seen_ids == {r['patent_id'] for r in patents},
            f'Collection {len(seen_ids)} vs CSV {len(patents)}')

    print('\n=== 3. Embedding encoder consistency ===')
    for field in TEXT_FIELDS:
        stored = sig_by_field.get(field, set())
        current = encoders[field]['dense'].signature()
        c.check(f'Indexing encoder == search encoder ({field})',
                len(stored) == 1 and current in stored, f'stored={stored}, current={current}')
    c.info('Dense encoder', encoders['title_claims']['dense'].describe())
    c.check('Dense backend is a pretrained semantic model',
            encoders['title_claims']['dense'].name != 'hashing',
            encoders['title_claims']['dense'].name)

    print('\n=== 4. Idempotence ===')
    sample = list(client.scroll(settings.patents_collection, limit=5,
                                with_payload=True, with_vectors=True)[0])
    before = client.count(settings.patents_collection).count
    client.upsert(settings.patents_collection, points=[
        models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in sample], wait=True)
    after = client.count(settings.patents_collection).count
    c.check('Re-upserting identical points preserves point count', before == after, f'{before} -> {after}')

    print('\n=== 5. Retrieval filter correctness ===')
    retriever = PatentRetriever(client, settings,
                                {f: encoders[f]['dense'] for f in TEXT_FIELDS},
                                {f: encoders[f]['sparse'] for f in TEXT_FIELDS})
    try:
        retriever.search_dense('x', None)
        c.check('Unfiltered search is blocked', False, 'No exception was raised')
    except ValueError:
        c.check('Unfiltered search is blocked', True, 'prior_art_filter is required')

    citations = defaultdict(set)
    for r in load_rows('examiner_citation'):
        citations[r['query_patent_id']].add(r['cited_patent_id'])
    by_id = {r['patent_id']: r for r in patents}
    all_ids = [r['patent_id'] for r in patents]
    pub_of = {r['patent_id']: r['pub_date'] for r in patents}
    app_of = {r['patent_id']: r['app_date'] for r in patents}
    fam_of = {r['patent_id']: r['family_id'] for r in patents}

    sample_queries, reasons = pick_boundary_queries(patents, citations)
    c.info('Deterministic filter verification sample', {q: reasons[q] for q in sample_queries})

    # (a) Compare complete filtered ID sets against CSV-derived candidate pools.
    set_mismatch = []
    equal_date_included = None
    for qid in sample_queries:
        q = by_id[qid]
        pf = build_prior_art_filter(qid, q['app_date'], q['family_id'])
        got = scroll_ids_with_filter(client, settings.patents_collection, pf.qdrant_filter)
        expected, _, _ = build_candidate_pool(qid, all_ids, pub_of, app_of, fam_of)
        expected = set(expected)
        if got != expected:
            set_mismatch.append({'query': qid, 'only_in_qdrant': sorted(got - expected)[:5],
                                 'only_in_expected': sorted(expected - got)[:5]})
        if reasons[qid].startswith('boundary_equal_date'):
            same = [p for p in expected if pub_of[p] == q['app_date']]
            equal_date_included = {'query': qid, 'equal_date_candidates': len(same),
                                   'all_present_in_qdrant_filter': all(p in got for p in same)}
    c.check('Filtered result set exactly matches the CSV-derived candidate pool', not set_mismatch,
            set_mismatch or f'{len(sample_queries)} query comparisons passed')
    if equal_date_included:
        c.check('Boundary: candidate.pub_date == query.app_date is included',
                equal_date_included['all_present_in_qdrant_filter']
                and equal_date_included['equal_date_candidates'] > 0, equal_date_included)
    else:
        c.info('Boundary condition (equal date)', 'No corresponding data case; verification skipped')

    # (b) Check all 3 actual retrieval modes for filter violations.
    for mode, field in (('bm25', 'claims'), ('bm25', 'title_claims'),
                        ('dense', 'claims'), ('dense', 'title_claims'),
                        ('hybrid', 'title_claims')):
        v = {'future': 0, 'family': 0, 'self': 0, 'hits': 0}
        for qid in sample_queries:
            q = by_id[qid]
            pf = build_prior_art_filter(qid, q['app_date'], q['family_id'])
            text, _ = build_field_text(q.get('title'), q.get('claims_text'), field)
            hits, _ = retriever.search(mode, text or 'chemical composition', pf, limit=300, field=field)
            v['hits'] += len(hits)
            for h in hits:
                pl = h['payload']
                if pl.get('patent_id') == qid:
                    v['self'] += 1
                if q['family_id'] and pl.get('family_id') == str(q['family_id']):
                    v['family'] += 1
                if q['app_date'] and pl.get('pub_date') and pl['pub_date'] > q['app_date']:
                    v['future'] += 1
        violations_summary[f'{mode}/{field}'] = v
        c.check(f'Zero filter violations ({mode}, {field})',
                v['future'] == 0 and v['family'] == 0 and v['self'] == 0, v)

    print('\n  --- Filter violation summary ---')
    for k, v in violations_summary.items():
        print(f'    {k}: future={v["future"]} family={v["family"]} self={v["self"]} (results checked {v["hits"]})')

    print('\n=== 6. Structure collection ===')
    try:
        sc = client.count(settings.structures_collection).count
    except Exception as exc:
        c.check('Structure collection exists', False, f'{type(exc).__name__}: {exc}')
        sc = -1
    c.check(f'Structure point count == {EXPECTED_STRUCTURES}', sc == EXPECTED_STRUCTURES, f'actual {sc}')

    pairs, dup, contradictions = set(), 0, defaultdict(int)
    tri = {'in_claim': defaultdict(int), 'in_figure': defaultdict(int),
           'figure_exist': defaultdict(int)}
    db_occ_sum = 0
    db_occ_missing = 0
    fig_exist_true_pats = set()
    occ_known = ccws_known = struct_id_mismatch = 0
    unverified_ev = 0
    for p in scroll_all(client, settings.structures_collection):
        pl = p.payload or {}
        key = (pl.get('patent_id'), pl.get('mol_id'))
        if key in pairs:
            dup += 1
        pairs.add(key)
        if str(p.id) != structure_point_id(key[0], key[1]):
            struct_id_mismatch += 1
        in_claim, in_figure = pl.get('in_claim'), pl.get('in_figure')
        occ, ccws = pl.get('claim_occurrence_count'), pl.get('claim_count_with_structure')
        ev = pl.get('location_evidence') or {}
        tri['in_claim'][in_claim] += 1
        tri['in_figure'][in_figure] += 1
        fe = pl.get('figure_exist')
        tri['figure_exist'][fe] += 1
        if fe == TRUE:
            fig_exist_true_pats.add(key[0])
        doc = pl.get('db_occurrence_count')
        if doc is None:
            db_occ_missing += 1
        else:
            db_occ_sum += doc
        if occ is not None:
            occ_known += 1
        if ccws is not None:
            ccws_known += 1
        if ev.get('unverified'):
            unverified_ev += 1

        if in_claim not in (TRUE, FALSE, UNKNOWN):
            contradictions['in_claim is not three-valued'] += 1
        if in_figure not in (TRUE, FALSE, UNKNOWN):
            contradictions['in_figure is not three-valued'] += 1
        if in_claim == TRUE and not ev.get('claim'):
            contradictions['in_claim=true without evidence'] += 1
        if in_figure == TRUE and not ev.get('figure'):
            contradictions['in_figure=true without evidence'] += 1
        if in_figure == TRUE and not (pl.get('figure_unit_refs') or
                                      any(e.get('fig_id') or e.get('unit_ref') for e in ev.get('figure') or [])):
            contradictions['in_figure=true without figure identifier'] += 1
        if in_claim == FALSE and occ is not None and occ > 0:
            contradictions['in_claim=false with occurrence>0'] += 1
        if in_claim == UNKNOWN and occ is not None:
            contradictions['in_claim=unknown with non-null occurrence'] += 1
        if occ is not None and ccws is not None and ccws > occ:
            contradictions['claim_count_with_structure > claim_occurrence_count'] += 1

    c.check('No duplicate structure (patent_id, mol_id) points', dup == 0, f'duplicates {dup}')
    c.check('Structure point IDs follow UUID5 policy', struct_id_mismatch == 0, f'mismatches {struct_id_mismatch}')
    c.check('in_claim/in_figure/occurrence states and evidence are consistent', not contradictions,
            dict(contradictions) or 'Zero contradictions')
    pm_unique = {(r['patent_id'], r['mol_id']) for r in load_rows('patent_molecule')}
    c.check('Structure point count == unique source patent_molecule pairs', sc == len(pm_unique), f'{sc} vs {len(pm_unique)}')
    c.info('in_claim distribution', dict(tri['in_claim']))
    c.info('in_figure distribution', dict(tri['in_figure']))
    # --- Verify patent-level auxiliary fields db_occurrence_count / figure_exist ---
    pm_rows_all = load_rows('patent_molecule')
    c.check('Sum of db_occurrence_count == source patent_molecule rows',
            db_occ_missing == 0 and db_occ_sum == len(pm_rows_all),
            f'sum {db_occ_sum} vs source {len(pm_rows_all)} (null {db_occ_missing})')
    fig_csv_pats = {r['patent_id'] for r in load_rows('figure')}
    stray = fig_exist_true_pats - fig_csv_pats
    c.check('All figure_exist=true patents occur in figure.csv',
            not stray, f'Patents absent from figure.csv: {len(stray)} records')
    c.info('figure_exist distribution', dict(tri['figure_exist']))
    c.info('Patents with figure_exist=true', len(fig_exist_true_pats))
    c.info('claim_occurrence_count known/unknown', f'{occ_known} / {sc - occ_known}')
    c.info('claim_count_with_structure known', ccws_known)
    c.info('Points with unverified figure evidence', unverified_ev)

    report = {'provenance': prov,
              'counts': {'patents': pc, 'structures': sc},
              'filter_violations': violations_summary,
              'passed': len(c.passed), 'failed': len(c.failed), 'results': c.results}
    out = os.path.normpath(args.json_out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\n=== Results: passed {len(c.passed)} / failed {len(c.failed)} ===')
    print(f'Saved: {out}')
    return 1 if c.failed else 0


if __name__ == '__main__':
    sys.exit(main())
