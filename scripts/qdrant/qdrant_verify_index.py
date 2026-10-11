"""full-index 검증 - 실제 4,123 / 6,928 포인트 DB가 있어야 통과한다.

DB가 없거나 비어 있으면 실패한다(skip 하지 않는다). 실패가 하나라도 있으면 종료 코드 1.

    python3 scripts/qdrant/qdrant_verify_index.py
    python3 scripts/qdrant/qdrant_verify_index.py --json-out benchmark/results/qdrant_verify_report.json
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
    """필터에 걸린 모든 포인트의 patent_id 집합(랭킹과 무관하게 필터만 검증)."""
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
    """경계조건을 포함한 결정적(deterministic) 검증 표본을 만든다."""
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

    # (1) 후보 중에 pub_date == query.app_date 인 경계 케이스(<= 이므로 포함되어야 함)
    for q in with_gold:
        if pub_dates.get(by_id[q]['app_date']):
            add(q, 'boundary_equal_date(candidate.pub_date == query.app_date)')
            break
    # (2) app_date 가 비어 있는 쿼리(시간 필터 미적용 정책)
    for q in with_gold:
        if not (by_id[q].get('app_date') or '').strip():
            add(q, 'missing_app_date')
            break
    # (3) family_id 가 비어 있는 쿼리(패밀리 필터 미적용 정책)
    for q in with_gold:
        if not (by_id[q].get('family_id') or '').strip():
            add(q, 'missing_family_id')
            break
    # (4) 같은 family 특허가 실제로 여러 건 있는 쿼리
    fam = defaultdict(list)
    for r in patents:
        fam[r['family_id']].append(r['patent_id'])
    for q in with_gold:
        if len(fam.get(by_id[q]['family_id'], [])) > 1:
            add(q, 'has_same_family_siblings')
            break
    # (5) 가장 이른/늦은 출원일
    for q in (min(with_gold, key=lambda x: by_id[x]['app_date']),
              max(with_gold, key=lambda x: by_id[x]['app_date'])):
        add(q, 'extreme_app_date')
    # (6) 정렬 기준 균등 표본(재현 가능)
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

    print('\n=== 0. 실행 환경 ===')
    prov = runtime_provenance(settings, ' '.join(sys.argv))
    c.info('Qdrant 모드/위치', f"{prov['qdrant']['mode']} @ {prov['qdrant']['url_or_path']}")
    c.info('python / qdrant-client', f"{prov['python_version']} / {prov['packages']['qdrant-client']}")

    print('\n=== 1. 원본 CSV 무결성 ===')
    checksum_path = os.path.join(DERIVED_DIR, 'original_csv_checksums.json')
    if os.path.exists(checksum_path):
        with open(checksum_path, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        changed = [fname for fname, meta in baseline.items()
                   if not os.path.exists(os.path.join(DB_EXPORT_DIR, fname))
                   or file_digest(os.path.join(DB_EXPORT_DIR, fname)) != meta['sha256']]
        c.check('원본 db_export CSV가 변경되지 않음', not changed, changed or f'{len(baseline)}개 확인')
    else:
        c.check('원본 CSV 체크섬 기준 파일 존재', False, f'{checksum_path} 없음')

    print('\n=== 2. 특허 컬렉션 ===')
    try:
        pc = client.count(settings.patents_collection).count
    except Exception as exc:
        c.check(f'특허 컬렉션({settings.patents_collection}) 존재', False, f'{type(exc).__name__}: {exc}')
        pc = -1
    c.check(f'특허 포인트 수 == {EXPECTED_PATENTS}', pc == EXPECTED_PATENTS, f'실제 {pc}')

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
        c.check(f'dense 벡터 존재/차원 일치 ({spec["dense"]})',
                spec['dense'] in vectors_cfg and vectors_cfg[spec['dense']].size == expected_dim,
                f'collection={vectors_cfg.get(spec["dense"]).size if spec["dense"] in vectors_cfg else None}, '
                f'encoder={expected_dim}')
        c.check(f'sparse 벡터 존재 ({spec["sparse"]})', spec['sparse'] in sparse_cfg)

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

    c.check('모든 특허 포인트에 patent_id 존재', missing_pid == 0, f'누락 {missing_pid}')
    c.check('모든 dense 벡터 차원이 스키마와 일치', dim_mismatch == 0, f'불일치 {dim_mismatch}')
    c.check('split.csv role/split/time_cutoff 결합 정확', role_mismatch == 0, f'불일치 {role_mismatch}')
    c.check('포인트 ID가 UUID5(patent_id) 정책과 일치', id_mismatch == 0, f'불일치 {id_mismatch}')
    c.check('특허 ID 집합이 patent_content.csv와 동일',
            seen_ids == {r['patent_id'] for r in patents},
            f'컬렉션 {len(seen_ids)} vs CSV {len(patents)}')

    print('\n=== 3. 임베딩 인코더 일치 ===')
    for field in TEXT_FIELDS:
        stored = sig_by_field.get(field, set())
        current = encoders[field]['dense'].signature()
        c.check(f'적재 시 인코더 == 검색 시 인코더 ({field})',
                len(stored) == 1 and current in stored, f'stored={stored}, current={current}')
    c.info('dense 인코더', encoders['title_claims']['dense'].describe())
    c.check('dense 백엔드가 실제 사전학습 semantic 모델',
            encoders['title_claims']['dense'].name != 'hashing',
            encoders['title_claims']['dense'].name)

    print('\n=== 4. 멱등성 ===')
    sample = list(client.scroll(settings.patents_collection, limit=5,
                                with_payload=True, with_vectors=True)[0])
    before = client.count(settings.patents_collection).count
    client.upsert(settings.patents_collection, points=[
        models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in sample], wait=True)
    after = client.count(settings.patents_collection).count
    c.check('동일 포인트 재적재 후 포인트 수 불변', before == after, f'{before} -> {after}')

    print('\n=== 5. 검색 필터 정확성 ===')
    retriever = PatentRetriever(client, settings,
                                {f: encoders[f]['dense'] for f in TEXT_FIELDS},
                                {f: encoders[f]['sparse'] for f in TEXT_FIELDS})
    try:
        retriever.search_dense('x', None)
        c.check('필터 없는 검색은 차단됨', False, '예외가 발생하지 않음')
    except ValueError:
        c.check('필터 없는 검색은 차단됨', True, 'prior_art_filter 필수')

    citations = defaultdict(set)
    for r in load_rows('examiner_citation'):
        citations[r['query_patent_id']].add(r['cited_patent_id'])
    by_id = {r['patent_id']: r for r in patents}
    all_ids = [r['patent_id'] for r in patents]
    pub_of = {r['patent_id']: r['pub_date'] for r in patents}
    app_of = {r['patent_id']: r['app_date'] for r in patents}
    fam_of = {r['patent_id']: r['family_id'] for r in patents}

    sample_queries, reasons = pick_boundary_queries(patents, citations)
    c.info('필터 검증 표본(결정적)', {q: reasons[q] for q in sample_queries})

    # (a) 필터 자체를 전수 대조: 필터로 걸러진 집합 == CSV로 계산한 후보 풀
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
    c.check('필터 결과 집합이 CSV 기준 후보 풀과 정확히 일치', not set_mismatch,
            set_mismatch or f'{len(sample_queries)}개 쿼리 전수 대조 통과')
    if equal_date_included:
        c.check('경계조건: candidate.pub_date == query.app_date 는 포함됨',
                equal_date_included['all_present_in_qdrant_filter']
                and equal_date_included['equal_date_candidates'] > 0, equal_date_included)
    else:
        c.info('경계조건(동일 날짜)', '해당 케이스가 데이터에 없어 검증 생략')

    # (b) 실제 검색 3종 결과에 필터 위반이 없는지
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
        c.check(f'필터 위반 0건 ({mode}, {field})',
                v['future'] == 0 and v['family'] == 0 and v['self'] == 0, v)

    print('\n  --- 필터 위반 요약 ---')
    for k, v in violations_summary.items():
        print(f'    {k}: future={v["future"]} family={v["family"]} self={v["self"]} (검사한 결과 {v["hits"]}건)')

    print('\n=== 6. 구조 컬렉션 ===')
    try:
        sc = client.count(settings.structures_collection).count
    except Exception as exc:
        c.check('구조 컬렉션 존재', False, f'{type(exc).__name__}: {exc}')
        sc = -1
    c.check(f'구조 포인트 수 == {EXPECTED_STRUCTURES}', sc == EXPECTED_STRUCTURES, f'실제 {sc}')

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
            contradictions['in_claim이 3-값이 아님'] += 1
        if in_figure not in (TRUE, FALSE, UNKNOWN):
            contradictions['in_figure가 3-값이 아님'] += 1
        if in_claim == TRUE and not ev.get('claim'):
            contradictions['in_claim=true 인데 근거 없음'] += 1
        if in_figure == TRUE and not ev.get('figure'):
            contradictions['in_figure=true 인데 근거 없음'] += 1
        if in_figure == TRUE and not (pl.get('figure_unit_refs') or
                                      any(e.get('fig_id') or e.get('unit_ref') for e in ev.get('figure') or [])):
            contradictions['in_figure=true 인데 figure 식별자 없음'] += 1
        if in_claim == FALSE and occ is not None and occ > 0:
            contradictions['in_claim=false 인데 occurrence>0'] += 1
        if in_claim == UNKNOWN and occ is not None:
            contradictions['in_claim=unknown 인데 occurrence가 null이 아님'] += 1
        if occ is not None and ccws is not None and ccws > occ:
            contradictions['claim_count_with_structure > claim_occurrence_count'] += 1

    c.check('구조 포인트 (patent_id, mol_id) 중복 없음', dup == 0, f'중복 {dup}')
    c.check('구조 포인트 ID가 UUID5 정책과 일치', struct_id_mismatch == 0, f'불일치 {struct_id_mismatch}')
    c.check('in_claim/in_figure/occurrence 상태-근거 무모순', not contradictions,
            dict(contradictions) or '모순 0건')
    pm_unique = {(r['patent_id'], r['mol_id']) for r in load_rows('patent_molecule')}
    c.check('구조 포인트 수 == 원본 patent_molecule 고유쌍 수', sc == len(pm_unique), f'{sc} vs {len(pm_unique)}')
    c.info('in_claim 분포', dict(tri['in_claim']))
    c.info('in_figure 분포', dict(tri['in_figure']))
    # --- 특허 단위 보조 필드(db_occurrence_count / figure_exist) 실검증 ---
    pm_rows_all = load_rows('patent_molecule')
    c.check('db_occurrence_count 합계 == patent_molecule 원본 행 수',
            db_occ_missing == 0 and db_occ_sum == len(pm_rows_all),
            f'합계 {db_occ_sum} vs 원본 {len(pm_rows_all)} (null {db_occ_missing})')
    fig_csv_pats = {r['patent_id'] for r in load_rows('figure')}
    stray = fig_exist_true_pats - fig_csv_pats
    c.check('figure_exist=true 특허가 모두 figure.csv에 존재',
            not stray, f'figure.csv에 없는 특허 {len(stray)}건')
    c.info('figure_exist 분포', dict(tri['figure_exist']))
    c.info('figure_exist=true 특허 수', len(fig_exist_true_pats))
    c.info('claim_occurrence_count 확인 가능/불가', f'{occ_known} / {sc - occ_known}')
    c.info('claim_count_with_structure 확인 가능', ccws_known)
    c.info('미검증 figure 근거 보유 포인트', unverified_ev)

    report = {'provenance': prov,
              'counts': {'patents': pc, 'structures': sc},
              'filter_violations': violations_summary,
              'passed': len(c.passed), 'failed': len(c.failed), 'results': c.results}
    out = os.path.normpath(args.json_out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\n=== 결과: 통과 {len(c.passed)} / 실패 {len(c.failed)} ===')
    print(f'저장됨: {out}')
    return 1 if c.failed else 0


if __name__ == '__main__':
    sys.exit(main())
