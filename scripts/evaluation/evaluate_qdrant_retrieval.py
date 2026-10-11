"""검색 방식 비교 평가 - 알고리즘 효과와 입력 필드 효과를 분리한다.

시스템 (모두 같은 쿼리 집합 / 같은 gold / 같은 후보 풀 / 같은 필터 / 같은 limit):

  A_jaccard_claims          기존 rag_engine 방식(Jaccard+overlap) - claims only
  B_bm25_claims             Qdrant BM25            - claims only
  C_bm25_title_claims       Qdrant BM25            - title + claims
  D_dense_claims            Qdrant dense(semantic) - claims only
  E_dense_title_claims      Qdrant dense(semantic) - title + claims
  F_hybrid_title_claims     BM25 + dense 하이브리드(RRF) - title + claims
  G_hybrid_rerank_title_claims  F + RDKit exact Tanimoto 재정렬 - title + claims

효과 분해
  A->B  Jaccard를 BM25로 바꾼 효과 (입력 필드 고정)
  B->C  title 추가 효과 (알고리즘 고정) / D->E 동일
  B->D, C->E  실제 semantic dense 임베딩 효과 (입력 필드 고정)
  C,E->F  hybrid fusion 효과 / F->G 구조 reranking 효과

주의: Qdrant 자체는 저장·실행 계층이다. 아래 수치 차이는 Qdrant 도입이 아니라
스코어링 알고리즘 / 입력 필드 / 임베딩 모델 / 융합 설정의 차이에서 나온다.

AP 정의: sum(정답 위치별 precision) / (전체 정답 수). 회수되지 않은 정답은 0 기여.

지표 이름(중요): 모든 시스템이 --search-limit(기본 1000) 밖은 반환하지 않으므로
이 스크립트가 내는 지표는 절단 없는 일반 MAP이 아니라 MAP@{cutoff}(기본 MAP@1000)이다.
요약 JSON의 mean_average_precision_at_{cutoff} 키가 정식 이름이고,
mean_average_precision_map 키는 이전 이름과의 호환을 위한 deprecated alias로,
항상 mean_average_precision_at_{cutoff}와 같은 값이다.
(scripts/evaluation/evaluate_full_corpus_retrieval.py를 단독 실행할 때는 후보 풀 전체를 랭킹하므로
 이 절단이 적용되지 않는다 - 그 스크립트의 mean_average_precision_map은 그대로 절단 없는 MAP이다.)

사용법:
    python3 scripts/evaluation/evaluate_qdrant_retrieval.py --systems all --workers 4
    python3 scripts/evaluation/evaluate_qdrant_retrieval.py --selftest
"""
import os
import sys
import json
import time
import math
import glob
import shutil
import argparse
import tempfile
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import DATA_DIR
from qdrant_engine.config import load_settings
from qdrant_engine.client import get_client, is_local_mode
from qdrant_engine.filters import build_prior_art_filter
from qdrant_engine.patent_retriever import PatentRetriever
from qdrant_engine.structure_repository import StructureRepository, max_tanimoto
from qdrant_engine.hybrid_fusion import rerank_with_structure
from qdrant_engine.schemas import TEXT_FIELDS, build_field_text
from qdrant_engine.provenance import runtime_provenance, source_version
from embeddings.registry import build_dense_encoder
from embeddings.sparse_encoder import BM25SparseEncoder

from scripts.evaluation.evaluate_full_corpus_retrieval import (
    build_indexes, precompute_token_counters, precompute_fingerprints,
    text_similarity_cached, max_tanimoto as baseline_max_tanimoto, combine_score,
    build_candidate_pool, evaluate_query as baseline_evaluate_query, load_csv,
)
from rag_engine.chemical_tokenizer import ChemicalTokenizer

RESULTS_DIR = os.path.normpath(os.path.join(DATA_DIR, '..', 'benchmark', 'results'))
DERIVED_DIR = os.path.join(DATA_DIR, 'db_export', 'derived')

SYSTEMS = {
    'A_jaccard_claims': ('jaccard', 'claims'),
    'B_bm25_claims': ('bm25', 'claims'),
    'C_bm25_title_claims': ('bm25', 'title_claims'),
    'D_dense_claims': ('dense', 'claims'),
    'E_dense_title_claims': ('dense', 'title_claims'),
    'F_hybrid_title_claims': ('hybrid', 'title_claims'),
    'G_hybrid_rerank_title_claims': ('hybrid_rerank', 'title_claims'),
}
ALL_SYSTEMS = tuple(SYSTEMS)


def bm25_state_path(field):
    return os.path.join(DERIVED_DIR, f'bm25_state_{field}.json')


def dcg_at_k(relevances, k):
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(relevances[:k]))


def evaluate_ranking(ranked_ids, gold_set):
    """AP = sum(정답 위치별 precision) / 전체 정답 수.

    검색 결과에 없는 정답은 자동으로 0 기여가 된다(BM25가 zero-overlap 문서를
    반환하지 않아 정답이 빠지는 경우도 동일하게 벌점 처리).
    """
    num_relevant = len(gold_set)
    if num_relevant == 0:
        return None
    relevant_ranks = [i for i, pid in enumerate(ranked_ids) if pid in gold_set]

    precisions, hit = [], 0
    for rank in relevant_ranks:
        hit += 1
        precisions.append(hit / (rank + 1))
    ap = sum(precisions) / num_relevant

    relevances = [1 if pid in gold_set else 0 for pid in ranked_ids[:10]]
    idcg = dcg_at_k([1] * min(num_relevant, 10), 10)
    return {
        'num_gold': num_relevant,
        'num_gold_retrieved': len(relevant_ranks),
        'retrieved': len(ranked_ids),
        'ap': round(ap, 6),
        'recall_at_10': round(sum(1 for r in relevant_ranks if r < 10) / num_relevant, 6),
        'recall_at_100': round(sum(1 for r in relevant_ranks if r < 100) / num_relevant, 6),
        'recall_at_1000': round(sum(1 for r in relevant_ranks if r < 1000) / num_relevant, 6),
        'mrr_at_10': round(1.0 / (relevant_ranks[0] + 1)
                           if relevant_ranks and relevant_ranks[0] < 10 else 0.0, 6),
        'ndcg_at_10_binary': round(dcg_at_k(relevances, 10) / idcg, 6) if idcg > 0 else 0.0,
        'best_rank': (relevant_ranks[0] + 1) if relevant_ranks else None,
    }


def metric_key_for_cutoff(cutoff):
    """MAP 요약 키 이름. cutoff=1000(기본)이면 'mean_average_precision_at_1000'."""
    return f'mean_average_precision_at_{int(cutoff)}'


def build_system_summary(per_query, algorithm, field, cutoff):
    """per-query 레코드 목록 -> 시스템 요약 dict. 실제 검색 루프와 --recompute-only 양쪽에서
    똑같이 이 함수를 쓴다(집계 로직 중복 방지 - 하나만 고치면 둘 다 반영됨)."""
    def agg(key, records=per_query):
        vals = [r[key] for r in records if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    lat = [r['latency_ms'] for r in per_query if r.get('latency_ms') is not None]
    map_value = agg('ap')
    mkey = metric_key_for_cutoff(cutoff)
    summary = {
        'algorithm': algorithm, 'text_field': field,
        'text_field_description': TEXT_FIELDS[field]['description'],
        'num_queries_evaluated': len(per_query),
        mkey: map_value,
        # deprecated alias - 항상 위 mkey 값과 동일. 즉시 제거하지 않고 하위 호환으로 유지.
        'mean_average_precision_map': map_value,
        'recall_at_10': agg('recall_at_10'),
        'recall_at_100': agg('recall_at_100'),
        'recall_at_1000': agg('recall_at_1000'),
        'mrr_at_10': agg('mrr_at_10'),
        'ndcg_at_10_binary': agg('ndcg_at_10_binary'),
        'gold_retrieved_ratio': round(
            sum(r['num_gold_retrieved'] for r in per_query) /
            max(sum(r['num_gold'] for r in per_query), 1), 4) if per_query else None,
        'latency_ms_p50': percentile(lat, 0.50),
        'latency_ms_p95': percentile(lat, 0.95),
        'avg_retrieved': agg('retrieved'),
    }
    if any('rerank' in r for r in per_query):
        rr = [r['rerank'] for r in per_query if 'rerank' in r]
        summary['rerank_avg_candidates_with_structure'] = round(
            sum(x['with_structure'] for x in rr) / len(rr), 3)
    return summary


def build_effects(summaries, mkey):
    def delta(a, b, key=mkey):
        if a in summaries and b in summaries and summaries[a].get(key) is not None:
            return {'from': a, 'to': b, 'values': [summaries[a][key], summaries[b][key]],
                    'delta': round(summaries[b][key] - summaries[a][key], 4)}
        return None

    return {
        'jaccard_to_bm25 (claims 고정)': delta('A_jaccard_claims', 'B_bm25_claims'),
        'title 추가 (BM25 고정)': delta('B_bm25_claims', 'C_bm25_title_claims'),
        'title 추가 (dense 고정)': delta('D_dense_claims', 'E_dense_title_claims'),
        'BM25 -> semantic dense (claims 고정)': delta('B_bm25_claims', 'D_dense_claims'),
        'BM25 -> semantic dense (title+claims 고정)': delta('C_bm25_title_claims', 'E_dense_title_claims'),
        'hybrid fusion vs BM25 (title+claims 고정)': delta('C_bm25_title_claims', 'F_hybrid_title_claims'),
        'hybrid fusion vs dense (title+claims 고정)': delta('E_dense_title_claims', 'F_hybrid_title_claims'),
        '구조 reranking': delta('F_hybrid_title_claims', 'G_hybrid_rerank_title_claims'),
    }


def build_consistency(summaries, out_prefix, mkey):
    """summary 값과 .partial.jsonl에서 재집계한 값을 대조. mkey와 deprecated alias가
    항상 같은 값인지도 함께 확인한다(이름만 바뀌고 값이 갈라지는 회귀를 잡기 위함)."""
    consistency = []
    for system in summaries:
        pq, _ = load_partial(os.path.join(RESULTS_DIR, f'{out_prefix}_{system}.partial.jsonl'))
        checks = {}
        for key in ('ap', 'recall_at_10', 'recall_at_100', 'recall_at_1000',
                    'mrr_at_10', 'ndcg_at_10_binary'):
            skey = mkey if key == 'ap' else key
            vals = [r[key] for r in pq if r.get(key) is not None]
            recomputed = round(sum(vals) / len(vals), 4) if vals else None
            checks[skey] = {'summary': summaries[system][skey], 'recomputed': recomputed,
                            'match': summaries[system][skey] == recomputed}
        alias_val = summaries[system].get('mean_average_precision_map')
        mkey_val = summaries[system].get(mkey)
        checks['mean_average_precision_map_alias'] = {
            'summary': alias_val, 'recomputed': mkey_val, 'match': alias_val == mkey_val}
        count_match = len(pq) == summaries[system]['num_queries_evaluated']
        consistency.append({'system': system, 'num_per_query_records': len(pq),
                            'count_match': count_match, 'metrics': checks,
                            'all_match': count_match and all(c['match'] for c in checks.values())})
    bad = [c['system'] for c in consistency if not c['all_match']]
    return consistency, bad


def resolve_recompute_cutoff(per_query, fallback_cutoff):
    """--recompute-only가 재계산에 쓸 실제 cutoff를 결정한다. per-query 레코드 자체의
    'cutoff' 필드(이번 패치 이후 새로 기록됨)를 최우선으로 쓰고, 레코드에 그 필드가 없는
    예전 partial 파일은 과거 summary의 config.search_limit(fallback_cutoff)을 근거로 쓴다.
    둘 다 없으면 절대 추정하지 않고 (None, 이유)를 반환해 호출자가 중단하게 한다.

    반환: (cutoff:int|None, problem:str|None)"""
    recorded = {r['cutoff'] for r in per_query if r.get('cutoff') is not None}
    if len(recorded) > 1:
        return None, f'per-query 레코드에 서로 다른 cutoff가 섞여 있음: {sorted(recorded)}'
    if len(recorded) == 1:
        return recorded.pop(), None
    if fallback_cutoff is not None:
        return fallback_cutoff, None
    return None, ('per-query 레코드에 cutoff 메타데이터가 없고(예전 실행분) 기존 summary도 없어 '
                  '어떤 cutoff로 계산됐는지 확인할 수 없음 - 추정하지 않고 중단')


def run_recompute_only(args):
    """기존 .partial.jsonl(전체 1,106쿼리 검색 결과)에서 summary/effect_decomposition/
    consistency만 다시 만든다. Qdrant 연결도, 검색도 하지 않는다 - MAP 지표 이름을
    바꾸는 것뿐이므로 전체 재평가가 필요 없기 때문. DB에 물어봐야 하는 필드
    (collection_counts, dense_encoder_by_field, encoder_signature_check,
    operational_latency_benchmark)는 이전 리포트 값을 그대로 이월하고
    '이번 실행에서 재검증하지 않음'으로 표시한다."""
    systems = ALL_SYSTEMS if args.systems == 'all' else tuple(s.strip() for s in args.systems.split(','))
    for s in systems:
        if s not in SYSTEMS:
            raise SystemExit(f'알 수 없는 시스템: {s} (가능: {ALL_SYSTEMS})')
    cutoff = args.search_limit
    mkey = metric_key_for_cutoff(cutoff)

    old_path = os.path.join(RESULTS_DIR, f'{args.out_prefix}_summary.json')
    old_report = None
    if os.path.exists(old_path):
        with open(old_path, 'r', encoding='utf-8') as f:
            old_report = json.load(f)
        backup_path = old_path + '.pre_map_at_cutoff_rename.json'
        if not os.path.exists(backup_path):
            shutil.copyfile(old_path, backup_path)
            print(f'  원본 백업: {backup_path}')

    fallback_cutoff = (old_report or {}).get('config', {}).get('search_limit')

    summaries = {}
    per_query_by_system = {}
    cutoff_problems = []
    for system in systems:
        algorithm, field = SYSTEMS[system]
        partial_path = os.path.join(RESULTS_DIR, f'{args.out_prefix}_{system}.partial.jsonl')
        pq, _ = load_partial(partial_path)
        if not pq:
            raise SystemExit(
                f'[{system}] 기존 per-query 결과가 없습니다: {partial_path}(.wN) - '
                f'--recompute-only는 전체 재실행을 하지 않으므로 최소 한 번은 정식 실행이 필요합니다.')
        per_query_by_system[system] = pq
        resolved_cutoff, problem = resolve_recompute_cutoff(pq, fallback_cutoff)
        if problem is not None:
            cutoff_problems.append(f'[{system}] {problem}')
        elif resolved_cutoff != cutoff:
            cutoff_problems.append(
                f'[{system}] 기존 per-query 결과는 top-{resolved_cutoff} 기준으로 이미 절단되어 '
                f'저장돼 있어 MAP@{cutoff}로 재계산할 수 없습니다(요청 cutoff={cutoff}). '
                f'top-{resolved_cutoff} 밖의 정답은 애초에 검색되지 않았으므로 top-{resolved_cutoff} '
                f'partial에서 그보다 넓은 cutoff를 만들어낼 수 없고, 반대로 더 좁은 cutoff로 다시 '
                f'자르는 것도 원래 검색이 그 깊이에서 반환했을 순위와 같다는 보장이 없어 하지 않습니다. '
                f'MAP@{cutoff}가 필요하면 --search-limit {cutoff}로 해당 시스템의 전체 검색을 '
                f'다시 실행하세요(--recompute-only 없이).')
    if cutoff_problems:
        raise SystemExit('[중단] --recompute-only cutoff 불일치/확인 불가:\n  - '
                         + '\n  - '.join(cutoff_problems))

    for system in systems:
        algorithm, field = SYSTEMS[system]
        pq = per_query_by_system[system]
        summaries[system] = build_system_summary(pq, algorithm, field, cutoff)
        print(f'  [{system}] {mkey}={summaries[system][mkey]} (n={len(pq)}, 기존 파일에서 재집계)')

    effects = build_effects(summaries, mkey)
    consistency, bad = build_consistency(summaries, args.out_prefix, mkey)

    src_paths = [os.path.join(DATA_DIR, 'db_export', f) for f in
                 ('patent_content.csv', 'split.csv', 'examiner_citation.csv',
                  'molecule.csv', 'patent_molecule.csv')]
    src_ver, src_detail = source_version(src_paths)

    report = dict(old_report) if old_report else {}
    report['recomputed_from_partial'] = True
    report['recompute_note'] = (
        f'MAP@{cutoff} 지표명 변경을 위해 기존 .partial.jsonl(시스템당 {[len(load_partial(os.path.join(RESULTS_DIR, f"{args.out_prefix}_{s}.partial.jsonl"))[0]) for s in systems]}쿼리)에서 '
        f'summary/effect_decomposition/consistency만 재생성함. 전체 검색은 재실행하지 않음. '
        f'collection_counts/dense_encoder_by_field/encoder_signature_check/'
        f'operational_latency_benchmark는 이전 리포트 값을 그대로 이월(이번 실행에서 재검증 안 함).')
    report['previous_provenance'] = (old_report or {}).get('provenance')
    report['provenance'] = runtime_provenance(load_settings(), ' '.join(sys.argv))
    report['source_version'] = src_ver
    report['source_csv_checksums'] = src_detail
    report.setdefault('config', {})
    report['config'] = dict((old_report or {}).get('config', {}))
    report['config']['metric_name'] = f'MAP@{cutoff}'
    report['config']['cutoff'] = cutoff
    report['config']['ap_definition'] = (
        f'AP@{cutoff} = sum(정답 위치별 precision, 상위 {cutoff}위 이내에서만) / 전체 정답 수. '
        f'{cutoff}위 밖이거나 검색 결과에 없는 정답은 0 기여. '
        f'mean_average_precision_map는 mean_average_precision_at_{cutoff}의 deprecated alias(항상 동일값).')
    report['config']['search_limit'] = cutoff
    report['config']['recomputed_fields_not_reverified_this_run'] = [
        'collection_counts', 'config.dense_encoder_by_field', 'config.encoder_signature_check',
        'operational_latency_benchmark']
    report['summaries'] = summaries
    report['effect_decomposition'] = effects
    report['effect_decomposition_metric'] = mkey
    report['summary_vs_per_query_consistency'] = {'all_ok': not bad, 'mismatched_systems': bad,
                                                   'detail': consistency}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(old_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('\n[recompute-only] 결과')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print('\n효과 분해(' + mkey + ' 기준): ' + json.dumps(effects, ensure_ascii=False, indent=1))
    print(f'\nsummary/per-query 일치: {"OK" if not bad else "불일치 " + str(bad)}')
    print(f'저장됨: {old_path}')
    return 1 if bad else 0


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[int(k)], 2) if lo == hi else round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def run_selftest():
    ranked = [f'P{i}' for i in range(1, 11)]
    gold = {'P2', 'P5'}
    m = evaluate_ranking(ranked, gold)
    assert abs(m['ap'] - 0.45) < 1e-9, m['ap']
    base = baseline_evaluate_query('Q', [(p, 1.0) for p in ranked], gold)
    for k in ('recall_at_100', 'recall_at_1000', 'mrr_at_10', 'best_rank', 'ap'):
        assert abs(float(m[k]) - float(base[k])) < 1e-4, f'{k}: {m[k]} vs {base[k]}'

    partial = evaluate_ranking(['P2'], gold)          # (1/1)/2 = 0.5
    assert abs(partial['ap'] - 0.5) < 1e-9, partial['ap']
    assert partial['recall_at_10'] == 0.5 and partial['num_gold_retrieved'] == 1

    none_hit = evaluate_ranking(['ZZZ'], gold)
    assert none_hit['ap'] == 0.0 and none_hit['num_gold_retrieved'] == 0
    assert none_hit['best_rank'] is None

    empty = evaluate_ranking([], gold)                 # 검색 결과가 아예 없는 경우
    assert empty['ap'] == 0.0 and empty['retrieved'] == 0
    print('[selftest] AP 정의(sum(P)/전체 정답 수) + 부분/무회수/빈결과 처리 검증 OK')

    # --- MAP@cutoff 이름/절단 의미론 검증 ---
    assert metric_key_for_cutoff(1000) == 'mean_average_precision_at_1000'
    assert metric_key_for_cutoff(500) == 'mean_average_precision_at_500'

    # 1000위 밖 정답은 0 기여 - 호출자가 상위 cutoff개로 자른 리스트를 넘긴다는 계약을
    # build_system_summary까지 이어서 검증한다(검색 리스트 자체가 이미 cutoff로 잘려있다고 가정).
    gold2 = {'P1', 'P1500'}                      # P1500은 상위 1000위 밖에 있다고 가정
    ranked_full = [f'P{i}' for i in range(1, 1501)]
    ap_uncut = evaluate_ranking(ranked_full, gold2)['ap']          # 절단 없이 전부 넘기면 둘 다 회수
    ap_cut = evaluate_ranking(ranked_full[:1000], gold2)['ap']      # 상위 1000만 넘기면 P1500 미회수
    assert ap_uncut > ap_cut, (ap_uncut, ap_cut)
    assert abs(ap_cut - 0.5) < 1e-9, ap_cut          # P1만 회수: (1/1)/2 = 0.5, P1500은 0 기여

    # build_system_summary가 mkey/alias를 항상 동일값으로 채우는지
    per_query = [
        {'ap': 0.5, 'recall_at_10': 1.0, 'recall_at_100': 1.0, 'recall_at_1000': 1.0,
         'mrr_at_10': 1.0, 'ndcg_at_10_binary': 1.0, 'num_gold': 1, 'num_gold_retrieved': 1,
         'retrieved': 10, 'latency_ms': 5.0},
        {'ap': 0.25, 'recall_at_10': 0.0, 'recall_at_100': 1.0, 'recall_at_1000': 1.0,
         'mrr_at_10': 0.0, 'ndcg_at_10_binary': 0.0, 'num_gold': 1, 'num_gold_retrieved': 1,
         'retrieved': 10, 'latency_ms': 7.0},
    ]
    summ = build_system_summary(per_query, 'bm25', 'title_claims', 1000)
    assert summ['mean_average_precision_at_1000'] == 0.375
    assert summ['mean_average_precision_map'] == summ['mean_average_precision_at_1000']
    summ2 = build_system_summary(per_query, 'bm25', 'title_claims', 500)
    assert 'mean_average_precision_at_500' in summ2 and 'mean_average_precision_at_1000' not in summ2
    assert summ2['mean_average_precision_map'] == summ2['mean_average_precision_at_500']
    print('[selftest] MAP@cutoff 이름/절단 의미론(1000위 밖 0 기여) + summary alias 일관성 검증 OK')


def load_partial(partial_path):
    per_query, done = [], set()
    for path in sorted(glob.glob(partial_path) + glob.glob(partial_path + '.w*')):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec['query_id'] in done:
                    continue
                per_query.append(rec)
                done.add(rec['query_id'])
    return per_query, done


def _search_one(retriever, algorithm, field, query_text, pf, search_limit, prefetch_limit):
    if algorithm == 'bm25':
        return retriever.search_bm25(query_text, pf, limit=search_limit, field=field)
    if algorithm == 'dense':
        return retriever.search_dense(query_text, pf, limit=search_limit, field=field)
    return retriever.search_hybrid(query_text, pf, limit=search_limit,
                                   prefetch_limit=prefetch_limit, field=field)


def _worker_run(task):
    """워커 프로세스: 할당된 쿼리만 검색해 자기 파일에 기록.
    로컬 임베디드 모드는 저장소에 배타 락이 걸리므로 워커마다 사본을 쓴다."""
    import json as _json
    from qdrant_engine.config import load_settings as _load
    from qdrant_engine.client import get_client as _client

    (system, algorithm, field, queries, out_path, storage_path,
     search_limit, prefetch_limit, deadline) = task

    settings = _load()
    if storage_path:
        settings.url = ''
        settings.local_path = storage_path
    client = _client(settings)
    dense = {field: build_dense_encoder(settings, idf_state_path=bm25_state_path(field))}
    sparse = {field: BM25SparseEncoder.load(bm25_state_path(field))}
    retriever = PatentRetriever(client, settings, dense, sparse, default_field=field)

    repo = smiles_index = None
    if algorithm == 'hybrid_rerank':
        repo = StructureRepository(client, settings)
        smiles_index = repo.load_smiles_index()

    written = 0
    with open(out_path, 'a', encoding='utf-8') as f:
        for qid, app_date, family_id, query_text, gold, pool_size, split_label in queries:
            if deadline and time.time() > deadline:
                break
            pf = build_prior_art_filter(qid, app_date, family_id)
            algo = 'hybrid' if algorithm == 'hybrid_rerank' else algorithm
            hits, latency_ms = _search_one(retriever, algo, field, query_text, pf,
                                           search_limit, prefetch_limit)
            rerank_stats = None
            if algorithm == 'hybrid_rerank':
                t0 = time.perf_counter()
                q_fps = repo.fingerprints_for_patent(qid, smiles_index)

                def tanimoto_fn(cand_id, _q=q_fps):
                    if not _q:
                        return None
                    cf = repo.fingerprints_for_patent(cand_id, smiles_index)
                    return max_tanimoto(_q, cf) if cf else None

                hits, rerank_stats = rerank_with_structure(
                    hits, tanimoto_fn, settings.text_weight,
                    settings.structure_rerank_weight, settings.structure_rerank_top_k)
                latency_ms += (time.perf_counter() - t0) * 1000.0

            rec = {'query_id': qid, 'system': system, 'pool_size': pool_size,
                   'latency_ms': round(latency_ms, 2), 'original_split_label': split_label,
                   'parallel_worker': True, 'cutoff': search_limit,
                   **evaluate_ranking([h['patent_id'] for h in hits], set(gold))}
            if rerank_stats:
                rec['rerank'] = rerank_stats
            f.write(_json.dumps(rec, ensure_ascii=False) + '\n')
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', default='all')
    ap.add_argument('--limit-queries', type=int, default=None)
    ap.add_argument('--search-limit', type=int, default=1000,
                    help='모든 시스템이 동일하게 상위 N개만 반환/평가한다(Jaccard baseline의 '
                         '전수 랭킹도 같은 N으로 자른다). R@1000을 재려면 1000 이상이어야 한다.')
    ap.add_argument('--prefetch-limit', type=int, default=1000)
    ap.add_argument('--latency-bench-limit', type=int, default=100)
    ap.add_argument('--latency-bench-queries', type=int, default=50)
    ap.add_argument('--time-budget-seconds', type=float, default=None)
    ap.add_argument('--text-weight', type=float, default=0.5)
    ap.add_argument('--tanimoto-weight', type=float, default=0.3)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out-prefix', default='qdrant_eval')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--recompute-only', action='store_true',
                    help='Qdrant에 연결하거나 검색을 실행하지 않고, 이미 있는 '
                         '.partial.jsonl에서 summary/effect_decomposition만 다시 만든다 '
                         '(지표 이름 변경처럼 재평가가 필요 없는 변경에 사용).')
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return 0

    if args.recompute_only:
        return run_recompute_only(args)

    systems = ALL_SYSTEMS if args.systems == 'all' else tuple(s.strip() for s in args.systems.split(','))
    for s in systems:
        if s not in SYSTEMS:
            raise SystemExit(f'알 수 없는 시스템: {s} (가능: {ALL_SYSTEMS})')

    settings = load_settings()

    print('[1/5] db_export 인덱스 로딩...')
    idx = build_indexes()
    all_patent_ids = idx['all_patent_ids']
    title_of = {r['patent_id']: (r.get('title') or '') for r in load_csv('patent_content')}
    query_ids = [pid for pid in all_patent_ids if idx['gold_of'].get(pid)]
    if args.limit_queries:
        query_ids = query_ids[: args.limit_queries]

    def field_text(pid, field):
        text, _ = build_field_text(title_of.get(pid), idx['claims_text_of'].get(pid), field)
        return text

    print('[2/5] 후보 풀/gold 사전 계산 (모든 시스템 공통)...')
    pools, golds = {}, {}
    excluded = 0
    for qid in query_ids:
        pool_ids, _, _ = build_candidate_pool(qid, all_patent_ids, idx['pub_date_of'],
                                              idx['app_date_of'], idx['family_id_of'])
        gold_in_pool = idx['gold_of'][qid] & set(pool_ids)
        if not gold_in_pool:
            excluded += 1
            continue
        pools[qid] = pool_ids
        golds[qid] = gold_in_pool
    eval_query_ids = [q for q in query_ids if q in golds]
    avg_pool = round(sum(len(p) for p in pools.values()) / max(len(pools), 1), 2)
    print(f'  평가 쿼리 {len(eval_query_ids)}건 (gold가 풀에 없어 제외 {excluded}건), 평균 풀 {avg_pool}')

    need_jaccard = any(SYSTEMS[s][0] == 'jaccard' for s in systems)
    need_qdrant = any(SYSTEMS[s][0] != 'jaccard' for s in systems)
    fields_used = sorted({SYSTEMS[s][1] for s in systems if SYSTEMS[s][0] != 'jaccard'})

    token_counters = fps_of = None
    if need_jaccard:
        print('[3/5] Jaccard baseline 사전계산(토큰/지문)...')
        token_counters = precompute_token_counters(all_patent_ids, idx['claims_text_of'],
                                                   ChemicalTokenizer())
        fps_of = precompute_fingerprints(all_patent_ids, idx['smiles_of'])

    retriever = structure_repo = smiles_index = None
    dense_encoders, sparse_encoders = {}, {}
    encoder_signature_check = None
    if need_qdrant:
        print('[4/5] Qdrant 연결 및 필드별 인코더 준비...')
        client = get_client(settings)
        for field in fields_used:
            if not os.path.exists(bm25_state_path(field)):
                raise SystemExit(f'{bm25_state_path(field)} 없음 - scripts/qdrant/qdrant_index_patents.py 실행 필요')
            sparse_encoders[field] = BM25SparseEncoder.load(bm25_state_path(field))
            dense_encoders[field] = build_dense_encoder(settings, idf_state_path=bm25_state_path(field))
            print(f'  {field}: dense={dense_encoders[field].describe()["model"]} '
                  f'sig={dense_encoders[field].signature()}')
        retriever = PatentRetriever(client, settings, dense_encoders, sparse_encoders)
        if any(SYSTEMS[s][0] == 'hybrid_rerank' for s in systems):
            structure_repo = StructureRepository(client, settings)
            smiles_index = structure_repo.load_smiles_index()
            print(f'  구조 보유 특허 {len(smiles_index)}건')

        # 적재 시 인코더 == 검색 시 인코더 확인
        sample = client.scroll(settings.patents_collection, limit=1, with_payload=True)[0]
        encoder_signature_check = {}
        if sample:
            stored = (sample[0].payload or {}).get('embedding_signature_by_field') or {}
            for field in fields_used:
                got, want = dense_encoders[field].signature(), stored.get(field)
                encoder_signature_check[field] = {'indexed': want, 'search': got, 'match': got == want}
                if want and got != want:
                    raise SystemExit(f'인코더 불일치({field}): 적재 {want} != 검색 {got}. 재색인이 필요합니다.')
            print(f'  인코더 서명 일치: {encoder_signature_check}')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()
    stopped_early = False
    summaries = {}

    for system in systems:
        algorithm, field = SYSTEMS[system]
        partial_path = os.path.join(RESULTS_DIR, f'{args.out_prefix}_{system}.partial.jsonl')
        per_query, done = load_partial(partial_path)
        remaining = [q for q in eval_query_ids if q not in done]
        print(f'\n[{system}] algorithm={algorithm} field={field} | 남은 {len(remaining)}/{len(eval_query_ids)}')

        if remaining and args.workers > 1 and algorithm != 'jaccard':
            deadline = (t_start + args.time_budget_seconds) if args.time_budget_seconds else None
            chunks = [remaining[i::args.workers] for i in range(args.workers)]
            tmp_root = tempfile.mkdtemp(prefix='chemrag_qdrant_workers_')
            tasks = []
            for i, chunk in enumerate(chunks):
                if not chunk:
                    continue
                storage = ''
                if is_local_mode(settings):
                    storage = os.path.join(tmp_root, f'w{i}')
                    shutil.copytree(settings.local_path, storage)
                payload = [(qid, idx['app_date_of'].get(qid), idx['family_id_of'].get(qid),
                            field_text(qid, field), sorted(golds[qid]), len(pools[qid]),
                            idx['split_of'].get(qid)) for qid in chunk]
                tasks.append((system, algorithm, field, payload, f'{partial_path}.w{i}', storage,
                              args.search_limit, args.prefetch_limit, deadline))
            print(f'  병렬 워커 {len(tasks)}개')
            try:
                with mp.get_context('fork').Pool(len(tasks)) as pool:
                    print(f'  워커별 처리: {pool.map(_worker_run, tasks)}')
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)
            per_query, done = load_partial(partial_path)
            if len(done) < len(eval_query_ids):
                print(f'  [미완료] {len(done)}/{len(eval_query_ids)} - 다시 실행하면 이어집니다.')
                stopped_early = True
        elif remaining:
            f_out = open(partial_path, 'a', encoding='utf-8')
            try:
                for n, qid in enumerate(remaining):
                    if args.time_budget_seconds and (time.time() - t_start) > args.time_budget_seconds:
                        stopped_early = True
                        break
                    gold = golds[qid]
                    rerank_stats = None
                    if algorithm == 'jaccard':
                        t0 = time.perf_counter()
                        q_tc, q_fps = token_counters[qid], fps_of[qid]
                        scored = []
                        for pid in pools[qid]:
                            ts = text_similarity_cached(q_tc, token_counters[pid])
                            tan = baseline_max_tanimoto(q_fps, fps_of[pid])
                            scored.append((pid, combine_score(ts, tan, args.text_weight,
                                                              args.tanimoto_weight)))
                        scored.sort(key=lambda x: x[1], reverse=True)
                        # 모든 시스템과 동일한 깊이로 자른다(공정 비교)
                        ranked = [p for p, _ in scored[:args.search_limit]]
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                    else:
                        pf = build_prior_art_filter(qid, idx['app_date_of'].get(qid),
                                                    idx['family_id_of'].get(qid))
                        algo = 'hybrid' if algorithm == 'hybrid_rerank' else algorithm
                        hits, latency_ms = _search_one(retriever, algo, field, field_text(qid, field),
                                                       pf, args.search_limit, args.prefetch_limit)
                        if algorithm == 'hybrid_rerank':
                            t0 = time.perf_counter()
                            q_fps = structure_repo.fingerprints_for_patent(qid, smiles_index)

                            def tanimoto_fn(cand_id, _q=q_fps):
                                if not _q:
                                    return None
                                cf = structure_repo.fingerprints_for_patent(cand_id, smiles_index)
                                return max_tanimoto(_q, cf) if cf else None

                            hits, rerank_stats = rerank_with_structure(
                                hits, tanimoto_fn, settings.text_weight,
                                settings.structure_rerank_weight, settings.structure_rerank_top_k)
                            latency_ms += (time.perf_counter() - t0) * 1000.0
                        ranked = [h['patent_id'] for h in hits]

                    rec = {'query_id': qid, 'system': system, 'pool_size': len(pools[qid]),
                           'latency_ms': round(latency_ms, 2),
                           'original_split_label': idx['split_of'].get(qid),
                           'cutoff': args.search_limit,
                           **evaluate_ranking(ranked, gold)}
                    if rerank_stats:
                        rec['rerank'] = rerank_stats
                    per_query.append(rec)
                    f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    if (n + 1) % 100 == 0:
                        f_out.flush()
                        print(f'  {n + 1}/{len(remaining)} ({time.time() - t_start:.0f}s)')
            finally:
                f_out.close()

        if stopped_early:
            print('\n[시간 예산 도달] 같은 명령을 다시 실행하면 이어서 진행됩니다.')
            return 0

        mkey = metric_key_for_cutoff(args.search_limit)
        summaries[system] = build_system_summary(per_query, algorithm, field, args.search_limit)
        print(f'  {mkey}={summaries[system][mkey]} '
              f'R@10={summaries[system]["recall_at_10"]} R@100={summaries[system]["recall_at_100"]}')

    latency_bench = {}
    if need_qdrant and not stopped_early:
        print('\n[추가] 운영 조건(top-k, 단일 프로세스) 지연 벤치마크...')
        bench_ids = eval_query_ids[: args.latency_bench_queries]
        for system in systems:
            algorithm, field = SYSTEMS[system]
            if algorithm == 'jaccard':
                continue
            algo = 'hybrid' if algorithm == 'hybrid_rerank' else algorithm
            lats = []
            for qid in bench_ids:
                pf = build_prior_art_filter(qid, idx['app_date_of'].get(qid),
                                            idx['family_id_of'].get(qid))
                _, ms = _search_one(retriever, algo, field, field_text(qid, field), pf,
                                    args.latency_bench_limit, args.latency_bench_limit * 4)
                lats.append(ms)
            latency_bench[system] = {'limit': args.latency_bench_limit, 'queries': len(lats),
                                     'p50_ms': percentile(lats, 0.50), 'p95_ms': percentile(lats, 0.95),
                                     'note': '구조 재정렬 제외한 순수 검색 지연'}

    mkey = metric_key_for_cutoff(args.search_limit)
    effects = build_effects(summaries, mkey)
    consistency, bad = build_consistency(summaries, args.out_prefix, mkey)

    src_paths = [os.path.join(DATA_DIR, 'db_export', f) for f in
                 ('patent_content.csv', 'split.csv', 'examiner_citation.csv',
                  'molecule.csv', 'patent_molecule.csv')]
    src_ver, src_detail = source_version(src_paths)

    collection_counts = {}
    if need_qdrant:
        for name in (settings.patents_collection, settings.structures_collection):
            try:
                collection_counts[name] = client.count(name).count
            except Exception as exc:
                collection_counts[name] = f'error: {type(exc).__name__}'

    report = {
        'provenance': runtime_provenance(settings, ' '.join(sys.argv)),
        'source_version': src_ver,
        'source_csv_checksums': src_detail,
        'collection_counts': collection_counts,
        'config': {
            'systems': {s: {'algorithm': SYSTEMS[s][0], 'text_field': SYSTEMS[s][1]} for s in systems},
            'metric_name': f'MAP@{args.search_limit}',
            'cutoff': args.search_limit,
            'ap_definition': f'AP@{args.search_limit} = sum(precision at each retrieved relevant '
                             f'rank, within top {args.search_limit}) / total relevant '
                             f'(unretrieved gold, or gold beyond rank {args.search_limit}, '
                             f'contributes 0). mean_average_precision_map is a deprecated alias '
                             f'for mean_average_precision_at_{args.search_limit} (always equal).',
            'query_set': 'examiner_citation gold 보유 & gold가 후보 풀 안에 존재',
            'num_candidate_queries_with_gold': len(query_ids),
            'excluded_gold_not_in_pool': excluded,
            'num_queries_evaluated': len(eval_query_ids),
            'avg_pool_size': avg_pool,
            'filters': ['candidate.pub_date <= query.app_date',
                        'candidate.family_id != query.family_id',
                        'candidate.patent_id != query.patent_id'],
            'search_limit': args.search_limit,
            'search_depth_note': '모든 시스템(Jaccard baseline 포함)이 동일하게 상위 search_limit '
                                 '개만 반환/평가한다. 그 밖의 정답은 AP/Recall에서 0 기여로 반영된다.',
            'hybrid_prefetch_limit': args.prefetch_limit,
            'fusion': settings.fusion,
            'bm25': {'k1': settings.bm25_k1, 'b': settings.bm25_b},
            'dense_encoder_by_field': {f: dense_encoders[f].describe() for f in dense_encoders},
            'encoder_signature_check': encoder_signature_check,
            'jaccard_weights': {'text': args.text_weight, 'tanimoto': args.tanimoto_weight},
            'structure_rerank': {'top_k': settings.structure_rerank_top_k,
                                 'structure_weight': settings.structure_rerank_weight,
                                 'text_weight': settings.text_weight,
                                 'similarity': 'RDKit exact Tanimoto (Morgan ECFP4 r=2, 2048bit)'},
            'eval_workers': args.workers,
            'notes': [
                'Qdrant는 저장·실행 계층이다. 시스템 간 수치 차이는 스코어링 알고리즘/입력 필드/'
                '임베딩 모델/융합 설정의 차이이지 Qdrant 도입 자체의 효과가 아니다.',
                'ndcg는 examiner_citation category가 전부 X 계열이라 binary 기준으로만 계산한다.',
                'workers>1 실행 시 per-query latency는 프로세스 경합 영향을 받는다 - '
                'operational_latency_benchmark(단일 프로세스)를 볼 것.',
            ],
        },
        'summaries': summaries,
        'effect_decomposition': effects,
        'effect_decomposition_metric': mkey,
        'operational_latency_benchmark': latency_bench,
        'summary_vs_per_query_consistency': {'all_ok': not bad, 'mismatched_systems': bad,
                                             'detail': consistency},
    }
    out = os.path.join(RESULTS_DIR, f'{args.out_prefix}_summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('\n[5/5] 결과 (' + mkey + ')')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f'\n효과 분해({mkey} 기준): ' + json.dumps(effects, ensure_ascii=False, indent=1))
    print(f'\nsummary/per-query 일치: {"OK" if not bad else "불일치 " + str(bad)}')
    print(f'저장됨: {out}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
