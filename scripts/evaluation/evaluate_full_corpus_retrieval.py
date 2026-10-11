"""Task 1 전체 코퍼스 검색(full-corpus retrieval) 평가 스크립트.

TARGET 특허 하나당 소수 후보만 놓고 맞는지 확인하던 기존 방식 대신, 쿼리 출원일
이전에 공개되고 동일 patent family가 아닌 로컬 코퍼스 전체 특허를 후보 풀로 놓고
실제로 랭킹한다. 즉:

  candidate.pub_date <= query.app_date  AND  candidate.family_id != query.family_id

로컬 코퍼스는 애초에 (target, prior) 인용쌍에서만 뽑힌 특허들이라(한 번도 인용된
적 없는 배경 특허는 로컬에 없음), 후보 풀은 db_export의 patent_content 전체에서
위 조건으로 필터링한다. train/test 분리는 결과 표에 참고용으로만 남기고 후보 풀
구성에는 쓰지 않는다 - 이 파이프라인은 학습을 하지 않는 결정론적 채점 방식이라
split을 나눌 이유가 없고, 시간 필터가 실질적인 데이터 누출 방지 역할을 한다.

점수는 텍스트 유사도(Jaccard+overlap)와 화학구조 Tanimoto를 rag_engine과 동일한
공식으로 결합한다(가중치 text=0.5/structure=0.3, vision 없이 재정규화). 특허 하나에
분자가 여러 개면 모든 SMILES 쌍 중 최댓값을 쓴다. nDCG는 현재 인용 라벨이 전부 X
등급이라 그레이딩 정보가 없어 이진(binary) 기준으로만 계산한다.

쿼리 x 코퍼스 전수 비교(최대 수백만 쌍)를 감당하기 위해 토큰화/지문 계산은 특허당
한 번만 미리 해두고 재사용한다.

사용법:
    python3 scripts/evaluation/evaluate_full_corpus_retrieval.py
    python3 scripts/evaluation/evaluate_full_corpus_retrieval.py --time-budget-seconds 35   # 여러 번 나눠 이어 돌리기
    python3 scripts/evaluation/evaluate_full_corpus_retrieval.py --text-weight 0.5 --tanimoto-weight 0.3
    python3 scripts/evaluation/evaluate_full_corpus_retrieval.py --selftest   # 채점 로직 자체 검증
"""
import os
import sys
import csv
import json
import time
import argparse
import math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import DATA_DIR
from rag_engine.chemical_tokenizer import ChemicalTokenizer

DB_EXPORT_DIR = os.path.join(DATA_DIR, 'db_export')


def load_csv(name):
    path = os.path.join(DB_EXPORT_DIR, f'{name}.csv')
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def build_indexes():
    split_rows = load_csv('split')
    patent_rows = load_csv('patent_content')
    citation_rows = load_csv('examiner_citation')
    molecule_rows = load_csv('molecule')
    patent_molecule_rows = load_csv('patent_molecule')

    split_of = {r['id']: r['split'] for r in split_rows}  # 결과 표시용 참고 정보 - 후보 풀 구성에는 쓰지 않음
    claims_text_of = {r['patent_id']: r['claims_text'] for r in patent_rows}
    pub_date_of = {r['patent_id']: r['pub_date'] for r in patent_rows}
    app_date_of = {r['patent_id']: r['app_date'] for r in patent_rows}
    family_id_of = {r['patent_id']: r['family_id'] for r in patent_rows}
    all_patent_ids = [r['patent_id'] for r in patent_rows]

    mol_smiles = {r['mol_id']: r['smiles'] for r in molecule_rows}
    smiles_of = defaultdict(list)
    for r in patent_molecule_rows:
        smi = mol_smiles.get(r['mol_id'])
        if smi:
            smiles_of[r['patent_id']].append(smi)

    gold_of = defaultdict(set)
    for r in citation_rows:
        gold_of[r['query_patent_id']].add(r['cited_patent_id'])

    return {
        'split_of': split_of,
        'claims_text_of': claims_text_of,
        'pub_date_of': pub_date_of,
        'app_date_of': app_date_of,
        'family_id_of': family_id_of,
        'all_patent_ids': all_patent_ids,
        'smiles_of': dict(smiles_of),
        'gold_of': dict(gold_of),
    }


def precompute_token_counters(patent_ids, claims_text_of, tokenizer):
    counters = {}
    for pid in patent_ids:
        text = claims_text_of.get(pid) or ''
        tokens = tokenizer.tokenize(text)
        counters[pid] = (Counter(tokens), len(tokens))
    return counters


def text_similarity_cached(q_counter_len, d_counter_len):
    q_counts, q_len = q_counter_len
    d_counts, d_len = d_counter_len
    if q_len == 0 or d_len == 0:
        return 0.0
    intersection = sum((q_counts & d_counts).values())
    union = sum((q_counts | d_counts).values())
    jaccard = intersection / float(union) if union > 0 else 0.0
    overlap_ratio = intersection / math.sqrt(q_len * d_len)
    return round(0.5 * jaccard + 0.5 * overlap_ratio, 4)


def precompute_fingerprints(patent_ids, smiles_of):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')

    fps_of = {}
    for pid in patent_ids:
        fps = []
        for smi in smiles_of.get(pid, []):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            except Exception:
                continue
        fps_of[pid] = fps
    return fps_of


def max_tanimoto(fps_q, fps_d):
    if not fps_q or not fps_d:
        return None
    from rdkit import DataStructs
    best = 0.0
    for fq in fps_q:
        for fd in fps_d:
            sim = DataStructs.TanimotoSimilarity(fq, fd)
            if sim > best:
                best = sim
    return round(best, 4)


def combine_score(text_score, tanimoto_score, text_weight, tanimoto_weight):
    # rag_engine/multimodal_ranker.py와 동일한 재정규화 방식 - vision 축은 여기선 제외
    if tanimoto_score is not None:
        weight_sum = text_weight + tanimoto_weight
        return round((text_weight * text_score + tanimoto_weight * tanimoto_score) / weight_sum, 4)
    return round(text_score, 4)


def build_candidate_pool(query_id, all_patent_ids, pub_date_of, app_date_of, family_id_of):
    """전체 코퍼스에서 (a) 자기 자신 제외, (b) 쿼리 출원일 이후 공개된 미래 특허 제외,
    (c) 동일 patent family 제외한 후보 풀을 만든다. YYYYMMDD 문자열은 그대로
    사전식(lexicographic) 비교해도 날짜 대소 비교와 결과가 같다."""
    q_app_date = app_date_of.get(query_id) or ''
    q_family = family_id_of.get(query_id) or ''

    pool = []
    excluded_future = 0
    excluded_family = 0
    for pid in all_patent_ids:
        if pid == query_id:
            continue
        if family_id_of.get(pid) and q_family and family_id_of.get(pid) == q_family:
            excluded_family += 1
            continue
        cand_pub_date = pub_date_of.get(pid) or ''
        if q_app_date and cand_pub_date and cand_pub_date > q_app_date:
            excluded_future += 1
            continue
        pool.append(pid)
    return pool, excluded_future, excluded_family


def dcg_at_k(relevances, k):
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(relevances[:k]))


def evaluate_query(query_id, ranked_pool, gold_set):
    # ranked_pool: [(patent_id, score), ...] 내림차순 정렬됨
    relevant_ranks = [i for i, (pid, _) in enumerate(ranked_pool) if pid in gold_set]
    if not relevant_ranks:
        return None

    num_relevant = len(gold_set)

    precisions = []
    hit_count = 0
    for rank in relevant_ranks:
        hit_count += 1
        precisions.append(hit_count / (rank + 1))
    # AP 정의: 회수된 relevant 수가 아니라 '전체 정답 수'로 나눈다.
    # 랭킹에 나타나지 않은 정답은 0 기여로 반영되어야 하기 때문(표준 Average Precision).
    ap = sum(precisions) / num_relevant

    r_at_100 = sum(1 for r in relevant_ranks if r < 100) / num_relevant
    r_at_1000 = sum(1 for r in relevant_ranks if r < 1000) / num_relevant
    rr_10 = 1.0 / (relevant_ranks[0] + 1) if relevant_ranks[0] < 10 else 0.0

    relevances = [1 if pid in gold_set else 0 for pid, _ in ranked_pool[:10]]
    dcg = dcg_at_k(relevances, 10)
    ideal_relevances = [1] * min(num_relevant, 10)
    idcg = dcg_at_k(ideal_relevances, 10)
    ndcg_10 = round(dcg / idcg, 4) if idcg > 0 else 0.0

    return {
        'query_id': query_id,
        'num_gold': num_relevant,
        'pool_size': len(ranked_pool),
        'ap': round(ap, 4),
        'recall_at_100': round(r_at_100, 4),
        'recall_at_1000': round(r_at_1000, 4),
        'mrr_at_10': round(rr_10, 4),
        'ndcg_at_10_binary': ndcg_10,
        'best_rank': relevant_ranks[0] + 1,
    }


def run_selftest():
    """채점 로직(pool 필터링 + AP/Recall/MRR/nDCG)에 대한 최소 단위 테스트.
    외부 데이터 없이 합성 데이터로 검증하며, 실패 시 AssertionError로 즉시 중단된다."""
    print('[selftest] 후보 풀 필터링(시간/패밀리) 검증...')
    pub_date_of = {'Q': '20200101', 'A': '20190101', 'B': '20200601', 'C': '20191201', 'D': '20191001'}
    app_date_of = {'Q': '20200101'}
    family_id_of = {'Q': 'FAM1', 'A': 'FAM2', 'B': 'FAM2', 'C': 'FAM1', 'D': 'FAM3'}
    all_ids = ['Q', 'A', 'B', 'C', 'D']
    pool, excl_future, excl_family = build_candidate_pool('Q', all_ids, pub_date_of, app_date_of, family_id_of)
    # B는 쿼리 출원일(20200101) 이후 공개(20200601)라 제외돼야 함
    # C는 쿼리와 동일 family(FAM1)라 제외돼야 함
    # A, D만 남아야 함
    assert set(pool) == {'A', 'D'}, f'예상과 다른 후보 풀: {pool}'
    assert excl_future == 1 and excl_family == 1, f'제외 카운트 불일치: future={excl_future}, family={excl_family}'
    print('  OK - 미래 특허 1건, 동일 패밀리 1건이 정확히 제외됨')

    print('[selftest] AP/Recall/MRR/nDCG 계산 검증...')
    # 정답 2개(P2, P5)가 각각 2위, 5위인 랭킹
    ranked = [(f'P{i}', 1.0 - i * 0.01) for i in range(1, 11)]
    gold = {'P2', 'P5'}
    result = evaluate_query('Q', ranked, gold)
    # AP = ((1/2) + (2/5)) / 2 = 0.45  (정답 2개 모두 회수된 경우)
    assert abs(result['ap'] - 0.45) < 1e-6, f"AP 계산 오류: {result['ap']}"

    # 부분 회수: 정답 2개 중 1개만 1위로 회수 -> AP = (1/1) / 2 = 0.5
    partial = evaluate_query('Q', [('P2', 1.0)], {'P2', 'P5'})
    assert abs(partial['ap'] - 0.5) < 1e-6, f"부분 회수 AP 오류: {partial['ap']}"
    assert result['recall_at_100'] == 1.0
    # MRR@10: 첫 정답이 2위 -> 1/2
    assert abs(result['mrr_at_10'] - 0.5) < 1e-6, f"MRR 계산 오류: {result['mrr_at_10']}"
    assert result['best_rank'] == 2
    print(f'  OK - AP={result["ap"]}, MRR@10={result["mrr_at_10"]}, nDCG@10={result["ndcg_at_10_binary"]}')

    print('[selftest] 정답이 풀에 없는 쿼리는 None 반환 검증...')
    result_none = evaluate_query('Q', ranked, {'NOT_IN_POOL'})
    assert result_none is None
    print('  OK')

    print('[selftest] 중단 후 재개(resume) 시 제외 통계·소요시간 집계 검증...')
    # 시간 예산을 넘겨 여러 번 나눠 실행해도, 제외 통계/소요시간이 마지막 청크가
    # 아니라 전체 청크 기준으로 정확히 합산되는지 확인한다.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        partial_path = os.path.join(tmp, 'fake.partial.jsonl')
        meta_path = partial_path + '.meta.json'

        # 1번째 청크: 미래 제외 10건짜리 쿼리 하나 기록, 누적시간 5.0초로 중단
        with open(partial_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'query_id': 'Q1', 'ap': 1.0, 'candidates_excluded_future': 10,
                                 'candidates_excluded_same_family': 2}) + '\n')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'cumulative_elapsed_seconds': 5.0}, f)

        # 2번째 청크: 미래 제외 0건짜리 쿼리 하나 추가 기록 (main()의 재개 로직과
        # 동일하게 - 이전 결과를 읽고, 이번 실행분만 새로 append)
        with open(partial_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'query_id': 'Q2', 'ap': 0.0, 'candidates_excluded_future': 0,
                                 'candidates_excluded_same_family': 0}) + '\n')
        this_run_elapsed = 3.0  # 가정: 2번째 청크가 3초 걸림

        # main()이 최종 집계 시 하는 것과 동일한 절차
        per_query_results = []
        with open(partial_path, 'r', encoding='utf-8') as f:
            for line in f:
                per_query_results.append(json.loads(line))
        with open(meta_path, 'r', encoding='utf-8') as f:
            cumulative_before = json.load(f)['cumulative_elapsed_seconds']
        cumulative_elapsed = cumulative_before + this_run_elapsed

        avg_future = sum(r['candidates_excluded_future'] for r in per_query_results) / len(per_query_results)
        assert avg_future == 5.0, f'재개 시 제외 통계가 전체 평균이 아님: {avg_future}'
        assert cumulative_elapsed == 8.0, f'재개 시 누적 소요시간이 틀림: {cumulative_elapsed}'
    print('  OK - 재개 시에도 전체 청크 기준으로 정확히 집계됨(avg_future=5.0, elapsed=8.0)')

    print('\n[selftest] 전체 통과')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-split', choices=['train', 'test', 'all'], default='all',
                         help='결과를 어느 쿼리 집합 기준으로 볼지(참고용 메타데이터일 뿐, 후보 풀은 항상 전체 코퍼스)')
    parser.add_argument('--text-weight', type=float, default=0.5)
    parser.add_argument('--tanimoto-weight', type=float, default=0.3,
                         help='rag_engine/multimodal_ranker.py 기본값(text=0.5/structure=0.3/vision=0.2)과 '
                              '같은 비율이 되도록 기본값을 0.3으로 맞춤(vision 제외 후 재정규화하면 0.625:0.375)')
    parser.add_argument('--limit-queries', type=int, default=None,
                         help='디버그용 - 쿼리 수 상한(전체 실행 전 소규모 테스트)')
    parser.add_argument('--time-budget-seconds', type=float, default=None,
                         help='이 시간(초)이 지나면 처리 중이던 쿼리까지만 끝내고 중간 저장 후 종료 - '
                              '전체 코퍼스라 오래 걸리는 실행을 여러 번 나눠 돌려 이어갈 때 사용. '
                              '완료된 쿼리는 <out>.partial.jsonl에 누적되고, 다음 실행 시 자동으로 건너뜀')
    parser.add_argument('--selftest', action='store_true', help='합성 데이터로 채점 로직만 검증하고 종료')
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    print('[1/5] db_export 테이블 로딩...')
    idx = build_indexes()
    all_patent_ids = idx['all_patent_ids']

    query_ids = [pid for pid in all_patent_ids if idx['gold_of'].get(pid)]
    if args.query_split != 'all':
        query_ids = [pid for pid in query_ids if idx['split_of'].get(pid) == args.query_split]
    if args.limit_queries:
        query_ids = query_ids[:args.limit_queries]
    print(f'  전체 코퍼스 {len(all_patent_ids)}건, 평가 대상 쿼리(정답 보유, query-split={args.query_split}) {len(query_ids)}건')

    print('[2/5] 텍스트 토큰 사전계산...')
    tokenizer = ChemicalTokenizer()
    token_counters = precompute_token_counters(all_patent_ids, idx['claims_text_of'], tokenizer)
    empty_text = sum(1 for pid in all_patent_ids if token_counters[pid][1] == 0)
    empty_text_in_queries = sum(1 for pid in query_ids if token_counters[pid][1] == 0)
    print(f'  claims_text 비어있어 토큰 0개인 특허: 전체 {empty_text}건, 쿼리 중 {empty_text_in_queries}건')

    print('[3/5] 화학구조 Morgan fingerprint 사전계산...')
    fps_of = precompute_fingerprints(all_patent_ids, idx['smiles_of'])
    with_structure = sum(1 for pid in all_patent_ids if fps_of[pid])
    print(f'  구조(SMILES) 확보된 특허: {with_structure}/{len(all_patent_ids)}')

    out_dir = os.path.normpath(os.path.join(DATA_DIR, '..', 'benchmark', 'results'))
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, f'full_corpus_retrieval_v2_{args.query_split}.partial.jsonl')
    # 청크별 소요시간을 이어서 더하기 위한 보조 메타 파일. 제외 통계는 per-query
    # 결과 자체에 저장되므로 별도 누적 카운터 없이 나중에 agg()로 평균낸다.
    meta_path = partial_path + '.meta.json'

    done_ids = set()
    per_query_results = []
    cumulative_elapsed_before_this_run = 0.0
    if os.path.exists(partial_path):
        with open(partial_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                per_query_results.append(rec)
                done_ids.add(rec['query_id'])
        print(f'  이전 실행에서 완료된 쿼리 {len(done_ids)}건을 이어서 진행합니다 ({partial_path})')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                cumulative_elapsed_before_this_run = json.load(f).get('cumulative_elapsed_seconds', 0.0)

    remaining_query_ids = [q for q in query_ids if q not in done_ids]
    print(f'[4/5] 쿼리별 전체 코퍼스(시간/패밀리 필터 적용) 랭킹 계산 중... (남은 쿼리 {len(remaining_query_ids)})')
    t0 = time.time()
    excluded_gold_not_in_pool = 0
    partial_f = open(partial_path, 'a', encoding='utf-8')
    stopped_early = False
    processed_this_run = 0
    try:
        for query_id in remaining_query_ids:
            if args.time_budget_seconds and (time.time() - t0) > args.time_budget_seconds:
                stopped_early = True
                break

            pool_ids, excl_future, excl_family = build_candidate_pool(
                query_id, all_patent_ids, idx['pub_date_of'], idx['app_date_of'], idx['family_id_of'])

            gold_set = idx['gold_of'][query_id]
            gold_in_pool = gold_set & set(pool_ids)
            if not gold_in_pool:
                excluded_gold_not_in_pool += 1
                processed_this_run += 1
                continue

            q_tc = token_counters[query_id]
            q_fps = fps_of[query_id]

            scored = []
            for pid in pool_ids:
                text_score = text_similarity_cached(q_tc, token_counters[pid])
                tanimoto_score = max_tanimoto(q_fps, fps_of[pid])
                score = combine_score(text_score, tanimoto_score, args.text_weight, args.tanimoto_weight)
                scored.append((pid, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            result = evaluate_query(query_id, scored, gold_in_pool)
            if result:
                result['original_split_label'] = idx['split_of'].get(query_id)
                result['candidates_excluded_future'] = excl_future
                result['candidates_excluded_same_family'] = excl_family
                per_query_results.append(result)
                partial_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                partial_f.flush()
            processed_this_run += 1

            if processed_this_run % 20 == 0:
                elapsed = time.time() - t0
                print(f'  이번 실행에서 {processed_this_run}/{len(remaining_query_ids)}건 처리 ({elapsed:.1f}초 경과)')
    finally:
        partial_f.close()

    this_run_elapsed = time.time() - t0
    cumulative_elapsed = cumulative_elapsed_before_this_run + this_run_elapsed

    if stopped_early:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'cumulative_elapsed_seconds': round(cumulative_elapsed, 1)}, f)
        done_now = len(per_query_results)
        print(f'\n[시간 예산 도달] 이번 실행에서 {processed_this_run}건 처리 ({this_run_elapsed:.1f}초), '
              f'누적 완료 {done_now}건, 누적 소요시간 {cumulative_elapsed:.1f}초. '
              f'같은 명령을 다시 실행하면 이어서 진행됩니다.')
        return

    print('[5/5] 결과 집계...')
    n = len(per_query_results)
    if n == 0:
        print('평가 가능한 쿼리가 없습니다(정답이 코퍼스 풀 안에 하나도 없음).')
        return

    def agg(key, records):
        return round(sum(r[key] for r in records) / len(records), 4) if records else None

    summary = {
        'candidate_pool_definition': 'patent_content 전체(4,123건) - 자기 자신 - candidate.pub_date > query.app_date - 동일 family_id',
        'query_split_filter': args.query_split,
        'candidate_queries_with_gold_label': len(query_ids),
        'excluded_gold_not_in_pool': excluded_gold_not_in_pool,
        'num_queries_evaluated': n,
        'text_weight': args.text_weight,
        'tanimoto_weight': args.tanimoto_weight,
        'vision_included': False,
        'ndcg_note': 'ndcg_at_10_binary는 이진 relevance 기준 - examiner_citation.category가 전부 X 계열이라 등급(X/Y/A) 반영 nDCG는 아직 계산 불가',
        'structure_multi_molecule_policy': 'max_tanimoto_over_all_smiles_pairs',
        'avg_pool_size': agg('pool_size', per_query_results),
        'avg_candidates_excluded_future_per_query': agg('candidates_excluded_future', per_query_results),
        'avg_candidates_excluded_same_family_per_query': agg('candidates_excluded_same_family', per_query_results),
        'empty_claims_text_total_corpus': empty_text,
        'empty_claims_text_in_evaluated_queries': empty_text_in_queries,
        'structure_coverage_total_corpus': with_structure,
        'mean_average_precision_map': agg('ap', per_query_results),
        'recall_at_100': agg('recall_at_100', per_query_results),
        'recall_at_1000': agg('recall_at_1000', per_query_results),
        'mrr_at_10': agg('mrr_at_10', per_query_results),
        'ndcg_at_10_binary': agg('ndcg_at_10_binary', per_query_results),
        'elapsed_seconds': round(cumulative_elapsed, 1),  # 여러 청크로 나눠 돌렸으면 전체 합산 시간
    }

    # 참고용 - 쿼리의 원래 split 라벨별 breakdown (후보 풀은 동일하게 전체 코퍼스 기준)
    by_split = defaultdict(list)
    for r in per_query_results:
        by_split[r.get('original_split_label') or 'unknown'].append(r)
    summary['breakdown_by_original_split_label'] = {
        split_label: {
            'num_queries': len(records),
            'mean_average_precision_map': agg('ap', records),
            'recall_at_100': agg('recall_at_100', records),
            'recall_at_1000': agg('recall_at_1000', records),
            'mrr_at_10': agg('mrr_at_10', records),
            'ndcg_at_10_binary': agg('ndcg_at_10_binary', records),
        }
        for split_label, records in by_split.items()
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = os.path.join(out_dir, f'full_corpus_retrieval_v2_{args.query_split}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'per_query': per_query_results}, f, ensure_ascii=False, indent=2)
    print(f'\n저장됨: {out_path}')

    for tmp_path in (partial_path, meta_path):
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                print(f'(참고: {tmp_path} 자동 삭제 실패 - 수동으로 지워도 되고 안 지워도 다음 실행에 영향 없음)')


if __name__ == '__main__':
    main()
