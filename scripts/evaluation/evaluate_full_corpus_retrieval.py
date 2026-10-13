"""Task 1 full-corpus retrieval evaluation. Rank all local patents published before query filing and outside the query family, rather than a few paired candidates:
candidate.pub_date <= query.app_date AND candidate.family_id != query.family_id.
The corpus itself is citation-pair-derived; unrelated uncited background patents are unavailable. Filter all patent_content records accordingly. Train/test labels provide reporting breakdowns, not candidate-pool selection; scoring is deterministic and untrained, with time filtering providing chronological protection.
Fuse Jaccard/overlap text and maximum molecule-pair Tanimoto as in rag_engine, weights text=0.5/structure=0.3 renormalized without vision. All current citation categories are X, so nDCG is binary, not graded.
Precompute tokens/fingerprints once per patent for potentially millions of query-corpus comparisons.
Usage: python3 scripts/evaluation/evaluate_full_corpus_retrieval.py [--time-budget-seconds 35] [--text-weight 0.5 --tanimoto-weight 0.3] [--selftest]. Time-bounded runs resume; --selftest checks scoring logic.
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

    split_of = {r['id']: r['split'] for r in split_rows}  # Reporting metadata only; not used to build the candidate pool.
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
    # Renormalize as in rag_engine/multimodal_ranker.py, excluding vision here.
    if tanimoto_score is not None:
        weight_sum = text_weight + tanimoto_weight
        return round((text_weight * text_score + tanimoto_weight * tanimoto_score) / weight_sum, 4)
    return round(text_score, 4)


def build_candidate_pool(query_id, all_patent_ids, pub_date_of, app_date_of, family_id_of):
    """Build a corpus-wide candidate pool excluding self, publications after query filing, and the same patent family. YYYYMMDD strings support lexicographic date comparison."""
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
    # ranked_pool: [(patent_id, score), ...], sorted descending.
    relevant_ranks = [i for i, (pid, _) in enumerate(ranked_pool) if pid in gold_set]
    if not relevant_ranks:
        return None

    num_relevant = len(gold_set)

    precisions = []
    hit_count = 0
    for rank in relevant_ranks:
        hit_count += 1
        precisions.append(hit_count / (rank + 1))
    # AP divides by all gold documents, not only retrieved relevant documents.
    # Unretrieved gold contributes zero, following standard Average Precision.
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
    """Minimal synthetic checks for pool filters and AP/Recall/MRR/nDCG without external data. AssertionError stops immediately on failure."""
    print('[selftest] Checking time/family candidate filters...')
    pub_date_of = {'Q': '20200101', 'A': '20190101', 'B': '20200601', 'C': '20191201', 'D': '20191001'}
    app_date_of = {'Q': '20200101'}
    family_id_of = {'Q': 'FAM1', 'A': 'FAM2', 'B': 'FAM2', 'C': 'FAM1', 'D': 'FAM3'}
    all_ids = ['Q', 'A', 'B', 'C', 'D']
    pool, excl_future, excl_family = build_candidate_pool('Q', all_ids, pub_date_of, app_date_of, family_id_of)
    # Exclude B, published 20200601 after query filing on 20200101.
    # Exclude C, sharing the query family FAM1.
    # Only A and D should remain.
    assert set(pool) == {'A', 'D'}, f'Unexpected candidate pool: {pool}'
    assert excl_future == 1 and excl_family == 1, f'Exclusion count mismatch: future={excl_future}, family={excl_family}'
    print('  OK - exactly 1 future patent and 1 same-family patent excluded')

    print('[selftest] Checking AP/Recall/MRR/nDCG calculations...')
    # Gold P2/P5 at ranks 2/5.
    ranked = [(f'P{i}', 1.0 - i * 0.01) for i in range(1, 11)]
    gold = {'P2', 'P5'}
    result = evaluate_query('Q', ranked, gold)
    # AP = ((1/2) + (2/5)) / 2 = 0.45 when both gold documents are retrieved.
    assert abs(result['ap'] - 0.45) < 1e-6, f"AP calculation error: {result['ap']}"

    # Partial retrieval: 1 of 2 gold documents at rank 1 -> AP = (1/1) / 2 = 0.5.
    partial = evaluate_query('Q', [('P2', 1.0)], {'P2', 'P5'})
    assert abs(partial['ap'] - 0.5) < 1e-6, f"Partial-retrieval AP error: {partial['ap']}"
    assert result['recall_at_100'] == 1.0
    # MRR@10: first gold at rank 2 -> 1/2.
    assert abs(result['mrr_at_10'] - 0.5) < 1e-6, f"MRR calculation error: {result['mrr_at_10']}"
    assert result['best_rank'] == 2
    print(f'  OK - AP={result["ap"]}, MRR@10={result["mrr_at_10"]}, nDCG@10={result["ndcg_at_10_binary"]}')

    print('[selftest] Checking None for queries with no gold in the pool...')
    result_none = evaluate_query('Q', ranked, {'NOT_IN_POOL'})
    assert result_none is None
    print('  OK')

    print('[selftest] Checking exclusion statistics and elapsed time across resumed runs...')
    # Time-bounded runs must aggregate exclusion statistics and elapsed time
    # across all chunks, not just the final chunk.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        partial_path = os.path.join(tmp, 'fake.partial.jsonl')
        meta_path = partial_path + '.meta.json'

        # Chunk 1: one query with 10 future exclusions; stop at cumulative 5.0 seconds.
        with open(partial_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'query_id': 'Q1', 'ap': 1.0, 'candidates_excluded_future': 10,
                                 'candidates_excluded_same_family': 2}) + '\n')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'cumulative_elapsed_seconds': 5.0}, f)

        # Chunk 2: append one query with 0 future exclusions, matching main() resume:
        # read previous results and append only this run's work.
        with open(partial_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'query_id': 'Q2', 'ap': 0.0, 'candidates_excluded_future': 0,
                                 'candidates_excluded_same_family': 0}) + '\n')
        this_run_elapsed = 3.0  # Assume chunk 2 takes 3 seconds.

        # Use the same final aggregation procedure as main().
        per_query_results = []
        with open(partial_path, 'r', encoding='utf-8') as f:
            for line in f:
                per_query_results.append(json.loads(line))
        with open(meta_path, 'r', encoding='utf-8') as f:
            cumulative_before = json.load(f)['cumulative_elapsed_seconds']
        cumulative_elapsed = cumulative_before + this_run_elapsed

        avg_future = sum(r['candidates_excluded_future'] for r in per_query_results) / len(per_query_results)
        assert avg_future == 5.0, f'Resumed exclusion statistics are not the overall mean: {avg_future}'
        assert cumulative_elapsed == 8.0, f'Incorrect cumulative elapsed time after resume: {cumulative_elapsed}'
    print('  OK - all chunks aggregated correctly on resume (avg_future=5.0, elapsed=8.0)')

    print('\n[selftest] All checks passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-split', choices=['train', 'test', 'all'], default='all',
                         help='Query subset for reporting only; candidate pool always uses the entire corpus')
    parser.add_argument('--text-weight', type=float, default=0.5)
    parser.add_argument('--tanimoto-weight', type=float, default=0.3,
                         help='Match rag_engine/multimodal_ranker.py defaults (text=0.5/structure=0.3/vision=0.2); '
                              'structure defaults to 0.3, giving 0.625:0.375 after excluding vision and renormalizing')
    parser.add_argument('--limit-queries', type=int, default=None,
                         help='Debug query limit for a small trial before a full run')
    parser.add_argument('--time-budget-seconds', type=float, default=None,
                         help='After this many seconds, finish the current query, save intermediate results, and stop; '
                              'split lengthy full-corpus runs into resumable chunks. '
                              'Completed queries accumulate in <out>.partial.jsonl and are skipped on resume')
    parser.add_argument('--selftest', action='store_true', help='Check scoring logic on synthetic data and exit')
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    print('[1/5] Loading db_export tables...')
    idx = build_indexes()
    all_patent_ids = idx['all_patent_ids']

    query_ids = [pid for pid in all_patent_ids if idx['gold_of'].get(pid)]
    if args.query_split != 'all':
        query_ids = [pid for pid in query_ids if idx['split_of'].get(pid) == args.query_split]
    if args.limit_queries:
        query_ids = query_ids[:args.limit_queries]
    print(f'  Full corpus: {len(all_patent_ids)}; eligible gold-bearing queries (query-split={args.query_split}) {len(query_ids)} records')

    print('[2/5] Precomputing text tokens...')
    tokenizer = ChemicalTokenizer()
    token_counters = precompute_token_counters(all_patent_ids, idx['claims_text_of'], tokenizer)
    empty_text = sum(1 for pid in all_patent_ids if token_counters[pid][1] == 0)
    empty_text_in_queries = sum(1 for pid in query_ids if token_counters[pid][1] == 0)
    print(f'  Patents with empty claims_text and zero tokens: total {empty_text}; among queries {empty_text_in_queries} records')

    print('[3/5] Precomputing chemical Morgan fingerprints...')
    fps_of = precompute_fingerprints(all_patent_ids, idx['smiles_of'])
    with_structure = sum(1 for pid in all_patent_ids if fps_of[pid])
    print(f'  Patents with SMILES: {with_structure}/{len(all_patent_ids)}')

    out_dir = os.path.normpath(os.path.join(DATA_DIR, '..', 'benchmark', 'results'))
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, f'full_corpus_retrieval_v2_{args.query_split}.partial.jsonl')
    # Auxiliary metadata accumulates elapsed time across chunks. Per-query results
    # already store exclusions; agg() computes means without extra counters.
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
        print(f'  Resuming completed queries from previous runs: {len(done_ids)} ({partial_path})')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                cumulative_elapsed_before_this_run = json.load(f).get('cumulative_elapsed_seconds', 0.0)

    remaining_query_ids = [q for q in query_ids if q not in done_ids]
    print(f'[4/5] Ranking the time/family-filtered corpus per query... (queries remaining {len(remaining_query_ids)})')
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
                print(f'  This run processed {processed_this_run}/{len(remaining_query_ids)} ({elapsed:.1f} seconds elapsed)')
    finally:
        partial_f.close()

    this_run_elapsed = time.time() - t0
    cumulative_elapsed = cumulative_elapsed_before_this_run + this_run_elapsed

    if stopped_early:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'cumulative_elapsed_seconds': round(cumulative_elapsed, 1)}, f)
        done_now = len(per_query_results)
        print(f'\n[TIME BUDGET REACHED] This run processed {processed_this_run} ({this_run_elapsed:.1f} seconds), '
              f'completed cumulatively: {done_now}; cumulative elapsed time {cumulative_elapsed:.1f} seconds. '
              f'Run the same command again to resume.')
        return

    print('[5/5] Aggregating results...')
    n = len(per_query_results)
    if n == 0:
        print('No evaluable queries: no gold documents are present in the corpus pool.')
        return

    def agg(key, records):
        return round(sum(r[key] for r in records) / len(records), 4) if records else None

    summary = {
        'candidate_pool_definition': 'All patent_content (4,123 records), excluding self, candidate.pub_date > query.app_date, and matching family_id',
        'query_split_filter': args.query_split,
        'candidate_queries_with_gold_label': len(query_ids),
        'excluded_gold_not_in_pool': excluded_gold_not_in_pool,
        'num_queries_evaluated': n,
        'text_weight': args.text_weight,
        'tanimoto_weight': args.tanimoto_weight,
        'vision_included': False,
        'ndcg_note': 'ndcg_at_10_binary uses binary relevance; examiner_citation.category contains only X variants, so graded X/Y/A nDCG is unavailable',
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
        'elapsed_seconds': round(cumulative_elapsed, 1),  # Sum elapsed time over all chunks.
    }

    # Original query split labels provide reporting breakdowns; the corpus pool stays identical.
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
    print(f'\nSaved: {out_path}')

    for tmp_path in (partial_path, meta_path):
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                print(f'(Note: {tmp_path} automatic deletion failed; manual removal is optional and does not affect the next run)')


if __name__ == '__main__':
    main()
