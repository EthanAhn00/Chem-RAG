"""Compare retrieval systems while separating algorithm and input-field effects. All systems share queries, gold, candidate pools, filters, and limits.
A_jaccard_claims: lexical Jaccard/overlap, claims only.
B_bm25_claims: BM25, claims only.
C_bm25_title_claims: BM25, title+claims.
D_dense_claims: dense, claims only.
E_dense_title_claims: dense, title+claims.
F_hybrid_title_claims: BM25+dense RRF, title+claims.
G_hybrid_rerank_title_claims: F plus RDKit exact Tanimoto reranking.
A->B isolates Jaccard/BM25; B->C and D->E isolate title inclusion; B->D and C->E compare semantic dense representations at fixed inputs; C,E->F isolates fusion and F->G structure reranking. Qdrant is the storage/execution layer, not the source of algorithmic improvements.
AP = sum(precision at gold ranks) / all gold; missing gold contributes zero.
All systems truncate to --search-limit (default 1000), so report MAP@{cutoff}, not untruncated MAP. mean_average_precision_at_{cutoff} is the canonical JSON key; mean_average_precision_map remains a deprecated equal-valued alias. Standalone evaluate_full_corpus_retrieval.py ranks the entire pool and reports untruncated MAP.
Usage: python3 scripts/evaluation/evaluate_qdrant_retrieval.py --systems all --workers 4 or --selftest.
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
    """AP = sum(precision at gold ranks) / all gold documents. Missing results contribute zero, including gold omitted by BM25 due to zero lexical overlap."""
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
    """MAP summary key; default cutoff=1000 gives 'mean_average_precision_at_1000'."""
    return f'mean_average_precision_at_{int(cutoff)}'


def build_system_summary(per_query, algorithm, field, cutoff):
    """Aggregate per-query records into a system summary. Shared by live retrieval and --recompute-only to avoid duplicated aggregation logic."""
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
        # Deprecated alias, always equal to mkey; retain for backward compatibility.
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

    # Legacy result-schema identifiers, not UI labels. Keep byte-identical keys
    # for compatibility with stored reports and their readers.
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
    """Compare summary with aggregation from .partial.jsonl, including equality of mkey and the deprecated alias to detect regressions."""
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
    """Determine --recompute-only cutoff from per-query cutoff metadata first. For legacy partials without it, use historical summary config.search_limit (fallback_cutoff). If neither exists, return (None, reason) rather than guessing so callers stop. Return (cutoff:int|None, problem:str|None)."""
    recorded = {r['cutoff'] for r in per_query if r.get('cutoff') is not None}
    if len(recorded) > 1:
        return None, f'Mixed cutoffs in per-query records: {sorted(recorded)}'
    if len(recorded) == 1:
        return recorded.pop(), None
    if fallback_cutoff is not None:
        return fallback_cutoff, None
    return None, ('Legacy per-query records have no cutoff metadata and no previous summary is available; '
                  'the original cutoff cannot be verified - stopping without guessing')


def run_recompute_only(args):
    """Rebuild summary/effect_decomposition/consistency from existing .partial.jsonl (1,106 query results), without Qdrant access or retrieval. Metric renaming does not require a full rerun. Carry forward DB-dependent collection_counts, dense_encoder_by_field, encoder_signature_check, and operational_latency_benchmark with an explicit note that they were not reverified in this run."""
    systems = ALL_SYSTEMS if args.systems == 'all' else tuple(s.strip() for s in args.systems.split(','))
    for s in systems:
        if s not in SYSTEMS:
            raise SystemExit(f'Unknown system: {s} (available: {ALL_SYSTEMS})')
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
            print(f'  Original backup: {backup_path}')

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
                f'[{system}] No previous per-query results: {partial_path}(.wN) - '
                f'--recompute-only does not rerun retrieval; a prior full run is required.')
        per_query_by_system[system] = pq
        resolved_cutoff, problem = resolve_recompute_cutoff(pq, fallback_cutoff)
        if problem is not None:
            cutoff_problems.append(f'[{system}] {problem}')
        elif resolved_cutoff != cutoff:
            cutoff_problems.append(
                f'[{system}] Previous results were already truncated to top-{resolved_cutoff}, '
                f'so they cannot be used to recompute MAP@{cutoff} (requested cutoff={cutoff}). '
                f'top-{resolved_cutoff} is the stored retrieval limit; gold beyond it was never retrieved. A top-{resolved_cutoff} '
                f' partial cannot provide a wider cutoff; a narrower cutoff is also unsupported because '
                f'it need not reproduce the ranks originally returned at that depth. '
                f'MAP@{cutoff} requires a full system rerun with --search-limit {cutoff} rather than recomputation '
                f'(omit --recompute-only).')
    if cutoff_problems:
        raise SystemExit('[STOP] --recompute-only cutoff mismatch/unverified:\n  - '
                         + '\n  - '.join(cutoff_problems))

    for system in systems:
        algorithm, field = SYSTEMS[system]
        pq = per_query_by_system[system]
        summaries[system] = build_system_summary(pq, algorithm, field, cutoff)
        print(f'  [{system}] {mkey}={summaries[system][mkey]} (n={len(pq)}, reaggregated from existing files)')

    effects = build_effects(summaries, mkey)
    consistency, bad = build_consistency(summaries, args.out_prefix, mkey)

    src_paths = [os.path.join(DATA_DIR, 'db_export', f) for f in
                 ('patent_content.csv', 'split.csv', 'examiner_citation.csv',
                  'molecule.csv', 'patent_molecule.csv')]
    src_ver, src_detail = source_version(src_paths)

    report = dict(old_report) if old_report else {}
    report['recomputed_from_partial'] = True
    report['recompute_note'] = (
        f'MAP@{cutoff} metric renaming: using existing .partial.jsonl (per system {[len(load_partial(os.path.join(RESULTS_DIR, f"{args.out_prefix}_{s}.partial.jsonl"))[0]) for s in systems]} queries), '
        f'regenerated only summary/effect_decomposition/consistency without rerunning retrieval. '
        f'collection_counts/dense_encoder_by_field/encoder_signature_check/'
        f'operational_latency_benchmark carried forward without revalidation in this run.')
    report['previous_provenance'] = (old_report or {}).get('provenance')
    report['provenance'] = runtime_provenance(load_settings(), ' '.join(sys.argv))
    report['source_version'] = src_ver
    report['source_csv_checksums'] = src_detail
    report.setdefault('config', {})
    report['config'] = dict((old_report or {}).get('config', {}))
    report['config']['metric_name'] = f'MAP@{cutoff}'
    report['config']['cutoff'] = cutoff
    report['config']['ap_definition'] = (
        f'AP@{cutoff} = sum(precision at gold ranks within top {cutoff}) / all gold documents. '
        f'{cutoff} or absent from retrieval contribute zero. '
        f'mean_average_precision_map is a deprecated alias of mean_average_precision_at_{cutoff} (always equal).')
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

    print('\n[recompute-only] Results')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print('\nEffect decomposition (' + mkey + ' basis): ' + json.dumps(effects, ensure_ascii=False, indent=1))
    print(f'\nSummary/per-query consistency: {"OK" if not bad else "mismatch " + str(bad)}')
    print(f'Saved: {old_path}')
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

    empty = evaluate_ranking([], gold)                 # No search results at all.
    assert empty['ap'] == 0.0 and empty['retrieved'] == 0
    print('[selftest] AP definition (sum(P)/all gold), partial/missing retrieval, and empty-result checks OK')

    # --- MAP@cutoff naming and truncation semantics ---
    assert metric_key_for_cutoff(1000) == 'mean_average_precision_at_1000'
    assert metric_key_for_cutoff(500) == 'mean_average_precision_at_500'

    # Gold beyond rank 1000 contributes zero. Check the contract through
    # build_system_summary, assuming callers already truncate results to cutoff.
    gold2 = {'P1', 'P1500'}                      # Assume P1500 lies outside the top 1000.
    ranked_full = [f'P{i}' for i in range(1, 1501)]
    ap_uncut = evaluate_ranking(ranked_full, gold2)['ap']          # Untruncated input retrieves both gold documents.
    ap_cut = evaluate_ranking(ranked_full[:1000], gold2)['ap']      # Top 1000 input omits P1500.
    assert ap_uncut > ap_cut, (ap_uncut, ap_cut)
    assert abs(ap_cut - 0.5) < 1e-9, ap_cut          # Only P1 retrieved: (1/1)/2 = 0.5; P1500 contributes zero.

    # Check that build_system_summary gives mkey and alias identical values.
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
    print('[selftest] MAP@cutoff naming/truncation (zero contribution beyond rank 1000) and summary alias consistency OK')


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
    """Worker: retrieve assigned queries and write its own file. Embedded storage uses exclusive locks, so each worker gets a separate copy."""
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
                    help='All systems return/evaluate the same top N results, including truncation of the exhaustive Jaccard baseline. '
                         'Use at least 1000 to measure R@1000.')
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
                    help='Without connecting to Qdrant or rerunning search, rebuild '
                         'summary/effect_decomposition from existing .partial.jsonl '
                         'for changes such as metric renaming that do not require reevaluation.')
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return 0

    if args.recompute_only:
        return run_recompute_only(args)

    systems = ALL_SYSTEMS if args.systems == 'all' else tuple(s.strip() for s in args.systems.split(','))
    for s in systems:
        if s not in SYSTEMS:
            raise SystemExit(f'Unknown system: {s} (available: {ALL_SYSTEMS})')

    settings = load_settings()

    print('[1/5] Loading db_export indexes...')
    idx = build_indexes()
    all_patent_ids = idx['all_patent_ids']
    title_of = {r['patent_id']: (r.get('title') or '') for r in load_csv('patent_content')}
    query_ids = [pid for pid in all_patent_ids if idx['gold_of'].get(pid)]
    if args.limit_queries:
        query_ids = query_ids[: args.limit_queries]

    def field_text(pid, field):
        text, _ = build_field_text(title_of.get(pid), idx['claims_text_of'].get(pid), field)
        return text

    print('[2/5] Precomputing candidate pools/gold shared by all systems...')
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
    print(f'  Evaluation queries: {len(eval_query_ids)} (excluded because gold is absent from pool: {excluded}); mean pool size {avg_pool}')

    need_jaccard = any(SYSTEMS[s][0] == 'jaccard' for s in systems)
    need_qdrant = any(SYSTEMS[s][0] != 'jaccard' for s in systems)
    fields_used = sorted({SYSTEMS[s][1] for s in systems if SYSTEMS[s][0] != 'jaccard'})

    token_counters = fps_of = None
    if need_jaccard:
        print('[3/5] Precomputing Jaccard baseline tokens/fingerprints...')
        token_counters = precompute_token_counters(all_patent_ids, idx['claims_text_of'],
                                                   ChemicalTokenizer())
        fps_of = precompute_fingerprints(all_patent_ids, idx['smiles_of'])

    retriever = structure_repo = smiles_index = None
    dense_encoders, sparse_encoders = {}, {}
    encoder_signature_check = None
    if need_qdrant:
        print('[4/5] Connecting to Qdrant and preparing field-specific encoders...')
        client = get_client(settings)
        for field in fields_used:
            if not os.path.exists(bm25_state_path(field)):
                raise SystemExit(f'{bm25_state_path(field)} missing - run scripts/qdrant/qdrant_index_patents.py first')
            sparse_encoders[field] = BM25SparseEncoder.load(bm25_state_path(field))
            dense_encoders[field] = build_dense_encoder(settings, idf_state_path=bm25_state_path(field))
            print(f'  {field}: dense={dense_encoders[field].describe()["model"]} '
                  f'sig={dense_encoders[field].signature()}')
        retriever = PatentRetriever(client, settings, dense_encoders, sparse_encoders)
        if any(SYSTEMS[s][0] == 'hybrid_rerank' for s in systems):
            structure_repo = StructureRepository(client, settings)
            smiles_index = structure_repo.load_smiles_index()
            print(f'  Patents with structures: {len(smiles_index)} records')

        # Verify indexing encoder == search encoder.
        sample = client.scroll(settings.patents_collection, limit=1, with_payload=True)[0]
        encoder_signature_check = {}
        if sample:
            stored = (sample[0].payload or {}).get('embedding_signature_by_field') or {}
            for field in fields_used:
                got, want = dense_encoders[field].signature(), stored.get(field)
                encoder_signature_check[field] = {'indexed': want, 'search': got, 'match': got == want}
                if want and got != want:
                    raise SystemExit(f'Encoder mismatch ({field}): indexed {want} != search {got}. Reindexing required.')
            print(f'  Encoder signatures match: {encoder_signature_check}')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()
    stopped_early = False
    summaries = {}

    for system in systems:
        algorithm, field = SYSTEMS[system]
        partial_path = os.path.join(RESULTS_DIR, f'{args.out_prefix}_{system}.partial.jsonl')
        per_query, done = load_partial(partial_path)
        remaining = [q for q in eval_query_ids if q not in done]
        print(f'\n[{system}] algorithm={algorithm} field={field} | remaining {len(remaining)}/{len(eval_query_ids)}')

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
            print(f'  Parallel workers: {len(tasks)} entries')
            try:
                with mp.get_context('fork').Pool(len(tasks)) as pool:
                    print(f'  Per-worker processing: {pool.map(_worker_run, tasks)}')
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)
            per_query, done = load_partial(partial_path)
            if len(done) < len(eval_query_ids):
                print(f'  [INCOMPLETE] {len(done)}/{len(eval_query_ids)} - run again to resume.')
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
                        # Truncate to the same depth as all systems for a fair comparison.
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
            print('\n[TIME BUDGET REACHED] Run the same command again to resume.')
            return 0

        mkey = metric_key_for_cutoff(args.search_limit)
        summaries[system] = build_system_summary(per_query, algorithm, field, args.search_limit)
        print(f'  {mkey}={summaries[system][mkey]} '
              f'R@10={summaries[system]["recall_at_10"]} R@100={summaries[system]["recall_at_100"]}')

    latency_bench = {}
    if need_qdrant and not stopped_early:
        print('\n[EXTRA] Operational latency benchmark (top-k, single process)...')
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
                                     'note': 'Retrieval latency excluding structure reranking'}

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
            'query_set': 'Has examiner_citation gold with at least one gold document in the candidate pool',
            'num_candidate_queries_with_gold': len(query_ids),
            'excluded_gold_not_in_pool': excluded,
            'num_queries_evaluated': len(eval_query_ids),
            'avg_pool_size': avg_pool,
            'filters': ['candidate.pub_date <= query.app_date',
                        'candidate.family_id != query.family_id',
                        'candidate.patent_id != query.patent_id'],
            'search_limit': args.search_limit,
            'search_depth_note': 'All systems, including the Jaccard baseline, return/evaluate the same top search_limit '
                                 'results. Gold outside that depth contributes zero to AP/Recall.',
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
                'Qdrant is the storage/execution layer. Differences reflect scoring algorithms/input fields/'
                'embedding models/fusion settings, not introduction of Qdrant itself.',
                'nDCG is binary because all examiner_citation categories are X variants.',
                'With workers>1, per-query latency reflects process contention; '
                'use operational_latency_benchmark (single process).',
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

    print('\n[5/5] Results (' + mkey + ')')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f'\nEffect decomposition ({mkey} basis): ' + json.dumps(effects, ensure_ascii=False, indent=1))
    print(f'\nSummary/per-query consistency: {"OK" if not bad else "mismatch " + str(bad)}')
    print(f'Saved: {out}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
