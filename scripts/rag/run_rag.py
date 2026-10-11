#!/usr/bin/env python3
"""End-to-end RAG — four experimental retrievers + BM25 baseline + two
reference conditions.

  T1_nv  T1_qwen  T2_morgan  T3_m4  BM25   ← retrievers (precomputed rankings)
  oracle  closedbook                       ← reference conditions, not guaranteed bounds

Judgments are cached under the (query, candidate) key and shared across
conditions. Use a fresh output/cache directory when changing the run settings.

Examples
  python -m scripts.rag.run_rag --retriever all --task citation --k 50
  python -m scripts.rag.run_rag --retriever all --task citation --k 50 --sample-neg 5
  python -m scripts.rag.run_rag --retriever T1_nv --task conflict_x --k 50
  python -m scripts.rag.run_rag --plan --retriever all --task citation --k 50
"""
import argparse, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from . import config, data, prompts

PLAN_CACHE = {}
from .llm import LLM

CACHE_PATH = os.path.join(config.CACHE, "judgments.jsonl")
_lock = threading.Lock()


# ───────────────────────── judgment cache ─────────────────────────
def load_cache():
    c = {}
    if os.path.exists(CACHE_PATH):
        for line in open(CACHE_PATH):
            try:
                d = json.loads(line)
                c[(d["q"], d["c"])] = d["v"]
            except Exception:
                pass
    return c


def cache_put(f, q, c, v):
    with _lock:
        f.write(json.dumps({"q": q, "c": c, "v": v}, ensure_ascii=False) + "\n")
        f.flush()


# ───────────────────────── candidate construction ─────────────────────────
def candidates_for(name, q, k, task, rankings, rng):
    if name == "closedbook":
        return None                                   # no candidates — separate path
    if name == "oracle":
        g = sorted(data.gold(q, task) & data.candidates())
        pool = [p for p in sorted(data.candidates()) if p != q and p not in g]
        rng.shuffle(pool)
        c = g + pool[:max(0, k - len(g))]
        rng.shuffle(c)
        return c[:k]
    R = rankings[name]
    return R[q]["top"][:k] if q in R else []


def pick_negatives(cands, gold_set, sample_neg, rng):
    """Choose which candidates to judge. With sample_neg>0, only a sample of
    the non-gold candidates is judged."""
    if not sample_neg:
        return cands, 1.0
    pos = [c for c in cands if c in gold_set]
    neg = [c for c in cands if c not in gold_set]
    rng.shuffle(neg)
    take = neg[:sample_neg]
    scale = len(neg) / len(take) if take else 1.0      # correction factor for FP estimates
    return pos + take, scale


# ───────────────────────── run ─────────────────────────
def run(name, task, k, n_queries, workers, llm, sample_neg, plan_only=False):
    rankings = {n: data.load_rankings(n) for n in config.RETRIEVERS}
    qs = data.eval_queries(task)
    if name in config.RETRIEVERS:
        qs = [q for q in qs if q in rankings[name]]
    if n_queries:
        qs = qs[:n_queries]
    folds = data.load_folds()
    rng = random.Random(config.SEED)

    cache = PLAN_CACHE if plan_only else load_cache()
    plan = []                                           # (q, [candidates to judge], scale, all candidates)
    for q in qs:
        if name == "closedbook":
            plan.append((q, None, 1.0, None)); continue
        cands = candidates_for(name, q, k, task, rankings, rng)
        sel, scale = pick_negatives(cands, data.gold(q, task), sample_neg, rng)
        plan.append((q, sel, scale, cands))

    if name == "closedbook":
        need = len(qs)
    else:
        need = sum(1 for q, sel, _, _ in plan for c in sel if (q, c) not in cache)
        if plan_only:                      # record into the cache so later conditions can reuse
            for q, sel, _, _ in plan:
                for c in sel: cache[(q, c)] = True
    total = len(qs) if name == "closedbook" else sum(len(sel) for _, sel, _, _ in plan)
    print(f"[{name:10s}] {task} k={k} | queries {len(qs):,} | to judge {total:,} "
          f"| cache hits {total-need:,} | new calls {need:,}", file=sys.stderr)
    if plan_only:
        return need

    os.makedirs(config.OUT, exist_ok=True); os.makedirs(config.CACHE, exist_ok=True)
    out = os.path.join(config.OUT, f"rag_{name}_{task}_k{k}.jsonl")
    done = {json.loads(l)["query_id"] for l in open(out) if l.strip()} if os.path.exists(out) else set()
    plan = [p for p in plan if p[0] not in done]
    meta = dict(retriever=name, desc=config.RETRIEVERS.get(name, (None,)*4)[3] or config.REFERENCE.get(name),
                task=task, k=k, model=llm.model, thinking=llm.thinking,
                sample_neg=sample_neg, n_eval=len(qs))
    if not plan:
        print(f"  already complete -> {out}", file=sys.stderr); return 0

    cf = open(CACHE_PATH, "a")

    def judge(q, cand):
        hit = cache.get((q, cand))
        if hit is not None:
            return dict(cited_id=cand, cached=True, **hit)
        qt, ct = data.patent_text(q), data.patent_text(cand)
        if ct is None:
            return dict(cited_id=cand, pred=None, error="text missing")
        r = llm.chat(prompts.system_prompt(), prompts.user_prompt(qt, ct))
        p = prompts.parse_json(r["text"]) or {}
        v = dict(pred=prompts.normalize(p.get("label")),
                 rel_claims=[int(x) for x in (p.get("relevant_query_claims") or []) if str(x).isdigit()],
                 evidence_location=p.get("evidence_location"), parse_ok=bool(p),
                 latency=r["latency"], completion_tokens=r["completion_tokens"])
        cache_put(cf, q, cand, v)
        with _lock: cache[(q, cand)] = v
        return dict(cited_id=cand, cached=False, **v)

    def work(item):
        q, sel, scale, cands = item
        g_all, g_x = data.gold(q, "citation"), data.gold(q, "conflict_x")
        base = dict(query_id=q, fold=folds[q]["fold"],
                    gold_citation=sorted(g_all), gold_conflict_x=sorted(g_x), meta=meta)
        if name == "closedbook":
            r = llm.chat(prompts.system_prompt(), prompts.closedbook_prompt(data.patent_text(q), config.CLOSEDBOOK_N))
            p = prompts.parse_json(r["text"]) or {}
            docs, labels = [], {}
            for d in (p.get("documents") or []):
                if not isinstance(d, dict): continue
                pid = re.sub(r"\s", "", str(d.get("publication_number", ""))).upper()
                if pid: docs.append(pid); labels[pid] = prompts.normalize(d.get("label")) or "CITED"
            docs = docs[:config.CLOSEDBOOK_N]
            judged = [dict(cited_id=d, pred=labels.get(d, "CITED"),
                           gold=("X" if d in g_x else "CITED" if d in g_all else "NOT_CITED")) for d in docs]
            return dict(**base, candidates=docs, judged=judged, scale=1.0,
                        confidence=p.get("confidence"), raw=r["text"])
        judged = [judge(q, c) for c in sel]
        for j in judged:
            j["gold"] = "X" if j["cited_id"] in g_x else ("CITED" if j["cited_id"] in g_all else "NOT_CITED")
        return dict(**base, candidates=cands, judged=judged, scale=scale,
                    gold_ranks=(rankings[name][q]["ranks"] if name in config.RETRIEVERS else {}))

    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(work, it): it[0] for it in plan}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=name):
            try: rec = fut.result()
            except Exception as e: rec = dict(query_id=futs[fut], error=repr(e), meta=meta)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    cf.close()
    print(f"  saved -> {out}", file=sys.stderr)
    return need


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="all",
                    help="T1_nv|T1_qwen|T2_morgan|T3_m4|oracle|closedbook|all|experiment")
    ap.add_argument("--task", default="citation", choices=["citation", "conflict_x"])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n", "--n-queries", dest="n", type=int, default=None)
    ap.add_argument("--sample-neg", type=int, default=0,
                    help="judge N non-gold candidates per query (0=all); precision uses weighted FP counts")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--model"); ap.add_argument("--base-url"); ap.add_argument("--backend")
    ap.add_argument("--plan", action="store_true", help="estimate call counts without running")
    ap.add_argument("--calls-per-second", type=float, default=None,
                    help="optional user-measured throughput for a planning-only time estimate")
    a = ap.parse_args()
    if a.calls_per_second is not None and a.calls_per_second <= 0:
        ap.error("--calls-per-second must be positive")

    names = (config.ALL_CONDITIONS if a.retriever == "all"
             else list(config.RETRIEVERS) if a.retriever == "experiment"
             else [a.retriever])
    llm = None if a.plan else LLM(base_url=a.base_url, model=a.model, backend=a.backend, thinking=a.thinking)
    tot = 0
    for n in names:
        tot += run(n, a.task, a.k, a.n, a.workers, llm, a.sample_neg, a.plan)
    if a.plan:
        print(f"\ntotal planned new calls {tot:,}", file=sys.stderr)
        if a.calls_per_second is not None:
            print(f"estimated time {tot/a.calls_per_second/3600:.1f} h "
                  f"at user-supplied {a.calls_per_second:g} calls/s", file=sys.stderr)
