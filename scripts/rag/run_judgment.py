#!/usr/bin/env python3
"""Judgment layer — retrieval is skipped and the examiner-cited document is
provided directly; the model only assigns the citation category.

The label is 1 if data.is_x finds an exact comma-separated X token, else 0.
Counts are obtained from the configured input files at execution time.

Four conditions
  base      default prompt
  think     Qwen3 thinking enabled            (does reasoning help?)
  nodefs    EPO category definitions removed  (does stating the criteria help?)
  3class    ask for X/Y/A instead of X/NOT_X  (does a finer question help?)

Examples
  python -m scripts.rag.run_judgment --cond all --workers 16
  python -m scripts.rag.run_judgment --cond base --n 50
Output: CHEMRAG_OUT/judge_<cond>.jsonl (existing pair IDs are skipped)
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from . import config, data, prompts
from .llm import LLM

CONDS = {
    "base":   dict(scheme="binary", defs=True,  thinking=False, desc="base"),
    "think":  dict(scheme="binary", defs=True,  thinking=True,  desc="thinking"),
    "nodefs": dict(scheme="binary", defs=False, thinking=False, desc="no defs"),
    "3class": dict(scheme="3class", defs=True,  thinking=False, desc="3-class (X/Y/A)"),
}


def run(cond, n, workers, all_pairs, base_llm_kw):
    c = CONDS[cond]
    llm = LLM(thinking=c["thinking"], **base_llm_kw)
    pairs = data.judgment_pairs(restrict_eval=not all_pairs)
    if n: pairs = pairs[:n]

    os.makedirs(config.OUT, exist_ok=True)
    out = os.path.join(config.OUT, f"judge_{cond}.jsonl")
    done = {json.loads(l)["pair_id"] for l in open(out) if l.strip()} if os.path.exists(out) else set()
    todo = [p for p in pairs if p["pair_id"] not in done]
    meta = dict(cond=cond, desc=c["desc"], scheme=c["scheme"], examiner_defs=c["defs"],
                thinking=c["thinking"], model=llm.model, n_pairs=len(pairs),
                n_positive=sum(x["label"] for x in pairs))
    print(f"[{cond:7s}] {c['desc']:16s} pairs {len(pairs):,} (done {len(done):,}, todo {len(todo):,})",
          file=sys.stderr)
    if not todo:
        return out

    sysmsg = prompts.judge_system(c["defs"])

    def work(p):
        qt, ct = data.patent_text(p["query_id"]), data.patent_text(p["cited_id"])
        if ct is None or qt is None:
            return dict(**p, pred=None, error="text missing", meta=meta)
        r = llm.chat(sysmsg, prompts.judge_user(qt, ct, c["scheme"]))
        j = prompts.parse_json(r["text"]) or {}
        return dict(**p,
                    pred=prompts.judge_normalize(j.get("category"), c["scheme"]),
                    pred_raw=j.get("category"),
                    rel_claims=[int(x) for x in (j.get("relevant_query_claims") or []) if str(x).isdigit()],
                    evidence_location=j.get("evidence_location"), rationale=j.get("rationale"),
                    parse_ok=bool(j), latency=r["latency"],
                    completion_tokens=r["completion_tokens"], meta=meta)

    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(work, p): p["pair_id"] for p in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=cond):
            try: rec = fut.result()
            except Exception as e: rec = dict(pair_id=futs[fut], error=repr(e), meta=meta)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    print(f"  saved -> {out}", file=sys.stderr)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="all", help="base|think|nodefs|3class|all")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--all-pairs", action="store_true", help="all input GT pairs, without fold-membership restriction")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model"); ap.add_argument("--base-url"); ap.add_argument("--backend")
    a = ap.parse_args()
    kw = dict(base_url=a.base_url, model=a.model, backend=a.backend)
    for cond in (list(CONDS) if a.cond == "all" else [a.cond]):
        run(cond, a.n, a.workers, a.all_pairs, kw)
