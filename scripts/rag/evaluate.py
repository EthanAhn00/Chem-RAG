#!/usr/bin/env python3
"""Evaluation — retrieval-only and end-to-end.

Retrieval metrics are computed from stored rankings; no LLM is called.

  python -m scripts.rag.evaluate --retrieval
  python -m scripts.rag.evaluate --retrieval --task conflict_x
  python -m scripts.rag.evaluate --judgment
  python -m scripts.rag.evaluate 'structure_retrieval_outputs/rag/results/rag_*.jsonl'
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict

from . import config, data

KS = (1, 5, 10, 20, 50, 100)
KS_MAIN = (50,)   # the main experiment uses k=50 only


# ───────────────────────── retrieval only ─────────────────────────
def retrieval_metrics(name, task, ks=KS):
    R = data.load_rankings(name)
    qs = [q for q in data.eval_queries(task) if q in R]
    folds = data.load_folds()
    per_fold = defaultdict(lambda: defaultdict(list))
    agg = defaultdict(list)
    maxrank = max((r for q in qs for r in R[q]["ranks"].values()), default=0)

    for q in qs:
        g = data.gold(q, task)
        if not g: continue
        ranks = sorted(R[q]["ranks"].get(c, math.inf) for c in g)
        f = folds[q]["fold"]
        for k in ks:
            hit = sum(1 for r in ranks if r <= k)
            rec = hit / len(g)
            agg[f"R@{k}"].append(rec); per_fold[f][f"R@{k}"].append(rec)
        rr = 1.0 / ranks[0] if ranks[0] != math.inf else 0.0
        agg["MRR"].append(rr); per_fold[f]["MRR"].append(rr)
        # nDCG@10 (binary relevance)
        dcg = sum(1 / math.log2(r + 1) for r in ranks if r <= 10)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))
        agg["nDCG@10"].append(dcg / idcg if idcg else 0.0)
        per_fold[f]["nDCG@10"].append(dcg / idcg if idcg else 0.0)

    mean = {m: sum(v) / len(v) for m, v in agg.items()}
    fold_means = {m: [sum(per_fold[f][m]) / len(per_fold[f][m]) for f in sorted(per_fold)] for m in agg}
    std = {m: (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
           for m, v in fold_means.items() if len(v) > 1}
    return dict(n=len(qs), maxrank=maxrank, mean=mean, fold_std=std)


def print_retrieval(task):
    print(f"\n{'='*104}")
    print(f"Retrieval performance — task: {task}  "
          f"(eligible queries {len(data.eval_queries(task))}; query mean ± sample SD of fold means)")
    print("=" * 104)
    hdr = f"{'condition':11s} {'n':>5s} " + " ".join(f"{'R@'+str(k):>12s}" for k in KS) + f" {'MRR':>12s} {'nDCG@10':>12s}"
    print(hdr); print("-" * 104)
    for name in config.RETRIEVERS:
        r = retrieval_metrics(name, task)
        cells = []
        for k in KS:
            m = r["mean"][f"R@{k}"]; s = r["fold_std"].get(f"R@{k}", 0)
            cells.append(f"{m:.3f}±{s:.3f}" if (name != "T3_m4" or k <= 100) else "     —      ")
        for m_ in ("MRR", "nDCG@10"):
            cells.append(f"{r['mean'][m_]:.3f}±{r['fold_std'].get(m_,0):.3f}")
        print(f"{name:11s} {r['n']:5d} " + " ".join(f"{c:>12s}" for c in cells))
    print("-" * 104)
    print("* T3_m4 rankings only store the top-100; gold beyond that counts as not retrieved.")


# ───────────────────────── end-to-end ─────────────────────────
def e2e(path):
    """End-to-end evaluation.

    With --sample-neg, only a sample of the non-gold candidates was judged, so
    precision scales false positives by the correction factor to the full
    candidate set. The reported FP rate is a pooled within-sample proportion;
    with unequal per-query sampling fractions it need not equal the full-pool rate.
    """
    recs = [json.loads(l) for l in open(path) if l.strip() and '"error"' not in l[:60]]
    if not recs: return None
    meta = recs[0]["meta"]; task = meta["task"]; k = meta["k"]
    is_cb = meta["retriever"] == "closedbook"
    pos  = (lambda p: p == "X") if task == "conflict_x" else (lambda p: p in ("X", "CITED"))
    gpos = (lambda g: g == "X") if task == "conflict_x" else (lambda g: g in ("X", "CITED"))

    n = len(recs); ret_hit = found = 0
    tp = fn = tn = 0; fp_raw = 0.0; fp_scaled = 0.0
    parse_fail = 0; lat = []; cached = 0
    for r in recs:
        gset = set(r["gold_conflict_x"] if task == "conflict_x" else r["gold_citation"])
        ret_hit += bool(gset & set(r.get("candidates") or []))
        sc = r.get("scale", 1.0)
        hit = False
        for j in r["judged"]:
            if j.get("cached"): cached += 1
            if j.get("pred") is None: parse_fail += 1; continue
            if j.get("latency"): lat.append(j["latency"])
            g, p = gpos(j["gold"]), pos(j["pred"])
            if g and p: tp += 1; hit = True
            elif g: fn += 1
            elif p: fp_raw += 1; fp_scaled += sc
            else: tn += 1
        found += hit
    prec = tp / (tp + fp_scaled) if tp + fp_scaled else 0
    rec_ = tp / (tp + fn) if tp + fn else 0
    out = {"retriever": meta["retriever"], "task": task, "k": ("—" if is_cb else k),
           "n_queries": n,
           "retrieval R@k": ret_hit / n, "final found": found / n,
           "judged|retrieved": (found / ret_hit) if ret_hit else float("nan"),
           "cand_P": prec, "cand_R": rec_,
           "cand_F1": (2 * prec * rec_ / (prec + rec_)) if prec + rec_ else 0,
           "FP rate (non-gold)": fp_raw / (fp_raw + tn) if fp_raw + tn else float("nan"),
           "neg sample": meta.get("sample_neg") or "all",
           "parse_fail": parse_fail, "cache hits": cached,
           "avg latency(s)": sum(lat) / len(lat) if lat else 0}
    out["file"] = os.path.basename(path)
    return out


# ───────────────────────── judgment layer ─────────────────────────
def judgment(path):
    """Evaluation under the binary labels stored in the judgment results.
    The baseline is the majority class: always non-X."""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if "error" not in r and r.get("pred") is not None]
    if not recs: return None
    m = recs[0]["meta"]
    tp = sum(1 for r in recs if r["label"] == 1 and r["pred"] == "X")
    fn = sum(1 for r in recs if r["label"] == 1 and r["pred"] != "X")
    fp = sum(1 for r in recs if r["label"] == 0 and r["pred"] == "X")
    tn = sum(1 for r in recs if r["label"] == 0 and r["pred"] != "X")
    n = tp + fn + fp + tn
    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    lat = [r["latency"] for r in recs if r.get("latency")]
    tok = [r["completion_tokens"] for r in recs if r.get("completion_tokens")]
    return {"condition": m["desc"], "n": n, "pred X": f"{(tp+fp)/n:.1%}",
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "accuracy": acc, "kappa": kappa, "MCC": mcc,
            "parse_fail": m["n_pairs"] - len(recs),
            "latency(s)": sum(lat) / len(lat) if lat else 0,
            "tokens": int(sum(tok) / len(tok)) if tok else 0}


def print_judgment(paths):
    pairs = data.judgment_pairs()
    P = sum(x["label"] for x in pairs); N = len(pairs) - P
    rows = [{"condition": "baseline (always non-X)", "n": len(pairs), "pred X": "0.0%", "recall": 0.0,
             "specificity": 1.0, "precision": 0.0, "accuracy": N / len(pairs), "kappa": 0.0, "MCC": 0.0,
             "parse_fail": 0, "latency(s)": 0, "tokens": 0}]
    rows += [r for r in (judgment(p) for p in paths) if r]
    cols = ["condition", "n", "pred X", "recall", "specificity", "precision", "accuracy",
            "kappa", "MCC", "parse_fail", "latency(s)", "tokens"]
    print(f"\n{'='*130}")
    print(f"Judgment layer — gold document given (label: exact comma-separated X token = 1)   "
          f"pairs {len(pairs):,} · X {P} ({P/len(pairs):.1%})")
    print("=" * 130)
    print(f"{cols[0]:24s}" + "".join(f"{c:>12s}" for c in cols[1:]))
    print("-" * 130)
    for r in rows:
        cells = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols[1:]]
        print(f"{r['condition']:24s}" + "".join(f"{v:>12s}" for v in cells))
    print("-" * 130)
    print(f"* X accounts for {P/len(pairs):.1%} of input pairs; the baseline always predicts non-X.")
    print("  Read this table with kappa and MCC, not accuracy.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--retrieval", action="store_true", help="retrieval metrics only (no LLM)")
    ap.add_argument("--judgment", action="store_true", help="evaluate judgment-layer results")
    ap.add_argument("--task", default=None, choices=["citation", "conflict_x"])
    a = ap.parse_args()

    if a.judgment:
        print_judgment(sorted(sum([glob.glob(f) for f in a.files], [])) or
                       sorted(glob.glob(os.path.join(config.OUT, "judge_*.jsonl"))))
        sys.exit(0)
    if a.retrieval or not a.files:
        for t in ([a.task] if a.task else ["citation", "conflict_x"]):
            print_retrieval(t)
        sys.exit(0)

    rows = [e2e(f) for f in sorted(sum([glob.glob(f) for f in a.files], []))]
    rows = [r for r in rows if r]
    if not rows: sys.exit("no results")
    cols = ["retriever", "task", "k", "n_queries", "retrieval R@k", "final found",
            "judged|retrieved", "cand_P", "cand_R", "cand_F1", "FP rate (non-gold)",
            "neg sample", "parse_fail", "cache hits", "avg latency(s)"]
    w = max(len(c) for c in cols) + 1
    print(f"\n{'metric':{w}s}" + "".join(f"{r['retriever']+'/'+str(r['k']):>16s}" for r in rows))
    print("-" * (w + 16 * len(rows)))
    for c in cols[1:]:
        vals = []
        for r in rows:
            v = r[c]
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{c:{w}s}" + "".join(f"{v:>16s}" for v in vals))
