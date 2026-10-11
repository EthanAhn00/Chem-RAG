#!/usr/bin/env python3
"""Confusion-matrix breakdown of the judgment layer — what does the model
confuse X with?

Splits available result records by the true EPO category.

  python -m scripts.rag.analyze_3class
"""
import collections, glob, json, os, sys
from . import config

ORDER = ["X", "Y", "A", "other"]


def true_grade(cat):
    """True EPO category — X takes precedence, then Y, then A."""
    c = (cat or "").upper()
    if "X" in c: return "X"
    if "Y" in c: return "Y"
    if "A" in c: return "A"
    return "other"


def load(cond):
    p = os.path.join(config.OUT, f"judge_{cond}.jsonl")
    if not os.path.exists(p): return None
    out = []
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l)
        if "error" in r or r.get("pred") is None: continue
        out.append(r)
    return out


def pred_grade(r):
    """Predicted category. 3-class uses pred_raw (X/Y/A); binary uses X / non-X."""
    raw = (r.get("pred_raw") or "").upper().strip()
    for g in ("X", "Y", "A"):
        if raw == g or raw.startswith(g): return g
    return "X" if r["pred"] == "X" else "non-X"


def grid(title, rows, cols, counts, rowtot=True):
    w = max(12, max(len(str(r)) for r in rows) + 2)
    print(f"\n{title}")
    print("-" * (w + 10 * (len(cols) + 1)))
    heading = "true \\ pred"
    print(f"{heading:<{w}}" + "".join(f"{c:>10}" for c in cols) + (f"{'total':>10}" if rowtot else ""))
    for r in rows:
        tot = sum(counts.get((r, c), 0) for c in cols)
        line = f"{str(r):<{w}}" + "".join(f"{counts.get((r,c),0):>10,}" for c in cols)
        print(line + (f"{tot:>10,}" if rowtot else ""))
    print("-" * (w + 10 * (len(cols) + 1)))
    print(f"{'total':<{w}}" + "".join(f"{sum(counts.get((r,c),0) for r in rows):>10,}" for c in cols))


def main():
    print("=" * 78)
    print("Judgment-layer confusion-matrix breakdown")
    print("=" * 78)

    # ── 3-class: true X/Y/A × predicted X/Y/A ─────────────────────────────
    recs = load("3class")
    if recs:
        c = collections.Counter((true_grade(r.get("category")), pred_grade(r)) for r in recs)
        grid(f"[3-class] true category x predicted category  (n={len(recs):,})",
             ORDER, ["X", "Y", "A", "non-X"], c)

        xp = [r for r in recs if pred_grade(r) == "X"]
        if xp:
            b = collections.Counter(true_grade(r.get("category")) for r in xp)
            print(f"\nTrue categories of the {len(xp):,} documents the model called X")
            for g in ORDER:
                if b[g]:
                    print(f"  {g:<6} {b[g]:>6,}  ({b[g]/len(xp):6.1%})")
        xg = [r for r in recs if true_grade(r.get("category")) == "X"]
        if xg:
            b = collections.Counter(pred_grade(r) for r in xg)
            print(f"\nWhat the model called the {len(xg):,} true X documents")
            for g in ["X", "Y", "A", "non-X"]:
                if b[g]:
                    print(f"  {g:<6} {b[g]:>6,}  ({b[g]/len(xg):6.1%})")
    else:
        print("\n[3-class] results/judge_3class.jsonl not found", file=sys.stderr)

    # ── binary conditions: true categories of X predictions ───────────────
    for cond, desc in [("base", "base"), ("think", "thinking"), ("nodefs", "no defs")]:
        recs = load(cond)
        if not recs: continue
        xp = [r for r in recs if r["pred"] == "X"]
        print(f"\n[{desc}] n={len(recs):,} · X predictions {len(xp):,}", end="")
        if not xp:
            print(" — nothing to break down"); continue
        b = collections.Counter(true_grade(r.get("category")) for r in xp)
        print(", true categories: " + " · ".join(f"{g} {b[g]}" for g in ORDER if b[g]))

    # ── true category distribution (denominator check) ────────────────────
    from . import data
    pairs = data.judgment_pairs()
    b = collections.Counter(true_grade(p["category"]) for p in pairs)
    print(f"\nTrue category distribution of the {len(pairs):,} evaluation pairs")
    for g in ORDER:
        if b[g]: print(f"  {g:<6} {b[g]:>6,}  ({b[g]/len(pairs):6.1%})")


if __name__ == "__main__":
    main()
