#!/usr/bin/env python3
"""Generate BM25 rankings in the same jsonl format as the released ranking
files.

Candidate identifiers come from the eligible-structure CSV. Queries use the
Citation eligibility flag in the fold CSV. The query itself is excluded.
Text is the stored title plus individual claims in claim-number order.

  python -m scripts.rag.build_bm25   # writes CHEMRAG_BM25 or its default path
"""
import json, os, re, sys, time
import numpy as np
from rank_bm25 import BM25Okapi

from . import config, data

OUT = config.RANK_BM25
TOK = re.compile(r"[a-z0-9]+")


def text_of(pid, units):
    e = units.get(pid)
    if not e: return ""
    claims = " ".join(e["claims"][n] for n in sorted(e["claims"]))
    return f"{e.get('title','')} {claims}"


def main():
    units = data._units()
    cands = sorted(data.candidates())
    qs = data.eval_queries("citation")
    print(f"candidates {len(cands):,} · queries {len(qs):,}", file=sys.stderr)

    t = time.time()
    docs = [TOK.findall(text_of(p, units).lower()) for p in cands]
    empty = sum(1 for d in docs if not d)
    bm = BM25Okapi(docs)
    print(f"indexed in {time.time()-t:.1f}s (empty docs {empty})", file=sys.stderr)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pos = {p: i for i, p in enumerate(cands)}
    with open(OUT, "w") as f:
        t = time.time()
        for i, q in enumerate(qs):
            sc = bm.get_scores(TOK.findall(text_of(q, units).lower()))
            if q in pos: sc[pos[q]] = -np.inf          # exclude the query itself
            order = np.argsort(-sc)
            ranking = [{"patent_id": cands[j], "rank": r + 1, "score": float(sc[j])}
                       for r, j in enumerate(order) if np.isfinite(sc[j])]
            f.write(json.dumps({"query_patent_id": q, "ranking": ranking,
                                "source": "bm25_title_claims"}, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1:,}/{len(qs):,}  ({time.time()-t:.0f}s)", file=sys.stderr)
    print(f"saved -> {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
