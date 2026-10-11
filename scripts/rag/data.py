"""Data loaders — ground truth, folds, precomputed rankings, patent full text.

Ranking files are streamed and cached as (a) top-N candidates used in prompts
and (b) the stored ranks of gold documents. Missing ranks are not reconstructed.
"""
import csv, io, json, os, pickle, re, sqlite3
from functools import lru_cache

from . import config

TOPN_CACHE = 200          # top candidates kept in the cache (k must not exceed this)


# ───────────────────────── GT · folds ─────────────────────────
@lru_cache(maxsize=1)
def load_gt():
    """query -> {cited_id: category}; is_x defines the binary category rule."""
    gt = {}
    for r in csv.DictReader(open(config.GT)):
        gt.setdefault(r["query_patent_id"], {})[r["cited_patent_id"]] = r["category"]
    return gt


def is_x(category):
    return "X" in str(category).split(",")


@lru_cache(maxsize=1)
def load_folds():
    """query -> dict(outer_fold, citation_eligible, conflict_x_eligible)"""
    out = {}
    for r in csv.DictReader(open(config.FOLDS)):
        out[r["query_id"]] = dict(fold=int(r["outer_fold"]),
                                  citation=r["citation_eligible"] == "1",
                                  conflict_x=r["conflict_x_eligible"] == "1")
    return out


def eval_queries(task="citation"):
    """Queries selected by the corresponding eligibility flag in the fold CSV."""
    f = load_folds()
    key = "citation" if task == "citation" else "conflict_x"
    return sorted(q for q, v in f.items() if v[key])


def gold(query, task="citation"):
    """Set of gold cited documents for the query."""
    cites = load_gt().get(query, {})
    if task == "conflict_x":
        return {c for c, cat in cites.items() if is_x(cat)}
    return set(cites)


@lru_cache(maxsize=1)
def candidates():
    """Unique patent identifiers in the configured eligible-structure CSV."""
    return {r["patent_id"] for r in csv.DictReader(open(config.STRUCT))}


# ───────────────────────── precomputed rankings ─────────────────────────
def load_rankings(name, topn=TOPN_CACHE):
    """condition name -> {query: {"top": [id...], "ranks": {cited_id: rank}}}

    `ranks` stores only the gold documents' ranks (1-based). Recall@k and MRR
    are computed from available stored ranks, without inventing missing ranks.
    """
    path, fkey, fval, _ = config.RETRIEVERS[name]
    os.makedirs(config.CACHE, exist_ok=True)
    cache = os.path.join(config.CACHE, f"rank_{name}_top{topn}.pkl")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        with open(cache, "rb") as f:
            return pickle.load(f)

    gt = load_gt()
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if fkey and d.get(fkey) != fval:
                continue
            q = d["query_patent_id"]
            want = set(gt.get(q, {}))
            top, ranks = [], {}
            for e in d["ranking"]:
                pid, rk = e["patent_id"], e["rank"]
                if rk <= topn:
                    top.append(pid)
                if pid in want:
                    ranks[pid] = rk
            out[q] = {"top": top, "ranks": ranks}
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


# ───────────────────────── patent full text ─────────────────────────
class _Dummy:
    def __init__(self, *a, **k): self.__dict__.update(k)
    def __setstate__(self, st):
        if isinstance(st, dict): self.__dict__.update(st.get("__dict__", st))
        elif isinstance(st, tuple):
            for d in st:
                if isinstance(d, dict): self.__dict__.update(d.get("__dict__", d))

class _Unp(pickle.Unpickler):
    def find_class(self, m, n):
        try: return super().find_class(m, n)
        except Exception: return type(n, (_Dummy,), {})


@lru_cache(maxsize=1)
def _units():
    """text units -> {patent_id: {"doc": title+abstract, "claims": {no: text}}}"""
    out = {}
    con = sqlite3.connect(f"file:{config.TEXT_UNITS_SQLITE}?mode=ro", uri=True)
    for _, blob in con.execute("select id, point from points"):
        p = _Unp(io.BytesIO(blob)).load().__dict__["payload"]
        e = out.setdefault(p["patent_id"], {"doc": "", "claims": {}, "title": p.get("title", "")})
        if p["unit_type"] == "document":
            e["doc"] = p["text"]
        elif p["unit_type"] == "claim":
            e["claims"][int(p["claim_no"])] = p["text"]
    con.close()
    return out


def patent_text(pid, max_chars=None, with_claims=True):
    """Patent text for prompts — title + abstract + numbered claims."""
    max_chars = max_chars or config.MAX_DOC_CHARS
    e = _units().get(pid)
    if e is None:
        return None
    parts = [f"Patent ID: {pid}", e["doc"].strip()]
    if with_claims and e["claims"]:
        body = "\n".join(f"[Claim {n}] {e['claims'][n].strip()}" for n in sorted(e["claims"]))
        parts.append("Claims:\n" + body)
    t = "\n".join(x for x in parts if x)
    return t[:max_chars] + "\n[... truncated ...]" if len(t) > max_chars else t


# ───────────────────────── auxiliary-task gold ─────────────────────────
@lru_cache(maxsize=1)
def load_grounding():
    """(query, cited) -> {rel_claims:set, evidence_type:set}  — grounding_gold.csv"""
    g = {}
    for r in csv.DictReader(open(config.GROUNDING)):
        k = (r["query_patent_id"], r["cited_patent_id"])
        e = g.setdefault(k, {"rel_claims": set(), "evidence_type": set()})
        for tok in re.split(r"[,\s]+", r.get("target_rel_claims") or ""):
            tok = tok.strip()
            m = re.match(r"^(\d+)-(\d+)$", tok)
            if m: e["rel_claims"].update(range(int(m.group(1)), int(m.group(2)) + 1))
            elif tok.isdigit(): e["rel_claims"].add(int(tok))
        if r.get("evidence_type"): e["evidence_type"].add(r["evidence_type"])
    return g


# ───────────────────────── judgment pairs ─────────────────────────
def judgment_pairs(restrict_eval=True):
    """Judgment-layer pairs — retrieval is skipped and the examiner-cited
    document is provided directly.

    Label: 1 if is_x finds an exact comma-separated X token, else 0.
    With restrict_eval=True, membership in the fold CSV is required;
    citation_eligible is not checked here.
    """
    folds = load_folds()
    out = []
    for q, cites in load_gt().items():
        if restrict_eval and q not in folds:
            continue
        for c, cat in cites.items():
            out.append(dict(pair_id=f"{q}__{c}", query_id=q, cited_id=c,
                            category=cat, label=1 if is_x(cat) else 0,
                            fold=folds[q]["fold"] if q in folds else -1))
    out.sort(key=lambda d: d["pair_id"])
    return out
