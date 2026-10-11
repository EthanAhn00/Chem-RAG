"""Prompts for the RAG judge and the judgment layer.

Labels follow the dataset specification:
  Citation labels : per-citation EPO category + binary Conflict-X label
                    (1 if the category contains X, else 0)

The RAG judge is asked a 3-way question and the answer is folded into two tasks
at evaluation time:
  Task 1 Citation  : X or CITED  → cited
  Task 2 Conflict-X: X           → citation containing X
"""
import json, re

EPO_DEFS = """EPO search report citation categories:
- X: particularly relevant if taken alone — the document alone anticipates the claimed invention (lack of novelty) or renders it obvious (lack of inventive step).
- Y: particularly relevant if combined with another document of the same category.
- A: technological background — defines the general state of the art, not of particular relevance."""

SYSTEM = ("You are a European Patent Office search examiner specialised in chemistry and materials. "
          "Given a patent application and one candidate document retrieved from the search corpus, "
          "you decide whether you would cite that candidate in the search report, and with which category. "
          "Follow EPO practice strictly.\n\n" + EPO_DEFS)

SCHEMA = """Respond with a single JSON object and nothing else:
{
  "label": "X" | "CITED" | "NOT_CITED",
  "relevant_query_claims": [integer claim numbers of the QUERY affected; [] if none],
  "evidence_location": short phrase locating the evidence in the candidate (e.g. "claims 1-3", "abstract", "paragraph [0021]"),
  "rationale": 2-3 sentences: which claim features the candidate discloses, which are missing, and why the label follows
}"""

TASK = """### TASK
Decide which label applies to the CANDIDATE document.
- "X"          : you would cite it as category X — on its own it destroys novelty or inventive step of at least one query claim.
- "CITED"      : you would cite it, but not as X (e.g. Y or A).
- "NOT_CITED"  : you would not cite it in the search report at all.

Compare claim features one by one. Do not rely on publication dates.
Most retrieved candidates are topically similar but would not be cited at all.
Among the ones you would cite, judge X on its merits using the criterion above —
X is common in real search reports, so do not avoid it when the criterion is met."""


def system_prompt(examiner_defs=True):
    return SYSTEM if examiner_defs else SYSTEM.split("\n\nEPO search report")[0]


def user_prompt(query_text, cand_text):
    return f"""### QUERY PATENT (application under examination)
{query_text}

### CANDIDATE DOCUMENT (retrieved from the search corpus)
{cand_text}

{TASK}

{SCHEMA}"""


def parse_json(text):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception:
        try: return json.loads(m.group(0).replace("'", '"'))
        except Exception: return None


def normalize(raw):
    s = str(raw or "").strip().upper().replace(" ", "_")
    if s in ("X", "CITED", "NOT_CITED"): return s
    if s in ("Y", "A"): return "CITED"
    if s.startswith("NOT"): return "NOT_CITED"
    return None


# ───────────── closed-book (no candidates given) ─────────────
def closedbook_prompt(query_text, n=10):
    return f"""### QUERY PATENT (application under examination)
{query_text}

### TASK
No candidate documents are provided. From your own knowledge only, name up to {n} published
patent documents that a European Patent Office examiner would cite in the search report for
these claims. List the most likely first.

Use full publication numbers (e.g. EP1234567A1). Do not invent numbers you are not confident
about — return fewer, or an empty list, rather than guessing.

Respond with a single JSON object and nothing else:
{{
  "documents": [
    {{"publication_number": "<number>", "label": "X" | "CITED", "why": "<one sentence>"}}
  ],
  "confidence": "high" | "medium" | "low"
}}"""


# ───────────── judgment layer (gold document given, no retrieval) ─────────────
#   label: 1 if the category contains X, else 0
#   binary : X / NOT_X      3class : X / Y / A

JUDGE_SYSTEM = ("You are a European Patent Office search examiner specialised in chemistry and materials. "
                "A prior-art document has already been cited against a patent application. "
                "Your task is to assign the citation category, following EPO practice strictly.")

def judge_system(examiner_defs=True):
    return JUDGE_SYSTEM + ("\n\n" + EPO_DEFS if examiner_defs else "")

def judge_user(query_text, cited_text, scheme="binary"):
    if scheme == "binary":
        task = ('Decide whether this is a category X citation.\n'
                '- "X"     : on its own the document destroys novelty or inventive step of at least one query claim.\n'
                '- "NOT_X" : any other category (Y, A, ...) — it does not defeat a claim on its own.')
        cat = '"X" or "NOT_X"'
    else:
        task = 'Assign the EPO citation category this document would receive: "X", "Y" or "A".'
        cat = '"X", "Y" or "A"'
    return f"""### QUERY PATENT (application under examination)
{query_text}

### CITED PRIOR-ART DOCUMENT
{cited_text}

### TASK
{task}
Compare claim features one by one. Do not rely on publication dates.

Respond with a single JSON object and nothing else:
{{
  "category": {cat},
  "relevant_query_claims": [integer claim numbers of the QUERY affected; [] if none],
  "evidence_location": short phrase locating the evidence (e.g. "claims 1-3", "abstract"),
  "rationale": 2-3 sentences: which claim features are disclosed, which are missing, and why the category follows
}}"""

def judge_normalize(raw, scheme="binary"):
    """Reduce to positive (=X) vs not. Returns 'X' | 'NOT_X' | None."""
    s = str(raw or "").strip().upper().replace(" ", "_")
    if s == "X": return "X"
    if scheme == "binary":
        return "NOT_X" if s.startswith("NOT") else None
    return "NOT_X" if s in ("Y", "A") else None
