"""Adversarial hypothesis checker (generic, no math engine).
Channel A = model recall (books named, unverified; only steers search).
Channel B = local retrieval over corpus/ (real text; the only citable evidence).
Pipeline: attack plan -> retrieval + reconcile -> hostile review -> quote verification -> verdict.
"""
from __future__ import annotations
import glob, json, math, os, re
from typing import Literal
from pydantic import BaseModel


# ------------------------------------------------------------------ schemas
class Claim(BaseModel):
    id: str
    text: str

class Query(BaseModel):
    text: str
    kind: Literal["impossibility", "failure_mode", "contradiction", "limit"]

class RecallItem(BaseModel):
    constraint: str      # the limit/principle, from memory (unverified)
    book: str            # best-guess source book
    search_terms: str    # keywords likely to appear in the book text

class AttackPlan(BaseModel):
    claims: list[Claim]
    queries: list[Query]
    recall: list[RecallItem]

class Issue(BaseModel):
    id: str
    severity: Literal["critical", "major", "minor"]
    category: str
    claim_id: str
    claim: str           # the objection
    passage_id: str      # P# the objection rests on
    quote: str           # verbatim from that passage
    edge_condition: str  # what would flip it (qualitative)

class Review(BaseModel):
    issues: list[Issue]
    untested_claims: list[str]

class Entail(BaseModel):
    label: Literal["supports", "partly", "not", "unclear"]
    why: str


# ------------------------------------------------------------------ LLM backends
class GeminiLLM:
    """pip install google-genai ; env: GEMINI_API_KEY, GEMINI_MODEL"""
    def __init__(self, model=None):
        from google import genai
        self.client = genai.Client()
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def json(self, system, user, schema, temperature=0.0):
        from google.genai import types
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                          response_mime_type="application/json", response_schema=schema)
        r = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
        return json.loads(r.text)


class AnthropicLLM:
    """pip install anthropic ; env: ANTHROPIC_API_KEY, ANTHROPIC_MODEL"""
    def __init__(self, model=None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    def json(self, system, user, schema, temperature=0.0):
        sys_ = system + "\nReturn ONLY JSON matching this schema:\n" + json.dumps(schema.model_json_schema())
        r = self.client.messages.create(model=self.model, max_tokens=4000, temperature=temperature,
                                        system=sys_, messages=[{"role": "user", "content": user}])
        txt = "".join(b.text for b in r.content if b.type == "text")
        return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])


# ------------------------------------------------------------------ Channel B: local BM25 retrieval
class LocalBM25:
    """Paragraph-aware chunks (never splits a paragraph, so quotes stay intact). Swap for any .search(q,k)."""
    STOP = set("the a an of in to is and or for on at by with as it its be that this from are".split())

    def __init__(self, corpus_dir, chunk_words=180):
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(corpus_dir, "**", "*.*"), recursive=True)):
            if not path.lower().endswith((".txt", ".md")):
                continue
            paras = [p.strip() for p in re.split(r"\n\s*\n", open(path, encoding="utf-8", errors="ignore").read()) if p.strip()]
            buf, n = [], 0
            for p in paras:
                buf.append(p)
                n += len(p.split())
                if n >= chunk_words:
                    self.chunks.append((os.path.basename(path), "\n\n".join(buf)))
                    buf, n = [], 0
            if buf:
                self.chunks.append((os.path.basename(path), "\n\n".join(buf)))
        self.toks = [self._tok(c[1]) for c in self.chunks]
        self.N = len(self.chunks)
        self.avg = sum(map(len, self.toks)) / max(self.N, 1)
        self.df = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1

    def _tok(self, s):
        return [w for w in re.findall(r"\w+", s.lower()) if w not in self.STOP]

    def search(self, q, k=3):
        """-> [(source, text, score)] best first, score > 0 only."""
        qs, scores = self._tok(q), []
        for i, t in enumerate(self.toks):
            s = 0.0
            for w in qs:
                f = t.count(w)
                if not f:
                    continue
                idf = math.log(1 + (self.N - self.df[w] + .5) / (self.df[w] + .5))
                s += idf * f * 2.2 / (f + 1.2 * (.25 + .75 * len(t) / self.avg))
            scores.append((s, i))
        return [(*self.chunks[i], s) for s, i in sorted(scores, reverse=True)[:k] if s > 0]


# ------------------------------------------------------------------ prompts
PLAN_SYS = """You prepare an adversarial review of a scientific hypothesis. You do NOT judge it.
1) Split it into a chain of testable claims (ids C1, C2, ...).
2) Write 6-10 adversarial search queries (kinds: impossibility, failure_mode, contradiction, limit). NEVER query for support. 3-10 words each, using domain terms.
3) RECALL: from your memory of these textbooks: {books}
   list up to 8 hard constraints or principles that bear on the claims. This is UNVERIFIED memory. For each give the best-guess book and 3-8 search_terms: keywords likely to appear in that book's own text (not a paraphrase of the claim). No page numbers, no quotes."""

REVIEW_SYS = """You are a skeptical, hostile domain expert refereeing a hypothesis. Your incentive is to find fatal flaws.
Inputs: HYPOTHESIS, CLAIMS (C#), PASSAGES (P#, real text retrieved from textbooks).
Rules:
1. Every issue must rest on ONE passage: give its passage_id and a VERBATIM quote (6-40 words) copied exactly from that passage. Do not paraphrase inside `quote`. Do not use outside memory as evidence.
2. If you suspect a problem but no passage supports it, put it in untested_claims. Never guess.
3. Do not introduce numbers that are not in the hypothesis or passages. If a number matters, say 'eyeball-check needed' instead of computing.
4. Severity: critical = the passage shows the mechanism unworkable or an invariant violated; major = large penalty with plausible workaround; minor = otherwise.
5. edge_condition: what change to the hypothesis would flip the issue, stated qualitatively.
6. Also list any hypothesis claim that no passage tests in untested_claims."""

ENTAIL_SYS = "Decide whether the QUOTE supports the CLAIM. Use only the quote. Answer supports / partly / not / unclear with a one-sentence reason."


# ------------------------------------------------------------------ helpers
def norm(s):
    s = s.lower()
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def verify_quote(quote, passage_text, all_chunks):
    nq = norm(quote)
    if len(nq.split()) < 4:
        return "too_short"
    if nq in norm(passage_text):
        return "verified"
    if any(nq in norm(t) for _, t in all_chunks):
        return "misattributed"   # real text, but not in the cited passage
    return "failed"


NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(text):
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def eyeball_numbers(text, grounded_texts, rel=0.03):
    pool = [n for t in grounded_texts for n in numbers_in(t)]
    bad = []
    for n in numbers_in(text):
        if abs(n) <= 10 and float(n).is_integer():
            continue
        if not any(abs(n - p) <= rel * max(abs(p), 1e-12) for p in pool):
            bad.append(n)
    return bad


# ------------------------------------------------------------------ pipeline
def run(hyp, books, gen, critic, retriever, min_recall_score=2.0, auto_entail=False, check_math=False, log=print):
    degraded = []

    log("[1] attack plan + recall (channel A)")
    plan = AttackPlan.model_validate(gen.json(PLAN_SYS.format(books=books), hyp, AttackPlan, 0.3))

    log("[2] retrieval (channel B) + reconcile")
    passages, by_text = [], {}

    def add(src, text, score, via):
        if text in by_text:
            by_text[text]["via"].add(via)
            return by_text[text]["id"]
        p = dict(id=f"P{len(passages) + 1}", source=src, text=text, score=round(score, 2), via={via})
        passages.append(p)
        by_text[text] = p
        return p["id"]

    for q in plan.queries:
        for src, txt, sc in retriever.search(q.text, 2):
            add(src, txt, sc, "adv")
    reconcile = []
    for r in plan.recall:
        hits = retriever.search(r.search_terms, 1)
        if hits and hits[0][2] >= min_recall_score:
            pid = add(hits[0][0], hits[0][1], hits[0][2], "recall")
            reconcile.append(dict(item=r.model_dump(), bucket="agree", passage_id=pid, score=round(hits[0][2], 2)))
        else:
            reconcile.append(dict(item=r.model_dump(), bucket="recall_only", passage_id=None,
                                  score=round(hits[0][2], 2) if hits else 0.0))
    retrieval_only = [p["id"] for p in passages if p["via"] == {"adv"}]
    if not passages:
        degraded.append("no passages retrieved: literature falsification NOT performed")

    log("[3] hostile review (reviewer sees passages only, not recall output)")
    ctx = (f"HYPOTHESIS:\n{hyp}\n\nCLAIMS:\n{json.dumps([c.model_dump() for c in plan.claims])}\n\n"
           f"PASSAGES:\n{json.dumps([dict(id=p['id'], source=p['source'], text=p['text']) for p in passages], indent=1)}")
    rev = Review.model_validate(critic.json(REVIEW_SYS, ctx, Review, 0.0))

    log("[4] quote verification")
    pmap = {p["id"]: p for p in passages}
    grounded = [hyp] + [p["text"] for p in passages]
    issues = []
    for i in rev.issues:
        d = i.model_dump()
        p = pmap.get(i.passage_id)
        d["quote_status"] = verify_quote(i.quote, p["text"], retriever.chunks) if p else "bad_passage_id"
        d["eyeball_numbers"] = eyeball_numbers(i.claim + " " + i.edge_condition, grounded) if check_math else []
        d["auto_entail"] = None
        if auto_entail and d["quote_status"] == "verified":
            d["auto_entail"] = Entail.model_validate(critic.json(
                ENTAIL_SYS, f"CLAIM: {i.claim}\nQUOTE: {i.quote}", Entail, 0.0)).model_dump()
        issues.append(d)

    rep = dict(books=books, hypothesis=hyp, plan=plan.model_dump(),
               passages=[{**p, "via": sorted(p["via"])} for p in passages],
               reconcile=reconcile, retrieval_only=retrieval_only, issues=issues,
               untested_claims=rev.untested_claims, degraded=degraded)
    rep["verdict"] = verdict(issues)
    return rep


def verdict(issues):
    ok = [i for i in issues if i["quote_status"] == "verified"]
    unv = [i for i in issues if i["quote_status"] != "verified"]
    if any(i["severity"] == "critical" for i in ok):
        return "FAIL"
    if any(i["severity"] == "major" for i in ok):
        return "WEAK"
    if any(i["severity"] == "critical" for i in unv):
        return "WEAK (critical concern with unverified citation)"
    return "SURVIVES (not endorsed)"


def render(rep):
    L = [f"# Verdict: {rep['verdict']}", ""]
    if rep["degraded"]:
        L += ["**Degraded:** " + "; ".join(rep["degraded"]), ""]
    L += ["## Issue board"]
    for i in rep["issues"]:
        flag = "" if i["quote_status"] == "verified" else f" **[quote {i['quote_status']}]**"
        eye = f" _(eyeball-check numbers: {i['eyeball_numbers']})_" if i["eyeball_numbers"] else ""
        ent = f" _(auto-entail: {i['auto_entail']['label']})_" if i.get("auto_entail") else ""
        L.append(f"- **{i['id']}** {i['severity']} / {i['category']} / {i['passage_id']}{flag}{eye}{ent}\n"
                 f"  {i['claim']}\n  > {i['quote']}\n  _flips if:_ {i['edge_condition']}")
    L += ["", "## Channel A vs B (reconcile)"]
    for r in rep["reconcile"]:
        L.append(f"- **{r['bucket']}** ({r['item']['book']}): {r['item']['constraint']}"
                 + (f" -> {r['passage_id']}" if r["passage_id"] else " -> NOT found in corpus (gap, bad search terms, or invented)"))
    if rep["retrieval_only"]:
        L.append(f"- retrieval-only passages (recall silent): {', '.join(rep['retrieval_only'])}")
    L += ["", "## Untested"] + [f"- {u}" for u in rep["untested_claims"]]
    return "\n".join(L)
