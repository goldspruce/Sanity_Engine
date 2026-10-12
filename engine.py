"""Adversarial hypothesis checker (generic, no math engine).
Channel A = model recall (books named, unverified; only steers search).
Channel B = local retrieval over corpus/ (real text; the only citable evidence).
Pipeline: attack plan -> retrieval + reconcile -> hostile review -> quote verification -> verdict.
"""
from __future__ import annotations
import glob, json, math, os, re, warnings
from typing import Literal

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

from pydantic import BaseModel


def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                m = re.match(r"^[\"'](.*?)[\"']", v)
                if m:
                    v = m.group(1)
                else:
                    v = v.split("#", 1)[0].strip()
                if k and k not in os.environ:
                    os.environ[k] = v

_load_dotenv()


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
    scientific_synthesis: str = ""  # Narrative: what is biophysically/physically plausible vs what is empirically contested
    falsification_test: str = ""    # A concrete empirical experiment or clinical test to settle the claim
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
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

    def json(self, system, user, schema, temperature=0.0):
        import time
        from google.genai import types, errors
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                          response_mime_type="application/json", response_schema=schema)
        candidates = [self.model]
        for alt in ["gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.8-flash"]:
            if alt not in candidates:
                candidates.append(alt)
        last_err = None
        for model_name in candidates:
            for attempt in range(3):
                try:
                    r = self.client.models.generate_content(model=model_name, contents=user, config=cfg)
                    return json.loads(r.text)
                except (errors.ServerError, errors.ClientError) as e:
                    last_err = e
                    if "404" in str(e) or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        break
                    time.sleep(2.0 * (attempt + 1))
        raise last_err


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

REVIEW_SYS = """You are a rigorous, constructive-adversarial scientific referee evaluating a hypothesis against literature passages.
Inputs: HYPOTHESIS, CLAIMS (C#), PASSAGES (P#, real text retrieved from textbooks).

Deliverables:
1. scientific_synthesis: Provide a 2-4 sentence narrative synthesis of the review: acknowledge what mechanisms in the hypothesis are biophysically or physically plausible, and clearly explain where the hypothesis faces empirical friction, causal ambiguity, or competing explanations in the literature.
2. falsification_test: Describe a concrete, realistic empirical experiment, clinical study, or benchtop measurement that would decisively test or falsify this hypothesis.
3. issues: Structured objections. Every issue must rest on ONE passage with a VERBATIM quote (6-40 words) copied exactly from that passage. Do not paraphrase inside `quote`. Do not introduce numbers not in the text.
   Severity:
   - critical = the passage shows an established physical/biological invariant violated or mechanism impossible (results in FAIL).
   - major = significant theoretical or empirical friction, but workable edge conditions exist (results in CHALLENGED).
   - minor = peripheral objection, scope limitation, or slight discrepancy.
   edge_condition: what change to the hypothesis or experimental condition would flip the issue, stated qualitatively.
4. untested_claims: Claims where the passages are silent and provide no definitive test. Never guess."""

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
               reconcile=reconcile, retrieval_only=retrieval_only,
               scientific_synthesis=rev.scientific_synthesis,
               falsification_test=rev.falsification_test,
               issues=issues,
               untested_claims=rev.untested_claims, degraded=degraded)
    rep["verdict"] = verdict(issues)
    return rep


def verdict(issues):
    ok = [i for i in issues if i["quote_status"] == "verified"]
    unv = [i for i in issues if i["quote_status"] != "verified"]
    if any(i["severity"] == "critical" for i in ok):
        return "FAIL"
    if any(i["severity"] == "major" for i in ok):
        return "CHALLENGED"
    if any(i["severity"] == "critical" for i in unv):
        return "CHALLENGED (critical concern with unverified citation)"
    return "SURVIVES (not endorsed)"


VERDICT_EXPLANATIONS = {
    "FAIL": "The hypothesis is falsified by an established physical invariant or documented empirical limit verified in the literature.",
    "CHALLENGED": "The hypothesis faces significant theoretical or empirical challenges documented in the literature, but key edge conditions remain plausible and open to test.",
    "CHALLENGED (critical concern with unverified citation)": "A critical objection was raised, but the citation could not be verified verbatim in the corpus.",
    "SURVIVES (not endorsed)": "The hypothesis was not falsified by any verified passage in the current corpus. This does NOT mean the hypothesis is endorsed or scientifically proven; rather, the current library contained no definitive counter-evidence to refute it (see Untested claims below)."
}


def render(rep):
    v = rep["verdict"]
    exp = VERDICT_EXPLANATIONS.get(v, "")
    hyp = rep.get("hypothesis", "").strip()

    L = []
    if hyp:
        L += ["## 🔬 Hypothesis / Prompt", f"> {hyp}", ""]

    L += [f"# Verdict: {v}"]
    if exp:
        L += [f"> **Plain-language explanation:** {exp}", ""]
    else:
        L += [""]

    synth = rep.get("scientific_synthesis", "").strip()
    test = rep.get("falsification_test", "").strip()
    if synth or test:
        L += ["## 🧠 Scientific Synthesis & Evaluation"]
        if synth:
            L += [f"> {synth}", ""]
        if test:
            L += [f"> 🧪 **Decisive Falsification Test:** {test}", ""]
    if rep["degraded"]:
        L += ["**Degraded:** " + "; ".join(rep["degraded"]), ""]
    L += ["## Issue board"]
    if not rep["issues"]:
        L += ["_No disqualifying issues or rule violations were found in the cited passages._", ""]
    else:
        for i in rep["issues"]:
            flag = "" if i["quote_status"] == "verified" else f" **[quote {i['quote_status']}]**"
            eye = f" _(eyeball-check numbers: {i['eyeball_numbers']})_" if i["eyeball_numbers"] else ""
            ent = f" _(auto-entail: {i['auto_entail']['label']})_" if i.get("auto_entail") else ""
            L.append(f"- **{i['id']}** {i['severity']} / {i['category']} / {i['passage_id']}{flag}{eye}{ent}\n"
                     f"  {i['claim']}\n  > {i['quote']}\n  _flips if:_ {i['edge_condition']}")
    L += ["", "## Channel A vs B (Reconciliation)",
          "> **Channel A (Model Recall):** Theoretical principles and textbooks recalled from model memory (unverified).",
          "> **Channel B (Corpus Retrieval):** Real text passages retrieved deterministically from your library (`corpus/`).",
          "> **Reconciliation:**",
          "> - `agree`: The recalled principle was confirmed by verified passages in your library.",
          "> - `recall_only`: Recalled by model memory, but no supporting passage was found in your library (indicates a literature gap or unverified memory).",
          "> - `retrieval_only`: Relevant literature was found by search that model recall did not anticipate.",
          ""]
    for r in rep["reconcile"]:
        L.append(f"- **{r['bucket']}** ({r['item']['book']}): {r['item']['constraint']}"
                 + (f" -> {r['passage_id']}" if r["passage_id"] else " -> NOT found in corpus (gap, bad search terms, or invented)"))
    if rep["retrieval_only"]:
        L.append(f"- retrieval-only passages (recall silent): {', '.join(rep['retrieval_only'])}")
    L += ["", "## Untested Claims",
          "> Specific claims in the hypothesis where the current corpus lacked relevant experimental or theoretical literature:"]
    for u in rep["untested_claims"]:
        L.append(f"- {u}")

    # Recommended Reading & NotebookLM section
    cited_sources = sorted({p["source"].replace(".txt", "").replace(".md", "").replace("_", " ") for p in rep.get("passages", [])})
    recalled_books = sorted({r["item"]["book"] for r in rep.get("reconcile", []) if r.get("item", {}).get("book")})

    nb_url = os.environ.get("NOTEBOOKLM_URL", "").strip()
    nb_link_msg = (
        f"> 🔗 **Preloaded NotebookLM Workspace:** [{nb_url}]({nb_url})"
        if nb_url
        else "> 🔗 **Preloaded NotebookLM Workspace:** A direct shareable link with the complete textbook corpus preloaded is provided for reviewers and researchers."
    )

    L += ["", "## 📚 Recommended Reading & NotebookLM Investigation",
          "> To investigate untested claims or examine biological/physical mechanisms further,",
          "> you can cross-examine these textbooks in a **Google NotebookLM** notebook for cited Q&A.",
          nb_link_msg,
          ""]
    if cited_sources:
        L += ["### Directly Cited in Corpus Passages:"]
        for s in cited_sources:
            L += [f"- **{s}**"]
        L += [""]
    if recalled_books:
        L += ["### Relevant Domain Books (from Model Recall):"]
        for b in recalled_books:
            L += [f"- **{b}**"]
        L += [""]
    L += ["> 💡 *Tip: See `TEXTBOOKS.md` for full textbook catalog and NotebookLM sharing details. Run `python pack_for_notebook.py --corpus corpus --out notebook_pack` to generate upload packages.*"]

    return "\n".join(L)
