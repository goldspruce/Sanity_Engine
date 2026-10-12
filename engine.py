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

from pydantic import BaseModel, Field


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
    viability_score: int = Field(default=5, ge=0, le=9)  # Referee assessment on 0-9 scale
    score_rationale: str = Field(default="")             # 1-2 sentence justification for the assigned score
    scientific_synthesis: str = ""  # Narrative: what is biophysically/physically plausible vs what is empirically contested
    falsification_test: str = ""    # A concrete empirical experiment or clinical test to settle the claim
    issues: list[Issue]
    untested_claims: list[str]

class FollowUpPlan(BaseModel):
    new_claims: list[Claim] = Field(default_factory=list)
    adversarial_queries: list[Query] = Field(default_factory=list)
    recall: list[RecallItem] = Field(default_factory=list)

class FollowUpReview(BaseModel):
    viability_score: int = Field(default=5, ge=0, le=9)
    score_rationale: str = Field(default="")
    score_delta_explanation: str = Field(default="")
    resolved_issue_ids: list[str] = Field(default_factory=list)
    scientific_synthesis: str = ""
    falsification_test: str = ""
    new_issues: list[Issue] = Field(default_factory=list)
    untested_claims: list[str] = Field(default_factory=list)

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

    def text(self, system, user, temperature=0.3):
        import time
        from google.genai import types, errors
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        candidates = [self.model]
        for alt in ["gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.8-flash"]:
            if alt not in candidates:
                candidates.append(alt)
        last_err = None
        for model_name in candidates:
            for attempt in range(3):
                try:
                    r = self.client.models.generate_content(model=model_name, contents=user, config=cfg)
                    return r.text
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

    def text(self, system, user, temperature=0.3):
        r = self.client.messages.create(model=self.model, max_tokens=4000, temperature=temperature,
                                        system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")


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
1. viability_score (integer 0 to 9): Assign an empirical viability / sanity score:
   - 0: Flatly impossible or definitively falsified by established conservation laws or biophysical invariants.
   - 1-3: Severely contradicted; core mechanisms directly conflict with verified literature passages.
   - 4-6: Challenged; notable empirical friction, competing models, or rate/frequency trade-offs exist (the hypothesis requires specific edge conditions or constraints).
   - 7-8: Plausible; largely consistent with textbook mechanisms, surviving with only minor caveats or scope limits.
   - 9: Highly robust invariant; fully consistent with canonical principles and unrefuted.
2. score_rationale: 1-2 concise sentences explaining why this score was assigned on the 0-9 scale.
3. scientific_synthesis: Provide a 2-4 sentence narrative synthesis of the review: acknowledge what mechanisms in the hypothesis are biophysically or physically plausible, and clearly explain where the hypothesis faces empirical friction, causal ambiguity, or competing explanations in the literature.
4. falsification_test: Describe a concrete, realistic empirical experiment, clinical study, or benchtop measurement that would decisively test or falsify this hypothesis.
5. issues: Structured objections. Every issue must rest on ONE passage with a VERBATIM quote (6-40 words) copied exactly from that passage. Do not paraphrase inside `quote`. Do not introduce numbers not in the text.
   Severity:
   - critical = the passage shows an established physical/biological invariant violated or mechanism impossible.
   - major = significant theoretical or empirical friction, but workable edge conditions exist.
   - minor = peripheral objection, scope limitation, or slight discrepancy.
   edge_condition: what change to the hypothesis or experimental condition would flip the issue, stated qualitatively.
6. untested_claims: Claims where the passages are silent and provide no definitive test. Never guess."""

ENTAIL_SYS = "Decide whether the QUOTE supports the CLAIM. Use only the quote. Answer supports / partly / not / unclear with a one-sentence reason."

FOLLOW_UP_SYS = """You are a rigorous, constructive scientific referee in an interactive hypothesis evaluation session.
Inputs available to you:
- Original Hypothesis
- Current Verdict Score (0 to 9 scale) and Plain-language classification
- Scientific Synthesis of Plausible vs Contested Mechanisms
- Decisive Falsification Test
- Key Issues and Edge Conditions
- Retrieved Textbook Passages (P#) from the literature
- Untested Claims

The user will provide a follow-up prompt. This may be:
1. A refined or amended hypothesis with new constraints, biological subpopulations, or edge conditions.
2. A direct scientific question or probing counter-argument regarding specific passages, issues, or claims.
3. A request for experimental designs or literature verification.

Instructions:
- Ground your response in the provided passages and established physical/biological principles.
- Maintain an objective, constructive-adversarial scientific tone.
"""

FOLLOWUP_PLAN_SYS = """You prepare an adversarial follow-up investigation for a scientific hypothesis refinement or probe.
Context provided:
- Original Hypothesis
- Prior Verdict Score and Issues
- The user's new follow-up prompt or refined hypothesis

Your tasks:
1) Identify any new testable claims introduced by the follow-up prompt (ids C1_F{turn}, C2_F{turn}, ...).
2) Formulate 3-6 new adversarial search queries (kinds: impossibility, failure_mode, contradiction, limit) targeting the new claims, edge conditions, or specific questions raised. Never query for support. 3-10 words each.
3) RECALL: list up to 4 hard constraints or textbook principles bearing on the follow-up from: {books}"""

FOLLOWUP_REVIEW_SYS = """You are a rigorous scientific referee evaluating a follow-up refinement or probe of a hypothesis.
Inputs:
- ORIGINAL HYPOTHESIS & PREVIOUS VERDICT
- PREVIOUS ISSUES (with their IDs and edge conditions)
- FOLLOW-UP PROMPT / REFINED HYPOTHESIS
- CUMULATIVE PASSAGES (both previous passages and newly retrieved passages)

Deliverables:
1. viability_score (integer 0 to 9): Assign the updated empirical viability score for the refined hypothesis.
2. score_rationale: 1-2 sentences justifying the updated 0-9 score.
3. score_delta_explanation: Explicitly explain how and why the score changed from the previous turn (or why it remained unchanged).
4. resolved_issue_ids: List of issue IDs from previous turns that are now flipped or resolved by the proposed refinement/edge condition.
5. scientific_synthesis: Cumulative narrative synthesis: what mechanisms in the refined hypothesis are now plausible, and where empirical friction remains.
6. falsification_test: A concrete empirical test or measurement tailored to test the refined hypothesis.
7. new_issues: Any NEW objections grounded in the newly retrieved or cumulative passages with VERBATIM quotes (6-40 words).
8. untested_claims: Claims where the passages remain silent."""



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
               viability_score=getattr(rev, "viability_score", 5),
               score_rationale=getattr(rev, "score_rationale", ""),
               scientific_synthesis=rev.scientific_synthesis,
               falsification_test=rev.falsification_test,
               issues=issues,
               untested_claims=rev.untested_claims, degraded=degraded)
    rep["verdict"] = verdict(issues, getattr(rev, "viability_score", None), getattr(rev, "score_rationale", ""))
    m = re.match(r"^(\d+)/9", rep["verdict"])
    rep["score"] = int(m.group(1)) if m else getattr(rev, "viability_score", 5)
    return rep


def score_band(score):
    if score == 0:
        return "FAIL (Definitively Falsified)"
    elif 1 <= score <= 3:
        return "FAIL / SEVERELY CONTRADICTED"
    elif 4 <= score <= 6:
        return "CHALLENGED (Empirical Friction / Edge Conditions Required)"
    elif 7 <= score <= 8:
        return "SURVIVES (Plausible / Minor Caveats)"
    else:
        return "SURVIVES (Highly Robust / Uncontested)"


def score_explanation(score):
    if score == 0:
        return "The hypothesis violates established physical or biological invariants verified directly in the literature."
    elif 1 <= score <= 3:
        return "The hypothesis faces severe contradictions and critical failure modes documented in verified literature passages; core mechanisms appear unworkable without substantial revision."
    elif 4 <= score <= 6:
        return "The hypothesis faces notable empirical friction, counter-mechanisms, or conflicting findings in the literature. It is viable only under specific, constrained edge conditions or subpopulations."
    elif 7 <= score <= 8:
        return "The hypothesis is plausible and largely consistent with canonical literature. It survives adversarial challenge with only minor caveats or peripheral scope limits."
    else:
        return "The hypothesis is fully consistent with established principles in the current corpus; no disqualifying counter-evidence or contradictions were found."


def verdict(issues, referee_score=None, referee_rationale=""):
    ok_crit = sum(1 for i in issues if i["severity"] == "critical" and i["quote_status"] == "verified")
    ok_maj = sum(1 for i in issues if i["severity"] == "major" and i["quote_status"] == "verified")
    ok_min = sum(1 for i in issues if i["severity"] == "minor" and i["quote_status"] == "verified")
    unv_crit = sum(1 for i in issues if i["severity"] == "critical" and i["quote_status"] != "verified")

    if referee_score is not None:
        try:
            score = int(referee_score)
        except (ValueError, TypeError):
            score = 5
        # Reconcile referee assessment against verified issues
        if ok_crit >= 2:
            score = min(score, 1)
        elif ok_crit == 1:
            score = min(score, 3)
        elif ok_maj >= 2:
            score = min(score, 4)
        elif ok_maj == 1:
            score = min(score, 5)
        elif ok_min >= 1:
            score = min(score, 7)
        elif not issues:
            score = max(score, 8)

        if unv_crit >= 1 and ok_crit == 0:
            score = min(score, 6)
    else:
        if ok_crit >= 2:
            score = 1
        elif ok_crit == 1:
            score = 2
        elif ok_maj >= 2:
            score = 4
        elif ok_maj == 1:
            score = 5
        elif ok_min >= 1:
            score = 7
        elif unv_crit >= 1:
            score = 5
        else:
            score = 9

    score = max(0, min(9, int(score)))
    band = score_band(score)
    return f"{score}/9 ({band})"


def render(rep):
    v = rep["verdict"]
    m = re.match(r"^(\d+)/9", str(v))
    score_val = int(m.group(1)) if m else rep.get("score", 5)
    exp = score_explanation(score_val)
    hyp = rep.get("hypothesis", "").strip()

    L = []
    if hyp:
        L += ["## 🔬 Hypothesis / Prompt", f"> {hyp}", ""]

    L += [f"# Verdict Score: {v}"]
    if rep.get("score_rationale"):
        L += [f"> **Referee assessment rationale:** {rep['score_rationale']}", ""]
    L += [f"> **Plain-language explanation:** {exp}", ""]

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

    nb_url = os.environ.get("NOTEBOOKLM_URL", "").strip() or "https://notebooklm.google.com"
    L += ["", "## 📚 Recommended Reading & NotebookLM Investigation",
          f"> To investigate untested claims or examine biological/physical mechanisms further, you can cross-examine here is a [link to a Google NotebookLM pre-loaded with these canonical textbooks]({nb_url})",
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


def follow_up(history, user_message, rep, llm):
    """Answers follow-up prompts or evaluates hypothesis refinements against the current review."""
    hyp = rep.get("hypothesis", "")
    verdict_str = rep.get("verdict", "")
    synth = rep.get("scientific_synthesis", "")
    test = rep.get("falsification_test", "")
    issues_summary = "\n".join(
        f"- {i['id']} ({i['severity']}): {i['claim']} [passage: {i['passage_id']}, quote: '{i['quote']}'] (flips if: {i['edge_condition']})"
        for i in rep.get("issues", [])
    )
    passages_summary = "\n".join(
        f"- {p['id']} ({p['source']}): {p['text']}"
        for p in rep.get("passages", [])
    )
    untested_summary = "\n".join(f"- {u}" for u in rep.get("untested_claims", []))

    ctx = (
        f"ORIGINAL HYPOTHESIS:\n{hyp}\n\n"
        f"CURRENT VERDICT SCORE: {verdict_str}\n\n"
        f"SCIENTIFIC SYNTHESIS:\n{synth}\n\n"
        f"DECISIVE FALSIFICATION TEST:\n{test}\n\n"
        f"KEY ISSUES:\n{issues_summary or '(None)'}\n\n"
        f"RETRIEVED PASSAGES:\n{passages_summary or '(None)'}\n\n"
        f"UNTESTED CLAIMS:\n{untested_summary or '(None)'}\n\n"
    )

    hist_text = ""
    if history:
        hist_text = "CONVERSATION HISTORY:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        ) + "\n\n"

    prompt = f"{ctx}{hist_text}USER FOLLOW-UP PROMPT:\n{user_message}"
    return llm.text(FOLLOW_UP_SYS, prompt, temperature=0.3)


def run_followup(parent_rep, follow_up_prompt, books, gen, critic, retriever, turn_idx=1, auto_entail=False, check_math=False, log=print):
    """Executes a continuous adversarial research pipeline on a follow-up prompt or refined hypothesis."""
    log(f"[followup {turn_idx}] 1. Targeted adversarial plan for follow-up")
    prior_hyp = parent_rep.get("hypothesis") or parent_rep.get("parent_hypothesis") or ""
    prior_score = parent_rep.get("verdict", "")
    prior_issues_str = json.dumps([
        {"id": i["id"], "severity": i["severity"], "claim": i["claim"], "edge_condition": i.get("edge_condition", "")}
        for i in parent_rep.get("issues", []) if i.get("status", "active") == "active"
    ], indent=1)

    plan_ctx = (
        f"ORIGINAL HYPOTHESIS:\n{prior_hyp}\n\n"
        f"PRIOR VERDICT SCORE: {prior_score}\n\n"
        f"PRIOR ACTIVE ISSUES:\n{prior_issues_str}\n\n"
        f"FOLLOW-UP PROMPT / REFINEMENT:\n{follow_up_prompt}\n"
    )

    f_plan = FollowUpPlan.model_validate(
        gen.json(FOLLOWUP_PLAN_SYS.format(books=books, turn=turn_idx), plan_ctx, FollowUpPlan, 0.3)
    )

    log(f"[followup {turn_idx}] 2. Cumulative retrieval (Channel B) + reconcile")
    passages = list(parent_rep.get("passages", []))
    by_text = {p["text"]: p for p in passages}
    new_passage_ids = []

    def add_p(src, text, score, via):
        if text in by_text:
            p_obj = by_text[text]
            v = p_obj.get("via", [])
            if isinstance(v, list):
                p_obj["via"] = set(v)
            p_obj["via"].add(via)
            return p_obj["id"]
        pid = f"P{len(passages) + 1}"
        p = dict(id=pid, source=src, text=text, score=round(score, 2), via={via})
        passages.append(p)
        by_text[text] = p
        new_passage_ids.append(pid)
        return pid

    for q in f_plan.adversarial_queries:
        for src, txt, sc in retriever.search(q.text, 2):
            add_p(src, txt, sc, f"followup_{turn_idx}")

    for r in f_plan.recall:
        hits = retriever.search(r.search_terms, 1)
        if hits and hits[0][2] >= 2.0:
            add_p(hits[0][0], hits[0][1], hits[0][2], f"recall_{turn_idx}")

    log(f"[followup {turn_idx}] 3. Continuous adversarial review with issue tracking")
    review_ctx = (
        f"ORIGINAL HYPOTHESIS:\n{prior_hyp}\n\n"
        f"PREVIOUS VERDICT:\n{prior_score}\n\n"
        f"PREVIOUS ACTIVE ISSUES:\n{prior_issues_str}\n\n"
        f"FOLLOW-UP PROMPT / REFINEMENT:\n{follow_up_prompt}\n\n"
        f"NEW CLAIMS:\n{json.dumps([c.model_dump() for c in f_plan.new_claims])}\n\n"
        f"CUMULATIVE PASSAGES:\n{json.dumps([dict(id=p['id'], source=p['source'], text=p['text']) for p in passages], indent=1)}"
    )

    f_rev = FollowUpReview.model_validate(critic.json(FOLLOWUP_REVIEW_SYS, review_ctx, FollowUpReview, 0.0))

    log(f"[followup {turn_idx}] 4. Quote verification on new issues")
    pmap = {p["id"]: p for p in passages}
    grounded = [prior_hyp, follow_up_prompt] + [p["text"] for p in passages]
    new_issues = []
    for i in f_rev.new_issues:
        d = i.model_dump()
        p = pmap.get(i.passage_id)
        d["quote_status"] = verify_quote(i.quote, p["text"], retriever.chunks) if p else "bad_passage_id"
        d["eyeball_numbers"] = eyeball_numbers(i.claim + " " + i.edge_condition, grounded) if check_math else []
        d["auto_entail"] = None
        if auto_entail and d["quote_status"] == "verified":
            d["auto_entail"] = Entail.model_validate(critic.json(
                ENTAIL_SYS, f"CLAIM: {i.claim}\nQUOTE: {i.quote}", Entail, 0.0)).model_dump()
        new_issues.append(d)

    resolved_set = set(f_rev.resolved_issue_ids)
    cumulative_issues = []
    for old_i in parent_rep.get("issues", []):
        d = dict(old_i)
        if d["id"] in resolved_set:
            d["status"] = "resolved"
        else:
            d["status"] = d.get("status", "active")
        cumulative_issues.append(d)

    for ni in new_issues:
        ni["status"] = "active"
        cumulative_issues.append(ni)

    active_issues = [i for i in cumulative_issues if i["status"] == "active"]

    updated_verdict = verdict(active_issues, f_rev.viability_score, f_rev.score_rationale)
    m = re.match(r"^(\d+)/9", updated_verdict)
    score_val = int(m.group(1)) if m else f_rev.viability_score

    norm_passages = []
    for p in passages:
        via = p.get("via", [])
        if isinstance(via, set):
            via = sorted(via)
        norm_passages.append({**p, "via": via})

    followup_rep = dict(
        turn=turn_idx,
        parent_hypothesis=prior_hyp,
        follow_up_prompt=follow_up_prompt,
        previous_verdict=prior_score,
        verdict=updated_verdict,
        score=score_val,
        score_rationale=f_rev.score_rationale,
        score_delta_explanation=f_rev.score_delta_explanation,
        resolved_issue_ids=list(resolved_set),
        scientific_synthesis=f_rev.scientific_synthesis,
        falsification_test=f_rev.falsification_test,
        new_issues=new_issues,
        issues=cumulative_issues,
        active_issues=active_issues,
        passages=norm_passages,
        new_passage_ids=new_passage_ids,
        untested_claims=f_rev.untested_claims,
    )
    return followup_rep


def render_followup(rep):
    turn = rep.get("turn", 1)
    v = rep["verdict"]
    prev_v = rep.get("previous_verdict", "N/A")
    m = re.match(r"^(\d+)/9", str(v))
    score_val = int(m.group(1)) if m else rep.get("score", 5)
    exp = score_explanation(score_val)
    hyp = rep.get("parent_hypothesis", "").strip()
    f_prompt = rep.get("follow_up_prompt", "").strip()

    L = []
    L += [f"# Follow-Up Adversarial Review Report (Turn {turn})", ""]
    L += ["## 🔬 Research Lineage & Continuity",
          f"- **Original Hypothesis:** {hyp}",
          f"- **Follow-Up Prompt / Refinement:** {f_prompt}",
          f"- **Score Trajectory:** `{prev_v}` ➔ `{v}`", ""]

    L += [f"# Updated Verdict Score: {v}"]
    if rep.get("score_delta_explanation"):
        L += [f"> **Score Delta & Impact:** {rep['score_delta_explanation']}", ""]
    if rep.get("score_rationale"):
        L += [f"> **Referee assessment rationale:** {rep['score_rationale']}", ""]
    L += [f"> **Plain-language explanation:** {exp}", ""]

    synth = rep.get("scientific_synthesis", "").strip()
    test = rep.get("falsification_test", "").strip()
    if synth or test:
        L += ["## 🧠 Updated Scientific Synthesis & Evaluation"]
        if synth:
            L += [f"> {synth}", ""]
        if test:
            L += [f"> 🧪 **Refined Decisive Falsification Test:** {test}", ""]

    L += ["## Issue Board Status (Continuity Tracking)"]
    resolved_ids = set(rep.get("resolved_issue_ids", []))
    if resolved_ids:
        L += [f"### ✅ Resolved Issues ({len(resolved_ids)}):",
              "> The follow-up refinement or edge condition successfully resolved these prior objections:"]
        for i in rep.get("issues", []):
            if i["id"] in resolved_ids:
                L.append(f"- **{i['id']}** [RESOLVED] {i['severity']} / {i['category']}: {i['claim']}\n  _edge condition fulfilled:_ {i.get('edge_condition', '')}")
        L += [""]

    active = [i for i in rep.get("issues", []) if i.get("status") == "active"]
    if not active:
        L += ["_No active disqualifying issues or rule violations remain in the cumulative cited passages._", ""]
    else:
        L += [f"### ⚠️ Active Issues ({len(active)}):"]
        for i in active:
            is_new = " *(new this turn)*" if i["id"] in [ni["id"] for ni in rep.get("new_issues", [])] else ""
            flag = "" if i["quote_status"] == "verified" else f" **[quote {i['quote_status']}]**"
            eye = f" _(eyeball-check numbers: {i['eyeball_numbers']})_" if i.get("eyeball_numbers") else ""
            ent = f" _(auto-entail: {i['auto_entail']['label']})_" if i.get("auto_entail") else ""
            L.append(f"- **{i['id']}** {i['severity']} / {i['category']} / {i['passage_id']}{is_new}{flag}{eye}{ent}\n"
                     f"  {i['claim']}\n  > {i['quote']}\n  _flips if:_ {i['edge_condition']}")
        L += [""]

    new_pids = set(rep.get("new_passage_ids", []))
    if new_pids:
        L += ["## 📖 New Literature Retrieved in This Follow-Up",
              "> Deterministic corpus search retrieved the following new passages to test the follow-up claims:"]
        for p in rep.get("passages", []):
            if p["id"] in new_pids:
                excerpt = p["text"].replace("\n", " ")[:200] + "..." if len(p["text"]) > 200 else p["text"]
                L.append(f"- **{p['id']}** ({p['source']}): \"{excerpt}\"")
        L += [""]

    if rep.get("untested_claims"):
        L += ["## Untested Claims Remaining",
              "> Specific claims where the corpus remains silent:"]
        for u in rep["untested_claims"]:
            L.append(f"- {u}")
        L += [""]

    nb_url = os.environ.get("NOTEBOOKLM_URL", "").strip() or "https://notebooklm.google.com"
    L += ["## 📚 Recommended Reading & NotebookLM Investigation",
          f"> To investigate untested claims or examine biological/physical mechanisms further, you can cross-examine here is a [link to a Google NotebookLM pre-loaded with these canonical textbooks]({nb_url})",
          ""]

    return "\n".join(L)


