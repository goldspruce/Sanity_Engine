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

from pydantic import BaseModel, Field, model_validator


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
    id: str = "C1"
    text: str = ""

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            return {
                "id": str(data.get("id") or "C1"),
                "text": str(data.get("text") or "")
            }
        return data

class Query(BaseModel):
    text: str = ""
    kind: Literal["impossibility", "failure_mode", "contradiction", "limit"] = "limit"

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            k = str(data.get("kind") or "limit").lower().strip()
            if k not in ("impossibility", "failure_mode", "contradiction", "limit"):
                if "impos" in k: k = "impossibility"
                elif "fail" in k: k = "failure_mode"
                elif "contra" in k: k = "contradiction"
                else: k = "limit"
            return {
                "text": str(data.get("text") or ""),
                "kind": k
            }
        return data

class RecallItem(BaseModel):
    constraint: str = ""      # the limit/principle, from memory (unverified)
    book: str = ""            # best-guess source book
    search_terms: str = ""    # keywords likely to appear in the book text

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            return {
                "constraint": str(data.get("constraint") or ""),
                "book": str(data.get("book") or ""),
                "search_terms": str(data.get("search_terms") or "")
            }
        return data

class AttackPlan(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    queries: list[Query] = Field(default_factory=list)
    recall: list[RecallItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            out = dict(data)
            for f in ("claims", "queries", "recall"):
                if out.get(f) is None: out[f] = []
            return out
        return data

class GreenFlag(BaseModel):
    id: str = "G1"
    mechanism: str = ""       # What aspect of the hypothesis is biophysically or physically sound
    rationale: str = ""       # Why it is plausible according to canonical principles
    passage_id: str = ""      # P# if grounded in a passage
    quote: str = ""           # Verbatim quote if grounded

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            return {
                "id": str(data.get("id") or "G1"),
                "mechanism": str(data.get("mechanism") or ""),
                "rationale": str(data.get("rationale") or ""),
                "passage_id": str(data.get("passage_id") or ""),
                "quote": str(data.get("quote") or "")
            }
        return data

class Issue(BaseModel):
    id: str = "I1"
    flag: Literal["red", "yellow"] = "yellow"
    severity: Literal["critical", "major", "minor"] = "major"
    category: str = "Mechanism"
    claim_id: str = ""
    claim: str = ""           # the tension or objection
    passage_id: str = ""      # P# the objection rests on
    quote: str = ""           # verbatim from that passage
    edge_condition: str = ""  # what qualitative parameter change or boundary condition flips/resolves it
    status: Literal["active", "resolved"] = "active"

    @model_validator(mode="before")
    @classmethod
    def normalize_flags(cls, data):
        if isinstance(data, dict):
            out = dict(data)
            out["id"] = str(out.get("id") or "I1")
            flag = str(out.get("flag") or "").lower().strip()
            sev = str(out.get("severity") or "").lower().strip()
            if flag in ("red", "yellow"):
                out["flag"] = flag
            elif sev == "critical":
                out["flag"] = "red"
            else:
                out["flag"] = "yellow"

            if sev in ("critical", "major", "minor"):
                out["severity"] = sev
            else:
                out["severity"] = "critical" if out["flag"] == "red" else "major"

            for k in ("category", "claim_id", "claim", "passage_id", "quote", "edge_condition"):
                out[k] = str(out.get(k) or "")

            stat = str(out.get("status") or "active").lower().strip()
            out["status"] = stat if stat in ("active", "resolved") else "active"
            return out
        return data

class Review(BaseModel):
    diagnostic_stance: str = "" # Executive stance on hypothesis viability & tension
    scientific_synthesis: str = ""  # Narrative: biophysically plausible mechanisms vs empirical friction
    falsification_test: str = ""    # Concrete empirical experiment or measurement to test the yellow/red flags
    green_flags: list[GreenFlag] = Field(default_factory=list) # Sound biophysical anchors
    issues: list[Issue] = Field(default_factory=list)          # Red and Yellow flags
    untested_claims: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            out = dict(data)
            for k in ("diagnostic_stance", "scientific_synthesis", "falsification_test"):
                out[k] = str(out.get(k) or "")
            for k in ("green_flags", "issues", "untested_claims"):
                if out.get(k) is None: out[k] = []
            return out
        return data

class FollowUpPlan(BaseModel):
    new_claims: list[Claim] = Field(default_factory=list)
    adversarial_queries: list[Query] = Field(default_factory=list)
    recall: list[RecallItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            out = dict(data)
            for k in ("new_claims", "adversarial_queries", "recall"):
                if out.get(k) is None: out[k] = []
            return out
        return data

class FollowUpReview(BaseModel):
    diagnostic_stance: str = ""
    continuity_assessment: str = "" # How the follow-up changes the flag profile from previous inquiries
    resolved_issue_ids: list[str] = Field(default_factory=list) # Prior issue IDs now resolved/flipped
    green_flags: list[GreenFlag] = Field(default_factory=list)
    new_issues: list[Issue] = Field(default_factory=list)
    scientific_synthesis: str = ""
    falsification_test: str = ""
    untested_claims: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            out = dict(data)
            for k in ("diagnostic_stance", "continuity_assessment", "scientific_synthesis", "falsification_test"):
                out[k] = str(out.get(k) or "")
            for k in ("resolved_issue_ids", "green_flags", "new_issues", "untested_claims"):
                if out.get(k) is None: out[k] = []
            return out
        return data

class Entail(BaseModel):
    label: Literal["supports", "partly", "not", "unclear"] = "unclear"
    why: str = ""

    @model_validator(mode="before")
    @classmethod
    def clean(cls, data):
        if isinstance(data, dict):
            lbl = str(data.get("label") or "unclear").lower().strip()
            if lbl not in ("supports", "partly", "not", "unclear"):
                lbl = "unclear"
            return {"label": lbl, "why": str(data.get("why") or "")}
        return data


# ------------------------------------------------------------------ LLM backends
class GeminiLLM:
    """pip install google-genai ; env: GEMINI_API_KEY, GEMINI_MODEL"""
    def __init__(self, model=None):
        from google import genai
        self.client = genai.Client()
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def json(self, system, user, schema, temperature=0.0):
        import time
        from google.genai import types, errors
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                          response_mime_type="application/json", response_schema=schema)
        candidates = [self.model]
        for alt in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.8-flash"]:
            if alt not in candidates:
                candidates.append(alt)
        last_err = None
        for model_name in candidates:
            for attempt in range(3):
                try:
                    r = self.client.models.generate_content(model=model_name, contents=user, config=cfg)
                    txt = r.text or ""
                    if "```" in txt:
                        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.MULTILINE)
                        txt = re.sub(r"\s*```$", "", txt, flags=re.MULTILINE)
                    txt = txt.strip()
                    if not txt:
                        raise ValueError("Empty response from Gemini")
                    return json.loads(txt)
                except (errors.ServerError, errors.ClientError, json.JSONDecodeError, ValueError, TypeError) as e:
                    last_err = e
                    if "404" in str(e):
                        break
                    time.sleep(3.0 * (attempt + 1))
        raise last_err

    def text(self, system, user, temperature=0.3):
        import time
        from google.genai import types, errors
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        candidates = [self.model]
        for alt in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.8-flash"]:
            if alt not in candidates:
                candidates.append(alt)
        last_err = None
        for model_name in candidates:
            for attempt in range(3):
                try:
                    r = self.client.models.generate_content(model=model_name, contents=user, config=cfg)
                    return r.text or ""
                except (errors.ServerError, errors.ClientError) as e:
                    last_err = e
                    if "404" in str(e):
                        break
                    time.sleep(3.0 * (attempt + 1))
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

REVIEW_SYS = """You are a rigorous, constructive scientific referee stress-testing a hypothesis against literature passages.
Inputs: HYPOTHESIS, CLAIMS (C#), PASSAGES (P#, real text retrieved from textbooks).

Your role is NOT to pass/fail or compute an arbitrary score, but to act as a high-resolution issue-spotting radar:
- 🟩 GREEN FLAGS: Identify solid scientific ground, biophysically sound mechanisms, and alignments with canonical principles.
- 🟨 YELLOW FLAGS: Pinpoint empirical friction, conflicting studies, and parameter regime constraints; formulate workable edge conditions.
- 🚩 RED FLAGS: Highlight any violation of established physical invariants or biological conservation laws.

Deliverables:
1. diagnostic_stance: A 1-2 sentence executive assessment of the hypothesis's empirical standing.
2. scientific_synthesis: A 2-4 sentence narrative synthesis: acknowledge biophysically plausible mechanisms and explain where empirical friction or competing models exist.
3. falsification_test: Describe a concrete, realistic empirical experiment, clinical measurement, or benchtop test that would decisively test the flagged tensions.
4. green_flags: 1-4 points detailing aspects of the hypothesis that are biophysically plausible, mathematically sound, or supported by canonical principles.
5. issues: Structured red and yellow flags. Every issue must rest on ONE passage with a VERBATIM quote (6-40 words) copied exactly from that passage. Do not paraphrase inside `quote`. Do not introduce numbers not in the text.
   Flag types:
   - red = Fatal invariant or conservation law violation (e.g. conservation of energy, absolute thermodynamic impossibility).
   - yellow = Empirical friction, competing literature, or parameter regime constraints. Must state a qualitative `edge_condition` describing what parameter change or boundary condition flips/resolves it.
6. untested_claims: Claims where the passages are silent and provide no definitive test. Never guess."""

ENTAIL_SYS = "Decide whether the QUOTE supports the CLAIM. Use only the quote. Answer supports / partly / not / unclear with a one-sentence reason."

FOLLOW_UP_SYS = """You are a rigorous, constructive scientific referee in an interactive hypothesis evaluation session.
Inputs available to you:
- Original Hypothesis
- Diagnostic Flags Summary (Red, Yellow, Green)
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
- Prior Diagnostic Flags and Issues
- The user's new follow-up prompt or refined hypothesis

Your tasks:
1) Identify any new testable claims introduced by the follow-up prompt (ids C1_F{turn}, C2_F{turn}, ...).
2) Formulate 3-6 new adversarial search queries (kinds: impossibility, failure_mode, contradiction, limit) targeting the new claims, edge conditions, or specific questions raised. Never query for support. 3-10 words each.
3) RECALL: list up to 4 hard constraints or textbook principles bearing on the follow-up from: {books}"""

FOLLOWUP_REVIEW_SYS = """You are a rigorous scientific referee evaluating a follow-up refinement or probe of a hypothesis.
Inputs:
- ORIGINAL HYPOTHESIS & PREVIOUS FLAGS
- PREVIOUS ISSUES (with their IDs and edge conditions)
- FOLLOW-UP PROMPT / REFINED HYPOTHESIS
- CUMULATIVE PASSAGES (both previous passages and newly retrieved passages)

Deliverables:
1. diagnostic_stance: Updated 1-2 sentence executive diagnostic of the refined hypothesis.
2. continuity_assessment: Explicitly explain how the follow-up changes the flag profile (e.g., "Issue I1 resolved: adding quarter-wave impedance matching removes the red flag boundary reflection; yellow flag attenuation remains active").
3. resolved_issue_ids: List of issue IDs from previous inquiries that are now resolved or flipped by the follow-up.
4. green_flags: Updated or newly confirmed biophysical anchors and synergies.
5. scientific_synthesis: Cumulative narrative synthesis of plausible vs contested mechanisms.
6. falsification_test: A concrete empirical test tailored to the refined hypothesis.
7. new_issues: Any NEW red or yellow flags grounded in newly retrieved passages with VERBATIM quotes (6-40 words).
8. untested_claims: Claims where the literature remains silent."""



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

    green_flags = []
    for g in getattr(rev, "green_flags", []):
        gd = g.model_dump()
        if gd.get("passage_id") and gd.get("quote"):
            p = pmap.get(gd["passage_id"])
            gd["quote_status"] = verify_quote(gd["quote"], p["text"], retriever.chunks) if p else "bad_passage_id"
        else:
            gd["quote_status"] = "unquoted"
        green_flags.append(gd)

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

    counts = count_flags(issues, green_flags)
    diag_badge = diagnostic_badge(counts)
    diag_class = diagnostic_classification(counts)

    rep = dict(books=books, hypothesis=hyp, plan=plan.model_dump(),
               passages=[{**p, "via": sorted(p["via"])} for p in passages],
               reconcile=reconcile, retrieval_only=retrieval_only,
               diagnostic_stance=getattr(rev, "diagnostic_stance", ""),
               counts=counts,
               badge=diag_badge,
               classification=diag_class,
               scientific_synthesis=rev.scientific_synthesis,
               falsification_test=rev.falsification_test,
               green_flags=green_flags,
               issues=issues,
               untested_claims=rev.untested_claims, degraded=degraded)
    rep["verdict"] = verdict(issues, green_flags)
    return rep


def count_flags(issues, green_flags=None):
    active = [i for i in issues if i.get("status", "active") == "active"]
    red_verified = sum(1 for i in active if i.get("flag") == "red" and i.get("quote_status") == "verified")
    yellow_verified = sum(1 for i in active if i.get("flag") == "yellow" and i.get("quote_status") == "verified")
    red_unv = sum(1 for i in active if i.get("flag") == "red" and i.get("quote_status") != "verified")
    yellow_unv = sum(1 for i in active if i.get("flag") == "yellow" and i.get("quote_status") != "verified")

    # Graceful fallback for legacy severity labels if flag wasn't set
    if red_verified == 0 and red_unv == 0:
        red_verified = sum(1 for i in active if i.get("severity") == "critical" and i.get("quote_status") == "verified")
        red_unv = sum(1 for i in active if i.get("severity") == "critical" and i.get("quote_status") != "verified")
    if yellow_verified == 0 and yellow_unv == 0:
        yellow_verified = sum(1 for i in active if i.get("severity") in ("major", "minor") and i.get("quote_status") == "verified")
        yellow_unv = sum(1 for i in active if i.get("severity") in ("major", "minor") and i.get("quote_status") != "verified")

    green_count = len(green_flags or [])
    return {
        "red": red_verified,
        "yellow": yellow_verified,
        "green": green_count,
        "red_unverified": red_unv,
        "yellow_unverified": yellow_unv,
    }


def diagnostic_badge(counts):
    return f"🚩 {counts['red']} Red | 🟨 {counts['yellow']} Yellow | 🟩 {counts['green']} Green"


def diagnostic_classification(counts):
    if counts["red"] > 0:
        return "CRITICAL INVARIANT FRICTION (Violation of physical/biological conservation laws)"
    elif counts["yellow"] > 0:
        return "CONDITIONAL VIABILITY (Empirical friction / edge conditions required)"
    elif counts["green"] > 0:
        return "CANONICALLY ANCHORED (Plausible mechanisms, unrefuted by literature)"
    else:
        return "OPEN / UNCONSTRAINED (No canonical flags identified)"


def diagnostic_explanation(classification):
    if "CRITICAL" in classification:
        return "The hypothesis encounters verified Red Flags indicating fundamental physical, thermodynamic, or biophysical conservation law violations."
    elif "CONDITIONAL" in classification:
        return "The hypothesis encounters verified Yellow Flags representing empirical friction, conflicting studies, or parameter regime constraints; viable only under specific boundary edge conditions."
    elif "CANONICALLY" in classification:
        return "The hypothesis is anchored on verified Green Flags consistent with canonical principles, with no disqualifying Red Flags in the current corpus."
    else:
        return "The hypothesis mechanisms are currently unconstrained by the available literature corpus."


def verdict(issues, green_flags=None, *args, **kwargs):
    counts = count_flags(issues, green_flags)
    badge = diagnostic_badge(counts)
    classification = diagnostic_classification(counts)
    return f"{badge} — {classification}"


def render(rep):
    counts = rep.get("counts") or count_flags(rep.get("issues", []), rep.get("green_flags", []))
    badge = rep.get("badge") or diagnostic_badge(counts)
    classification = rep.get("classification") or diagnostic_classification(counts)
    stance = (rep.get("diagnostic_stance") or "").strip()
    explanation = diagnostic_explanation(classification)
    hyp = (rep.get("hypothesis") or "").strip()

    L = []
    if hyp:
        L += ["## 🔬 Original Prompt", f"> {hyp}", ""]

    L += [f"# Diagnostic Radar: {badge}"]
    L += [f"> **Classification:** {classification}", ""]
    if stance:
        L += [f"> **Diagnostic Stance:** {stance}", ""]
    L += [f"> **Diagnostic Analysis:** {explanation}", ""]

    synth = (rep.get("scientific_synthesis") or "").strip()
    test = (rep.get("falsification_test") or "").strip()
    if synth or test:
        L += ["## 🧠 Scientific Synthesis & Evaluation"]
        if synth:
            L += [f"> {synth}", ""]
        if test:
            L += [f"> 🧪 **Decisive Falsification Test:** {test}", ""]
    if rep.get("degraded"):
        L += ["**Degraded:** " + "; ".join(rep["degraded"]), ""]

    # Issue Spotting Board
    L += ["## 🎯 Issue Spotting Board"]

    # 🟩 Green Flags
    green_flags = rep.get("green_flags", [])
    L += [f"### 🟩 Green Flags ({len(green_flags)})",
          "> Biophysically plausible mechanisms, mathematical consistency, or canonical textbook alignments:"]
    if not green_flags:
        L += ["_No explicit green flags identified in the cited passages._", ""]
    else:
        for idx, g in enumerate(green_flags, 1):
            gid = g.get("id") or f"G{idx}"
            pass_str = f" / {g['passage_id']}" if g.get("passage_id") else ""
            quote_str = f"\n  > \"{g['quote']}\"" if g.get("quote") else ""
            L.append(f"- **{gid}** {g.get('mechanism', '')}{pass_str}\n  {g.get('rationale', '')}{quote_str}")
        L += [""]

    # 🟨 Yellow Flags
    yellow_flags = [i for i in rep.get("issues", []) if i.get("flag") == "yellow" or (not i.get("flag") and i.get("severity") in ("major", "minor"))]
    L += [f"### 🟨 Yellow Flags ({len(yellow_flags)})",
          "> Empirical friction, competing literature, or parameter regime constraints (paired with workable edge conditions):"]
    if not yellow_flags:
        L += ["_No yellow flags encountered._", ""]
    else:
        for i in yellow_flags:
            flag_warn = "" if i.get("quote_status") == "verified" else f" **[quote {i.get('quote_status')}]**"
            eye = f" _(eyeball-check numbers: {i['eyeball_numbers']})_" if i.get("eyeball_numbers") else ""
            ent = f" _(auto-entail: {i['auto_entail']['label']})_" if i.get("auto_entail") else ""
            L.append(f"- **{i['id']}** {i.get('category', 'Friction')} / {i.get('passage_id', '')}{flag_warn}{eye}{ent}\n"
                     f"  {i['claim']}\n  > \"{i['quote']}\"\n  _flips if (edge condition):_ {i.get('edge_condition', 'N/A')}")
        L += [""]

    # 🚩 Red Flags
    red_flags = [i for i in rep.get("issues", []) if i.get("flag") == "red" or (not i.get("flag") and i.get("severity") == "critical")]
    L += [f"### 🚩 Red Flags ({len(red_flags)})",
          "> Invariant or conservation law violations (thermodynamics, energy conservation, physical impossibility):"]
    if not red_flags:
        L += ["_No fatal red flags encountered._", ""]
    else:
        for i in red_flags:
            flag_warn = "" if i.get("quote_status") == "verified" else f" **[quote {i.get('quote_status')}]**"
            eye = f" _(eyeball-check numbers: {i['eyeball_numbers']})_" if i.get("eyeball_numbers") else ""
            ent = f" _(auto-entail: {i['auto_entail']['label']})_" if i.get("auto_entail") else ""
            L.append(f"- **{i['id']}** {i.get('category', 'Invariant')} / {i.get('passage_id', '')}{flag_warn}{eye}{ent}\n"
                     f"  {i['claim']}\n  > \"{i['quote']}\"\n  _flips if (edge condition):_ {i.get('edge_condition', 'N/A')}")
        L += [""]

    L += ["## Channel A vs B (Reconciliation)",
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
    badge = rep.get("badge") or rep.get("verdict", "")
    stance = rep.get("diagnostic_stance", "")
    synth = rep.get("scientific_synthesis", "")
    test = rep.get("falsification_test", "")
    green_summary = "\n".join(
        f"- {g.get('id', 'G')}: {g.get('mechanism', '')} ({g.get('rationale', '')})"
        for g in rep.get("green_flags", [])
    )
    issues_summary = "\n".join(
        f"- {i['id']} [{i.get('flag', i.get('severity', 'yellow')).upper()}]: {i['claim']} [passage: {i['passage_id']}, quote: '{i['quote']}'] (flips if: {i.get('edge_condition', 'N/A')})"
        for i in rep.get("issues", [])
    )
    passages_summary = "\n".join(
        f"- {p['id']} ({p['source']}): {p['text']}"
        for p in rep.get("passages", [])
    )
    untested_summary = "\n".join(f"- {u}" for u in rep.get("untested_claims", []))

    ctx = (
        f"ORIGINAL HYPOTHESIS:\n{hyp}\n\n"
        f"CURRENT DIAGNOSTIC FLAGS: {badge}\n\n"
        f"DIAGNOSTIC STANCE:\n{stance}\n\n"
        f"GREEN FLAGS:\n{green_summary or '(None)'}\n\n"
        f"SCIENTIFIC SYNTHESIS:\n{synth}\n\n"
        f"DECISIVE FALSIFICATION TEST:\n{test}\n\n"
        f"FLAGGED ISSUES (RED & YELLOW):\n{issues_summary or '(None)'}\n\n"
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
    log(f"[follow-up #{turn_idx}] 1. Targeted adversarial plan for follow-up")
    prior_hyp = parent_rep.get("hypothesis") or parent_rep.get("parent_hypothesis") or ""
    prior_badge = parent_rep.get("badge") or parent_rep.get("verdict", "")
    prior_active_issues = [i for i in parent_rep.get("issues", []) if i.get("status", "active") == "active"]
    prior_issues_str = json.dumps([
        {
            "id": i.get("id", "I"),
            "flag": i.get("flag", "yellow"),
            "severity": i.get("severity", "major"),
            "claim": i.get("claim", ""),
            "edge_condition": i.get("edge_condition", "")
        }
        for i in prior_active_issues
    ], indent=1)

    plan_ctx = (
        f"ORIGINAL HYPOTHESIS:\n{prior_hyp}\n\n"
        f"PRIOR DIAGNOSTIC FLAGS: {prior_badge}\n\n"
        f"PRIOR ACTIVE ISSUES:\n{prior_issues_str}\n\n"
        f"FOLLOW-UP PROMPT / REFINEMENT:\n{follow_up_prompt}\n"
    )

    f_plan = FollowUpPlan.model_validate(
        gen.json(FOLLOWUP_PLAN_SYS.format(books=books, turn=turn_idx), plan_ctx, FollowUpPlan, 0.3)
    )

    log(f"[follow-up #{turn_idx}] 2. Cumulative retrieval (Channel B) + reconcile")
    # Clone passages so parent_rep is NEVER mutated in-place
    passages = [{**p, "via": list(p.get("via", []))} for p in parent_rep.get("passages", [])]
    by_text = {p["text"]: p for p in passages}
    new_passage_ids = []

    def add_p(src, text, score, via):
        if text in by_text:
            p_obj = by_text[text]
            v = list(p_obj.get("via", []))
            if via not in v:
                v.append(via)
            p_obj["via"] = v
            return p_obj["id"]
        pid = f"P{len(passages) + 1}"
        p = dict(id=pid, source=src, text=text, score=round(score, 2), via=[via])
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

    log(f"[follow-up #{turn_idx}] 3. Continuous adversarial review with issue tracking")
    review_ctx = (
        f"ORIGINAL HYPOTHESIS:\n{prior_hyp}\n\n"
        f"PREVIOUS DIAGNOSTIC FLAGS:\n{prior_badge}\n\n"
        f"PREVIOUS ACTIVE ISSUES:\n{prior_issues_str}\n\n"
        f"FOLLOW-UP PROMPT / REFINEMENT:\n{follow_up_prompt}\n\n"
        f"NEW CLAIMS:\n{json.dumps([c.model_dump() for c in f_plan.new_claims])}\n\n"
        f"CUMULATIVE PASSAGES:\n{json.dumps([dict(id=p['id'], source=p['source'], text=p['text']) for p in passages], indent=1)}"
    )

    f_rev = FollowUpReview.model_validate(critic.json(FOLLOWUP_REVIEW_SYS, review_ctx, FollowUpReview, 0.0))

    log(f"[follow-up #{turn_idx}] 4. Quote verification on new issues & green flags")
    pmap = {p["id"]: p for p in passages}
    grounded = [prior_hyp, follow_up_prompt] + [p["text"] for p in passages]

    new_issues = []
    for i in f_rev.new_issues:
        d = i.model_dump()
        p = pmap.get(i.passage_id)
        d["quote_status"] = verify_quote(i.quote, p["text"], retriever.chunks) if p else "bad_passage_id"
        d["eyeball_numbers"] = eyeball_numbers((i.claim or "") + " " + (i.edge_condition or ""), grounded) if check_math else []
        d["auto_entail"] = None
        if auto_entail and d["quote_status"] == "verified":
            d["auto_entail"] = Entail.model_validate(critic.json(
                ENTAIL_SYS, f"CLAIM: {i.claim}\nQUOTE: {i.quote}", Entail, 0.0)).model_dump()
        new_issues.append(d)

    prior_resolved = set(parent_rep.get("resolved_issue_ids", []))
    resolved_set = prior_resolved | set(f_rev.resolved_issue_ids or [])
    cumulative_issues = []
    for old_i in parent_rep.get("issues", []):
        d = dict(old_i)
        if d.get("id") in resolved_set:
            d["status"] = "resolved"
        else:
            d["status"] = d.get("status", "active")
        cumulative_issues.append(d)

    for ni in new_issues:
        ni["status"] = "active"
        cumulative_issues.append(ni)

    # Merge green flags across follow-up inquiries
    parent_greens = [dict(g) for g in parent_rep.get("green_flags", [])]
    new_greens = []
    for g in getattr(f_rev, "green_flags", []):
        gd = g.model_dump() if hasattr(g, "model_dump") else dict(g)
        if gd.get("passage_id") and gd.get("quote"):
            p = pmap.get(gd["passage_id"])
            gd["quote_status"] = verify_quote(gd.get("quote", ""), p["text"], retriever.chunks) if p else "bad_passage_id"
        else:
            gd["quote_status"] = "unquoted"
        new_greens.append(gd)

    cumulative_greens = list(parent_greens)
    known_green_mech = {(g.get("mechanism") or "").strip().lower() for g in parent_greens if (g.get("mechanism") or "").strip()}
    for ng in new_greens:
        mech = (ng.get("mechanism") or "").strip().lower()
        if not mech or mech not in known_green_mech:
            cumulative_greens.append(ng)
            if mech:
                known_green_mech.add(mech)

    active_issues = [i for i in cumulative_issues if i["status"] == "active"]
    counts = count_flags(active_issues, cumulative_greens)
    updated_badge = diagnostic_badge(counts)
    updated_class = diagnostic_classification(counts)

    norm_passages = []
    for p in passages:
        via = p.get("via", [])
        if isinstance(via, (set, list)):
            via = sorted(set(via))
        else:
            via = [via]
        norm_passages.append({**p, "via": list(via)})

    followup_rep = dict(
        turn=turn_idx,
        parent_hypothesis=prior_hyp,
        follow_up_prompt=follow_up_prompt,
        previous_badge=prior_badge,
        badge=updated_badge,
        classification=updated_class,
        counts=counts,
        verdict=f"{updated_badge} — {updated_class}",
        diagnostic_stance=getattr(f_rev, "diagnostic_stance", ""),
        continuity_assessment=getattr(f_rev, "continuity_assessment", ""),
        resolved_issue_ids=list(resolved_set),
        scientific_synthesis=f_rev.scientific_synthesis,
        falsification_test=f_rev.falsification_test,
        green_flags=cumulative_greens,
        new_green_flags=new_greens,
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
    badge = rep.get("badge") or diagnostic_badge(rep.get("counts", count_flags(rep.get("active_issues", rep.get("issues", [])), rep.get("green_flags", []))))
    prev_badge = rep.get("previous_badge") or rep.get("previous_verdict", "N/A")
    classification = rep.get("classification") or diagnostic_classification(rep.get("counts", count_flags(rep.get("active_issues", rep.get("issues", [])), rep.get("green_flags", []))))
    stance = (rep.get("diagnostic_stance") or "").strip()
    continuity = (rep.get("continuity_assessment") or "").strip()
    explanation = diagnostic_explanation(classification)
    hyp = (rep.get("parent_hypothesis") or "").strip()
    f_prompt = (rep.get("follow_up_prompt") or "").strip()

    L = []
    L += [f"# Follow-Up Adversarial Review Report (Follow-up #{turn})", ""]
    L += ["## 🔬 Research Lineage & Continuity",
          f"- **Original Prompt:** {hyp}",
          f"- **Follow-up #{turn} Prompt:** {f_prompt}",
          f"- **Flag Trajectory:** `{prev_badge}` ➔ `{badge}`", ""]

    L += [f"# Updated Diagnostic Radar: {badge}"]
    L += [f"> **Classification:** {classification}", ""]
    if stance:
        L += [f"> **Diagnostic Stance:** {stance}", ""]
    if continuity:
        L += [f"> **Continuity & Flag Trajectory Assessment:** {continuity}", ""]
    L += [f"> **Diagnostic Analysis:** {explanation}", ""]

    synth = (rep.get("scientific_synthesis") or "").strip()
    test = (rep.get("falsification_test") or "").strip()
    if synth or test:
        L += ["## 🧠 Updated Scientific Synthesis & Evaluation"]
        if synth:
            L += [f"> {synth}", ""]
        if test:
            L += [f"> 🧪 **Refined Decisive Falsification Test:** {test}", ""]

    # Issue Spotting Board Status (Continuity Tracking)
    L += ["## 🎯 Issue Spotting Board Status (Continuity Tracking)"]

    # 🟩 Green Flags
    green_flags = rep.get("green_flags", [])
    L += [f"### 🟩 Green Flags ({len(green_flags)})",
          "> Anchoring biophysical mechanisms, mathematical viability, and confirmed principles:"]
    if not green_flags:
        L += ["_No green flags identified._", ""]
    else:
        for idx, g in enumerate(green_flags, 1):
            gid = g.get("id") or f"G{idx}"
            pass_str = f" / {g['passage_id']}" if g.get("passage_id") else ""
            quote_str = f"\n  > \"{g['quote']}\"" if g.get("quote") else ""
            L.append(f"- **{gid}** {g.get('mechanism', '')}{pass_str}\n  {g.get('rationale', '')}{quote_str}")
        L += [""]

    # Resolved Issues
    resolved_ids = set(rep.get("resolved_issue_ids", []))
    if resolved_ids:
        L += [f"### ✅ Resolved Prior Issues ({len(resolved_ids)})",
              "> The follow-up refinement or edge condition successfully resolved these prior objections:"]
        for i in rep.get("issues", []):
            if i["id"] in resolved_ids:
                flag_type = i.get("flag", "yellow").upper()
                L.append(f"- **{i['id']}** [RESOLVED] [{flag_type}] {i.get('category', '')}: {i['claim']}\n  _edge condition fulfilled:_ {i.get('edge_condition', '')}")
        L += [""]

    # Active Issues
    active = rep.get("active_issues", [i for i in rep.get("issues", []) if i.get("status") == "active"])
    active_yellow = [i for i in active if i.get("flag") == "yellow" or (not i.get("flag") and i.get("severity") in ("major", "minor"))]
    active_red = [i for i in active if i.get("flag") == "red" or (not i.get("flag") and i.get("severity") == "critical")]

    L += [f"### 🟨 Active Yellow Flags ({len(active_yellow)})",
          "> Remaining empirical friction, competing models, or parameter regime constraints:"]
    if not active_yellow:
        L += ["_No active yellow flags remain._", ""]
    else:
        for i in active_yellow:
            is_new = " *(new in this follow-up)*" if i["id"] in [ni["id"] for ni in rep.get("new_issues", [])] else ""
            flag_warn = "" if i.get("quote_status") == "verified" else f" **[quote {i.get('quote_status')}]**"
            eye = f" _(eyeball-check numbers: {i['eyeball_numbers']})_" if i.get("eyeball_numbers") else ""
            ent = f" _(auto-entail: {i['auto_entail']['label']})_" if i.get("auto_entail") else ""
            L.append(f"- **{i['id']}** {i.get('category', 'Friction')} / {i.get('passage_id', '')}{is_new}{flag_warn}{eye}{ent}\n"
                     f"  {i['claim']}\n  > \"{i['quote']}\"\n  _flips if:_ {i.get('edge_condition', 'N/A')}")
        L += [""]

    L += [f"### 🚩 Active Red Flags ({len(active_red)})",
          "> Unresolved physical/biological invariant violations:"]
    if not active_red:
        L += ["_No active red flags remain._", ""]
    else:
        for i in active_red:
            is_new = " *(new in this follow-up)*" if i["id"] in [ni["id"] for ni in rep.get("new_issues", [])] else ""
            flag_warn = "" if i.get("quote_status") == "verified" else f" **[quote {i.get('quote_status')}]**"
            eye = f" _(eyeball-check numbers: {i['eyeball_numbers']})_" if i.get("eyeball_numbers") else ""
            ent = f" _(auto-entail: {i['auto_entail']['label']})_" if i.get("auto_entail") else ""
            L.append(f"- **{i['id']}** {i.get('category', 'Invariant')} / {i.get('passage_id', '')}{is_new}{flag_warn}{eye}{ent}\n"
                     f"  {i['claim']}\n  > \"{i['quote']}\"\n  _flips if:_ {i.get('edge_condition', 'N/A')}")
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


