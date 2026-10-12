"""Optional expert triage. build() writes blind cards + response sheet; merge() folds rulings into a final report."""
import csv, json, os, re
import engine

OPTIONS = {
    "E": ["supports", "partly", "not", "unclear"],
    "R": ["in_books", "real_not_in_books", "not_real", "unclear"],
    "I": ["serious", "minor", "not_flaw", "unclear"],
    "U": ["settled", "open_gap", "out_of_expertise"],
}
QUESTION = {
    "E": "Does this passage actually support the claim?",
    "R": "A model recalled this limit but our search did not find it. Is it real, and is it in the books?",
    "I": "Is this a real flaw in the hypothesis, and how serious?",
    "U": "Nothing in the corpus tested this. Can you settle it from your knowledge?",
}


def _search_string(text, n=8):
    return " ".join(re.findall(r"\w+", text)[:n])


def build_cards(rep, max_cards=12):
    pm = {p["id"]: p for p in rep["passages"]}
    cards = []

    def card(typ, **kw):
        cards.append(dict(type=typ, question=QUESTION[typ], options=OPTIONS[typ], response=None, note=None,
                          reviewer=None, **kw))

    for sev in ("critical", "major"):
        for i in rep["issues"]:
            if i["severity"] == sev:
                p = pm.get(i["passage_id"])
                card("I", issue_id=i["id"], claim=i["claim"], passage_id=i["passage_id"],
                     source=p["source"] if p else None, quote=i["quote"],
                     notebook_search=_search_string(i["quote"]),
                     hidden=dict(severity=i["severity"], quote_status=i["quote_status"], auto_entail=i.get("auto_entail")))
    for r in rep["reconcile"]:
        if r["bucket"] == "recall_only":
            card("R", claim=r["item"]["constraint"], source=r["item"]["book"], quote=None, passage_id=None,
                 notebook_search=r["item"]["search_terms"], hidden=dict(bucket="recall_only"))
    for i in rep["issues"]:
        if i["quote_status"] == "verified":
            p = pm[i["passage_id"]]
            card("E", issue_id=i["id"], claim=i["claim"], passage_id=i["passage_id"], source=p["source"],
                 quote=i["quote"], notebook_search=_search_string(i["quote"]),
                 hidden=dict(severity=i["severity"], auto_entail=i.get("auto_entail")))
    for u in rep["untested_claims"]:
        card("U", claim=u, source=None, quote=None, passage_id=None, notebook_search=_search_string(u),
             hidden=dict())
    for n, c in enumerate(cards, 1):
        c["id"] = f"C{n}"
        c["rank"] = n
    return cards[:max_cards], cards[max_cards:]


def build(rep, outdir, max_cards=12):
    shown, overflow = build_cards(rep, max_cards)
    json.dump(dict(cards=shown, overflow=overflow), open(f"{outdir}/triage.json", "w"), indent=1)
    hyp = rep.get("hypothesis", "").strip()
    L = ["# Expert triage (blind view)", ""]
    if hyp:
        L += ["## 🔬 Hypothesis Under Review", f"> {hyp}", ""]
    L += ["Read REVIEWER_GUIDE.md first. Do NOT open report.md until all cards are answered.",
         "Record answers in responses.csv (or the Google Sheet copy). 'unclear' and 'out_of_expertise' are valid answers.", ""]
    for c in shown:
        L += [f"## {c['id']}  [{c['type']}]  rank {c['rank']}", f"**{c['question']}**", "",
              f"CLAIM: {c['claim']}"]
        if c.get("quote"):
            qs = c.get("hidden", {}).get("quote_status")
            warn = "" if qs in (None, "verified") else f"  **[WARNING: this quote was NOT found in the cited passage ({qs}); it may be invented]**"
            L += [f"PASSAGE {c['passage_id']} ({c['source']}):{warn}", f"> {c['quote']}"]
        elif c.get("source"):
            L += [f"BOOK (model's guess, unverified): {c['source']}"]
        L += [f"NOTEBOOK SEARCH: `{c['notebook_search']}`", f"OPTIONS: {' | '.join(c['options'])}", ""]
    if overflow:
        L += [f"_{len(overflow)} further items were not queued; they will be listed as NOT REVIEWED._"]
    open(f"{outdir}/triage.md", "w").write("\n".join(L))
    with open(f"{outdir}/responses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "summary", "options", "response", "note", "reviewer"])
        for c in shown:
            w.writerow([c["id"], c["type"], c["claim"][:120], " | ".join(c["options"]), "", "", ""])
    return len(shown), len(overflow)


def merge(outdir, responses_path=None):
    rep = json.load(open(f"{outdir}/report.json"))
    tri = json.load(open(f"{outdir}/triage.json"))
    rows = {r["id"]: r for r in csv.DictReader(open(responses_path or f"{outdir}/responses.csv"))}
    problems, ruled, rejected = [], [], set()
    for c in tri["cards"]:
        r = rows.get(c["id"], {})
        resp = (r.get("response") or "").strip()
        if not resp:
            continue
        if resp not in c["options"]:
            problems.append(f"{c['id']}: '{resp}' not in {c['options']}")
            continue
        ruled.append(dict(card=c, response=resp, note=r.get("note", ""), reviewer=r.get("reviewer", "")))
        if (c["type"] == "I" and resp == "not_flaw") or (c["type"] == "E" and resp == "not"):
            rejected.add(c["issue_id"])
    answered = {x["card"]["id"] for x in ruled}
    unreviewed = [c for c in tri["cards"] if c["id"] not in answered] + tri["overflow"]
    unresolved = [x for x in ruled if x["response"] in ("unclear", "out_of_expertise")]
    remaining = [i for i in rep["issues"] if i["id"] not in rejected]
    adj = engine.verdict(remaining)
    if ruled:
        adj += "  [expert-adjusted]"
    crit_open = [i for i in remaining if i["severity"] == "critical" and i["quote_status"] == "verified"
                 and not any(x["card"].get("issue_id") == i["id"] and x["response"] in ("serious", "supports")
                             for x in ruled)]
    hyp = rep.get("hypothesis", "").strip()
    L = ["# Final Adversarial Review Report", ""]
    if hyp:
        L += ["## 🔬 Hypothesis / Prompt", f"> {hyp}", ""]
    L += [f"## Final verdict: {adj}", f"Engine verdict before review: {rep['verdict']}", ""]
    if crit_open and ruled:
        L += [f"**{len(crit_open)} critical issue(s) not confirmed by a reviewer:** {[i['id'] for i in crit_open]}", ""]
    if problems:
        L += ["**Invalid responses ignored:** " + "; ".join(problems), ""]
    L += ["## Expert rulings"]
    for x in ruled:
        c = x["card"]
        L.append(f"- {c['id']} [{c['type']}] **{x['response']}** ({x['reviewer'] or 'n/a'}): {c['claim'][:140]}"
                 + (f" _note: {x['note']}_" if x["note"] else ""))
    L += ["", "## NOT REVIEWED (do not read as cleared)"] + [f"- {c['id']} [{c['type']}] {c['claim'][:140]}" for c in unreviewed]
    L += ["", "## Unresolved"] + [f"- {x['card']['id']}: {x['response']}" for x in unresolved]
    L += ["", "## Engine report", "", engine.render(rep)]
    open(f"{outdir}/report_final.md", "w").write("\n".join(L))
    return adj
