"""Expert review cards generator & response merger.
Generates blind cards and response sheets for Original Prompts and Follow-ups.
Folds expert rulings into adjusted reports.
"""
import csv
import io
import json
import os
import re
import engine

OPTIONS = {
    "E": ["supports", "partly", "not", "unclear"],
    "R": ["in_books", "real_not_in_books", "not_real", "unclear"],
    "I": ["serious", "minor", "not_flaw", "unclear"],
    "U": ["settled", "open_gap", "out_of_expertise"],
    "S": ["resolved", "partly_resolved", "still_active", "unclear"],
}
QUESTION = {
    "E": "Does this passage actually support the claim?",
    "R": "A model recalled this limit but our search did not find it. Is it real, and is it in the books?",
    "I": "Is this a real flaw in the hypothesis, and how serious?",
    "U": "Nothing in the corpus tested this. Can you settle it from your knowledge?",
    "S": "The follow-up prompt claims this prior issue is now resolved. Is it genuinely resolved?",
}


def _search_string(text, n=8):
    return " ".join(re.findall(r"\w+", text or "")[:n])


class BuildResult(dict):
    """Dictionary holding card files and metadata, unpackable as (n_shown, n_overflow)."""
    def __iter__(self):
        return iter((len(self.get("shown", [])), len(self.get("overflow", []))))


def build_cards(rep, max_cards=12):
    pm = {p["id"]: p for p in rep.get("passages", [])}
    cards = []

    def card(typ, **kw):
        cards.append(dict(
            type=typ,
            question=QUESTION[typ],
            options=OPTIONS[typ],
            response=None,
            note=None,
            reviewer=None,
            **kw
        ))

    is_followup = bool(rep.get("turn") or rep.get("follow_up_prompt"))

    if is_followup:
        # 1. New issues spotted in this follow-up
        for sev in ("critical", "major", "minor"):
            for i in rep.get("new_issues", []):
                flag = (i.get("flag") or "").lower()
                sev_match = (
                    i.get("severity") == sev
                    or (sev == "critical" and flag == "red")
                    or (sev == "major" and flag == "yellow")
                )
                if sev_match:
                    p = pm.get(i.get("passage_id"))
                    card("I", issue_id=i.get("id"), claim=i.get("claim", ""), passage_id=i.get("passage_id"),
                         source=p["source"] if p else None, quote=i.get("quote"),
                         notebook_search=_search_string(i.get("quote") or i.get("claim")),
                         hidden=dict(severity=i.get("severity"), flag=i.get("flag"),
                                     quote_status=i.get("quote_status"), auto_entail=i.get("auto_entail"), is_new=True))

        # 2. Entailment check on verified quotes for new issues
        for i in rep.get("new_issues", []):
            if i.get("quote_status") == "verified" and i.get("passage_id") in pm:
                p = pm[i["passage_id"]]
                card("E", issue_id=i.get("id"), claim=i.get("claim", ""), passage_id=i.get("passage_id"),
                     source=p["source"], quote=i.get("quote"),
                     notebook_search=_search_string(i.get("quote")),
                     hidden=dict(severity=i.get("severity"), flag=i.get("flag"),
                                 auto_entail=i.get("auto_entail"), is_new=True))

        # 3. Check resolved issues claimed by this follow-up
        resolved_ids = set(rep.get("resolved_issue_ids", []))
        for i in rep.get("issues", []):
            if i.get("id") in resolved_ids:
                edge = i.get("edge_condition") or "Edge condition in follow-up"
                card("S", issue_id=i.get("id"),
                     claim=f"Prior issue {i.get('id')} ({i.get('claim', '')}) claimed resolved via edge condition: {edge}",
                     passage_id=i.get("passage_id"), source=None, quote=None,
                     notebook_search=_search_string(i.get("claim")),
                     hidden=dict(prior_issue_id=i.get("id"), edge_condition=edge))

        # 4. Remaining active critical/major issues from previous prompts
        new_ids = {ni.get("id") for ni in rep.get("new_issues", [])}
        for sev in ("critical", "major"):
            for i in rep.get("active_issues", []):
                flag = (i.get("flag") or "").lower()
                sev_match = (
                    i.get("severity") == sev
                    or (sev == "critical" and flag == "red")
                    or (sev == "major" and flag == "yellow")
                )
                if i.get("id") not in new_ids and sev_match:
                    p = pm.get(i.get("passage_id"))
                    card("I", issue_id=i.get("id"), claim=i.get("claim", ""), passage_id=i.get("passage_id"),
                         source=p["source"] if p else None, quote=i.get("quote"),
                         notebook_search=_search_string(i.get("quote") or i.get("claim")),
                         hidden=dict(severity=i.get("severity"), flag=i.get("flag"),
                                     quote_status=i.get("quote_status"), auto_entail=i.get("auto_entail"), is_new=False))

        # 5. Untested claims in this follow-up
        for u in rep.get("untested_claims", []):
            card("U", claim=u, source=None, quote=None, passage_id=None,
                 notebook_search=_search_string(u), hidden=dict())

    else:
        # Original Prompt cards
        for sev in ("critical", "major"):
            for i in rep.get("issues", []):
                flag = (i.get("flag") or "").lower()
                if i.get("severity") == sev or (sev == "critical" and flag == "red") or (sev == "major" and flag == "yellow"):
                    p = pm.get(i.get("passage_id"))
                    card("I", issue_id=i.get("id"), claim=i.get("claim", ""), passage_id=i.get("passage_id"),
                         source=p["source"] if p else None, quote=i.get("quote"),
                         notebook_search=_search_string(i.get("quote") or i.get("claim")),
                         hidden=dict(severity=i.get("severity"), flag=i.get("flag"),
                                     quote_status=i.get("quote_status"), auto_entail=i.get("auto_entail")))

        for r in rep.get("reconcile", []):
            if r.get("bucket") == "recall_only":
                card("R", claim=r["item"]["constraint"], source=r["item"]["book"], quote=None, passage_id=None,
                     notebook_search=r["item"]["search_terms"], hidden=dict(bucket="recall_only"))

        for i in rep.get("issues", []):
            if i.get("quote_status") == "verified" and i.get("passage_id") in pm:
                p = pm[i["passage_id"]]
                card("E", issue_id=i.get("id"), claim=i.get("claim", ""), passage_id=i.get("passage_id"),
                     source=p["source"], quote=i.get("quote"),
                     notebook_search=_search_string(i.get("quote")),
                     hidden=dict(severity=i.get("severity"), flag=i.get("flag"), auto_entail=i.get("auto_entail")))

        for u in rep.get("untested_claims", []):
            card("U", claim=u, source=None, quote=None, passage_id=None,
                 notebook_search=_search_string(u), hidden=dict())

    for n, c in enumerate(cards, 1):
        c["id"] = f"C{n}"
        c["rank"] = n

    return cards[:max_cards], cards[max_cards:]


def render_cards_md(shown, overflow, rep, label=None):
    turn = rep.get("turn")
    if not label:
        label = f"Follow-up #{turn}" if turn else "Original Prompt"

    parent_hyp = (rep.get("parent_hypothesis") or "").strip()
    hyp = (rep.get("hypothesis") or "").strip()
    f_prompt = (rep.get("follow_up_prompt") or "").strip()

    L = [f"# Expert Review Cards — {label} (Blind View)", ""]
    if parent_hyp and f_prompt:
        L += [
            "## 🔬 Research Lineage",
            f"- **Original Prompt:** {parent_hyp}",
            f"- **{label}:** {f_prompt}",
            ""
        ]
    elif hyp:
        L += ["## 🔬 Original Prompt", f"> {hyp}", ""]
    elif f_prompt:
        L += [f"## 💬 {label}", f"> {f_prompt}", ""]

    resp_file = f"responses_followup_{turn}.csv" if turn else "responses_original.csv"
    L += [
        "Read REVIEWER_GUIDE.md first. Do NOT open the report until all cards are answered.",
        f"Record answers in `{resp_file}` (or the Google Sheet copy). 'unclear' and 'out_of_expertise' are valid answers.",
        ""
    ]

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

    return "\n".join(L)


def render_responses_csv(shown):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["id", "type", "summary", "options", "response", "note", "reviewer"])
    for c in shown:
        w.writerow([c["id"], c["type"], c["claim"][:120], " | ".join(c["options"]), "", "", ""])
    return out.getvalue()


def build(rep, outdir=None, max_cards=12, label=None, prefix=None):
    turn = rep.get("turn")
    if not label:
        label = f"Follow-up #{turn}" if turn else "Original Prompt"
    if not prefix:
        prefix = f"cards_followup_{turn}" if turn else "cards_original"
    resp_fn = f"responses_followup_{turn}.csv" if turn else "responses_original.csv"

    shown, overflow = build_cards(rep, max_cards)
    cards_json_data = dict(cards=shown, overflow=overflow, label=label, prefix=prefix)
    cards_md = render_cards_md(shown, overflow, rep, label=label)
    responses_csv = render_responses_csv(shown)
    cards_json = json.dumps(cards_json_data, indent=1)

    files = {
        f"{prefix}.md": cards_md,
        f"{resp_fn}": responses_csv,
        f"{prefix}.json": cards_json,
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        for fn, content in files.items():
            with open(os.path.join(outdir, fn), "w", encoding="utf-8") as f:
                f.write(content)
        # Also maintain legacy names for backward compatibility
        if not turn:
            with open(os.path.join(outdir, "triage.md"), "w", encoding="utf-8") as f:
                f.write(cards_md)
            with open(os.path.join(outdir, "responses.csv"), "w", encoding="utf-8") as f:
                f.write(responses_csv)
            with open(os.path.join(outdir, "triage.json"), "w", encoding="utf-8") as f:
                f.write(cards_json)

    return BuildResult(
        shown=shown,
        overflow=overflow,
        label=label,
        prefix=prefix,
        cards_md=cards_md,
        responses_csv=responses_csv,
        cards_json=cards_json_data,
        files=files,
    )


def merge(outdir, responses_path=None, report_path=None, triage_path=None):
    if not report_path:
        report_path = f"{outdir}/report.json"
        if not os.path.exists(report_path) and os.path.exists(f"{outdir}/report_original.json"):
            report_path = f"{outdir}/report_original.json"
    if not triage_path:
        triage_path = f"{outdir}/triage.json"
        if not os.path.exists(triage_path) and os.path.exists(f"{outdir}/cards_original.json"):
            triage_path = f"{outdir}/cards_original.json"
    if not responses_path:
        responses_path = f"{outdir}/responses.csv"
        if not os.path.exists(responses_path) and os.path.exists(f"{outdir}/responses_original.csv"):
            responses_path = f"{outdir}/responses_original.csv"

    rep = json.load(open(report_path))
    tri = json.load(open(triage_path))
    rows = {r["id"]: r for r in csv.DictReader(open(responses_path))}
    problems, ruled, rejected = [], [], set()

    for c in tri.get("cards", []):
        r = rows.get(c["id"], {})
        resp = (r.get("response") or "").strip()
        if not resp:
            continue
        if resp not in c["options"]:
            problems.append(f"{c['id']}: '{resp}' not in {c['options']}")
            continue
        ruled.append(dict(card=c, response=resp, note=r.get("note", ""), reviewer=r.get("reviewer", "")))
        if (c["type"] == "I" and resp == "not_flaw") or (c["type"] == "E" and resp == "not"):
            rejected.add(c.get("issue_id"))

    answered = {x["card"]["id"] for x in ruled}
    unreviewed = [c for c in tri.get("cards", []) if c["id"] not in answered] + tri.get("overflow", [])
    unresolved = [x for x in ruled if x["response"] in ("unclear", "out_of_expertise")]
    remaining = [i for i in rep.get("issues", []) if i.get("id") not in rejected]
    adj = engine.diagnostic_badge(engine.count_flags(remaining, rep.get("green_flags", [])))
    if ruled:
        adj += "  [expert-adjusted]"

    crit_open = [
        i for i in remaining
        if (i.get("severity") == "critical" or i.get("flag") == "red")
        and i.get("quote_status") == "verified"
        and not any(x["card"].get("issue_id") == i.get("id") and x["response"] in ("serious", "supports")
                    for x in ruled)
    ]

    hyp = (rep.get("hypothesis") or rep.get("parent_hypothesis") or "").strip()
    f_prompt = (rep.get("follow_up_prompt") or "").strip()
    lbl = tri.get("label", "Original Prompt")

    L = ["# Final Adversarial Review Report", ""]
    if hyp:
        L += ["## 🔬 Original Prompt", f"> {hyp}", ""]
    if f_prompt:
        L += [f"## 💬 {lbl}", f"> {f_prompt}", ""]

    orig_verdict = rep.get("badge") or rep.get("verdict", "N/A")
    L += [f"## Final Diagnostic Badge: {adj}", f"Engine diagnostic badge before review: {orig_verdict}", ""]
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
    L += ["", "## Engine report", "", engine.render(rep) if not rep.get("turn") else engine.render_followup(rep)]
    
    out_file = f"{outdir}/report_final.md" if not rep.get("turn") else f"{outdir}/report_followup_{rep['turn']}_final.md"
    open(out_file, "w").write("\n".join(L))
    return adj
