import argparse, json, os, sys
import engine, triage

def backend(name):
    if name == "mock":
        from mock_llm import MockLLM
        return MockLLM()
    return {"gemini": engine.GeminiLLM, "anthropic": engine.AnthropicLLM}[name]()

def books_default(corpus):
    return "; ".join(sorted({os.path.splitext(f)[0] for f in os.listdir(corpus) if f.lower().endswith((".txt", ".md"))})) or "(none listed)"

def safe_json_dump(obj, fp, indent=1):
    def default(o):
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    return json.dump(obj, fp, indent=indent, default=default)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Adversarial hypothesis checker")
    ap.add_argument("hypothesis", nargs="?")
    ap.add_argument("--gen", default="gemini", choices=["gemini", "anthropic", "mock"])
    ap.add_argument("--critic", default=None, help="second model family for the review step (default = --gen)")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--books", default=None, help="book names to focus on in model recall; default = corpus filenames")
    ap.add_argument("--chunk-words", type=int, default=180)
    ap.add_argument("--auto-entail", action="store_true")
    ap.add_argument("--check-math", action="store_true", default=False, help="Enable number/math check on critique")
    ap.add_argument("--review", action="store_true", help="OPTIONAL: write expert triage cards")
    ap.add_argument("--batch", action="store_true", help="Non-interactive batch mode: exit after rendering report")
    ap.add_argument("--interactive", action="store_true", help="Force interactive follow-up loop even if redirected")
    ap.add_argument("--max-cards", type=int, default=12)
    ap.add_argument("--out", default="out")
    ap.add_argument("--merge", metavar="CASE_DIR", help="fold responses.csv into report_final.md")
    ap.add_argument("--responses", default=None)
    a = ap.parse_args()
    if a.merge:
        print(triage.merge(a.merge, a.responses))
        sys.exit()
    if not a.hypothesis:
        print("Error: Please provide a hypothesis file path or text string.")
        sys.exit(1)
    if os.path.isfile(a.hypothesis):
        hyp = open(a.hypothesis, encoding="utf-8").read()
    else:
        hyp = a.hypothesis
    gen = backend(a.gen); crit = backend(a.critic) if a.critic else gen
    rep = engine.run(hyp, a.books or books_default(a.corpus), gen, crit,
                     engine.LocalBM25(a.corpus, a.chunk_words), auto_entail=a.auto_entail,
                     check_math=a.check_math)
    os.makedirs(a.out, exist_ok=True)
    safe_json_dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    open(f"{a.out}/report.md", "w").write(engine.render(rep))
    print(engine.render(rep))
    if a.review:
        n, o = triage.build(rep, a.out, a.max_cards)
        print(f"\n[triage] {n} cards queued, {o} overflow -> {a.out}/triage.md, responses.csv")

    # Interactive follow-up session with continuous research & report generation
    should_interact = a.interactive or (not a.batch and sys.stdin.isatty())
    if should_interact:
        print("\n" + "=" * 65)
        print("💬 INTERACTIVE FOLLOW-UP SESSION WITH CONTINUOUS RESEARCH")
        print("Enter follow-up prompts, probes, or refined hypotheses.")
        print("Each follow-up generates an additional verified report.")
        print("Type 'exit' or 'q' to end the session.")
        print("=" * 65)
        current_rep = rep
        turn_idx = 1
        while True:
            try:
                line = input(f"\n[follow-up turn {turn_idx}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break
            if not line or line.lower() in ("exit", "quit", "q"):
                print("Session ended.")
                break
            print(f"\n--> Researching follow-up turn {turn_idx} against literature...")
            f_rep = engine.run_followup(
                current_rep, line, a.books or books_default(a.corpus),
                gen, crit, engine.LocalBM25(a.corpus, a.chunk_words),
                turn_idx=turn_idx, auto_entail=a.auto_entail, check_math=a.check_math
            )
            f_md = engine.render_followup(f_rep)
            safe_json_dump(f_rep, open(f"{a.out}/report_followup_{turn_idx}.json", "w"), indent=1)
            open(f"{a.out}/report_followup_{turn_idx}.md", "w").write(f_md)
            print(f"[saved] -> {a.out}/report_followup_{turn_idx}.md, report_followup_{turn_idx}.json\n")
            print(f_md)
            current_rep = f_rep
            turn_idx += 1

