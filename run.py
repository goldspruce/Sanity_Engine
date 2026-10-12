import argparse, json, os, sys
import engine, triage

def backend(name):
    if name == "mock":
        from mock_llm import MockLLM
        return MockLLM()
    return {"gemini": engine.GeminiLLM, "anthropic": engine.AnthropicLLM}[name]()

def books_default(corpus):
    return "; ".join(sorted({os.path.splitext(f)[0] for f in os.listdir(corpus) if f.lower().endswith((".txt", ".md"))})) or "(none listed)"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Adversarial hypothesis checker")
    ap.add_argument("hypothesis", nargs="?")
    ap.add_argument("--gen", default="gemini", choices=["gemini", "anthropic", "mock"])
    ap.add_argument("--critic", default=None, help="second model family for the review step (default = --gen)")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--books", default=None, help="book names for the recall channel; default = corpus filenames")
    ap.add_argument("--chunk-words", type=int, default=180)
    ap.add_argument("--auto-entail", action="store_true")
    ap.add_argument("--check-math", action="store_true", default=False, help="Enable number/math check on critique")
    ap.add_argument("--review", action="store_true", help="OPTIONAL: write expert triage cards")
    ap.add_argument("--max-cards", type=int, default=12)
    ap.add_argument("--out", default="out")
    ap.add_argument("--merge", metavar="CASE_DIR", help="fold responses.csv into report_final.md")
    ap.add_argument("--responses", default=None)
    a = ap.parse_args()
    if a.merge:
        print(triage.merge(a.merge, a.responses)); sys.exit()
    hyp = open(a.hypothesis).read()
    gen = backend(a.gen); crit = backend(a.critic) if a.critic else gen
    rep = engine.run(hyp, a.books or books_default(a.corpus), gen, crit,
                     engine.LocalBM25(a.corpus, a.chunk_words), auto_entail=a.auto_entail,
                     check_math=a.check_math)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    open(f"{a.out}/report.md", "w").write(engine.render(rep))
    print(engine.render(rep))
    if a.review:
        n, o = triage.build(rep, a.out, a.max_cards)
        print(f"\n[triage] {n} cards queued, {o} overflow -> {a.out}/triage.md, responses.csv")
