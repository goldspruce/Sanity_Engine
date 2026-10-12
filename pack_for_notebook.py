"""Split corpus/ into Notebook/Drive-friendly parts (same text the engine searches, so quotes verify).
python pack_for_notebook.py --corpus corpus --out notebook_pack --max-words 200000
Check your notebook tool's current per-source size/count limits and set --max-words accordingly."""
import argparse, csv, glob, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="corpus")
ap.add_argument("--out", default="notebook_pack")
ap.add_argument("--max-words", type=int, default=200000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rows = []
for path in sorted(glob.glob(os.path.join(a.corpus, "**", "*.*"), recursive=True)):
    if not path.lower().endswith((".txt", ".md")):
        continue
    stem = os.path.splitext(os.path.basename(path))[0]
    paras = [p for p in re.split(r"\n\s*\n", open(path, encoding="utf-8", errors="ignore").read()) if p.strip()]
    part, buf, n = 1, [], 0
    def flush():
        global part, buf, n
        if buf:
            name = f"{stem}_part{part:02d}.txt"
            open(os.path.join(a.out, name), "w", encoding="utf-8").write("\n\n".join(buf))
            rows.append([name, os.path.basename(path), n])
            part, buf, n = part + 1, [], 0
    for p in paras:
        w = len(p.split())
        if n + w > a.max_words:
            flush()
        buf.append(p); n += w
    flush()
with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
    cw = csv.writer(f); cw.writerow(["notebook_file", "original_file", "words"]); cw.writerows(rows)
print(f"{len(rows)} files -> {a.out}/ (manifest.csv maps each part to its original file)")
