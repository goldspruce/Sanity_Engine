# Sanity Engine: Adversarial Scientific Hypothesis Checker

**Sanity Engine** is a rigorous, adversarial scientific referee pipeline designed to stress-test hypotheses against domain literature and physical invariants.

Rather than acting as a supportive brainstormer or allowing sycophantic hallucinations, the engine acts as a **hostile journal referee** whose explicit incentive is to find fatal flaws, violated invariants, and ungrounded claims.

---

## Key Principles & Design

1. **Dual-Channel Evidence (Channel A vs. Channel B)**:
   - **Channel A (Model Recall)**: The model recalls principles and textbook sources from parametric memory. This is strictly *unverified* and is only used to guide search terms.
   - **Channel B (Corpus Retrieval)**: Deterministic BM25 retrieval over verified literature in `corpus/`. Only passages in Channel B can serve as citable evidence.
   - **Reconciliation**: Every recalled item is matched against retrieval: marked as `agree`, `recall_only` (unverified gap), or `retrieval_only`.

2. **Verbatim Quote Verification in Code**:
   - The hostile reviewer must cite a passage ID and provide an exact, verbatim quote (6–40 words).
   - Python code verifies that the quote is an exact substring of the retrieved passage.
   - Any hallucinated, misattributed, or modified quotes are caught and **disqualified from affecting the verdict**.

3. **Rule-Based Verdicts (No LLM Discretion)**:
   - Verdicts are computed deterministically by fixed code rules:
     - Verified Critical Issue $\rightarrow$ **FAIL**
     - Verified Major Issue $\rightarrow$ **WEAK**
     - Unverified Critical Issue $\rightarrow$ **WEAK (critical concern with unverified citation)**
     - No verified critical/major issues $\rightarrow$ **SURVIVES (never means endorsed)**

4. **Number Linting & Deterministic Physics Ledger**:
   - Numbers appearing in critique objections that do not trace back to the hypothesis, the cited passages, or the math ledger are flagged as `"eyeball-check needed"`.
   - `physics_tools.py` provides a deterministic calculation ledger using **SymPy** and **Pint** for acoustic transmission, quarter-wave matching, atmospheric absorption, diffraction limits, and efficiency cascades.

5. **Cross-Family Model Decorrelation**:
   - Supports Gemini as the generator (`--gen gemini` via the modern `google-genai` SDK) paired with a different model family as the hostile critic (`--critic anthropic`) to eliminate correlated sycophancy and blind spots.

---

## Directory Structure

```text
Sanity Engine/
├── engine.py             # Core pipeline (AttackPlan, BM25, Hostile Review, Quote Verification)
├── run.py                # Command-line interface runner & case output
├── app.py                # Streamlit browser UI (Paste, .docx, .txt, .md, Google Doc link)
├── triage.py             # Expert review card generator & final response merger
├── physics_tools.py      # Deterministic calculation ledger (SymPy + Pint)
├── ingest_textbooks.py   # Extracts PDF textbooks into searchable corpus/ text files
├── pack_for_notebook.py  # Splits corpus into notebook/Google Drive chunks for NotebookLM
├── mock_llm.py           # Offline test mock simulating LLM responses with planted errors
├── hypotheses/           # Test hypotheses (e.g., ubeam.md)
├── corpus/               # Extracted textbook text files (used for BM25 search)
├── corpus_demo/          # Minimal test corpus for offline smoke tests
├── Textbooks/            # Local PDF textbooks (Alberts, Feynman, Murray, Lehninger, etc.)
├── TEXTBOOKS.md          # Complete catalog of reference textbooks & NotebookLM guide
├── requirements.txt      # Python dependencies
└── .env.example          # Template for GEMINI_API_KEY and ANTHROPIC_API_KEY
```

---

## Quick Start

### 1. Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` or export environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GEMINI_MODEL="gemini-2.5-flash"        # or gemini-2.5-pro

# Optional for cross-family review:
export ANTHROPIC_API_KEY="your-anthropic-key"
export ANTHROPIC_MODEL="claude-3-5-sonnet-latest"
```

### 3. Run Offline Smoke Test (No API keys needed)

```bash
.venv/bin/python run.py hypotheses/ubeam.md --gen mock --corpus corpus_demo --chunk-words 60 --review --out out
```

### 4. Ingest Local PDF Textbooks into Corpus

To convert PDFs in the `Textbooks/` directory into searchable text files:

```bash
.venv/bin/python ingest_textbooks.py --src Textbooks --dest corpus
```

### 5. Run Live Analysis

```bash
# Gemini generator + Gemini critic:
.venv/bin/python run.py hypotheses/ubeam.md --gen gemini --corpus corpus --out cases/ubeam

# Cross-family: Gemini generator + Anthropic critic:
.venv/bin/python run.py hypotheses/ubeam.md --gen gemini --critic anthropic --corpus corpus --out cases/ubeam
```

### 6. Launch the Local Browser Interface

```bash
.venv/bin/streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can:
- Paste a hypothesis directly
- Upload a `.docx`, `.txt`, or `.md` file
- Paste a public Google Doc link

---

## Textbook Corpus & Google NotebookLM Integration

Sanity Engine maintains an indexed library of 12 foundational textbooks across biomechanics, electrophysiology, molecular biology, biochemistry, dynamical systems, and physics. 

See [TEXTBOOKS.md](TEXTBOOKS.md) for the complete catalog of titles and editions.

### Sharing and Querying Textbooks in Google NotebookLM
- **🔗 Preloaded NotebookLM Link:** A ready-to-use NotebookLM workspace with the full textbook corpus preloaded is shared directly via link for reviewers and collaborators.
- **Custom Uploads:** You can also slice the corpus into formatted chunks for custom notebooks:
  ```bash
  .venv/bin/python pack_for_notebook.py --corpus corpus --out notebook_pack --max-words 200000
  ```
  Then upload `notebook_pack/` directly to [Google NotebookLM](https://notebooklm.google.com) to ask questions against the indexed literature!

---

## Expert Triage Workflow (`triage.py`)

When `--review` is passed, the engine generates:
1. `out/triage.md`: Blind review cards without revealing the final verdict, categorized by:
   - **I (Issue)**: Is this a real flaw, and how serious?
   - **R (Recall gap)**: Model recalled a limit not found in corpus. Is it real?
   - **E (Entailment)**: Does the cited passage genuinely support the objection?
   - **U (Untested)**: Can a domain expert settle this untested link?
2. `out/responses.csv`: Form for recording expert ratings.

Merge rulings back into the final report:
```bash
.venv/bin/python run.py --merge cases/ubeam --responses cases/ubeam/responses.csv
```
This produces `cases/ubeam/report_final.md` with expert-adjusted verdicts.
