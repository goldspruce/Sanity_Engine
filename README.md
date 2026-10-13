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

3. **High-Resolution Flag Spotting Radar (No Arbitrary Scores)**:
   - Groundbreaking hypotheses challenge conventional wisdom; assigning arbitrary numeric scores or authoritarian "FAIL" stamps stifles scientific novelty. Instead, Sanity Engine acts as a constructive radar mapping three distinct diagnostic flags:
     - 🟩 **Green Flags**: Biophysically sound mechanisms, mathematical consistency, and alignment with canonical textbook principles that anchor novel ideas.
     - 🟨 **Yellow Flags**: Empirical friction, conflicting studies, competing mechanisms, or parameter regime constraints; paired with qualitative `edge_condition`s describing what parameter changes or boundaries flip the objection.
     - 🚩 **Red Flags**: Fatal physical, thermodynamic, or biophysical conservation law violations (e.g. conservation of energy, absolute physical non-viability).
   - Reports provide diagnostic badges (`🚩 X Red | 🟨 Y Yellow | 🟩 Z Green`) and actionable classification without arbitrary numbers.

4. **Iterative Follow-Up Prompts & Conversational Probing**:
   - Users can iterate with follow-up prompts, challenge critiques, or propose refined hypotheses without losing state.
   - The referee maintains research continuity: tracking resolved flags vs. active flags across follow-up inquiries and querying newly relevant literature.
   - Sessions continue until the user chooses to exit (via `exit` command, "🛑 End Session" button, or closing).

5. **Number Linting & Deterministic Physics Ledger**:
   - Numbers appearing in critique objections that do not trace back to the hypothesis, the cited passages, or the math ledger are flagged as `"eyeball-check needed"`.
   - `physics_tools.py` provides a deterministic calculation ledger using **SymPy** and **Pint** for acoustic transmission, quarter-wave matching, atmospheric absorption, diffraction limits, and efficiency cascades.

6. **Cross-Family Model Decorrelation**:
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

Sanity Engine maintains an indexed library of 11 foundational canonical textbooks across electrophysiology, cell biology, biochemistry, dynamical systems, biomechanics, information theory, and physics.

All files in `Textbooks/` and `corpus/` follow the standardized naming convention: `Author_Title_Edition_Year`:

| # | Standardized Filename (`Author_Title_Edition_Year`) | Title & Author(s) | Primary Domain |
|:-:|:---|:---|:---|
| **1** | `Alberts_Molecular_Biology_of_the_Cell_7ed_2022` | *Molecular Biology of the Cell* (Alberts et al., 7th ed.) | Cell biology, cross-bridges, motors |
| **2** | `Berg_Random_Walks_in_Biology_1993` | *Random Walks in Biology* (Berg, 1993) | Diffusion, Brownian motion, motility |
| **3** | `Cover_Elements_of_Information_Theory_2ed_2006` | *Elements of Information Theory* (Cover & Thomas, 2nd ed.) | Channel capacity, rate distortion, entropy |
| **4** | `Feynman_Lectures_on_Physics_Vol1_1963` | *The Feynman Lectures on Physics: Vol 1* (Feynman et al., 1963) | Mechanics, radiation, heat, acoustics |
| **5** | `Feynman_Lectures_on_Physics_Vol2_1964` | *The Feynman Lectures on Physics: Vol 2* (Feynman et al., 1964) | Electromagnetism, waves, dielectric media |
| **6** | `Johnston_Foundations_of_Cellular_Neurophysiology_1ed_1995` | *Foundations of Cellular Neurophysiology* (Johnston & Wu, 1995) | Electrophysiology, cable theory, ion channels |
| **7** | `Kreyszig_Advanced_Engineering_Mathematics_10ed_2011` | *Advanced Engineering Mathematics* (Kreyszig, 10th ed.) | Wave equations, PDEs, Fourier transforms |
| **8** | `Murray_Mathematical_Biology_an_Introduction_2ed_1993` | *Mathematical Biology: An Introduction* (Murray, 2nd ed.) | Reaction-diffusion, population dynamics |
| **9** | `Nelson_Lehninger_Principles_of_Biochemistry_8ed_2021` | *Lehninger: Principles of Biochemistry* (Nelson & Cox, 8th ed.) | Bioenergetics, ATP kinetics, metabolic flux |
| **10** | `Strogatz_Nonlinear_Dynamics_and_Chaos_3ed_2024` | *Nonlinear Dynamics and Chaos* (Strogatz, 3rd ed., 2024) | Limit cycles, bifurcations, oscillators |
| **11** | `Winter_Biomechanics_and_Motor_Control_of_Human_Movement_5ed_2020` | *Biomechanics and Motor Control* (Winter, 5th ed., 2020) | Joint kinetics, EMG, muscle power, SSC |

See [TEXTBOOKS.md](TEXTBOOKS.md) for full descriptions and indexing instructions.

### Google NotebookLM Integration
To investigate untested claims or examine biological/physical mechanisms further, here is a link to a Google NotebookLM pre-loaded with these canonical textbooks:

- **🔗 Preloaded Workspace:** [Google NotebookLM Textbook Workspace](https://notebook.google.com/notebook/3eed2329-aeef-4e67-81ef-c5bc2c06f5bd) with the complete textbook corpus preloaded for immediate cited Q&A and verification. (Configurable via `NOTEBOOKLM_URL` in `.env`).
- **Custom Uploads:** You can also slice the corpus into formatted chunks for custom notebooks:
  ```bash
  .venv/bin/python pack_for_notebook.py --corpus corpus --out notebook_pack --max-words 200000
  ```
  Then upload `notebook_pack/` directly to [Google NotebookLM](https://notebooklm.google.com).

---

## Expert Review Cards Workflow (`triage.py`)

When `--review` is passed (or the checkbox is enabled in Streamlit), the engine generates blind review cards and response sheets for both the **Original Prompt** and every **Follow-up**:
1. **Original Prompt Cards** (`cards_original.md` / `responses_original.csv`):
   - **I (Issue)**: Is this a real flaw in the prompt/hypothesis, and how serious?
   - **R (Recall gap)**: Model recalled a constraint not found in the corpus. Is it real?
   - **E (Entailment)**: Does the cited passage genuinely support the objection?
   - **U (Untested)**: Can a domain expert settle this untested mechanism?
2. **Follow-up Cards** (`cards_followup_{N}.md` / `responses_followup_{N}.csv`):
   - **I (New Issue)**: Is this newly spotted red or yellow flag a real objection?
   - **E (New Entailment)**: Does the newly cited passage support the critique?
   - **S (Resolution Verification)**: The follow-up proposed an edge condition to resolve a prior objection. Is it genuinely resolved?
   - **U (Untested Claims)**: Untested claims introduced in this follow-up.

Merge rulings back into the final report:
```bash
.venv/bin/python run.py --merge cases/ubeam --responses cases/ubeam/responses_original.csv
```
This produces `cases/ubeam/report_final.md` with expert-adjusted diagnostic badges.
