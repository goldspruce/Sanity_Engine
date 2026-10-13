import os, sys, json, time, urllib.parse, urllib.request, re, io
from dotenv import load_dotenv
load_dotenv()
import docx
import streamlit as st
import engine, triage

def docx_text(b):
    d = docx.Document(io.BytesIO(b) if hasattr(b, 'read') else io.BytesIO(b))
    return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())

def gdoc_text(url):
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("Could not find a valid Google Doc ID in the link.")
    gid = m.group(1)
    req = urllib.request.Request(f"https://docs.google.com/document/d/{gid}/export?format=txt",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", "ignore")

def safe_json_dumps(obj, indent=1):
    def default(o):
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    return json.dumps(obj, indent=indent, default=default)

st.set_page_config(page_title="Sanity Engine", layout="wide")
st.title("Sanity Engine")

nb_url = os.getenv("NOTEBOOKLM_URL", "https://notebook.google.com/notebook/3eed2329-aeef-4e67-81ef-c5bc2c06f5bd")

with st.expander("📚 Pre-Loaded Canonical Textbooks in Corpus & NotebookLM (11)", expanded=False):
    st.markdown(f"🔗 **Google NotebookLM Workspace:** [{nb_url}]({nb_url})")
    st.markdown("""
    All reference textbooks are indexed in `corpus/` for deterministic BM25 retrieval and quote verification:
    - **Alberts et al. (7th ed., 2022)**: `Alberts_Molecular_Biology_of_the_Cell_7ed_2022` *(Cell biology, cross-bridges, motors)*
    - **Berg (1993)**: `Berg_Random_Walks_in_Biology_1993` *(Diffusion, Brownian motion, motility, low Reynolds number)*
    - **Cover & Thomas (2nd ed., 2006)**: `Cover_Elements_of_Information_Theory_2ed_2006` *(Information theory, channel capacity, rate distortion)*
    - **Feynman et al. (Vol. 1, 1963)**: `Feynman_Lectures_on_Physics_Vol1_1963` *(Mechanics, oscillations, radiation, heat, acoustics)*
    - **Feynman et al. (Vol. 2, 1964)**: `Feynman_Lectures_on_Physics_Vol2_1964` *(Electromagnetism, wave propagation, dielectric media)*
    - **Johnston & Wu (1st ed., 1995)**: `Johnston_Foundations_of_Cellular_Neurophysiology_1ed_1995` *(Cellular neurophysiology, cable theory, ion channels)*
    - **Kreyszig (10th ed., 2011)**: `Kreyszig_Advanced_Engineering_Mathematics_10ed_2011` *(Wave equations, PDEs, Fourier/Laplace transforms)*
    - **Murray (2nd ed., 1993)**: `Murray_Mathematical_Biology_an_Introduction_2ed_1993` *(Mathematical biology, reaction-diffusion, population dynamics)*
    - **Nelson & Cox (8th ed., 2021)**: `Nelson_Lehninger_Principles_of_Biochemistry_8ed_2021` *(Biochemistry, ATP kinetics, bioenergetics, metabolic flux)*
    - **Strogatz (3rd ed., 2024)**: `Strogatz_Nonlinear_Dynamics_and_Chaos_3ed_2024` *(Nonlinear dynamics, limit cycles, bifurcations, oscillators)*
    - **Winter (5th ed., 2020)**: `Winter_Biomechanics_and_Motor_Control_of_Human_Movement_5ed_2020` *(Joint kinetics, EMG, muscle power, stretch-shortening cycle)*
    """)

with st.sidebar:
    gen_name = st.selectbox("Generator model", ["gemini", "anthropic", "mock"])
    critic_name = st.selectbox("Reviewer model", ["same as generator", "gemini", "anthropic", "mock"])
    corpus = st.text_input("Corpus folder", "corpus")
    if os.path.isdir(corpus):
        with st.expander(f"Books in `{corpus}/`", expanded=False):
            for bf in sorted(os.listdir(corpus)):
                if bf.lower().endswith((".txt", ".md")):
                    st.caption(f"• `{os.path.splitext(bf)[0]}`")
    books = st.text_input("Book names to focus on in model recall?", "",
                          help="Optional. Leave blank to default to all books in your corpus folder.")
    review = st.checkbox("Create expert review cards (Original Prompt & Follow-ups)", value=True)
    entail = st.checkbox("Auto entailment pre-check")
    check_math = st.checkbox("Math / number verification (optional)", value=False)

    if st.button("🔄 Reset / New Prompt", help="Clear current session and start with a new original prompt"):
        st.session_state["report"] = None
        st.session_state["latest_report"] = None
        st.session_state["followup_reports"] = []
        st.session_state["follow_up_history"] = []
        st.session_state["original_cards"] = None
        st.session_state["session_active"] = True
        st.rerun()

    st.divider()
    if st.button("🛑 End Session (Exit)", help="Stop the local server and exit Python process"):
        st.session_state["session_active"] = False
        st.warning("Session ended. Server stopped. You can safely close this browser tab.")
        time.sleep(0.5)
        os._exit(0)

if not st.session_state.get("session_active", True):
    st.warning("🛑 Session has ended. You can review past results below, or restart a new session.")
    if st.button("🔄 Start New Session", type="primary"):
        st.session_state["session_active"] = True
        st.session_state["report"] = None
        st.session_state["latest_report"] = None
        st.session_state["followup_reports"] = []
        st.session_state["follow_up_history"] = []
        st.session_state["original_cards"] = None
        st.rerun()

text = st.text_area("Original Prompt", height=200)
up = st.file_uploader("...or upload .docx / .txt / .md", type=["docx", "txt", "md"])
link = st.text_input("...or Google Doc link (must be viewable by anyone with the link)")
if up:
    text = docx_text(up.getvalue()) if up.name.lower().endswith(".docx") else up.getvalue().decode("utf-8", "ignore")
elif link and not text:
    try:
        text = gdoc_text(link)
    except Exception as e:
        st.error(f"Could not read the Google Doc: {e}")
if text:
    text = st.text_area("Text that will be checked (edit if extraction dropped anything)", text, height=200, key="confirm")

def backend(name):
    if name == "mock":
        from mock_llm import MockLLM
        return MockLLM()
    return {"gemini": engine.GeminiLLM, "anthropic": engine.AnthropicLLM}[name]()

def books_default(corpus):
    return "; ".join(sorted({os.path.splitext(f)[0] for f in os.listdir(corpus) if f.lower().endswith((".txt", ".md"))})) or "(none listed)"

if st.button("Run", disabled=not text, type="primary"):
    try:
        gen = backend(gen_name)
        crit = gen if critic_name == "same as generator" else backend(critic_name)
        with st.spinner("Running adversarial review on Original Prompt..."):
            rep = engine.run(text, books or books_default(corpus), gen, crit, engine.LocalBM25(corpus), auto_entail=entail,
                             check_math=check_math, log=lambda m: None)
            st.session_state["report"] = rep
            st.session_state["latest_report"] = rep
            st.session_state["followup_reports"] = []
            st.session_state["follow_up_history"] = []
            st.session_state["session_active"] = True
            if review:
                t_res = triage.build(rep, label="Original Prompt", prefix="cards_original")
                st.session_state["original_cards"] = t_res["files"]
            else:
                st.session_state["original_cards"] = None
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

if st.session_state.get("report"):
    rep = st.session_state["report"]
    counts = rep.get("counts") or engine.count_flags(rep.get("issues", []), rep.get("green_flags", []))
    col_r, col_y, col_g = st.columns(3)
    col_r.metric("🚩 Red Flags", counts.get("red", 0), help="Fatal physical or biological invariant violations")
    col_y.metric("🟨 Yellow Flags", counts.get("yellow", 0), help="Empirical friction & parameter regime constraints (requires edge conditions)")
    col_g.metric("🟩 Green Flags", counts.get("green", 0), help="Biophysically sound mechanisms & canonical principles")

    rendered = engine.render(rep)
    st.markdown(rendered)
    st.divider()
    st.subheader("📥 Downloads & Review Cards — Original Prompt")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download report_original.md", rendered, "report_original.md", mime="text/markdown", key="dl_report_md")
    with col2:
        st.download_button("📥 Download report_original.json", safe_json_dumps(rep, indent=1), "report_original.json", mime="application/json", key="dl_report_json")
    
    if st.session_state.get("original_cards"):
        st.markdown("##### 🃏 Expert Review Cards — Original Prompt")
        orig_files = st.session_state["original_cards"]
        c_cols = st.columns(len(orig_files))
        for idx, (fn, content) in enumerate(orig_files.items()):
            with c_cols[idx]:
                st.download_button(f"📥 Download {fn}", content, fn, key=f"dl_orig_{fn}")
        with st.expander("🔍 Preview Review Cards (Original Prompt)", expanded=False):
            st.markdown(orig_files.get("cards_original.md", ""))

    # Continuous Follow-Up Section
    st.divider()
    st.subheader("💬 Continuous Follow-Up Research & Refinement")
    st.caption("Ask questions, test edge conditions, or refine the prompt. The engine maintains continuous research across follow-up inquiries: querying literature, tracking resolved vs. persisting flags, generating verified reports, and producing expert review cards for each follow-up. Type **exit** or click **🛑 End Session** in the sidebar to finish.")

    for idx, msg in enumerate(st.session_state.get("follow_up_history", [])):
        with st.chat_message(msg["role"]):
            if msg.get("report") and msg.get("turn"):
                turn_num = msg["turn"]
                f_rep = msg["report"]
                f_counts = f_rep.get("counts") or engine.count_flags(f_rep.get("active_issues", f_rep.get("issues", [])), f_rep.get("green_flags", []))
                st.markdown(f"### 📄 Follow-up #{turn_num} Evaluation")
                fcol_r, fcol_y, fcol_g = st.columns(3)
                fcol_r.metric("🚩 Red Flags", f_counts.get("red", 0))
                fcol_y.metric("🟨 Yellow Flags", f_counts.get("yellow", 0))
                fcol_g.metric("🟩 Green Flags", f_counts.get("green", 0))
            st.markdown(msg["content"])
            if msg.get("report") and msg.get("turn"):
                turn_num = msg["turn"]
                f_rep = msg["report"]
                f_md = msg["content"]
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        f"📥 Download report_followup_{turn_num}.md",
                        f_md,
                        f"report_followup_{turn_num}.md",
                        mime="text/markdown",
                        key=f"dl_f_{turn_num}_{idx}_md"
                    )
                with c2:
                    st.download_button(
                        f"📥 Download report_followup_{turn_num}.json",
                        safe_json_dumps(f_rep, indent=1),
                        f"report_followup_{turn_num}.json",
                        mime="application/json",
                        key=f"dl_f_{turn_num}_{idx}_json"
                    )
                if msg.get("cards"):
                    f_card_files = msg["cards"].get("files", {})
                    if f_card_files:
                        st.markdown(f"##### 🃏 Expert Review Cards — Follow-up #{turn_num}")
                        fc_cols = st.columns(len(f_card_files))
                        for c_idx, (fn, content) in enumerate(f_card_files.items()):
                            with fc_cols[c_idx]:
                                st.download_button(f"📥 Download {fn}", content, fn, key=f"dl_f_cards_{turn_num}_{idx}_{fn}")
                        with st.expander(f"🔍 Preview Review Cards (Follow-up #{turn_num})", expanded=False):
                            st.markdown(f_card_files.get(f"cards_followup_{turn_num}.md", ""))

    if st.session_state.get("session_active", True):
        prompt = st.chat_input("Enter a follow-up prompt, probe, or refined hypothesis (or 'exit')...")
        if prompt:
            if prompt.strip().lower() in ("exit", "quit", "q", ":q"):
                st.session_state["session_active"] = False
                st.warning("Session ended. You can safely close this browser tab or restart a session.")
                st.rerun()
            else:
                st.session_state.setdefault("follow_up_history", []).append({"role": "user", "content": prompt})
                try:
                    gen = backend(gen_name)
                    crit = gen if critic_name == "same as generator" else backend(critic_name)
                    turn_idx = len(st.session_state.get("followup_reports", [])) + 1
                    with st.spinner(f"Conducting continuous literature research for Follow-up #{turn_idx}..."):
                        parent_to_use = st.session_state.get("latest_report") or rep
                        f_rep = engine.run_followup(
                            parent_to_use,
                            prompt,
                            books or books_default(corpus),
                            gen,
                            crit,
                            engine.LocalBM25(corpus),
                            turn_idx=turn_idx,
                            auto_entail=entail,
                            check_math=check_math,
                            log=lambda m: None
                        )
                        f_cards = None
                        if review:
                            f_cards = triage.build(f_rep, label=f"Follow-up #{turn_idx}", prefix=f"cards_followup_{turn_idx}")
                    st.session_state.setdefault("followup_reports", []).append(f_rep)
                    st.session_state["latest_report"] = f_rep
                    f_md = engine.render_followup(f_rep)
                    st.session_state["follow_up_history"].append({
                        "role": "assistant",
                        "content": f_md,
                        "report": f_rep,
                        "turn": turn_idx,
                        "cards": f_cards
                    })
                    st.rerun()
                except Exception as e:
                    # Clean up the pending user prompt from history so it doesn't leave an orphaned message
                    if st.session_state.get("follow_up_history") and st.session_state["follow_up_history"][-1].get("role") == "user":
                        st.session_state["follow_up_history"].pop()
                    import traceback
                    traceback.print_exc()
                    st.error(f"Error evaluating follow-up #{turn_idx}: {e}")
                    with st.expander("🔍 Show Detailed Error Diagnostics", expanded=True):
                        st.exception(e)
