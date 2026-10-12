"""Local browser UI:  pip install streamlit python-docx ;  streamlit run app.py
Paste text, upload .docx/.txt/.md, or give a Google Doc link shared as 'anyone with the link can view'."""
import io, json, os, re, tempfile, time, urllib.request
import streamlit as st
import engine, triage
from run import backend, books_default


def docx_text(data):
    import docx
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n\n".join(parts)


def gdoc_text(url):
    m = re.search(r"/document/d/([\w-]+)", url)
    if not m:
        raise ValueError("Not a Google Doc link")
    with urllib.request.urlopen(f"https://docs.google.com/document/d/{m.group(1)}/export?format=txt", timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


st.set_page_config(page_title="Adversarial hypothesis checker")
st.title("Adversarial hypothesis checker")
with st.sidebar:
    gen_name = st.selectbox("Generator model", ["gemini", "anthropic", "mock"])
    critic_name = st.selectbox("Reviewer model", ["same as generator", "gemini", "anthropic", "mock"])
    corpus = st.text_input("Corpus folder", "corpus")
    books = st.text_input("Book names (recall channel)", "")
    review = st.checkbox("Create expert triage cards (optional)")
    entail = st.checkbox("Auto entailment pre-check")
    check_math = st.checkbox("Math / number verification (optional)", value=False)

    st.divider()
    if st.button("🛑 End Session (Exit)", help="Stop the local server and exit Python process"):
        st.warning("Session ended. Server stopped. You can safely close this browser tab.")
        time.sleep(0.5)
        os._exit(0)

text = st.text_area("Hypothesis", height=200)
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

if st.button("Run", disabled=not text, type="primary"):
    try:
        gen = backend(gen_name)
        crit = gen if critic_name == "same as generator" else backend(critic_name)
        with st.spinner("Running adversarial review..."):
            rep = engine.run(text, books or books_default(corpus), gen, crit, engine.LocalBM25(corpus), auto_entail=entail,
                             check_math=check_math, log=lambda m: None)
            st.session_state["report"] = rep
            if review:
                d = tempfile.mkdtemp()
                triage.build(rep, d)
                st.session_state["triage_files"] = {
                    fn: open(f"{d}/{fn}").read() for fn in ("triage.md", "responses.csv", "triage.json")
                }
            else:
                st.session_state["triage_files"] = None
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

if st.session_state.get("report"):
    rep = st.session_state["report"]
    rendered = engine.render(rep)
    st.markdown(rendered)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download report.md", rendered, "report.md", mime="text/markdown", key="dl_report_md")
    with col2:
        st.download_button("📥 Download report.json", json.dumps(rep, indent=1), "report.json", mime="application/json", key="dl_report_json")
    if st.session_state.get("triage_files"):
        st.subheader("Expert Triage Files")
        t_cols = st.columns(3)
        for idx, (fn, content) in enumerate(st.session_state["triage_files"].items()):
            with t_cols[idx]:
                st.download_button(f"📥 Download {fn}", content, fn, key=f"dl_{fn}")
