""" Initial framework, no longer being used. use app2.py"""
import os
import re
import json
import hashlib
import tempfile
import asyncio
from textwrap import dedent
from typing import TypedDict, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import pymupdf4llm

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
load_dotenv()

DEFAULT_MODEL     = "llama3.2:3b"
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_TOKEN_LIMIT = 1200
MAX_DOC_TOKENS    = 12_000
SUMMARY_TEMP      = 0.2
RISKS_TEMP        = 0.15
SUGGEST_TEMP      = 0.3

# ─────────────────────────────────────────────
# Keyword routing tables
# ─────────────────────────────────────────────
RISK_KEYWORDS = [
    "liabilit", "indemnif", "warrant", "terminat", "breach",
    "penalt", "damages", "default", "governing law", "force majeure",
    "consequential", "limitation", "negligence", "represent",
]
SUGGEST_KEYWORDS = [
    "payment", "intellectual property", "confidential", "dispute",
    "limitation", "exclusiv", "assign", "non-compet", "solicit",
    "audit", "renewal", "notice", "amendment", "severab",
]

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    original_text : str
    chunks        : list[str]
    summary       : str
    risks         : str
    suggestions   : str
    final_report  : str
    validation    : Dict[str, Any]

# ─────────────────────────────────────────────
# Session defaults
# ─────────────────────────────────────────────
def ensure_session_defaults():
    st.session_state.setdefault("model", DEFAULT_MODEL)
    st.session_state.setdefault("results_by_hash", {})

# ─────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────
def get_llm(model: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)


def call_llm(prompt: str, temperature: float, model: str | None = None) -> str:
    m = model or st.session_state.get("model", DEFAULT_MODEL)
    llm = get_llm(m, temperature)
    try:
        resp = llm.invoke(prompt)
        return (resp.content or "").strip()
    except Exception as e:
        return f"[LLM ERROR: {e}]"

# ─────────────────────────────────────────────
# Token / text utilities
# ─────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split())


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_document(text: str) -> list[str]:
    section_pattern = r"\n(?=(?:ARTICLE|SECTION|CLAUSE|SCHEDULE|EXHIBIT|ANNEX)\s+[\dIVXA-Z]|\d+\.\s+[A-Z])"
    raw_sections = re.split(section_pattern, text, flags=re.IGNORECASE)

    chunks: list[str] = []
    for sec in raw_sections:
        if word_count(sec) <= CHUNK_TOKEN_LIMIT:
            chunks.append(sec.strip())
        else:
            paras = re.split(r"\n\s*\n", sec)
            current, count = [], 0
            for p in paras:
                wc = word_count(p)
                if count + wc > CHUNK_TOKEN_LIMIT and current:
                    chunks.append("\n\n".join(current).strip())
                    current, count = [p], wc
                else:
                    current.append(p)
                    count += wc
            if current:
                chunks.append("\n\n".join(current).strip())

    return [c for c in chunks if c]


def extract_relevant_chunks(chunks: list[str], keywords: list[str]) -> tuple[str, bool]:
    """
    Score chunks by keyword hit count, return top matches.
    Falls back to ALL chunks if nothing matches — never silently drops content.
    Returns (text, was_fallback).
    """
    scored = []
    for chunk in chunks:
        lower = chunk.lower()
        score = sum(lower.count(kw.lower()) for kw in keywords)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    matched = [c for s, c in scored if s > 0][:5]

    if matched:
        return "\n\n---\n\n".join(matched), False

    # Fallback: no keywords found — send everything (already chunked/safe)
    return "\n\n---\n\n".join(chunks), True

# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
# Wide nets — llama3.2 paraphrases heavily and never echoes prompt keywords verbatim.
# We check that the response is on-topic, not that it used specific words.
VALIDATION_SIGNALS = {
    "summary": [
        "party", "agree", "shall", "term", "payment", "right",
        "oblig", "contract", "licens", "servic", "provid", "period",
        "-", "•", "*", "1.",
    ],
    "risks": [
        "risk", "concern", "issue", "problem", "exposure", "danger",
        "high", "medium", "low", "potential", "could", "may", "if",
        "breach", "terminat", "penalt", "loss", "fail", "lack",
    ],
    "suggestions": [
        "should", "could", "recommend", "consider", "add", "include",
        "improve", "update", "ensure", "require", "specify", "define",
        "protect", "strengthen", "provide", "clear", "explicit",
    ],
}

def validate_output(field: str, content: str) -> Dict[str, Any]:
    if not content or content.startswith("[LLM ERROR"):
        return {"ok": False, "reason": content if content.startswith("[LLM ERROR") else "Empty response"}

    if word_count(content) < 20:
        return {"ok": False, "reason": f"Too short ({word_count(content)} words) — model may have timed out"}

    signals = VALIDATION_SIGNALS.get(field, [])
    matched = [s for s in signals if s.lower() in content.lower()]
    # Require 2 matches for risks/suggestions (wide net), 1 for summary
    needed  = 2 if field in ("risks", "suggestions") else 1

    if len(matched) < needed:
        return {
            "ok": False,
            "reason": f"Response appears off-topic ({len(matched)}/{needed} signals matched)",
        }

    return {"ok": True, "reason": "Passed"}


def validate_all(state: AgentState) -> Dict[str, Any]:
    return {k: validate_output(k, state.get(k, "")) for k in ("summary", "risks", "suggestions")}

# ─────────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────────

def chunk_node(state: AgentState) -> Dict[str, Any]:
    chunks = chunk_document(state["original_text"])
    return {"chunks": chunks}


def summarize_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or chunk_document(state["original_text"])

    mini_prompt_tpl = dedent("""
    You are a legal assistant. Summarize this section in 3–4 bullet points.
    Focus only on: obligations, rights, key terms, and monetary values.
    Be concise. No preamble.

    Section:
    {chunk}
    """)

    mini_summaries: list[str] = [""] * len(chunks)

    def summarize_chunk(idx_chunk):
        idx, chunk = idx_chunk
        return idx, call_llm(mini_prompt_tpl.format(chunk=chunk), SUMMARY_TEMP)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(summarize_chunk, (i, c)): i for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            mini_summaries[idx] = result

    combined = "\n\n---\n\n".join(mini_summaries)
    synthesis_prompt = dedent(f"""
    You are a legal assistant. Below are section-by-section summaries of a legal document.
    Produce ONE cohesive executive summary (8–12 bullet points).
    Rules:
    - Eliminate redundancy
    - Preserve ALL material obligations, rights, deadlines, and monetary values
    - Sort bullets from most to least important
    - No preamble, no "here is a summary"

    Section summaries:
    {combined}
    """)

    return {"summary": call_llm(synthesis_prompt, SUMMARY_TEMP)}


def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or chunk_document(state["original_text"])
    relevant_text, was_fallback = extract_relevant_chunks(chunks, RISK_KEYWORDS)

    fallback_note = (
        "No specific risk clauses were found by keyword scan. "
        "Analyse the full document below for any legal risks, obligations, or exposures.\n\n"
        if was_fallback else ""
    )

    # Deliberately loose format instruction — llama3.2 ignores rigid templates.
    # Ask for numbered points with natural labels it will actually produce.
    prompt = dedent(f"""
    You are an expert legal risk analyst. {fallback_note}
    Read the document below and identify the key legal risks.

    For each risk write:
    1. A short name for the risk
    2. Why it matters (one sentence)
    3. How serious it is: Low, Medium, or High
    4. One action to reduce or avoid the risk

    Find at least 3 risks. Write clearly. No preamble.

    Document:
    {relevant_text}
    """)

    return {"risks": call_llm(prompt, RISKS_TEMP)}


def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or chunk_document(state["original_text"])
    relevant_text, was_fallback = extract_relevant_chunks(chunks, SUGGEST_KEYWORDS)

    fallback_note = (
        "No specific improvement-target clauses were found by keyword scan. "
        "Review the full document below and suggest any improvements or missing protections.\n\n"
        if was_fallback else ""
    )

    # Same principle — ask for numbered plain-language output, not rigid markdown templates.
    prompt = dedent(f"""
    You are an expert contract attorney. {fallback_note}
    Read the document below and suggest specific improvements.

    For each suggestion write:
    1. What the current problem or gap is
    2. What should be added or changed (be specific)
    3. Why this change protects the parties

    Give at least 3 suggestions. Cover areas like payment, termination, liability,
    confidentiality, IP, and dispute resolution where relevant. No preamble.

    Document:
    {relevant_text}
    """)

    return {"suggestions": call_llm(prompt, SUGGEST_TEMP)}


def compile_report_node(state: AgentState) -> Dict[str, Any]:
    validation = validate_all(state)
    warnings = ""
    for field, result in validation.items():
        if not result["ok"]:
            warnings += f"\n> ⚠️ **{field.title()}**: {result['reason']}"

    report = dedent(f"""
    # Legal Document Analysis (AI-Assisted)

    > **Disclaimer**: This analysis is AI-generated and does not constitute legal advice.
    > Consult a qualified attorney before acting on any findings.
    {warnings}

    ---

    ## 📝 Executive Summary
    {state.get("summary", "_Not generated_")}

    ---

    ## ⚠️ Identified Risks
    {state.get("risks", "_Not generated_")}

    ---

    ## 💡 Suggestions for Improvement
    {state.get("suggestions", "_Not generated_")}
    """).strip()

    return {"final_report": report, "validation": validation}

# ─────────────────────────────────────────────
# Parallel workflow
# ─────────────────────────────────────────────
@st.cache_resource
def get_workflow():
    wf = StateGraph(AgentState)
    wf.add_node("chunk",       chunk_node)
    wf.add_node("summarize",   summarize_node)
    wf.add_node("risks",       analyze_risks_node)
    wf.add_node("suggestions", suggest_improvements_node)
    wf.add_node("compile",     compile_report_node)

    wf.set_entry_point("chunk")
    wf.add_edge("chunk",       "summarize")
    wf.add_edge("chunk",       "risks")
    wf.add_edge("chunk",       "suggestions")
    wf.add_edge("summarize",   "compile")
    wf.add_edge("risks",       "compile")
    wf.add_edge("suggestions", "compile")
    wf.add_edge("compile",     END)
    return wf.compile()

# ─────────────────────────────────────────────
# Document loading
# ─────────────────────────────────────────────
def load_doc(uploaded_file) -> str:
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    data   = uploaded_file.getvalue()

    if not data:
        st.error("Uploaded file is empty.")
        return ""

    if suffix == "txt":
        text = data.decode("utf-8", errors="replace")
    else:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            text = pymupdf4llm.to_markdown(tmp_path)
        except Exception as e:
            st.error(f"PDF extraction failed: {e}. Try saving as TXT and re-uploading.")
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not text or not text.strip():
        st.error("No text could be extracted. The file may be a scanned image PDF.")
        return ""

    words = text.split()
    if len(words) > MAX_DOC_TOKENS:
        st.warning(
            f"⚠️ Document is large ({len(words):,} words). "
            f"Truncated to {MAX_DOC_TOKENS:,} words to stay within model context limits."
        )
        text = " ".join(words[:MAX_DOC_TOKENS])

    return text.strip()

# ─────────────────────────────────────────────
# UI — custom CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0e0f13;
    --surface:   #16181f;
    --border:    #2a2d38;
    --accent:    #c8a96e;
    --accent2:   #7b9cda;
    --danger:    #e07070;
    --ok:        #6ec87a;
    --text:      #e8e6e1;
    --muted:     #7a7d8a;
    --radius:    10px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1 { font-family: 'DM Serif Display', serif !important; color: var(--accent) !important; letter-spacing: -0.5px; }
h2, h3 { font-family: 'DM Serif Display', serif !important; color: var(--text) !important; }

button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #0e0f13 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 500 !important;
    transition: opacity 0.2s ease;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: var(--radius) !important;
}

.stProgress > div > div { background: var(--accent) !important; }

[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}

.stTextInput input {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: var(--accent);
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

.badge-ok   { color: var(--ok);     font-family: 'DM Mono', monospace; font-size: 0.75rem; }
.badge-fail { color: var(--danger); font-family: 'DM Mono', monospace; font-size: 0.75rem; }

hr { border-color: var(--border) !important; }
</style>
"""

# ─────────────────────────────────────────────
# UI rendering helpers
# ─────────────────────────────────────────────
def render_metric(label: str, value: str):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_validation_badge(field: str, result: Dict[str, Any]):
    if result["ok"]:
        st.markdown(f'<span class="badge-ok">✓ {field.title()} — passed</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<span class="badge-fail">✗ {field.title()} — {result["reason"]}</span>',
            unsafe_allow_html=True,
        )


def render_sidebar() -> Any:
    st.markdown("### ⚖️ Legal Analyzer")
    st.markdown(
        '<span style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#7a7d8a;">AI-ASSISTED · NOT LEGAL ADVICE</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    uploaded = st.file_uploader("Upload document", type=["pdf", "txt"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model**")
    st.session_state.model = st.text_input(
        "Ollama model", value=st.session_state.get("model", DEFAULT_MODEL), label_visibility="collapsed"
    )
    st.markdown(
        '<span style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#7a7d8a;">'
        'Recommended: llama3.2:3b or mistral:7b</span>',
        unsafe_allow_html=True,
    )
    return uploaded


def run_analysis(uploaded_file, h: str):
    prog = st.progress(0, text="📄 Extracting document text…")
    doc_text = load_doc(uploaded_file)

    if not doc_text:
        prog.empty()
        return

    chunks = chunk_document(doc_text)
    wc     = word_count(doc_text)

    c1, c2, c3 = st.columns(3)
    with c1: render_metric("Words", f"{wc:,}")
    with c2: render_metric("Chunks", str(len(chunks)))
    with c3: render_metric("Model", st.session_state.model.split(":")[0].upper())

    with st.expander("Document preview (first 30 lines)"):
        st.markdown("\n".join(doc_text.splitlines()[:30]) or "_No content_")

    prog.progress(20, text="⚙️ Chunking & routing…")

    with st.spinner("🤖 Running parallel analysis (summary · risks · suggestions)…"):
        prog.progress(40, text="🔄 Agents running in parallel…")
        app    = get_workflow()
        result = app.invoke({"original_text": doc_text, "chunks": chunks})

    prog.progress(90, text="✅ Validating outputs…")
    validation = result.get("validation") or validate_all(result)

    prog.progress(100, text="Done.")
    st.session_state.results_by_hash[h] = {
        **result,
        "validation":  validation,
        "word_count":  wc,
        "chunk_count": len(chunks),
    }


def display_results(h: str):
    result     = st.session_state.results_by_hash[h]
    validation = result.get("validation", {})

    st.markdown("---")

    vcols = st.columns(3)
    for col, field in zip(vcols, ["summary", "risks", "suggestions"]):
        with col:
            render_validation_badge(field, validation.get(field, {"ok": False, "reason": "No result"}))

    st.markdown("---")

    summary_tab, risks_tab, suggest_tab, report_tab = st.tabs(
        ["📝 SUMMARY", "⚠️ RISKS", "💡 SUGGESTIONS", "📄 FULL REPORT"]
    )

    with summary_tab:
        st.markdown(result.get("summary") or "_Not generated_")

    with risks_tab:
        st.markdown(result.get("risks") or "_Not generated_")

    with suggest_tab:
        st.markdown(result.get("suggestions") or "_Not generated_")

    with report_tab:
        st.markdown(result.get("final_report") or "_Not generated_")
        st.download_button(
            "⬇ Download Report (.md)",
            data=result.get("final_report", ""),
            file_name="legal_analysis.md",
            mime="text/markdown",
        )

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Legal Doc Analyzer",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    ensure_session_defaults()

    with st.sidebar:
        uploaded_file = render_sidebar()

    st.markdown("# Legal Document Analyzer")
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.8rem;color:#7a7d8a;">'
        'PARALLEL AI AGENTS · MAP-REDUCE SUMMARIZATION · KEYWORD ROUTING</p>',
        unsafe_allow_html=True,
    )

    if not uploaded_file:
        st.info("Upload a PDF or TXT legal document in the sidebar to begin.")
        return

    data = uploaded_file.getvalue()
    h    = file_hash(data)

    col_analyze, col_clear = st.columns([2, 1])
    with col_analyze:
        if st.button("⚡ Analyze Document"):
            run_analysis(uploaded_file, h)
    with col_clear:
        if st.button("🗑 Clear Cache") and h in st.session_state.results_by_hash:
            del st.session_state.results_by_hash[h]
            st.rerun()

    if h in st.session_state.results_by_hash:
        display_results(h)


if __name__ == "__main__":
    main()
