import os
<<<<<<< Updated upstream
import tempfile
import hashlib
from textwrap import dedent
from typing import TypedDict, Dict, Any
=======
import re
import json
import hashlib
import tempfile
import asyncio
from textwrap import dedent
from typing import TypedDict, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
>>>>>>> Stashed changes

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import pymupdf4llm

<<<<<<< Updated upstream
################## Configuration & Setup ##################
# Load environment variables from .env file
load_dotenv()

DEFAULT_MODEL = "llama3.2:1b"
# Retrieve OLLAMA_BASE_URL from environment, default to localhost
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SUMMARY_TEMPERATURE = 0.2
RISKS_TEMPERATURE = 0.2
SUGGESTIONS_TEMPERATURE = 0.3


def ensure_session_defaults():
    """Ensure default values are set in Streamlit session state.
    """
    st.session_state.setdefault("model", DEFAULT_MODEL)
    st.session_state.setdefault("temp_summary", SUMMARY_TEMPERATURE)
    st.session_state.setdefault("temp_risks", RISKS_TEMPERATURE)
    st.session_state.setdefault("temp_suggestions", SUGGESTIONS_TEMPERATURE)

################## Data Structures ###############


class AgentState(TypedDict, total=False):
    """
    Represents the state of the agent during processing.
    Contains the original text and generated outputs at each step.
    """
    original_text: str
    summary: str
    risks: str
    suggestions: str
    final_report: str

#################### Core Utilities ###################


def file_hash(data: bytes) -> str:
    """Generate a SHA256 hash for the given binary data. This hash is used to uniquely identify uploaded files for caching analysis results, preventing redundant processing of the same document."""
    # Generate a SHA256 hash for given binary data
    return hashlib.sha256(data).hexdigest()


@st.cache_resource
def get_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    """Initialize and cache Ollama LLM instance."""
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def call_ollama(prompt: str, model: str, base_url: str, temperature: float) -> str:
    """Invoke LLM with prompt and return response content."""
    llm = get_llm(model, base_url, temperature)
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()


######## AI Agent Nodes #######################

def summarize_node(state: AgentState) -> Dict[str, Any]:
    """Generate executive summary of legal document."""
    text = state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Produce a concise executive summary of this legal document (5–12 bullet points max).
    Focus on practical meaning and major commitments.

    Document:
    {text}
    """)
    return {
        "summary": call_ollama(
            prompt,
            st.session_state.model,
            OLLAMA_BASE_URL,
            SUMMARY_TEMPERATURE,
        )
    }


def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    """Identify legal risks and liabilities in document."""
    text = state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Identify key legal risks and liabilities. For each item include:
    - Risk
    - Why it matters
    - Severity: Low/Med/High
    - Likelihood: Low/Med/High
    - Suggested mitigation (1 line)

    Document:
    {text}
    """)
    return {
        "risks": call_ollama(
            prompt, st.session_state.model, OLLAMA_BASE_URL, RISKS_TEMPERATURE
        )
    }


def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    """Suggest improvements and missing protections for document."""
    text = state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Suggest improvements or missing protections. Prefer specific clause-level suggestions.
    Organize by topic (payment, termination, limitation of liability, indemnity, confidentiality, IP, dispute resolution, warranties).

    Document:
    {text}
    """)
    return {
        "suggestions": call_ollama(
            prompt,
            st.session_state.model,
            OLLAMA_BASE_URL,
            SUGGESTIONS_TEMPERATURE,
        )
    }


def compile_report_node(state: AgentState) -> Dict[str, Any]:
    """Compile final analysis report from summary, risks, and suggestions."""
    report = dedent(f"""
    # Legal Document Analysis (AI-Assisted)

    > Disclaimer: This is not legal advice. Consult a qualified attorney before acting.

    ## 📝 Document Summary
    {state.get("summary", "")}

    ## ⚠️ Identified Risks
    {state.get("risks", "")}

    ## 💡 Suggestions for Improvement
    {state.get("suggestions", "")}
    """).strip()
    return {"final_report": report}


################################ Workflow Management ################################


@st.cache_resource
def get_workflow():
    """Initialize and configure LangGraph workflow for document analysis."""
    workflow = StateGraph(AgentState)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("suggest_improvements", suggest_improvements_node)
    workflow.add_node("compile_report", compile_report_node)

    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "analyze_risks")
    workflow.add_edge("analyze_risks", "suggest_improvements")
    workflow.add_edge("suggest_improvements", "compile_report")
    workflow.add_edge("compile_report", END)

    return workflow.compile()


################################ Document Handling ################################


def load_doc(uploaded_file) -> str:
    """Extract text from uploaded PDF or TXT file."""
    suffix = uploaded_file.name.split(".")[-1].lower()
    data = uploaded_file.getvalue()

    if suffix == "txt":
        return data.decode("utf-8", errors="replace")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    text = pymupdf4llm.to_markdown(tmp_path)
    os.remove(tmp_path)
    return text.strip()


def analyze_document(doc_text: str) -> Dict[str, Any]:
    """Run AI workflow to analyze document text."""
    app = get_workflow()
    return app.invoke({"original_text": doc_text})


################################ Streamlit Application ################################

def render_sidebar_inputs():
    """Render sidebar with file upload and model settings."""
    st.header("Upload Document")
    result = st.file_uploader("Upload Legal Document (PDF or TXT)", type=["pdf", "txt"])

    st.divider()
    st.subheader("Model Settings")
    st.session_state.model = st.text_input("Model", value=DEFAULT_MODEL)

    return result

def analyze_uploaded_file(uploaded_file, h):
    """Process uploaded file and run AI analysis."""
    prog = st.progress(0, text="Reading document...")
    doc_text = load_doc(uploaded_file)

    preview_lines = doc_text.splitlines()
    preview_text = "\n".join(preview_lines[:30])
    with st.expander("Document preview (first 30 lines)"):
        st.markdown(preview_text or "[No content extracted]")

    prog.progress(50, text="Running analysis agents...")
    result = analyze_document(doc_text)

    prog.progress(100, text="Done.")
    st.session_state.results_by_hash[h] = result





def display_analysis_results(h):
    """Display analysis results in organized tabs with download option."""
    result = st.session_state.results_by_hash[h]

    st.markdown("## 📊 Analysis Report")
    summary_tab, risks_tab, suggestions_tab, report_tab = st.tabs(
        ["Summary", "Risks", "Suggestions", "Full report"]
    )

    with summary_tab:
        st.markdown("### 📝 Document Summary")
        st.markdown(result.get("summary", ""))

    with risks_tab:
        st.markdown("### ⚠️ Identified Risks")
        st.markdown(result.get("risks", ""))

    with suggestions_tab:
        st.markdown("### 💡 Suggestions for Improvement")
        st.markdown(result.get("suggestions", ""))

    with report_tab:
        st.markdown(result.get("final_report", ""))

    st.download_button(
        "Download Report", result.get("final_report", ""), file_name="legal_analysis.md"
    )




def main():
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Legal Doc Analyzer", layout="wide")
    ensure_session_defaults()

    with st.sidebar:
        uploaded_file = render_sidebar_inputs()
    st.title("⚖️ AI Legal Document Analyzer")

    if uploaded_file:
        data = uploaded_file.getvalue()
        h = file_hash(data)
        st.session_state.setdefault("results_by_hash", {})

        if st.button("Analyze Document"):
            analyze_uploaded_file(uploaded_file, h)
        if h in st.session_state.results_by_hash:
            display_analysis_results(h)
    else:
        st.warning("Upload a PDF or TXT document to begin.")


# Ensure the main function runs when the script is executed
if __name__ == "__main__":
    main()
=======
# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
load_dotenv()

DEFAULT_MODEL     = "llama3.2:3b"        # 3b is the minimum viable for legal reasoning
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_TOKEN_LIMIT = 1200                  # ~900 words per chunk — safe for 4k context models
MAX_DOC_TOKENS    = 12_000               # hard ceiling before chunking kicks in
SUMMARY_TEMP      = 0.2
RISKS_TEMP        = 0.15                 # lower = more deterministic for risk IDs
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
    """Fresh LLM instance — NOT cached, so temperature is always respected."""
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
    """
    Semantic chunking: split on legal section headers first,
    then enforce CHUNK_TOKEN_LIMIT by further splitting long sections
    on paragraph breaks.
    """
    # Split on common legal section patterns
    section_pattern = r"\n(?=(?:ARTICLE|SECTION|CLAUSE|SCHEDULE|EXHIBIT|ANNEX)\s+[\dIVXA-Z]|\d+\.\s+[A-Z])"
    raw_sections = re.split(section_pattern, text, flags=re.IGNORECASE)

    chunks: list[str] = []
    for sec in raw_sections:
        if word_count(sec) <= CHUNK_TOKEN_LIMIT:
            chunks.append(sec.strip())
        else:
            # Further split on blank lines (paragraph boundaries)
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


def extract_relevant_chunks(chunks: list[str], keywords: list[str]) -> str:
    """Return only chunks that contain at least one keyword. Falls back to first 3."""
    relevant = [c for c in chunks if any(kw.lower() in c.lower() for kw in keywords)]
    selected = relevant[:5] if relevant else chunks[:3]
    return "\n\n---\n\n".join(selected)

# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
REQUIRED_SECTIONS = {
    "summary":     ["bullet", "•", "-", "*"],
    "risks":       ["risk", "severit", "liabilit", "mitigat"],
    "suggestions": ["suggest", "clause", "recommend", "should"],
}

def validate_output(field: str, content: str) -> Dict[str, Any]:
    """
    Lightweight structural validation — checks that the LLM actually
    produced content relevant to the expected section.
    """
    if not content or content.startswith("[LLM ERROR"):
        return {"ok": False, "reason": "Empty or errored response"}

    if len(content.split()) < 30:
        return {"ok": False, "reason": "Response too short — likely truncated"}

    markers = REQUIRED_SECTIONS.get(field, [])
    if markers and not any(m.lower() in content.lower() for m in markers):
        return {
            "ok": False,
            "reason": f"Response missing expected structure (checked for: {markers})",
        }

    return {"ok": True, "reason": "Passed"}


def validate_all(state: AgentState) -> Dict[str, Any]:
    return {
        "summary":     validate_output("summary",     state.get("summary", "")),
        "risks":       validate_output("risks",       state.get("risks", "")),
        "suggestions": validate_output("suggestions", state.get("suggestions", "")),
    }

# ─────────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────────

def chunk_node(state: AgentState) -> Dict[str, Any]:
    """Pre-processing: chunk the document once, shared by all downstream nodes."""
    chunks = chunk_document(state["original_text"])
    return {"chunks": chunks}


def summarize_node(state: AgentState) -> Dict[str, Any]:
    """
    Map-reduce summarization:
      1. Mini-summarize each chunk independently (parallelised).
      2. Synthesize mini-summaries into one executive summary.
    """
    chunks = state.get("chunks") or chunk_document(state["original_text"])

    # ── Map phase (parallel) ──────────────────────────────────────
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

    # ── Reduce phase ─────────────────────────────────────────────
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

    summary = call_llm(synthesis_prompt, SUMMARY_TEMP)
    return {"summary": summary}


def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    """Keyword-routed risk analysis — only sends relevant chunks."""
    chunks = state.get("chunks") or chunk_document(state["original_text"])
    relevant_text = extract_relevant_chunks(chunks, RISK_KEYWORDS)

    prompt = dedent(f"""
    You are an expert legal risk analyst. Analyze the following clauses for legal risks.

    For EACH risk output EXACTLY this format:
    **Risk**: [name]
    **Why it matters**: [one sentence]
    **Severity**: Low | Med | High
    **Likelihood**: Low | Med | High
    **Mitigation**: [one concrete action]

    Identify at least 5 risks. Do not repeat yourself. No preamble.

    Clauses:
    {relevant_text}
    """)

    return {"risks": call_llm(prompt, RISKS_TEMP)}


def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    """Keyword-routed suggestion generation — only sends relevant chunks."""
    chunks = state.get("chunks") or chunk_document(state["original_text"])
    relevant_text = extract_relevant_chunks(chunks, SUGGEST_KEYWORDS)

    prompt = dedent(f"""
    You are an expert contract attorney. Suggest specific improvements to the clauses below.

    Organise suggestions by topic. For each suggestion:
    - State the current issue
    - Propose specific new or revised language
    - Explain the benefit in one sentence

    Topics to cover (only if present in the document):
    Payment · Termination · Limitation of Liability · Indemnity ·
    Confidentiality · IP · Dispute Resolution · Warranties · Non-Compete

    No preamble. Be specific — cite section numbers if visible.

    Clauses:
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
    """
    Graph topology:
        chunk_node
            │
            ├──► summarize_node ──────────────────────┐
            ├──► analyze_risks_node ──────────────────┤
            └──► suggest_improvements_node ───────────┤
                                                       ▼
                                               compile_report_node
    risks + suggestions run in parallel after chunking.
    LangGraph fans them out automatically via multiple edges from chunk_node.
    """
    wf = StateGraph(AgentState)
    wf.add_node("chunk",       chunk_node)
    wf.add_node("summarize",   summarize_node)
    wf.add_node("risks",       analyze_risks_node)
    wf.add_node("suggestions", suggest_improvements_node)
    wf.add_node("compile",     compile_report_node)

    wf.set_entry_point("chunk")

    # Fan-out: all three analysis nodes run after chunking
    wf.add_edge("chunk",     "summarize")
    wf.add_edge("chunk",     "risks")
    wf.add_edge("chunk",     "suggestions")

    # Fan-in: compile only after all three complete
    wf.add_edge("summarize",   "compile")
    wf.add_edge("risks",       "compile")
    wf.add_edge("suggestions", "compile")

    wf.add_edge("compile", END)
    return wf.compile()

# ─────────────────────────────────────────────
# Document loading
# ─────────────────────────────────────────────
def load_doc(uploaded_file) -> str:
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    data   = uploaded_file.getvalue()

    if suffix == "txt":
        text = data.decode("utf-8", errors="replace")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            text = pymupdf4llm.to_markdown(tmp_path)
        finally:
            os.remove(tmp_path)

    # Hard token guard
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

/* Tabs */
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

/* Buttons */
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

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: var(--radius) !important;
}

/* Progress */
.stProgress > div > div { background: var(--accent) !important; }

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}

/* Text input */
.stTextInput input {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Metric cards */
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

/* Validation badge */
.badge-ok   { color: var(--ok);     font-family: 'DM Mono', monospace; font-size: 0.75rem; }
.badge-fail { color: var(--danger); font-family: 'DM Mono', monospace; font-size: 0.75rem; }

/* Divider */
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
    chunks   = chunk_document(doc_text)
    wc       = word_count(doc_text)

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
    st.session_state.results_by_hash[h] = {**result, "validation": validation, "word_count": wc, "chunk_count": len(chunks)}


def display_results(h: str):
    result     = st.session_state.results_by_hash[h]
    validation = result.get("validation", {})

    st.markdown("---")

    # Validation strip
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
        risks_text = result.get("risks") or "_Not generated_"
        # Render **bold** risk headers nicely
        st.markdown(risks_text)

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

>>>>>>> Stashed changes
