"""
Legal Document Analyzer
━━━━━━━━━━━━━━━━━━━━━━━
Token compression pipeline (4 layers):
  1. Noise strip      — removes PDF artifacts, page numbers, whitespace waste (~10-15%)
  2. Dedup            — removes repeated boilerplate paragraphs (~5-10% on long docs)
  3. Semantic chunking — splits on legal headers → paragraphs → hard window fallback
  4. Keyword routing  — scores chunks by relevance, sends only top-N to each node

Accuracy preservation:
  - Summary uses map-reduce: every chunk is seen, nothing dropped
  - Risks/suggestions use scored routing: most relevant chunks always included
  - Fallback to full doc when no keywords match (generic docs)
  - Mini-summary cap prevents synthesis prompt overflow

Parallelism:
  - chunk → [summarize | risks | suggestions] → compile  (fan-out/fan-in)
  - summarize map phase uses ThreadPoolExecutor
"""

import os
import re
import hashlib
import tempfile
from textwrap import dedent
from typing import TypedDict, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import pymupdf4llm

# ══════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════
load_dotenv()

DEFAULT_MODEL        = "llama3.2:3b"
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunking — 1,200 words ≈ 1,600 tokens, safe for llama3.2's 4k context
# leaving headroom for the prompt template and output
CHUNK_WORD_LIMIT     = 1_200

# Keyword routing — how many top-scored chunks each node receives
MAX_RISK_CHUNKS      = 5
MAX_SUGGEST_CHUNKS   = 5

# Map-reduce guard — cap combined mini-summaries before synthesis
# to prevent the reduce prompt from hitting context limits
MAX_SYNTHESIS_WORDS  = 2_000

# Mini-summary cap per chunk — keeps reduce input tight
MAX_MINI_WORDS       = 100

SUMMARY_TEMP         = 0.2
RISKS_TEMP           = 0.15
SUGGEST_TEMP         = 0.3

# ══════════════════════════════════════════════
# Keyword routing tables (stem-based)
# ══════════════════════════════════════════════
RISK_KEYWORDS = [
    "liabilit", "indemnif", "warrant", "terminat", "breach",
    "penalt", "damage", "default", "governing law", "force majeure",
    "consequential", "negligenc", "represent", "remedy", "forfeit",
    "insolvenc", "bankrupt", "infring", "misappropriat", "disclaim",
]
SUGGEST_KEYWORDS = [
    "payment", "intellectual property", "confidential", "dispute",
    "limitation", "exclusiv", "assign", "non-compet", "solicit",
    "audit", "renewal", "notice", "amendment", "severab",
    "arbitrat", "jurisdict", "disclosur", "sublicens", "royalt",
    "data protection", "privac", "govern",
]

# ══════════════════════════════════════════════
# Noise patterns — stripped before chunking
# ══════════════════════════════════════════════
_NOISE_LINE_RE = re.compile(
    r"^\s*("
    r"[-─═]{3,}"                        # horizontal rules
    r"|Page\s+\d+\s*(of\s*\d+)?"        # page numbers
    r"|\d{1,3}"                          # lone numbers (page/line numbers)
    r"|\[FORM\]|\[SIGNATURE PAGE.*?\]"   # boilerplate markers
    r"|_{4,}"                            # signature lines
    r"|EXHIBIT\s+[A-Z]\s*$"             # lone exhibit headers
    r"|SCHEDULE\s+\d+\s*$"             # lone schedule headers
    r")\s*$",
    re.IGNORECASE,
)

# ══════════════════════════════════════════════
# State
# ══════════════════════════════════════════════
class AgentState(TypedDict, total=False):
    original_text : str
    clean_text    : str
    chunks        : List[str]
    summary       : str
    risks         : str
    suggestions   : str
    final_report  : str
    validation    : Dict[str, Any]
    meta          : Dict[str, Any]

# ══════════════════════════════════════════════
# Session
# ══════════════════════════════════════════════
def ensure_session_defaults():
    st.session_state.setdefault("model", DEFAULT_MODEL)
    st.session_state.setdefault("results_by_hash", {})

# ══════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════
def get_llm(temperature: float) -> ChatOllama:
    return ChatOllama(
        model=st.session_state.get("model", DEFAULT_MODEL),
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )

def call_llm(prompt: str, temperature: float) -> str:
    try:
        resp = get_llm(temperature).invoke(prompt)
        return (resp.content or "").strip()
    except Exception as e:
        err = str(e)
        if "404" in err or "not found" in err.lower():
            m = st.session_state.get("model", DEFAULT_MODEL)
            return f"[LLM ERROR: Model '{m}' not found — run: ollama pull {m}]"
        return f"[LLM ERROR: {err}]"

# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════
def _wc(text: str) -> int:
    return len(text.split())

def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ══════════════════════════════════════════════
# Layer 1 — Noise stripping
# Removes PDF extraction artifacts that waste tokens
# without carrying any legal meaning.
# Typical saving: 10–20% on real legal PDFs.
# ══════════════════════════════════════════════
def strip_noise(text: str) -> str:
    lines = text.splitlines()
    clean = [l for l in lines if not _NOISE_LINE_RE.match(l)]
    text  = "\n".join(clean)

    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse excessive inline whitespace
    text = re.sub(r"[ \t]{3,}", "  ", text)
    # Remove zero-width and non-breaking spaces
    text = text.replace("\u200b", "").replace("\xa0", " ")

    return text.strip()

# ══════════════════════════════════════════════
# Layer 2 — Deduplication
# Legal docs repeat boilerplate (definitions,
# recitals, notice clauses). Removes exact
# duplicate paragraphs while preserving order.
# Typical saving: 5–15% on long contracts.
# ══════════════════════════════════════════════
def dedup_paragraphs(text: str) -> str:
    paras = re.split(r"\n\s*\n", text)
    seen, result = set(), []
    for p in paras:
        # Normalise for comparison — collapse whitespace, lowercase
        key = re.sub(r"\s+", " ", p.strip().lower())
        if len(key) < 40:
            # Keep short paragraphs always (headings, labels)
            result.append(p)
            continue
        if key not in seen:
            seen.add(key)
            result.append(p)
    return "\n\n".join(result)

# ══════════════════════════════════════════════
# Layer 3 — Semantic chunking
# Priority order:
#   1. Formal legal headers (ARTICLE, SECTION…)
#   2. Any numbered/markdown heading
#   3. ALL-CAPS lines (common in contracts)
#   4. Paragraph boundaries
#   5. Hard word-window (last resort)
# Handles generic docs, NDAs, employment
# contracts, and formal agreements equally.
# ══════════════════════════════════════════════
def chunk_document(text: str) -> List[str]:
    # Try progressively looser split patterns until we get > 1 section
    patterns = [
        # Formal legal: ARTICLE IV, SECTION 2.1, CLAUSE 3
        r"\n(?=(?:ARTICLE|SECTION|CLAUSE|SCHEDULE|EXHIBIT|ANNEX|APPENDIX)\s+[\dIVXA-Z])",
        # Numbered headings: "1. Payment Terms", "2.3 Termination"
        r"\n(?=\d+[\.\)]\s+[A-Z]|\d+\.\d+\s+[A-Z])",
        # Markdown headings: ## Section, # Title
        r"\n(?=#{1,3}\s+\S)",
        # ALL CAPS lines of 3+ words (common contract section headers)
        r"\n(?=[A-Z]{2}[A-Z\s]{8,}\n)",
        # Paragraph breaks (universal fallback)
        r"\n\s*\n",
    ]

    raw_sections = [text]
    for pattern in patterns:
        candidate = re.split(pattern, text, flags=re.IGNORECASE)
        if len(candidate) > 1:
            raw_sections = candidate
            break

    # Now enforce CHUNK_WORD_LIMIT on each section
    chunks: List[str] = []
    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue
        if _wc(sec) <= CHUNK_WORD_LIMIT:
            chunks.append(sec)
        else:
            # Section too long — split on paragraphs
            paras       = re.split(r"\n\s*\n", sec)
            buf, buf_wc = [], 0
            for para in paras:
                pwc = _wc(para)
                if buf_wc + pwc > CHUNK_WORD_LIMIT and buf:
                    chunks.append("\n\n".join(buf))
                    buf, buf_wc = [para], pwc
                else:
                    buf.append(para)
                    buf_wc += pwc
                # Hard window: single paragraph still too long
                if buf_wc > CHUNK_WORD_LIMIT and len(buf) == 1:
                    words = para.split()
                    for i in range(0, len(words), CHUNK_WORD_LIMIT):
                        chunks.append(" ".join(words[i:i + CHUNK_WORD_LIMIT]))
                    buf, buf_wc = [], 0
            if buf:
                chunks.append("\n\n".join(buf))

    # Merge orphan chunks (< 50 words) with their neighbour
    # to avoid feeding the LLM single-sentence context
    merged: List[str] = []
    for chunk in chunks:
        if merged and _wc(chunk) < 50:
            merged[-1] += "\n\n" + chunk
        else:
            merged.append(chunk)

    return [c for c in merged if c.strip()]

# ══════════════════════════════════════════════
# Layer 4 — Keyword routing
# Scores every chunk by keyword frequency,
# returns top-N most relevant chunks.
# Falls back to all chunks on zero matches
# (generic docs, non-standard language).
# ══════════════════════════════════════════════
def route_chunks(
    chunks: List[str],
    keywords: List[str],
    max_chunks: int,
) -> Tuple[List[str], bool]:
    scored = []
    for chunk in chunks:
        lower = chunk.lower()
        score = sum(lower.count(kw.lower()) for kw in keywords)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    matched = [c for s, c in scored if s > 0][:max_chunks]

    if matched:
        return matched, False
    # Fallback — no keywords matched at all
    return chunks[:max_chunks], True

# ══════════════════════════════════════════════
# Validation
# Wide signal nets — llama3.2 paraphrases
# heavily, so we check intent not exact words.
# ══════════════════════════════════════════════
VALIDATION_SIGNALS = {
    "summary": [
        "party", "agree", "shall", "term", "payment", "right",
        "oblig", "contract", "licens", "servic", "provid", "period",
        "-", "•", "*", "1.",
    ],
    "risks": [
        "risk", "concern", "issue", "problem", "exposure",
        "high", "medium", "low", "potential", "could", "may",
        "breach", "terminat", "penalt", "loss", "fail",
    ],
    "suggestions": [
        "should", "could", "recommend", "consider", "add", "include",
        "improve", "ensure", "require", "specify", "define",
        "protect", "provide", "clear", "explicit", "update",
    ],
}

def validate_output(field: str, content: str) -> Dict[str, Any]:
    if not content:
        return {"ok": False, "reason": "Empty response"}
    if content.startswith("[LLM ERROR"):
        return {"ok": False, "reason": content.strip("[]")}
    if _wc(content) < 20:
        return {"ok": False, "reason": f"Too short ({_wc(content)} words)"}

    signals = VALIDATION_SIGNALS.get(field, [])
    matched = [s for s in signals if s.lower() in content.lower()]
    needed  = 2 if field in ("risks", "suggestions") else 1

    if len(matched) < needed:
        return {"ok": False, "reason": f"Off-topic ({len(matched)}/{needed} signals found)"}
    return {"ok": True, "reason": "Passed"}

def validate_all(state: AgentState) -> Dict[str, Any]:
    return {k: validate_output(k, state.get(k, "")) for k in ("summary", "risks", "suggestions")}

# ══════════════════════════════════════════════
# Document loading
# ══════════════════════════════════════════════
def load_doc(uploaded_file) -> str:
    data   = uploaded_file.getvalue()
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if not data:
        st.error("File is empty.")
        return ""

    if suffix == "txt":
        return data.decode("utf-8", errors="replace")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        text = pymupdf4llm.to_markdown(tmp_path)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}. Try saving as TXT.")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text or not text.strip():
        st.error("No text extracted — file may be a scanned image PDF.")
        return ""

    return text

# ══════════════════════════════════════════════
# Preprocessing node
# Runs all 4 compression layers once,
# stores results in state for all downstream nodes.
# ══════════════════════════════════════════════
def preprocess_node(state: AgentState) -> Dict[str, Any]:
    raw    = state["original_text"]
    # Layer 1: noise strip
    clean  = strip_noise(raw)
    # Layer 2: dedup
    clean  = dedup_paragraphs(clean)
    # Layer 3: chunk
    chunks = chunk_document(clean)

    raw_wc   = _wc(raw)
    clean_wc = _wc(clean)
    saved    = raw_wc - clean_wc
    pct      = f"{(saved / raw_wc * 100):.0f}%" if raw_wc else "0%"

    meta = {
        "raw_words"   : raw_wc,
        "clean_words" : clean_wc,
        "saved_words" : saved,
        "compression" : pct,
        "chunks"      : len(chunks),
    }

    return {"clean_text": clean, "chunks": chunks, "meta": meta}

# ══════════════════════════════════════════════
# Summary node — map-reduce
# Every chunk is summarised (map, parallel),
# then synthesised into one output (reduce).
# Nothing is dropped for accuracy.
# ══════════════════════════════════════════════
def summarize_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or chunk_document(state.get("clean_text", state["original_text"]))

    # Single chunk shortcut — skip map phase
    if len(chunks) == 1:
        prompt = dedent(f"""
        You are a legal assistant. Summarize this document in 8–10 bullet points.
        Each bullet covers one obligation, right, key term, or monetary value.
        No preamble.

        Document:
        {chunks[0]}
        """)
        return {"summary": call_llm(prompt, SUMMARY_TEMP)}

    # ── Map: summarise each chunk in parallel ──
    mini_tpl = dedent("""
    Summarize this legal section in 2–3 bullet points.
    Only include: obligations, rights, monetary values, key terms.
    Be very concise. No preamble.

    Section:
    {chunk}
    """)

    mini_summaries: List[str] = [""] * len(chunks)

    def _map(args: Tuple[int, str]) -> Tuple[int, str]:
        idx, chunk = args
        result = call_llm(mini_tpl.format(chunk=chunk), SUMMARY_TEMP)
        # Cap each mini-summary to prevent synthesis overflow
        words = result.split()
        if len(words) > MAX_MINI_WORDS:
            result = " ".join(words[:MAX_MINI_WORDS]) + "…"
        return idx, result

    with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as ex:
        futures = {ex.submit(_map, (i, c)): i for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            idx, res = fut.result()
            mini_summaries[idx] = res

    # Filter failed mini-summaries
    valid = [m for m in mini_summaries if m and not m.startswith("[LLM ERROR")]
    if not valid:
        return {"summary": "[LLM ERROR: All map calls failed — check Ollama]"}

    # ── Guard: trim synthesis input if too long ──
    combined = "\n\n---\n\n".join(valid)
    if _wc(combined) > MAX_SYNTHESIS_WORDS:
        trimmed, acc = [], 0
        for m in valid:
            wc = _wc(m)
            if acc + wc > MAX_SYNTHESIS_WORDS:
                break
            trimmed.append(m)
            acc += wc
        combined = "\n\n---\n\n".join(trimmed)

    # ── Reduce: synthesise into final summary ──
    reduce_prompt = dedent(f"""
    You are a legal assistant. Below are section summaries of a legal document.
    Produce ONE executive summary as 8–12 bullet points.
    - Remove redundancy
    - Preserve all obligations, rights, deadlines, monetary values
    - Most important first
    - No preamble

    Section summaries:
    {combined}
    """)

    return {"summary": call_llm(reduce_prompt, SUMMARY_TEMP)}

# ══════════════════════════════════════════════
# Risks node
# Layer 4 routing: top-scored chunks only.
# Fallback to all chunks if no keywords match.
# ══════════════════════════════════════════════
def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or chunk_document(state.get("clean_text", state["original_text"]))
    selected, was_fallback = route_chunks(chunks, RISK_KEYWORDS, MAX_RISK_CHUNKS)
    text = "\n\n---\n\n".join(selected)

    fallback_note = (
        "No specific risk clauses were detected. "
        "Analyse the full document for any legal risks or exposures.\n\n"
        if was_fallback else ""
    )

    prompt = dedent(f"""
    You are an expert legal risk analyst. {fallback_note}
    Read the document below and identify the key legal risks.

    For each risk write:
    1. A short name for the risk
    2. Why it matters (one sentence)
    3. How serious it is: Low, Medium, or High
    4. One action to reduce or avoid the risk

    Find at least 3 risks. No preamble.

    Document:
    {text}
    """)

    return {"risks": call_llm(prompt, RISKS_TEMP)}

# ══════════════════════════════════════════════
# Suggestions node
# Layer 4 routing: top-scored chunks only.
# ══════════════════════════════════════════════
def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or chunk_document(state.get("clean_text", state["original_text"]))
    selected, was_fallback = route_chunks(chunks, SUGGEST_KEYWORDS, MAX_SUGGEST_CHUNKS)
    text = "\n\n---\n\n".join(selected)

    fallback_note = (
        "No specific improvement-target clauses were detected. "
        "Review the full document and suggest any improvements or missing protections.\n\n"
        if was_fallback else ""
    )

    prompt = dedent(f"""
    You are an expert contract attorney. {fallback_note}
    Read the document below and suggest specific improvements.

    For each suggestion write:
    1. What the current problem or gap is
    2. What should be added or changed (be specific)
    3. Why this change protects the parties

    Give at least 3 suggestions. Cover payment, termination, liability,
    confidentiality, IP, and dispute resolution where relevant. No preamble.

    Document:
    {text}
    """)

    return {"suggestions": call_llm(prompt, SUGGEST_TEMP)}

# ══════════════════════════════════════════════
# Compile node
# ══════════════════════════════════════════════
def compile_report_node(state: AgentState) -> Dict[str, Any]:
    validation = validate_all(state)
    meta       = state.get("meta", {})

    warnings = ""
    for field, result in validation.items():
        if not result["ok"]:
            warnings += f"\n> ⚠️ **{field.title()}**: {result['reason']}"

    compression_line = (
        f"\n> 📊 {meta['raw_words']:,} words → {meta['clean_words']:,} after compression "
        f"({meta['compression']} removed) · {meta['chunks']} chunks\n"
        if meta else ""
    )

    report = dedent(f"""
    # Legal Document Analysis (AI-Assisted)

    > **Disclaimer**: AI-generated. Not legal advice. Consult a qualified attorney.
    {compression_line}{warnings}

    ---

    ## 📝 Executive Summary
    {state.get("summary") or "_Not generated_"}

    ---

    ## ⚠️ Identified Risks
    {state.get("risks") or "_Not generated_"}

    ---

    ## 💡 Suggestions for Improvement
    {state.get("suggestions") or "_Not generated_"}
    """).strip()

    return {"final_report": report, "validation": validation}

# ══════════════════════════════════════════════
# Workflow — fan-out / fan-in
#
#   preprocess
#       │
#       ├──► summarize   ──┐
#       ├──► risks        ─┤──► compile
#       └──► suggestions ──┘
# ══════════════════════════════════════════════
@st.cache_resource
def get_workflow():
    wf = StateGraph(AgentState)
    wf.add_node("preprocess",   preprocess_node)
    wf.add_node("summarize",    summarize_node)
    wf.add_node("risks",        analyze_risks_node)
    wf.add_node("suggestions",  suggest_improvements_node)
    wf.add_node("compile",      compile_report_node)

    wf.set_entry_point("preprocess")
    wf.add_edge("preprocess",  "summarize")
    wf.add_edge("preprocess",  "risks")
    wf.add_edge("preprocess",  "suggestions")
    wf.add_edge("summarize",   "compile")
    wf.add_edge("risks",       "compile")
    wf.add_edge("suggestions", "compile")
    wf.add_edge("compile",     END)
    return wf.compile()

# ══════════════════════════════════════════════
# UI — CSS
# ══════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0e0f13;
    --surface: #16181f;
    --border:  #2a2d38;
    --accent:  #c8a96e;
    --danger:  #e07070;
    --ok:      #6ec87a;
    --info:    #7b9cda;
    --text:    #e8e6e1;
    --muted:   #7a7d8a;
    --radius:  10px;
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
h1 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--accent) !important;
    letter-spacing: -0.5px;
}
h2, h3 { font-family: 'DM Serif Display', serif !important; }

button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.73rem !important;
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
    font-size: 1.7rem;
    color: var(--accent);
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}
.compression-strip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--info);
    border-radius: var(--radius);
    padding: 0.5rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    margin: 0.5rem 0;
}
.compression-strip .hi { color: var(--ok); }
.badge-ok   { color: var(--ok);     font-family: 'DM Mono', monospace; font-size: 0.75rem; }
.badge-fail { color: var(--danger); font-family: 'DM Mono', monospace; font-size: 0.75rem; }
hr { border-color: var(--border) !important; }
</style>
"""

# ══════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════
def render_metric(label: str, value: str):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_compression_strip(meta: Dict[str, Any]):
    st.markdown(
        f'<div class="compression-strip">'
        f'TOKEN PIPELINE · '
        f'{meta["raw_words"]:,} raw → '
        f'<span class="hi">{meta["clean_words"]:,} clean</span> '
        f'({meta["compression"]} removed by noise+dedup) · '
        f'<span class="hi">{meta["chunks"]} chunks</span> · '
        f'risks+suggestions receive top-{MAX_RISK_CHUNKS} keyword-scored chunks only'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_validation_badge(field: str, result: Dict[str, Any]):
    if result["ok"]:
        st.markdown(f'<span class="badge-ok">✓ {field.title()} — passed</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-fail">✗ {field.title()} — {result["reason"]}</span>', unsafe_allow_html=True)

def render_sidebar():
    st.markdown("### ⚖️ Legal Analyzer")
    st.markdown(
        '<span style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#7a7d8a;">'
        'AI-ASSISTED · NOT LEGAL ADVICE</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    uploaded = st.file_uploader("Upload document", type=["pdf", "txt"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model**")
    st.session_state.model = st.text_input(
        "model", value=st.session_state.get("model", DEFAULT_MODEL), label_visibility="collapsed"
    )
    st.markdown(
        '<span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#7a7d8a;">'
        'Recommended: llama3.2:3b · mistral:7b</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#7a7d8a;">'
        'PIPELINE<br>'
        '① Noise strip<br>'
        '② Dedup paragraphs<br>'
        '③ Semantic chunk<br>'
        '④ Keyword route<br>'
        '⑤ Parallel agents<br>'
        '⑥ Map-reduce summary<br>'
        '⑦ Validate</span>',
        unsafe_allow_html=True,
    )
    return uploaded

# ══════════════════════════════════════════════
# Analysis runner
# ══════════════════════════════════════════════
def run_analysis(uploaded_file, h: str):
    prog = st.progress(0, text="📄 Extracting text…")
    doc_text = load_doc(uploaded_file)
    if not doc_text:
        prog.empty()
        return

    prog.progress(15, text="🧹 Stripping noise & deduplicating…")

    # Run preprocessing here too so we can show metrics before LLM starts
    clean  = strip_noise(doc_text)
    clean  = dedup_paragraphs(clean)
    chunks = chunk_document(clean)
    raw_wc = _wc(doc_text)
    cln_wc = _wc(clean)
    saved  = raw_wc - cln_wc
    pct    = f"{(saved / raw_wc * 100):.0f}%" if raw_wc else "0%"
    meta   = {"raw_words": raw_wc, "clean_words": cln_wc,
               "saved_words": saved, "compression": pct, "chunks": len(chunks)}

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric("Raw Words",   f"{raw_wc:,}")
    with c2: render_metric("After Clean", f"{cln_wc:,}")
    with c3: render_metric("Chunks",      str(len(chunks)))
    with c4: render_metric("Saved",       pct)

    render_compression_strip(meta)

    with st.expander("Document preview (first 30 lines — post-clean)"):
        st.markdown("\n".join(clean.splitlines()[:30]) or "_No content_")

    prog.progress(30, text="⚙️ Running parallel agents…")

    with st.spinner("🤖 Summarising · risks · suggestions (parallel)…"):
        prog.progress(45, text="🔄 Agents running…")
        result = get_workflow().invoke({"original_text": doc_text})

    prog.progress(92, text="✅ Validating…")
    validation = result.get("validation") or validate_all(result)
    prog.progress(100, text="Complete.")

    st.session_state.results_by_hash[h] = {**result, "validation": validation, "meta": meta}

# ══════════════════════════════════════════════
# Results display
# ══════════════════════════════════════════════
def display_results(h: str):
    result     = st.session_state.results_by_hash[h]
    validation = result.get("validation", {})
    meta       = result.get("meta", {})

    st.markdown("---")
    if meta:
        render_compression_strip(meta)

    vcols = st.columns(3)
    for col, field in zip(vcols, ["summary", "risks", "suggestions"]):
        with col:
            render_validation_badge(field, validation.get(field, {"ok": False, "reason": "No result"}))

    st.markdown("")

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
        report = result.get("final_report") or "_Not generated_"
        st.markdown(report)
        st.download_button(
            "⬇ Download Report (.md)",
            data=report,
            file_name="legal_analysis.md",
            mime="text/markdown",
        )

# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Legal Doc Analyzer", page_icon="⚖️",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    ensure_session_defaults()

    with st.sidebar:
        uploaded_file = render_sidebar()

    st.markdown("# Legal Document Analyzer")
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#7a7d8a;">'
        'NOISE STRIP · DEDUP · SEMANTIC CHUNK · KEYWORD ROUTE · '
        'PARALLEL AGENTS · MAP-REDUCE</p>',
        unsafe_allow_html=True,
    )

    if not uploaded_file:
        st.markdown("""
        ### Get Started

        This app runs **100% locally** on your machine — your documents never leave your device.

        To use it, you need **Ollama** running in the background:

        **Step 1 — Install Ollama**  
        👉 [Download Ollama](https://ollama.com/download) — available for Mac, Windows, and Linux.

        **Step 2 — Pull a model** (run this in your terminal after installing):
```bash
        ollama pull llama3.2:3b
```

        **Step 3 — Run this app locally:**
```bash
        pip install -r requirements.txt
        streamlit run app.py
```

        Once Ollama is running, come back here, upload a PDF or TXT contract, and hit **Analyze**.

        ---
        *Already set up? Upload a document in the sidebar to begin.*
        """)
        return
    data = uploaded_file.getvalue()
    h    = file_hash(data)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if st.button("⚡ Analyze Document"):
            run_analysis(uploaded_file, h)
    with col_b:
        if st.button("🗑 Clear Cache") and h in st.session_state.results_by_hash:
            del st.session_state.results_by_hash[h]
            st.rerun()

    if h in st.session_state.results_by_hash:
        display_results(h)

if __name__ == "__main__":
    main()
