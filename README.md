---
title: Legislative Analyser
emoji: ⚖️
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
---
# Legal Document Analyzer
## Special focus on token compression. USE APP2.py

> **AI-assisted legal document review. Please note that this is not a substitute for qualified legal advice**

A local, legal document analyzer built with **Streamlit**, **LangGraph**, and **Ollama**. Upload a contract or legal PDF and get an executive summary, flagged risks, and improvement suggestions.
Everything is processed on your own machine, no data is sent to external APIs.
Processing might take some time so please be patient.
 ## Use this url to go on huggingface deployed app
 https://huggingface.co/spaces/Chaitali-24csu246/Legislative-Analyser
 Please note that file uploads on huggingface are not always reliable, and if it shows file upload error 403, please manually setup as given   below

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Token Compression Pipeline](#token-compression-pipeline)
- [Agent Graph](#agent-graph)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Usage](#usage)
- [Deployment](#deployment)
- [Technical Reference](#technical-reference)
- [Limitations](#limitations)

---

## Features

| Feature | Detail |
|---|---|
| **PDF & TXT support** | Extracts Markdown-quality text via PyMuPDF4LLM |
| **4-layer compression** | Noise strip → dedup → semantic chunk → keyword route |
| **Map-reduce summary** | Every chunk is read; nothing is dropped for the summary |
| **Parallel agents** | Summarize, Risks, and Suggestions run concurrently |
| **Keyword routing** | Only the most relevant chunks reach each analysis node |
| **Output validation** | Signal-based checks catch off-topic or empty LLM responses |
| **Result caching** | SHA-256 file hash prevents re-running the same document |
| **Local-only** | All inference via Ollama — no data leaves your machine |
| **Download report** | Full Markdown report downloadable in one click |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        Streamlit UI                      │
│   Sidebar (upload · model select)  │  Main (tabs · DL)  │
└───────────────────┬─────────────────────────────────────┘
                    │ PDF / TXT bytes
                    ▼
┌─────────────────────────────────────────────────────────┐
│                   load_doc()                             │
│   TXT → decode   │   PDF → pymupdf4llm → Markdown        │
└───────────────────┬─────────────────────────────────────┘
                    │ raw text
                    ▼
┌─────────────────────────────────────────────────────────┐
│               preprocess_node  (Layer 1-3)               │
│   strip_noise → dedup_paragraphs → chunk_document        │
└──────┬────────────────────────────────────────┬─────────┘
       │ chunks                                 │ meta
       ▼                                        ▼
┌──────────────────────────────────┐   ┌───────────────┐
│  route_chunks (Layer 4)          │   │  Metrics row  │
│  RISK_KEYWORDS / SUGGEST_KEYWORDS│   │  (UI only)    │
└──────┬──────────────┬────────────┘   └───────────────┘
       │              │
  ┌────▼────┐   ┌─────▼──────┐   ┌──────────────┐
  │summarize│   │   risks    │   │  suggestions │
  │  node   │   │   node     │   │    node      │
  │(map-red)│   │(top-5 chks)│   │(top-5 chks)  │
  └────┬────┘   └─────┬──────┘   └──────┬───────┘
       └──────────────┴─────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ compile_report │
              │    _node       │
              │ + validate_all │
              └────────┬───────┘
                       │
                       ▼
              final_report (MD)
```

---

## Token Compression Pipeline

The pipeline is designed to squeeze as many tokens as possible out of a document *before* it reaches the LLM — preserving legal meaning while cutting noise.

### Layer 1 — Noise Strip (`strip_noise`)

Removes lines that carry zero legal meaning:

- Horizontal rules (`---`, `═══`)
- Page numbers (`Page 3 of 14`, lone digits)
- Boilerplate markers (`[FORM]`, `[SIGNATURE PAGE]`)
- Signature underscores (`___________`)
- Lone exhibit/schedule headers
- Collapses 3+ blank lines to 2, strips zero-width spaces

**Typical saving: 0–10%**

### Layer 2 — Paragraph Deduplication (`dedup_paragraphs`)

Legal contracts repeat definitions, notice clauses, and recital boilerplate. This layer:

- Splits on double newlines
- Normalises each paragraph (lowercase, collapsed whitespace) as a lookup key
- Keeps only the first occurrence of any paragraph ≥ 40 chars
- Always keeps short paragraphs (headings, labels < 40 chars)

**Typical saving: 5–15% on long contracts**

### Layer 3 — Semantic Chunking (`chunk_document`)

Splits the document into LLM-sized pieces using a priority-ordered pattern list:

| Priority | Pattern | Example |
|---|---|---|
| 1 | Formal legal headers | `ARTICLE IV`, `SECTION 2.1`, `CLAUSE 3` |
| 2 | Numbered headings | `1. Payment Terms`, `2.3 Termination` |
| 3 | Markdown headings | `## Section`, `# Title` |
| 4 | ALL CAPS lines (3+ words) | `CONFIDENTIALITY OBLIGATIONS` |
| 5 | Paragraph breaks | Universal fallback |

Each section is then enforced to `CHUNK_WORD_LIMIT = 1,200` words (~1,600 tokens), leaving headroom for the prompt template inside llama3.2's 4k context. Orphan chunks under 50 words are merged with their neighbour.

### Layer 4 — Keyword Routing (`route_chunks`)

Instead of sending all chunks to every LLM node, each node receives only the **top-N highest-scoring** chunks for its task:

- **Risks node**: scored against `RISK_KEYWORDS` (liability, indemnif, terminat, breach…) → top 5
- **Suggestions node**: scored against `SUGGEST_KEYWORDS` (payment, IP, confidential, arbitrat…) → top 5
- **Summary node**: receives **all chunks** via map-reduce (accuracy priority)
- **Fallback**: if no keywords match (generic/non-standard docs), falls back to first N chunks

---

## Agent Graph

Built with **LangGraph** `StateGraph`. The graph is compiled once and cached via `@st.cache_resource`.

```
preprocess
    │
    ├──► summarize   ──┐
    ├──► risks        ─┤──► compile ──► END
    └──► suggestions ──┘
```

`summarize`, `risks`, and `suggestions` run **in parallel** (LangGraph fan-out). The `compile` node waits for all three before assembling the final report and running validation.

### Map-Reduce Summary

For multi-chunk documents:

1. **Map phase**: each chunk is summarised to ≤ `MAX_MINI_WORDS = 100` words via `ThreadPoolExecutor` (up to 6 workers)
2. **Guard**: combined mini-summaries are trimmed to `MAX_SYNTHESIS_WORDS = 2,000` words if needed
3. **Reduce phase**: one LLM call synthesises all mini-summaries into an 8–12 bullet executive summary

Single-chunk documents skip the map phase entirely.

### Output Validation

Each LLM output is checked for:

- Non-empty, non-error response
- Minimum 20 words
- Presence of domain-appropriate signal words (e.g. risks output must contain at least 2 of: `risk`, `concern`, `issue`, `breach`, `high/medium/low`…)

Failed validations are shown as warning badges in the UI without blocking the report.

---

## Setup Instructions

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 – 3.12 | 3.11 recommended |
| [Ollama](https://ollama.com) | Latest | Must be running locally |
| pip | ≥ 23 | `pip install --upgrade pip` |

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
irm https://ollama.com/install.ps1 | iex

```

### 2. Pull a model

```bash
# Recommended — fast, fits in 4 GB RAM
ollama pull llama3.2:3b

# Alternatively — better quality, needs ~8 GB RAM
ollama pull mistral:7b
```

Verify Ollama is running:

```bash
ollama list        # should show your pulled model
curl http://localhost:11434   # should return "Ollama is running"
```

### 3. Clone / download the project

```bash
# Navigate to where you want the project first
cd path/to/your/folder        # e.g. cd Documents  or  cd D:\Projects

# Then clone — it will create the folder there
git clone https://github.com/Chaitali-24csu246/Legislative-analyser
cd Legislative-analyser
```

### 4. Create a virtual environment

```bash
python -m venv .venv

# Activate
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell
```

### 5. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Configure environment (optional)

Create a `.env` file in the project root if your Ollama instance is not on the default port:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

The default is `http://localhost:11434` — you only need this file if you've changed the port or are running Ollama on a remote host.

### 7. Run the app

```bash
streamlit run app2.py
```

Open **http://localhost:8501** in your browser.

---

## Configuration

All tuneable constants are at the top of `app.py`:

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (overridable via `.env`) |
| `CHUNK_WORD_LIMIT` | `1200` | Max words per chunk (~1,600 tokens) |
| `MAX_RISK_CHUNKS` | `5` | Top-N chunks sent to the risks node |
| `MAX_SUGGEST_CHUNKS` | `5` | Top-N chunks sent to the suggestions node |
| `MAX_SYNTHESIS_WORDS` | `2000` | Max combined mini-summary words before reduce |
| `MAX_MINI_WORDS` | `100` | Max words per mini-summary (map phase) |
| `SUMMARY_TEMP` | `0.2` | LLM temperature for summary (factual) |
| `RISKS_TEMP` | `0.15` | LLM temperature for risks (conservative) |
| `SUGGEST_TEMP` | `0.3` | LLM temperature for suggestions (creative) |

---

## Usage

1. Open the app at `http://localhost:8501`
2. Upload a **PDF or TXT** legal document using the sidebar uploader
3. Optionally change the model name in the sidebar (must be already pulled in Ollama)
4. Click **⚡ Analyze Document**
5. View results across four tabs: **Summary · Risks · Suggestions · Full Report**
6. Click **⬇ Download Report (.md)** to save the Markdown report

Results are cached by file hash — re-uploading the same document skips re-analysis. Use **Clear Cache** to force a re-run.

---

## Deployment

Deployed on HuggingFace as model was local. Use this link: https://huggingface.co/spaces/Chaitali-24csu246/Legislative-Analyser
Please note that this is not always reliable, and you may face file upload issues (currently unable to resolve) , and if any issue persists, use manual setup on device.
## Deployment

### Live Demo — HuggingFace Spaces

The app is deployed and publicly accessible at:

** [huggingface.co/spaces/Chaitali-24csu246/Legislative-Analyser](https://huggingface.co/spaces/Chaitali-24csu246/Legislative-Analyser)**

No downloads or setup required — just open the link and upload a document.

#### How it works

Ollama and Streamlit run together inside a Docker container on HuggingFace's servers. Users interact with the Streamlit UI; all LLM inference happens inside the container.

```
User's browser
      │
      ▼
HuggingFace Space (Docker container)
      ├── Ollama     — runs on localhost:11434 inside the container
      └── Streamlit  — serves the UI on port 7860
```

#### Deployment files

| File | Purpose |
|---|---|
| `Dockerfile` | Installs Python, curl, zstd, and Ollama |
| `start.sh` | Starts Ollama, pulls the model, launches Streamlit |
| `requirements.txt` | Python dependencies installed during Docker build |
| `app2.py` | Main Streamlit application |

#### Notes
- First boot takes **5–10 minutes** to download the model (~2GB)
- Free tier may **sleep after inactivity** — first visitor after sleep waits ~30s for wake-up
- Hardware: 2 vCPU · 16GB RAM (free tier) — sufficient for `llama3.2:3b`

## Technical Reference

### Key Functions

| Function | Module location | Purpose |
|---|---|---|
| `strip_noise(text)` | preprocessing | Layer 1 — regex-based artifact removal |
| `dedup_paragraphs(text)` | preprocessing | Layer 2 — exact-duplicate paragraph removal |
| `chunk_document(text)` | preprocessing | Layer 3 — semantic + hard-window chunking |
| `route_chunks(chunks, keywords, max_n)` | routing | Layer 4 — keyword-scored chunk selection |
| `preprocess_node(state)` | LangGraph nodes | Runs layers 1–3, populates `chunks` + `meta` |
| `summarize_node(state)` | LangGraph nodes | Map-reduce summary over all chunks |
| `analyze_risks_node(state)` | LangGraph nodes | Risk analysis on top-N routed chunks |
| `suggest_improvements_node(state)` | LangGraph nodes | Suggestions on top-N routed chunks |
| `compile_report_node(state)` | LangGraph nodes | Assembles final report + runs validation |
| `validate_output(field, content)` | validation | Signal-based quality check per field |
| `get_workflow()` | graph | Builds + compiles the LangGraph StateGraph |

### State Schema (`AgentState`)

```python
class AgentState(TypedDict, total=False):
    original_text : str            # raw extracted text
    clean_text    : str            # after noise strip + dedup
    chunks        : List[str]      # semantic chunks
    summary       : str            # executive summary
    risks         : str            # identified risks
    suggestions   : str            # improvement suggestions
    final_report  : str            # compiled Markdown report
    validation    : Dict[str, Any] # per-field validation results
    meta          : Dict[str, Any] # compression metrics
```

### Supported File Types

| Type | Handler | Notes |
|---|---|---|
| `.pdf` | `pymupdf4llm.to_markdown()` | Extracts structured Markdown; fails gracefully on scanned/image PDFs |
| `.txt` | `data.decode("utf-8")` | UTF-8 with error replacement |

Scanned PDFs (image-only, no text layer) will return an error — convert them with OCR first (e.g. Adobe Acrobat, `ocrmypdf`).

---

## Limitations

- **Not legal advice** — outputs are AI-generated and may miss critical clauses or misinterpret legal language. Always have a qualified attorney review important documents.
- **Scanned PDFs** — image-only PDFs cannot be processed without an OCR step.
- **Context window** — very long documents (100k+ words) may hit chunk limits even after compression. Reduce `CHUNK_WORD_LIMIT` or switch to a model with a larger context window.
- **Model quality** — `llama3.2:3b` is fast but may paraphrase heavily or miss nuance. `mistral:7b` or `llama3:8b` give noticeably better results at the cost of speed.
- **Language** — optimised for English-language legal documents.
- **Privacy** — while inference is local, ensure your Ollama instance is not publicly exposed if processing sensitive documents.

---

## Project Structure

```
legal-analyzer/
├── app2.py               # Main application (all logic + UI)
├── requirements.txt     # Python dependencies
├── .env                 # Optional — override OLLAMA_BASE_URL
├── README.md            # This file
└── .gitignore           # Recommended: exclude .env, .venv/, __pycache__/
```

---

#

*Built for the HPE Gen AI for GenZ project.*
