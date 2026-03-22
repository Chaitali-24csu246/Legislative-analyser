# Legislative-analyser
HPE Gen AI for GenZ Project
#copy.py  structure
"""
Legal Document Analyzer
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
