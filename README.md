# Lumen — Book Deconstruction Engine

Convert PDF/EPUB/Markdown into structured Obsidian notes and Mermaid mind maps, powered by LLM-guided multi-pass analysis.

## Quick Start

```bash
pip install -e ".[dev]"
lumen init              # set up API keys and vault path
lumen run book.epub     # full pipeline: ingest → skeletonize → deep-read → digest
lumen sync              # sync completed books to Obsidian vault
lumen status            # dashboard of all books in progress
```

## Pipeline

```
Input → Phase 1: Ingest (parse + classify + chunk)
      → Phase 2: Skeleton (3-pass LLM extraction + GATE review)
      → Phase 3: Deep Read (parallel chunk analysis + synthesis)
      → Phase 4: Digest (Obsidian notes + Mermaid mind maps)
```

## Supported Formats

| Format | Parser | Notes |
|--------|--------|-------|
| PDF | PyMuPDF | text-layer only; OCR not supported |
| EPUB | EbookLib + BeautifulSoup | auto-extracts TOC structure |
| Markdown | built-in section splitter | heading-based chapter detection |

## LLM Backends

- **DeepSeek** (default, full feature set)
- **Claude** via Anthropic Messages API

## Configuration

API keys go in `.env` (not `config.yaml`):

```
DEEPSEEK_API_KEY=sk-your-key
```

Run `lumen init` to generate both files interactively.

## Content Types

Auto-detected and analyzed with type-specific frameworks:

| Type | Framework | Key Fields |
|------|-----------|-----------|
| Book | `book.yaml` | claims, relationships, prerequisites |
| Podcast | `podcast.yaml` | core_argument, tension_point, timestamp |
| Article | `article.yaml` | thesis, methodology, findings |
| Reference | `reference.yaml` | definitions, procedures, dependencies |

## Output

- **Obsidian notes** — main book note + per-concept `[[wikilinks]]`
- **Mind maps** — embedded Mermaid `flowchart LR` diagrams
- **Checkpoints** — resume from any phase via `.checkpoint.json`

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check lumen/
mypy lumen/
```

## Architecture

```
lumen/
├── core/           pipeline orchestration, chunking, classification
├── backends/       DeepSeek + Claude API adapters
├── parsers/        PDF / EPUB / MD format parsers
├── renderers/      Obsidian + Mermaid output renderers
└── frameworks/     YAML analysis templates per content type
```
