# ML Venue Insights

Analyze what top ML conference papers are *about* — themes, conceptual structure, and trends over time — **without downloading a single PDF**. Scrapes only titles and abstracts, embeds them once, and mines that embedding space for a thematic map, cluster relationships, conceptual trends, and semantic search.

Covers **five venues**. Each is scraped and analyzed **independently — one JSON file per conference** — and can also be compared against the others:

| Venue | Source | Backend | Coverage | Speed |
|---|---|---|---|---|
| **NeurIPS** | papers.nips.cc | `nips` (HTML) | 1987–2024 | slow |
| **ICML** | proceedings.mlr.press | `pmlr` (HTML) | 2015–2025 | slow |
| **ICLR** | OpenReview API | `openreview` | 2018–2026 | **fast** |
| **RLC** | OpenReview API | `openreview` | 2024–2026 | **fast** |
| **COLT** | proceedings.mlr.press | `pmlr` (HTML) | 2015–2025 | slow |

The headline stage is **`--analyze`**: it clusters papers in embedding space, picks the papers that *anchor* each cluster (closest to the centroid), builds a graph of which themes sit near each other, and tracks each theme's share over time. The LLM stage then **names** each theme by reading its representative paper titles — not keyword bags — and writes how the themes relate plus a field narrative.

With more than one venue selected, `--analyze` simply runs once per conference and writes a separate `<venue>_analysis.json` for each. Two extra stages then compare venues against each other: **`--compare`** (what vocabulary distinguishes each venue) and **`--theme-mix`** (which venues own which themes, and where each theme appeared first). These build a pooled view on demand without disturbing the per-venue files.

## Why embeddings, not keywords

Keyword counts tell you `training` appears 17 times. They don't tell you that thirty papers are really *the same idea* phrased thirty ways. Embeddings group papers by **meaning**, so paraphrases collapse together, themes emerge that no single keyword names, and "papers like this one" becomes a one-line query.

## Install

```bash
git clone https://github.com/yourname/ml-venue-insights
cd ml-venue-insights

pip install -e .              # core: scrape + stats + cross-venue compare
pip install -e ".[analyze]"   # DEEP analysis: embeddings, clusters, search  <- recommended
pip install -e ".[all]"       # everything incl. legacy BERTopic path
```

Then **verify the install before you spend hours scraping**:

```bash
python selftest.py       # or: make check
```

This runs the whole pipeline on synthetic data with a stubbed embedder — no network, no model download. If it passes, your install is good.

For `--llm`, install [Ollama](https://ollama.com): `ollama pull llama3.1`.

## Quick start

```bash
# ICLR is an API, so this takes seconds — good first real run
ml-venue-insights --venues iclr --scrape --start 2024 --end 2024 --limit-per-year 50
ml-venue-insights --venues iclr --stats
```

## Pipeline

```
PER VENUE (runs once per selected conference, writes that venue's own files):

  --scrape ──► <venue>_corpus.jsonl ──► --analyze ──► <venue>_analysis.json ──► --llm
   (titles+abstracts,                   (embed once, cache to disk:              (names themes from
    streaming, resumable)                clusters, anchors, graph, trends)        representative papers
                                                    │                             + narrative)
                                                    └──► --briefs   (per-theme technical briefs)
            --search "free text query"   (semantic retrieval)
            --like   "a paper title"     (nearest neighbors)
            --stats                       (shallow keyword view, no ML)

CROSS VENUE (2+ venues; builds a pooled view, leaves per-venue json untouched):

  <venue>_corpus.jsonl (×N) ──► <venues>_pooled_corpus.jsonl ──► --compare
                                                              └─► --theme-mix (venue ownership + diffusion)
```

Embeddings are computed once and cached (`data/emb_*.npy`), keyed on model + corpus size, so every analysis after the first is fast.

## Usage

Every stage takes `--venues`: either `all`, one venue, or a comma-separated list.

- **Per-venue stages** (`--scrape`, `--stats`, `--analyze`, `--llm`, `--briefs`, `--search`, `--like`) run **once per selected conference** and write that venue's own artifacts. `--venues all --analyze` produces `neurips_analysis.json`, `icml_analysis.json`, `iclr_analysis.json`, `rlc_analysis.json` and `colt_analysis.json` — five separate files.
- **Cross-venue stages** (`--compare`, `--theme-mix`) compare venues against each other, so they build a pooled corpus (`<venues>_pooled_corpus.jsonl`) on demand. They never overwrite the per-venue files.

```bash
# --- single venue: exactly like before ---
ml-venue-insights --venues icml --scrape --start 2020 --end 2025
ml-venue-insights --venues icml --analyze --llm

# --- the fast venues (APIs, seconds not hours) ---
ml-venue-insights --venues iclr,rlc --scrape --start 2020 --end 2025

# --- all five venues -> one separate json per conference ---
ml-venue-insights --venues all --scrape --start 2020 --end 2025   # overnight job
ml-venue-insights --venues all --analyze                          # writes 5 *_analysis.json
ml-venue-insights --venues all --analyze --llm                    # + names themes per venue

# --- cross-venue reports (pooled view; per-venue json untouched) ---
ml-venue-insights --venues all --compare                          # no ML deps needed
ml-venue-insights --venues all --theme-mix

# --- targeted pair ---
ml-venue-insights --venues icml,colt --compare --analyze --theme-mix

# Semantic search — find papers by meaning, not keywords
ml-venue-insights --venues all --search "diffusion models for video generation"

# "Papers like this one"
ml-venue-insights --venues all --like "Denoising Diffusion Probabilistic Models"

# Per-theme technical briefs (Methods / Novelty / Use Cases), for experts
ml-venue-insights --venues all --analyze --llm --briefs
ml-venue-insights --venues all --briefs --brief-top 10

# Force a bigger embedding model (GPU recommended)
ml-venue-insights --venues all --analyze --model-name best

# List what's available
ml-venue-insights --list-venues
```

### Makefile shortcuts

```bash
make check          # offline self-test — run this first
make smoke          # tiny real run (~50 papers, needs network)
make scrape-fast    # ICLR + RLC only (APIs, quick)
make analyze VENUES=icml,colt START=2021 END=2024
```

### Embedding models (`--model-name`)

| Value | Model | Notes |
|-------|-------|-------|
| `auto` (default) | bge-large if a GPU is visible, else bge-small | sensible default |
| `fast` | `BAAI/bge-small-en-v1.5` (384d) | CPU-friendly, strong |
| `best` | `BAAI/bge-large-en-v1.5` (1024d) | GPU recommended |
| any HF id | e.g. `sentence-transformers/all-mpnet-base-v2` | passed through |

### Flags

| Flag | Purpose |
|------|---------|
| `--venues` | `all`, one venue, or comma list (`neurips,icml,iclr,rlc,colt`) |
| `--list-venues` | Print the venue registry and exit |
| `--scrape` | Fetch titles + abstracts (resumable, per venue) |
| `--stats` | Shallow keyword frequencies + trends (no ML) |
| `--compare` | **Multi-venue**: volume table + distinctive vocabulary per venue |
| `--analyze` | **Deep, per venue**: embed → cluster → anchors → graph → trends → `<venue>_analysis.json` |
| `--theme-mix` | **Multi-venue**: which venues own which themes + diffusion order |
| `--llm` | Ollama names themes from representative papers, + relationships + narrative |
| `--briefs` | Per-theme **Methods / Novelty / Use Cases** briefs; writes `*_theme_briefs.md` |
| `--search QUERY` | Semantic search over the corpus |
| `--like TITLE` | Papers most similar to one matching `TITLE` |
| `--topics` | Legacy BERTopic/KMeans path |
| `--model-name` | Embedding model (`auto`/`fast`/`best`/HF id) |
| `--n-topics` | Target cluster count (`0` = auto) |
| `--reps` | Representative papers per cluster |
| `--top-k` | Results for `--search` / `--like` |
| `--limit-per-year` | Cap papers/year (for quick tests) |
| `--force-embed` | Ignore the embedding cache |
| `--model` | Ollama model name |

Each conference owns a separate set of artifacts (`icml_analysis.json`, `icml_corpus.jsonl`, `icml_theme_briefs.md`, ...), so venues never clobber each other and you can re-run or delete any single venue in isolation. The cross-venue reports write their own clearly-named pooled files (e.g. `colt_iclr_pooled_corpus.jsonl`).

## What `--analyze` gives you

The output is structured into **five deliverables**, written to `<venue>_analysis.json` (one per conference) and printed as a report:

1. **Core themes** — every cluster with size, share, top terms, anchor papers, and a trend arrow.
2. **Evolution of topics** — each theme's early→late share with a growth ratio (e.g. LLM reasoning `3.3% → 23.5%, ×7.0 rising`). Trends use per-year **share**, not raw counts, so conference growth doesn't make everything look like it's rising.
3. **Connected areas** — theme pairs linked by centroid similarity *plus evidence*: their shared vocabulary and the **bridge papers** that sit between them (high similarity to both centroids — often the papers importing a method across areas).
4. **Emerging trends** — an emergence score (recent share × growth acceleration) ranking what's genuinely heating up, not just what's biggest.
5. **Research niches & gaps** — a niche score (small + internally diverse + intermittent = unconsolidated opportunity), shown against a **saturated** list (large + flattening = mature/crowded).

Trends work even with **as few as two years** in the corpus — the trend tag falls back to an early-vs-late share ratio when there aren't enough years for a regression slope, so themes are never left untagged.

## What multi-venue adds

**`--compare`** — per-venue volume table, plus each venue's distinctive vocabulary scored by **weighted log-odds with an informative Dirichlet prior** (Monroe et al. 2008). Raw frequency ratios are dominated by rare terms — a word appearing 3× at COLT and 0× elsewhere looks infinitely distinctive. This estimator shrinks exactly those cases, so small venues like RLC and COLT get honest, lower-confidence scores instead of spurious ones. **Needs no ML dependencies.**

**`--theme-mix`** — for each pooled cluster, the share of papers from each venue. High concentration = venue specialty; even split = field-wide topic. Computed from *all* papers in a cluster during `--analyze`, not just representatives.

**Theme diffusion** — per-venue **onset year** for each theme: the year a venue's output becomes non-trivial (10th percentile, not first-ever paper, so one outlier can't set the clock). Venues holding <5% of a theme are excluded. A consistent lead suggests where an idea originated and how it spread across the field.

## Two correctness details worth knowing

**Venue-token scrubbing.** `corpus.doc_text()` strips venue names and abbreviations before embedding. Without this, the clusterer recovers the **venue** instead of the **research topic** — every theme comes back 100% single-venue, which looks like a strong finding and is really the model reading the letterhead. `--theme-mix` prints a warning if ≥80% of themes are single-venue, since that's the signature of leakage.

**Accepted papers only.** OpenReview keeps rejected, withdrawn, and desk-rejected submissions publicly queryable. The backend filters on `content.venueid`, which only accepted papers retain. For ICLR — where rejections outnumber acceptances — including them would distort every trend.

### Anti-hallucination LLM synthesis

`--llm` runs in **two passes**. Pass 1 names each cluster *strictly from its own representative paper titles* (strict-JSON reply, low temperature, batched in small groups with a deterministic term-based fallback so no theme is ever left unnamed). Pass 2 writes the five-section report over those **names plus the pre-computed numbers and relationships**, with an explicit instruction not to introduce any theme or statistic not in the evidence.

### Per-theme technical briefs

`--briefs` is a separate, depth-first mode for an expert audience. Where `--llm` reasons across all themes at once (for relationships and rankings), `--briefs` reasons about **one theme per call**, grounded in that theme's representative papers enriched with their **abstracts** (pulled from the corpus — titles alone underdetermine methodology). Each brief has three sections: **Methods** (the shared technical pattern), **Novelty** (the specific methodological shift), and **Use Cases** (immediate vs emerging applications, or downstream relevance if the cluster is theoretical). Generating one theme at a time is the structural guarantee against cross-cluster drift: the model only ever sees the papers it's asked to describe, and is told to cite each claim by paper number. Reuses the names from a prior `--llm` pass when available.

## Project structure

```
ml_venue_insights/
├── cli.py                # argparse entry, wires all stages, venue routing
├── venues.py             # NEW: venue registry (year → PMLR volume / OpenReview id)
├── config.py             # URLs, paths, Ollama settings
├── corpus.py             # streaming JSONL I/O + venue-token scrubbing
├── scrape.py             # NeurIPS title+abstract scraper (no PDFs)
├── scrape_pmlr.py        # NEW: ICML + COLT (proceedings.mlr.press)
├── scrape_openreview.py  # NEW: ICLR + RLC (OpenReview API v2, v1 fallback)
├── scrape_all.py         # NEW: backend dispatcher, resumable per venue
├── compare.py            # NEW: pooling, venue signatures, theme mix, diffusion
├── embeddings.py         # model auto-select + disk-cached vectors
├── analyze.py            # clustering + temporal analysis + search + venue mix
├── temporal.py           # trends, emergence/saturation/niche scoring, relationships
├── llm_deep.py           # two-pass evidence-grounded naming + 5-section report
├── briefs.py             # per-theme Methods/Novelty/Use-Cases briefs
├── stats.py              # shallow keyword view
├── topics.py             # legacy BERTopic/KMeans
└── llm.py                # legacy keyword-based synthesis
selftest.py               # NEW: offline install verification
```

## Design notes

- **Backends, not rewrites.** Everything downstream of the corpus only sees `title`/`abstract`/`year`, so adding four venues meant adding *fetch strategies*. `venues.py` is the registry; `scrape_all.py` dispatches.
- **Embed once, analyze many.** Vectors cache to disk keyed on (model, corpus size). Re-running `--analyze`, `--search`, `--like` reuses them.
- **Anchors over keywords.** Clusters are labeled by the papers nearest their centroid, so the LLM reasons over real titles, not term-frequency noise.
- **bge with the right prompts.** Documents and queries get bge's recommended instruction prefixes for retrieval-quality embeddings.
- **Bounded memory.** Only the `n_docs × dim` float32 matrix is held; the corpus itself streams from disk.

## Maintenance

PMLR assigns a new volume number per year in **publication order across all PMLR proceedings**, so the numbers jump around with no derivable pattern (ICML 2024 = v235, 2025 = v267). Adding next year's ICML/COLT means adding one line to the `years` dict in `venues.py` after checking [proceedings.mlr.press](https://proceedings.mlr.press/). ICLR/RLC follow a predictable venue-id pattern and are pre-populated through 2026.

## Practical notes

- Scraping is **resumable per venue** via append-only key logs — interrupt freely and re-run.
- HTML backends sleep 1.5–3.5s between papers. A full NeurIPS+ICML+COLT scrape is an **overnight job**; ICLR and RLC take seconds. Start with `--limit-per-year 25`.
- RLC has only 2–3 editions, so its solo time series is thin — it's far more informative pooled.
- Set a contact email in `config.HEADERS` before large scrapes.

## License

MIT
