"""Shared configuration and constants."""

from __future__ import annotations

# NeurIPS own proceedings site (used by the "nips" backend only)
BASE = "https://papers.nips.cc"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ml-venue-insights/0.2 "
        "(research; contact: set-your-email-here)"
    )
}

# Legacy single-venue defaults. Multi-venue runs derive per-venue filenames
# from venues.corpus_filename() / venues.log_filename() instead.
CORPUS_FILE = "neurips_corpus.jsonl"
SCRAPE_LOG = "scraped_hashes.log"
TOPICS_FILE = "topics.json"

POOLED_CORPUS = "pooled_corpus.jsonl"

# Polite scraping delay (seconds), uniform random in this range.
# Applies to the HTML backends (nips, pmlr). The OpenReview backend is an API
# and paginates instead, so it uses its own much smaller delay.
DELAY_MIN = 1.5
DELAY_MAX = 3.5

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
