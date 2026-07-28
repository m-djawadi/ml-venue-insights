"""Streaming JSONL corpus I/O. Never loads the whole corpus into memory."""

from __future__ import annotations

import json
import os
import re
from typing import Iterator, Dict, Any

_WS = re.compile(r"\s+")


def iter_corpus(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one paper record at a time. Constant memory."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus not found: {path}\nRun `ml-venue-insights --scrape` first."
        )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_record(path: str, record: Dict[str, Any]) -> None:
    """Append a single record and flush immediately (crash-safe)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def count_records(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# Venue-identifying tokens. On a POOLED multi-venue corpus these are poison:
# if "icml" or "neurips" survives into the embedded text, the clusterer will
# happily recover the VENUE instead of the RESEARCH TOPIC, and every theme comes
# back 100%-one-venue. That looks like a strong finding and is actually just the
# model reading the letterhead. We strip them before embedding so a cluster's
# venue mix is an independent measurement rather than a restatement of the input.
_VENUE_TOKENS = re.compile(
    r"\b("
    r"neur\s?ips|nips|"
    r"icml|iclr|colt|rlc|"
    r"conference on neural information processing systems|"
    r"international conference on machine learning|"
    r"international conference on learning representations|"
    r"conference on learning theory|"
    r"reinforcement learning conference|"
    r"proceedings of machine learning research|pmlr|openreview"
    r")\b",
    re.IGNORECASE,
)


def scrub_venue_terms(text: str) -> str:
    """Remove venue names/abbreviations so clustering can't cheat off them."""
    return _WS.sub(" ", _VENUE_TOKENS.sub(" ", text)).strip()


def doc_text(record: Dict[str, Any], scrub: bool = True) -> str:
    """Combine title + abstract into a single analysis document.

    scrub=True (default) removes venue-identifying tokens. Keep it on for
    anything that feeds embeddings or clustering; turn it off only if you
    specifically want the raw text back.
    """
    title = record.get("title", "") or ""
    abstract = record.get("abstract", "") or ""
    text = f"{title}. {abstract}".strip()
    return scrub_venue_terms(text) if scrub else text
