"""PMLR (proceedings.mlr.press) backend -- ICML, COLT.

PMLR volume index pages are cheap and complete: a single page lists every paper
in the volume, and each entry already carries the title and author list. The
abstract is NOT on the index page, so we follow each paper's own page.

Structure of a volume index entry:

    <div class="paper">
      <p class="title">Paper Title</p>
      <span class="authors">A, B, C</span>
      <p class="links">
        <a href="http://proceedings.mlr.press/v267/foo25a.html">abs</a>
        <a href=".../foo25a/foo25a.pdf">Download PDF</a>
      </p>
    </div>

and on the per-paper page the abstract sits in <div id="abstract"> (with a
citation_abstract meta tag on newer volumes).

Resumability works the same way as the NeurIPS scraper: we key on the paper's
stable URL slug and keep an append-only log.
"""

from __future__ import annotations

import random
import time
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

from .config import HEADERS, DELAY_MIN, DELAY_MAX

PMLR_BASE = "https://proceedings.mlr.press"


def _abs_url(href: str) -> str:
    if href.startswith("http"):
        # PMLR serves some links as http; normalize so the dedupe log is stable
        return href.replace("http://", "https://", 1)
    return PMLR_BASE + (href if href.startswith("/") else "/" + href)


def get_paper_links_for_volume(volume: str) -> List[Dict[str, str]]:
    """Return [{url, title, authors}] for every paper in a PMLR volume.

    Titles come from the index page so we still have them even if a per-paper
    fetch later fails.
    """
    url = f"{PMLR_BASE}/{volume}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ! failed to fetch PMLR volume index {volume}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for div in soup.find_all("div", class_="paper"):
        t = div.find("p", class_="title") or div.find("span", class_="title")
        title = t.get_text(" ", strip=True) if t else ""

        a_el = div.find("span", class_="authors") or div.find("p", class_="details")
        authors = []
        if a_el:
            authors = [x.strip() for x in a_el.get_text(" ", strip=True).split(",") if x.strip()]

        link = None
        for a in div.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".html") and "/assets/" not in href:
                link = _abs_url(href)
                break
        if not link:
            continue
        out.append({"url": link, "title": title, "authors": authors})

    # dedupe on url, preserve order
    seen, deduped = set(), []
    for rec in out:
        if rec["url"] not in seen:
            seen.add(rec["url"])
            deduped.append(rec)
    return deduped


def _extract_abstract(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", attrs={"name": "citation_abstract"})
    if meta and meta.get("content") and len(meta["content"].split()) > 10:
        return meta["content"].strip()

    div = soup.find("div", id="abstract") or soup.find("div", class_="abstract")
    if div:
        txt = div.get_text(" ", strip=True)
        if len(txt.split()) > 10:
            return txt

    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paras = [p for p in paras if len(p.split()) > 15]
    return max(paras, key=len) if paras else ""


def extract_paper(entry: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Fetch one PMLR paper page; merge index metadata with the abstract."""
    r = requests.get(entry["url"], headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title = entry.get("title") or ""
    if not title:
        m = soup.find("meta", attrs={"name": "citation_title"})
        if m and m.get("content"):
            title = m["content"].strip()

    authors = entry.get("authors") or [
        m["content"] for m in soup.find_all("meta", attrs={"name": "citation_author"})
        if m.get("content")
    ]

    return {
        "title": title,
        "abstract": _extract_abstract(soup),
        "authors": authors,
        "url": entry["url"],
    }


def scrape_year(volume: str, year: int, limit: Optional[int], done: set,
                on_record, delay=(DELAY_MIN, DELAY_MAX)) -> int:
    """Scrape one PMLR volume. Calls on_record(record, key) per new paper."""
    entries = get_paper_links_for_volume(volume)
    if limit:
        entries = entries[:limit]
    print(f"\n{volume} ({year}): {len(entries)} papers")

    new_count = 0
    for i, entry in enumerate(entries, 1):
        key = entry["url"].rstrip("/").split("/")[-1].replace(".html", "")
        key = f"{volume}:{key}"
        if key in done:
            continue
        try:
            rec = extract_paper(entry)
            if rec is None or not rec.get("title"):
                continue
            rec["year"] = year
            on_record(rec, key)
            new_count += 1
            if new_count % 50 == 0:
                print(f"  [{i}/{len(entries)}] {year} (+{new_count} new)")
        except Exception as e:
            print(f"  ! err {key}: {e}")
        time.sleep(random.uniform(*delay))

    print(f"Done {year}: +{new_count} new papers.")
    return new_count
