"""OpenReview backend -- ICLR, RLC.

This is an API, not a scrape, so it is dramatically faster than the HTML paths:
one paged request returns up to 1000 full records (title + abstract + authors)
with no per-paper fetch. A whole ICLR year lands in a few seconds.

Two API generations:

  v2 (api2.openreview.net)  -- 2024+ for ICLR, all RLC years. Content fields are
                               wrapped: content["title"]["value"].
  v1 (api.openreview.net)   -- ICLR 2023 and earlier. Content fields are bare:
                               content["title"].

We try v2 first and fall back to v1, normalizing both into the same record
shape the rest of the pipeline expects.

ACCEPTED PAPERS ONLY: OpenReview keeps rejected, withdrawn, and desk-rejected
submissions publicly queryable. Querying by `content.venueid == <venue id>`
returns only the accepted set, because non-accepted papers get reassigned to a
different venueid (e.g. .../Withdrawn_Submission). This matters a lot for ICLR,
where rejected submissions outnumber accepted ones -- including them would
badly distort every trend in the analysis.
"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional

import requests

API_V2 = "https://api2.openreview.net/notes"
API_V1 = "https://api.openreview.net/notes"

PAGE = 1000


def _val(content: Dict[str, Any], field: str) -> Any:
    """Read a content field from either API generation."""
    v = content.get(field)
    if isinstance(v, dict):          # v2 wraps values
        return v.get("value")
    return v                          # v1 stores them bare


def _fetch_page(api: str, venue_id: str, offset: int) -> Optional[List[dict]]:
    params = {
        "content.venueid": venue_id,
        "limit": PAGE,
        "offset": offset,
        "details": "",
    }
    try:
        r = requests.get(api, params=params, timeout=90)
        if r.status_code == 400:
            return None               # wrong API generation for this venue
        r.raise_for_status()
        return r.json().get("notes", [])
    except requests.exceptions.RequestException as e:
        print(f"  ! OpenReview request failed: {e}")
        return None


def fetch_venue_notes(venue_id: str) -> List[dict]:
    """All accepted notes for a venue id, trying API v2 then v1."""
    for api, label in ((API_V2, "v2"), (API_V1, "v1")):
        notes, offset = [], 0
        while True:
            page = _fetch_page(api, venue_id, offset)
            if page is None:
                break                 # try the other API
            notes.extend(page)
            if len(page) < PAGE:
                return notes if notes else []
            offset += PAGE
            time.sleep(0.3)           # be polite; the API is generous but not free
        if notes:
            return notes
    return []


def note_to_record(note: dict, year: int) -> Optional[Dict[str, Any]]:
    content = note.get("content") or {}
    title = (_val(content, "title") or "").strip()
    abstract = (_val(content, "abstract") or "").strip()
    if not title:
        return None

    authors = _val(content, "authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]

    return {
        "title": title,
        "abstract": abstract,
        "authors": list(authors),
        "url": f"https://openreview.net/forum?id={note.get('id', '')}",
        "year": year,
    }


def scrape_year(venue_id: str, year: int, limit: Optional[int], done: set,
                on_record) -> int:
    """Pull one OpenReview venue-year. Calls on_record(record, key) per paper."""
    print(f"\n{venue_id} ({year}): querying OpenReview ...")
    notes = fetch_venue_notes(venue_id)
    if not notes:
        print(f"  ! no accepted papers returned for {venue_id} "
              f"(venue id may be wrong, or the year isn't public yet)")
        return 0
    if limit:
        notes = notes[:limit]
    print(f"  {len(notes)} accepted papers")

    new_count = 0
    for note in notes:
        key = f"{venue_id}:{note.get('id','')}"
        if key in done:
            continue
        rec = note_to_record(note, year)
        if rec is None:
            continue
        on_record(rec, key)
        new_count += 1

    print(f"Done {year}: +{new_count} new papers.")
    return new_count
