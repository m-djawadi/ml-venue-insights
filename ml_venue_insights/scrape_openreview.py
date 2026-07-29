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

import os
import time
from typing import List, Dict, Any, Optional

import requests

API_V2 = "https://api2.openreview.net/notes"
API_V1 = "https://api.openreview.net/notes"
LOGIN_V2 = "https://api2.openreview.net/login"

PAGE = 1000

# OpenReview sits behind Cloudflare bot protection. Anonymous API requests get
# a "Verifying your browser" challenge (HTTP 403) that plain requests can't
# solve. Logging in with a free OpenReview account returns a bearer token that
# skips the challenge entirely -- this is the supported way to script access.
#
# Credentials are read from env vars (never hard-code them):
#   OPENREVIEW_USERNAME  -- the email on your OpenReview account
#   OPENREVIEW_PASSWORD  -- your password
# If they're unset, we still try anonymously (works from some networks, and for
# older venues), so nothing breaks for users who don't hit the challenge.
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/125.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://openreview.net/",
})

_LOGGED_IN = False   # becomes True once a token is set on the session


def _try_login() -> bool:
    """Authenticate if OPENREVIEW_USERNAME/PASSWORD are set. Idempotent."""
    global _LOGGED_IN
    if _LOGGED_IN:
        return True
    user = os.environ.get("OPENREVIEW_USERNAME")
    pw = os.environ.get("OPENREVIEW_PASSWORD")
    if not user or not pw:
        return False
    try:
        r = _SESSION.post(LOGIN_V2, json={"id": user, "password": pw},
                          timeout=30)
        if r.status_code == 200:
            token = r.json().get("token")
            if token:
                _SESSION.headers["Authorization"] = f"Bearer {token}"
                _LOGGED_IN = True
                print(f"  logged in to OpenReview as {user}")
                return True
        # Surface OpenReview's own explanation — the status code alone hides why.
        detail = ""
        try:
            body = r.json()
            detail = (body.get("message") or body.get("name")
                      or str(body))[:300]
        except Exception:
            detail = r.text[:300]
        print(f"  ! OpenReview login failed (HTTP {r.status_code}): {detail}")
        if r.status_code == 400:
            print("    A 400 usually means the account isn't activated yet.\n"
                  "    Check your email for an OpenReview activation link and\n"
                  "    click it, then wait a few minutes and re-run. New profiles\n"
                  "    can also need brief moderation before API login works.")
        elif r.status_code in (403, 401):
            print("    Wrong email or password. Note the field is your EMAIL,\n"
                  "    not a username. Reset at https://openreview.net if unsure.")
    except requests.exceptions.RequestException as e:
        print(f"  ! OpenReview login request failed: {e}")
    return False


# Between successful pages. OpenReview is generous but not free.
PAGE_DELAY = 1.0
# Retry schedule for 429/503 (seconds). Grows so a briefly-busy API recovers
# without hammering it.
_BACKOFF = [3, 6, 12, 20, 30]


class _RateLimited(Exception):
    """Raised when a page keeps returning 429/503 after all retries."""


class _Forbidden(Exception):
    """Raised on 403 — Cloudflare challenge or a network/proxy block."""


def _val(content: Dict[str, Any], field: str) -> Any:
    """Read a content field from either API generation."""
    v = content.get(field)
    if isinstance(v, dict):          # v2 wraps values
        return v.get("value")
    return v                          # v1 stores them bare


def _fetch_page(api: str, venue_id: str, offset: int) -> Optional[List[dict]]:
    """One page of notes.

    Returns a list on success (possibly empty), or None to signal "this API
    generation isn't right, try the other one" (HTTP 400). Raises _RateLimited
    on persistent 429/503, or _Forbidden on 403 (Cloudflare/proxy block).
    """
    params = {
        "content.venueid": venue_id,
        "limit": PAGE,
        "offset": offset,
    }
    for attempt, wait in enumerate([0] + _BACKOFF):
        if wait:
            time.sleep(wait)
        try:
            r = _SESSION.get(api, params=params, timeout=90)
            if r.status_code == 400:
                return None           # wrong API generation for this venue
            if r.status_code in (401, 403):
                # Try logging in once, then retry this page immediately.
                if not _LOGGED_IN and _try_login():
                    continue
                raise _Forbidden(venue_id)   # not retryable without creds
            if r.status_code in (429, 503):
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(min(int(ra), 60))
                if attempt < len(_BACKOFF):
                    print(f"    throttled ({r.status_code}); retrying "
                          f"in {_BACKOFF[attempt]}s ...")
                continue              # retry
            r.raise_for_status()
            body = r.json()
            notes = body.get("notes", [])
            if os.environ.get("OPENREVIEW_DEBUG"):
                total = body.get("count")
                print(f"    [debug] {api} offset={offset}: "
                      f"count={total}, notes_this_page={len(notes)}")
                if not notes and offset == 0:
                    # show what the API actually said, so we can see why
                    keys = list(body.keys())
                    print(f"    [debug] response keys: {keys}")
                    print(f"    [debug] first 400 chars: {str(body)[:400]}")
            return notes
        except _Forbidden:
            raise
        except requests.exceptions.RequestException as e:
            if attempt < len(_BACKOFF):
                print(f"    request error ({e}); retrying "
                      f"in {_BACKOFF[attempt]}s ...")
                continue
            print(f"  ! OpenReview request failed after retries: {e}")
            return None
    raise _RateLimited(venue_id)


def fetch_venue_notes(venue_id: str) -> List[dict]:
    """All accepted notes for a venue id, trying API v2 then v1.

    Raises _RateLimited if the API throttled us to death on the FIRST page of
    both generations (so the caller can report it honestly instead of claiming
    the venue is empty).
    """
    rate_limited = False
    _try_login()   # no-op if creds unset or already logged in
    for api, label in ((API_V2, "v2"), (API_V1, "v1")):
        notes, offset = [], 0
        try:
            while True:
                page = _fetch_page(api, venue_id, offset)
                if page is None:
                    break             # try the other API
                notes.extend(page)
                if len(page) < PAGE:
                    return notes if notes else []
                offset += PAGE
                time.sleep(PAGE_DELAY)
        except _Forbidden:
            raise                     # a 403 on v2 will 403 on v1 too; bail now
        except _RateLimited:
            rate_limited = True
            continue                  # try the other API before giving up
        if notes:
            return notes
    if rate_limited:
        raise _RateLimited(venue_id)
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
    try:
        notes = fetch_venue_notes(venue_id)
    except _Forbidden:
        print(f"  ! OpenReview blocked the request for {venue_id} (Cloudflare).\n"
              f"    OpenReview now requires a (free) login to script its API.\n"
              f"    Fix — one-time, takes 2 minutes:\n"
              f"      1. Make a free account at https://openreview.net (Sign Up).\n"
              f"      2. Set these in your terminal BEFORE running, then re-run:\n"
              f"         PowerShell:\n"
              f'           $env:OPENREVIEW_USERNAME = "you@email.com"\n'
              f'           $env:OPENREVIEW_PASSWORD = "yourpassword"\n'
              f"      3. python run.py --venues iclr,rlc --scrape --start 2020 --end 2025\n"
              f"    (ICML/COLT are already scraped and don't need this.)")
        raise SystemExit(1)
    except _RateLimited:
        print(f"  ! OpenReview kept rate-limiting us for {venue_id}.\n"
              f"    This is temporary — wait a minute and re-run the SAME "
              f"command; already-saved venues are skipped, so it resumes here.")
        return 0
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
