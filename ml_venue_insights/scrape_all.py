"""Backend dispatcher: turn a Venue + year range into a JSONL corpus.

Every backend gets the same two contracts:
  - `done`: a set of already-fetched keys, loaded from an append-only log
  - `on_record(record, key)`: append the record, then log the key

Doing the write and the log in that order means a crash can at worst re-fetch
one paper; it can never silently drop one. Each venue writes its own corpus
file so venues stay independently re-scrapable, and `--venues all` just runs
this once per venue.
"""

from __future__ import annotations

import os
from typing import Optional

from .corpus import append_record, count_records
from .venues import Venue, corpus_filename, log_filename


def _load_done(log_path: str) -> set:
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


def _make_writer(corpus_path: str, log_path: str, done: set):
    def on_record(rec, key):
        rec.setdefault("venue", None)
        append_record(corpus_path, rec)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(key + "\n")
            lf.flush()
        done.add(key)
    return on_record


def scrape_venue(venue: Venue, start: int, end: int, data_dir: str,
                 limit_per_year: Optional[int] = None) -> str:
    """Scrape one venue into data_dir/<key>_corpus.jsonl. Returns that path."""
    corpus_path = os.path.join(data_dir, corpus_filename(venue.key))
    log_path = os.path.join(data_dir, log_filename(venue.key))

    years, missing = venue.resolve(start, end)
    if missing:
        print(f"  (no {venue.key} edition indexed for: "
              f"{', '.join(str(m) for m in missing)})")
    if not years:
        print(f"  nothing to fetch for {venue.key} in {start}-{end}.")
        return corpus_path

    done = _load_done(log_path)
    print(f"Resuming {venue.key}: {len(done)} papers already fetched.")

    base_writer = _make_writer(corpus_path, log_path, done)

    def on_record(rec, key):
        rec["venue"] = venue.key      # stamp every record with its venue
        base_writer(rec, key)

    if venue.backend == "nips":
        from .scrape import get_paper_links_for_year, extract_paper
        import random, time
        from .config import DELAY_MIN, DELAY_MAX

        for year in years:
            links = get_paper_links_for_year(year)
            if limit_per_year:
                links = links[:limit_per_year]
            print(f"\n{venue.key} {year}: {len(links)} papers")
            new_count = 0
            for i, url in enumerate(links, 1):
                key = f"neurips:{url.rstrip('/').split('/')[-1]}"
                if key in done:
                    continue
                try:
                    rec = extract_paper(url)
                    if rec is None:
                        continue
                    rec["year"] = year
                    on_record(rec, key)
                    new_count += 1
                    if new_count % 50 == 0:
                        print(f"  [{i}/{len(links)}] {year} (+{new_count} new)")
                except Exception as e:
                    print(f"  ! err {key}: {e}")
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            print(f"Done {year}: +{new_count} new papers.")

    elif venue.backend == "pmlr":
        from .scrape_pmlr import scrape_year as pmlr_year
        for year in years:
            pmlr_year(venue.years[year], year, limit_per_year, done, on_record)

    elif venue.backend == "openreview":
        from .scrape_openreview import scrape_year as or_year
        for year in years:
            or_year(venue.years[year], year, limit_per_year, done, on_record)

    else:
        raise SystemExit(f"Unknown backend '{venue.backend}' for {venue.key}")

    print(f"\n{venue.key} corpus now holds {count_records(corpus_path)} papers.")
    return corpus_path
