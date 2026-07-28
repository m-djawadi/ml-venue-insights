"""Venue registry: which conferences exist, and how to fetch each one.

The whole point of this module is that everything downstream of the corpus
(stats, embeddings, clustering, temporal analysis, LLM synthesis) is already
venue-agnostic -- it only ever sees records with title/abstract/year. So adding
a conference means adding a *fetch strategy*, nothing else.

Three backends cover all five venues:

  nips       papers.nips.cc HTML (NeurIPS only -- its own proceedings site)
  pmlr       proceedings.mlr.press volume pages (ICML, COLT)
  openreview OpenReview API v2 (ICLR, RLC)

Why the split:
  - NeurIPS hosts its own proceedings with citation_* meta tags.
  - ICML and COLT publish through PMLR. PMLR assigns a NEW VOLUME NUMBER per
    year with no derivable pattern, so the year->volume map must be a literal
    table. That table is the one thing you maintain here each year.
  - ICLR has no print proceedings; OpenReview is the canonical source. RLC
    (new in 2024) likewise publishes via OpenReview under rl-conference.cc.

Adding next year's edition = add one line to the relevant YEARS dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Venue:
    key: str                      # cli name, lowercase
    name: str                     # display name
    backend: str                  # "nips" | "pmlr" | "openreview"
    years: Dict[int, str] = field(default_factory=dict)
    # pmlr: year -> volume id ("v267").  openreview: year -> venueid.
    # nips: unused (year is part of the URL path).
    nips_range: Optional[tuple] = None   # (first_year, last_year) for nips backend
    note: str = ""

    def available_years(self):
        if self.backend == "nips":
            lo, hi = self.nips_range
            return list(range(lo, hi + 1))
        return sorted(self.years)

    def resolve(self, start: int, end: int):
        """Years we can actually fetch inside [start, end], plus what we skipped."""
        have = set(self.available_years())
        want = list(range(start, end + 1))
        ok = [y for y in want if y in have]
        missing = [y for y in want if y not in have]
        return ok, missing


# --------------------------------------------------------------------------- #
# NeurIPS -- papers.nips.cc.  Abstracts exist as citation_abstract from ~2010s;
# earlier years still parse but the abstract fallback is less reliable.
# --------------------------------------------------------------------------- #
NEURIPS = Venue(
    key="neurips",
    name="NeurIPS (Conference on Neural Information Processing Systems)",
    backend="nips",
    nips_range=(1987, 2024),
    note="papers.nips.cc; the year index page lists every paper's Abstract page.",
)

# --------------------------------------------------------------------------- #
# ICML -- PMLR.  Volume numbers are assigned in publication order across ALL
# PMLR proceedings, so they jump around; there is no formula. Verify a new year
# at https://proceedings.mlr.press/ before adding it.
# --------------------------------------------------------------------------- #
ICML = Venue(
    key="icml",
    name="ICML (International Conference on Machine Learning)",
    backend="pmlr",
    years={
        2015: "v37",
        2016: "v48",
        2017: "v70",
        2018: "v80",
        2019: "v97",
        2020: "v119",
        2021: "v139",
        2022: "v162",
        2023: "v202",
        2024: "v235",
        2025: "v267",
    },
    note="proceedings.mlr.press volume per year.",
)

# --------------------------------------------------------------------------- #
# COLT -- also PMLR. Much smaller volumes (a few hundred papers/year), and the
# papers are theory: expect short, dense, math-heavy abstracts. That matters for
# clustering -- COLT alone will produce fewer, tighter clusters than ICML.
# --------------------------------------------------------------------------- #
COLT = Venue(
    key="colt",
    name="COLT (Conference on Learning Theory)",
    backend="pmlr",
    years={
        2015: "v40",
        2016: "v49",
        2017: "v65",
        2018: "v75",
        2019: "v99",
        2020: "v125",
        2021: "v134",
        2022: "v178",
        2023: "v195",
        2024: "v247",
        2025: "v291",
    },
    note="proceedings.mlr.press volume per year.",
)

# --------------------------------------------------------------------------- #
# ICLR -- OpenReview API v2. The `venueid` content field distinguishes accepted
# papers from withdrawn/rejected/desk-rejected ones; accepted papers keep the
# base venue id, which is exactly what we query for.
#
# ICLR 2023 and earlier live on API v1 with a different note schema; the
# openreview backend falls back to v1 automatically for those years.
# --------------------------------------------------------------------------- #
ICLR = Venue(
    key="iclr",
    name="ICLR (International Conference on Learning Representations)",
    backend="openreview",
    years={
        2018: "ICLR.cc/2018/Conference",
        2019: "ICLR.cc/2019/Conference",
        2020: "ICLR.cc/2020/Conference",
        2021: "ICLR.cc/2021/Conference",
        2022: "ICLR.cc/2022/Conference",
        2023: "ICLR.cc/2023/Conference",
        2024: "ICLR.cc/2024/Conference",
        2025: "ICLR.cc/2025/Conference",
        2026: "ICLR.cc/2026/Conference",
    },
    note="OpenReview; accepted papers only (venueid filter).",
)

# --------------------------------------------------------------------------- #
# RLC -- Reinforcement Learning Conference, first held 2024. Peer-reviewed with
# proceedings, hosted on OpenReview under rl-conference.cc. Small (~100-300
# papers/year) and only 2-3 editions exist, so temporal analysis on RLC ALONE
# is thin -- it is far more informative pooled with the others (--venues all)
# where it acts as an RL-density signal.
# --------------------------------------------------------------------------- #
RLC = Venue(
    key="rlc",
    name="RLC (Reinforcement Learning Conference)",
    backend="openreview",
    years={
        2024: "rl-conference.cc/RLC/2024/Conference",
        2025: "rl-conference.cc/RLC/2025/Conference",
        2026: "rl-conference.cc/RLC/2026/Conference",
    },
    note="OpenReview; conference began 2024, so the time series is short.",
)


VENUES: Dict[str, Venue] = {
    v.key: v for v in (NEURIPS, ICML, ICLR, RLC, COLT)
}

ALL_KEYS = list(VENUES)


def get_venue(key: str) -> Venue:
    k = key.strip().lower()
    if k not in VENUES:
        raise SystemExit(
            f"Unknown venue '{key}'. Known: {', '.join(ALL_KEYS)}"
        )
    return VENUES[k]


def parse_venue_list(spec: str):
    """'all' | 'icml,iclr' -> [Venue, ...] preserving registry order."""
    if not spec or spec.strip().lower() == "all":
        return [VENUES[k] for k in ALL_KEYS]
    keys = [s for s in (x.strip().lower() for x in spec.split(",")) if s]
    return [get_venue(k) for k in keys]


def corpus_filename(venue_key: str) -> str:
    return f"{venue_key}_corpus.jsonl"


def log_filename(venue_key: str) -> str:
    return f"{venue_key}_scraped.log"


def describe_all() -> str:
    rows = []
    for k in ALL_KEYS:
        v = VENUES[k]
        ys = v.available_years()
        span = f"{min(ys)}-{max(ys)}" if ys else "-"
        rows.append(f"  {k:<8} {v.backend:<11} {span:<12} {v.name}")
    return "\n".join(rows)
