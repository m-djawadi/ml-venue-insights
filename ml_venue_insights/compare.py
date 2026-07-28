"""Cross-venue analytics -- the part that only exists once you have >1 venue.

Two distinct things live here:

1. `build_pooled_corpus`  -- concatenate per-venue JSONL into one corpus so the
   existing clustering/temporal pipeline can run over all venues at once. Every
   record carries a `venue` field, so a cluster can be decomposed by venue
   afterwards. This is what makes questions like "is this theme an ICML theme
   or an ICLR theme?" answerable.

2. Comparison reports -- things that are meaningless for a single venue:

   - VENUE SIGNATURE: terms over-represented at a venue relative to the pooled
     corpus, scored by log-odds with an informative Dirichlet prior rather than
     raw frequency ratio. Raw ratios are dominated by rare terms (a word
     appearing 3 times at COLT and 0 elsewhere looks infinitely distinctive);
     the log-odds-with-prior estimator shrinks exactly those cases and is the
     standard fix from Monroe et al. (2008).

   - THEME MIX: for each cluster from a pooled --analyze run, the share of its
     papers coming from each venue, plus a concentration score. A theme at 90%
     one venue is a venue specialty; an evenly-split theme is field-wide.

   - DIFFUSION: for themes that are venue-concentrated, when did each venue's
     papers appear? If ICLR papers in a theme predate ICML papers by two years,
     that is evidence of where the idea originated and how it spread.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List

from .corpus import iter_corpus, doc_text, count_records
from .stats import tokens
from .venues import corpus_filename, VENUES


# ------------------------------ pooling ----------------------------------- #

def build_pooled_corpus(venue_keys: List[str], data_dir: str,
                        out_name: str = "pooled_corpus.jsonl") -> str:
    """Concatenate per-venue corpora into one file, stamping missing venues."""
    out_path = os.path.join(data_dir, out_name)
    n_total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for key in venue_keys:
            path = os.path.join(data_dir, corpus_filename(key))
            if not os.path.exists(path):
                print(f"  ! {key}: no corpus yet ({os.path.basename(path)}); skipping")
                continue
            n = 0
            for rec in iter_corpus(path):
                rec.setdefault("venue", key)
                if not rec.get("venue"):
                    rec["venue"] = key
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            print(f"  {key:<8} {n:6d} papers")
            n_total += n
    print(f"\nPooled corpus: {n_total} papers -> {out_path}")
    return out_path


def venue_counts(corpus_path: str):
    """(venue, year) -> count, from a pooled corpus."""
    by_venue = Counter()
    by_venue_year = defaultdict(Counter)
    for rec in iter_corpus(corpus_path):
        v = rec.get("venue", "?")
        by_venue[v] += 1
        by_venue_year[v][rec.get("year", 0)] += 1
    return by_venue, by_venue_year


# --------------------------- venue signature ------------------------------ #

def _log_odds_with_prior(counts_a: Counter, counts_rest: Counter,
                         prior: Counter, alpha0: float = 500.0):
    """Monroe et al. (2008) weighted log-odds, z-scored.

    Returns {term: z}. Positive z = over-represented in group A. The prior
    (pooled corpus frequencies, scaled to alpha0 pseudo-counts) is what keeps
    rare terms from dominating.
    """
    n_prior = sum(prior.values()) or 1
    n_a = sum(counts_a.values())
    n_b = sum(counts_rest.values())
    if n_a == 0 or n_b == 0:
        return {}

    z = {}
    for w in counts_a:
        a_w = prior[w] / n_prior * alpha0
        if a_w <= 0:
            continue
        ya, yb = counts_a[w], counts_rest.get(w, 0)
        # log odds ratio in each group, with prior pseudo-counts
        la = math.log((ya + a_w) / (n_a + alpha0 - ya - a_w))
        lb = math.log((yb + a_w) / (n_b + alpha0 - yb - a_w))
        delta = la - lb
        var = 1.0 / (ya + a_w) + 1.0 / (yb + a_w)
        z[w] = delta / math.sqrt(var)
    return z


def venue_signatures(corpus_path: str, top_n: int = 20, min_count: int = 20):
    """Terms most distinctive of each venue vs all others."""
    per_venue = defaultdict(Counter)
    pooled = Counter()
    for rec in iter_corpus(corpus_path):
        v = rec.get("venue", "?")
        for t in tokens(doc_text(rec)):
            per_venue[v][t] += 1
            pooled[t] += 1

    keep = {w for w, c in pooled.items() if c >= min_count}
    results = {}
    for v, counts in per_venue.items():
        rest = Counter()
        for other, oc in per_venue.items():
            if other != v:
                rest.update(oc)
        ca = Counter({w: c for w, c in counts.items() if w in keep})
        cb = Counter({w: c for w, c in rest.items() if w in keep})
        z = _log_odds_with_prior(ca, cb, pooled)
        results[v] = sorted(z.items(), key=lambda kv: -kv[1])[:top_n]
    return results


def run_compare(corpus_path: str, top_n: int = 20) -> None:
    """Headline cross-venue report: volume, then per-venue vocabulary."""
    by_venue, by_venue_year = venue_counts(corpus_path)
    total = sum(by_venue.values())

    print(f"\n{'='*64}\n POOLED CORPUS: {total} papers across "
          f"{len(by_venue)} venues\n{'='*64}")

    all_years = sorted({y for c in by_venue_year.values() for y in c})
    print("\nPapers per venue per year")
    header = "  venue     " + "".join(f"{y:>7}" for y in all_years) + "    total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for v in sorted(by_venue, key=lambda k: -by_venue[k]):
        row = "".join(f"{by_venue_year[v].get(y, 0):>7}" for y in all_years)
        print(f"  {v:<10}{row}{by_venue[v]:>9}")

    print(f"\n{'='*64}\n VENUE SIGNATURES (weighted log-odds vs other venues)\n{'='*64}")
    print(" Terms a venue uses far more than the rest of the field.")
    sigs = venue_signatures(corpus_path, top_n=top_n)
    for v in sorted(sigs):
        name = VENUES[v].name if v in VENUES else v
        print(f"\n  {name}")
        terms = ", ".join(f"{w} ({z:.1f})" for w, z in sigs[v][:top_n])
        print(f"    {terms}")


# ---------------------- theme mix over a pooled analysis ------------------- #

def theme_venue_mix(analysis_path: str, corpus_path: str) -> List[dict]:
    """For each cluster in a pooled analysis, decompose it by venue.

    Matches representative papers back to venues by title. We use the full
    cluster only if the analysis stored per-paper venues; otherwise the
    representative papers give a usable (if coarse) estimate, which is why we
    also report how many papers the mix is based on.
    """
    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)

    title_venue, title_year = {}, {}
    for rec in iter_corpus(corpus_path):
        t = (rec.get("title") or "").strip()
        if t:
            title_venue[t] = rec.get("venue", "?")
            title_year[t] = rec.get("year", 0)

    rows = []
    for c in analysis["clusters"]:
        # Prefer the exact mix computed during --analyze (all papers, not just
        # representatives). Fall back to title-matching representatives.
        exact = c.get("venue_counts")
        if exact:
            n = sum(exact.values())
            shares = {v: cnt / n for v, cnt in exact.items()}
            top_v, top_share = max(shares.items(), key=lambda kv: kv[1])
            # Exact onset years from the full cluster when --analyze stored
            # them; representatives are a last resort.
            first_year = c.get("venue_onset_year") or c.get("venue_first_year") or {}
            if not first_year:
                first_year = {}
                for p in c.get("representative_papers", []):
                    v = title_venue.get(p["title"])
                    y = title_year.get(p["title"], 0)
                    if v and y and (v not in first_year or y < first_year[v]):
                        first_year[v] = y
            rows.append({
                "id": c["id"],
                "name": c.get("name") or " ".join(
                    t.capitalize() for t in c.get("top_terms", [])[:3]),
                "size": c["size"], "basis": n, "mix": shares,
                "dominant": top_v, "concentration": top_share,
                "first_year": first_year,
            })
            continue

        mix = Counter()
        first_year = {}
        for p in c.get("representative_papers", []):
            v = title_venue.get(p["title"])
            if not v:
                continue
            mix[v] += 1
            y = title_year.get(p["title"], 0)
            if y and (v not in first_year or y < first_year[v]):
                first_year[v] = y
        n = sum(mix.values())
        if not n:
            continue
        shares = {v: mix[v] / n for v in mix}
        top_v, top_share = max(shares.items(), key=lambda kv: kv[1])
        rows.append({
            "id": c["id"],
            "name": c.get("name") or " ".join(
                t.capitalize() for t in c.get("top_terms", [])[:3]),
            "size": c["size"],
            "basis": n,
            "mix": shares,
            "dominant": top_v,
            "concentration": top_share,
            "first_year": first_year,
        })
    rows.sort(key=lambda r: -r["concentration"])
    return rows


def run_theme_mix(analysis_path: str, corpus_path: str) -> None:
    if not os.path.exists(analysis_path):
        raise SystemExit(f"{analysis_path} not found. Run `--analyze` on the "
                         f"pooled corpus first (--venues all --analyze).")
    rows = theme_venue_mix(analysis_path, corpus_path)
    if not rows:
        raise SystemExit("No venue attribution possible -- is this a pooled corpus?")

    print(f"\n{'='*64}\n THEME x VENUE MIX\n{'='*64}")
    print(" Which venues own which themes. High concentration = venue specialty;")
    print(" even mix = field-wide topic.\n")

    print("  VENUE-CONCENTRATED THEMES")
    for r in rows[:12]:
        mix = ", ".join(f"{v} {s*100:.0f}%" for v, s in
                        sorted(r["mix"].items(), key=lambda kv: -kv[1]))
        print(f"    #{r['id']:>3} {r['name'][:38]:<38} [{r['size']:>5}p]  {mix}")

    print("\n  FIELD-WIDE THEMES (most evenly shared)")
    for r in sorted(rows, key=lambda r: r["concentration"])[:12]:
        mix = ", ".join(f"{v} {s*100:.0f}%" for v, s in
                        sorted(r["mix"].items(), key=lambda kv: -kv[1]))
        print(f"    #{r['id']:>3} {r['name'][:38]:<38} [{r['size']:>5}p]  {mix}")

    # Diagnostic: if nearly every theme is single-venue, the clustering is
    # probably keying on venue-specific boilerplate rather than research
    # content, and the whole mix report is circular. Say so loudly.
    n_pure = sum(1 for r in rows if r["concentration"] >= 0.95)
    frac_pure = n_pure / len(rows)
    if frac_pure >= 0.8:
        print(f"\n  !! WARNING: {n_pure}/{len(rows)} themes are >=95% single-venue.")
        print("     That usually means clustering separated VENUES, not TOPICS")
        print("     (venue boilerplate leaking into the embedded text), so treat")
        print("     this mix as unreliable. Check that doc_text() scrubbing is on")
        print("     and consider re-running --analyze with --force-embed.")

    print(f"\n{'='*64}\n THEME DIFFUSION (onset year per venue)\n{'='*64}")
    print(" Onset = the year a venue's output in this theme becomes non-trivial.")
    print(" Venues with <5% of a theme are excluded so a stray paper can't set")
    print(" the clock. A consistent lead suggests where a theme originated.\n")
    with open(analysis_path, encoding="utf-8") as _f:
        analysis_clusters = json.load(_f)["clusters"]
    for r in rows:
        fy = r["first_year"]
        # Only compare venues that actually have a foothold in the theme;
        # a venue with 2 papers in a 400-paper cluster shouldn't set the clock.
        counts = None
        for cl in analysis_clusters:
            if cl["id"] == r["id"]:
                counts = cl.get("venue_counts")
                break
        if counts:
            tot = sum(counts.values())
            fy = {v: y for v, y in fy.items()
                  if counts.get(v, 0) >= max(3, 0.05 * tot)}
        if len(fy) < 2:
            continue
        order = sorted(fy.items(), key=lambda kv: kv[1])
        lead, lead_y = order[0]
        lag, lag_y = order[-1]
        if lag_y - lead_y >= 1:
            chain = " -> ".join(f"{v} {y}" for v, y in order)
            print(f"    #{r['id']:>3} {r['name'][:38]:<38} {chain}"
                  f"   (lead: {lead}, +{lag_y-lead_y}y)")
