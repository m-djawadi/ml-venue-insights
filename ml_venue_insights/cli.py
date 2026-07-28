"""Command-line interface for ML Venue Insights.

Analyzes NeurIPS, ICML, ICLR, RLC and COLT from titles + abstracts only.

Every stage takes `--venues`, which is either `all`, a single venue, or a
comma-separated list. Two modes follow from that:

  Per-venue stages (--stats, --topics, --analyze, --llm, --briefs, --search,
  --like) run ONCE PER VENUE and write that venue's own artifacts. So
  `--venues all --analyze` produces neurips_analysis.json, icml_analysis.json,
  iclr_analysis.json, rlc_analysis.json and colt_analysis.json side by side.

  Cross-venue stages (--compare, --theme-mix) compare venues against each
  other, so they build a pooled view on demand (<venues>_pooled_corpus.jsonl
  and <venues>_analysis.json) without touching the per-venue files.

Stages:
  --scrape     fetch titles + abstracts (resumable, per venue)
  --stats      streaming keyword frequencies + year-over-year trends
  --compare    cross-venue volume + distinctive-vocabulary report  [multi only]
  --topics     legacy topic modeling
  --analyze    embedding clusters, representative papers, cluster graph, trends
  --theme-mix  decompose pooled themes by venue + diffusion order   [multi only]
  --briefs     per-theme Methods/Novelty/Use-Cases technical briefs
  --llm        Ollama synthesis into named themes + narrative
  --search     semantic search;  --like  nearest-neighbour papers
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .config import OLLAMA_MODEL, TOPICS_FILE, POOLED_CORPUS
from .corpus import count_records
from .venues import (
    parse_venue_list, corpus_filename, describe_all, ALL_KEYS,
)


def build_parser() -> argparse.ArgumentParser:
    # prog=None lets argparse use the actual invoked name, so both console
    # scripts (ml-venue-insights and ml-venue-insights) report themselves
    # correctly in --help and --version.
    p = argparse.ArgumentParser(
        prog=None,
        description="Analyze ML conference papers from titles + abstracts only "
                    "(no PDFs). Venues: " + ", ".join(ALL_KEYS),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""known venues:
{describe_all()}

examples:
  # one venue, exactly like the original tool
  ml-venue-insights --venues icml --scrape --start 2020 --end 2025
  ml-venue-insights --venues icml --stats --analyze --llm

  # ICLR is an API, so it is fast
  ml-venue-insights --venues iclr --scrape --start 2020 --end 2025

  # all five venues -> one separate json per conference
  ml-venue-insights --venues all --scrape --start 2020 --end 2025
  ml-venue-insights --venues all --analyze          # writes 5 *_analysis.json
  ml-venue-insights --venues all --analyze --llm    # + names themes per venue

  # cross-venue reports (build a pooled view; per-venue json untouched)
  ml-venue-insights --venues all --compare
  ml-venue-insights --venues all --theme-mix

  # a targeted pair
  ml-venue-insights --venues icml,colt --compare --analyze --theme-mix

  # quick smoke test before committing to a full scrape
  ml-venue-insights --venues all --scrape --start 2023 --end 2024 --limit-per-year 25
""",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Venue selection
    p.add_argument("--venues", default="neurips",
                   help="'all', one venue, or comma list "
                        f"({'|'.join(ALL_KEYS)}). Default: neurips")
    p.add_argument("--list-venues", action="store_true",
                   help="print the venue registry and exit")

    # Stage flags
    p.add_argument("--scrape", action="store_true", help="fetch titles+abstracts")
    p.add_argument("--stats", action="store_true", help="keyword + trend stats (shallow)")
    p.add_argument("--compare", action="store_true",
                   help="cross-venue volume + distinctive vocabulary (multi-venue)")
    p.add_argument("--topics", action="store_true", help="legacy topic modeling")
    p.add_argument("--analyze", action="store_true",
                   help="DEEP: embedding clusters, representative papers, "
                        "cluster graph, conceptual trends")
    p.add_argument("--theme-mix", action="store_true",
                   help="decompose themes by venue + diffusion order "
                        "(multi-venue; builds its own pooled analysis)")
    p.add_argument("--llm", action="store_true",
                   help="Ollama synthesis (uses --analyze output if present, "
                        "else --topics output)")
    p.add_argument("--search", metavar="QUERY", default=None,
                   help="semantic search over the corpus")
    p.add_argument("--like", metavar="TITLE", default=None,
                   help="find papers similar to one matching this title")
    p.add_argument("--briefs", action="store_true",
                   help="per-theme Methods/Novelty/Use-Cases technical briefs")

    # Shared
    p.add_argument("--data-dir", default="data", help="where files live (default: data/)")
    p.add_argument("--model-name", default="auto",
                   help="embedding model: auto|fast|best or a HF id (default auto)")
    p.add_argument("--force-embed", action="store_true",
                   help="recompute embeddings, ignore cache")

    # Scrape options
    p.add_argument("--start", type=int, default=2020, help="start year (default 2020)")
    p.add_argument("--end", type=int, default=2025, help="end year (default 2025)")
    p.add_argument("--limit-per-year", type=int, default=None,
                   help="cap papers/year (for quick tests)")

    # Topic / analyze options
    p.add_argument("--n-topics", type=int, default=0,
                   help="target cluster count (0=auto)")
    p.add_argument("--no-bertopic", action="store_true",
                   help="legacy --topics: use embeddings+KMeans directly")
    p.add_argument("--reps", type=int, default=6,
                   help="representative papers per cluster (--analyze)")
    p.add_argument("--top-k", type=int, default=10,
                   help="results for --search / --like")

    # Stats options
    p.add_argument("--top-n", type=int, default=40, help="top terms to show")

    # LLM options
    p.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")

    # Briefs options
    p.add_argument("--brief-ids", default=None,
                   help="comma-separated cluster ids to brief (default: all)")
    p.add_argument("--brief-top", type=int, default=None,
                   help="brief only the N largest themes")

    return p


def _require(packages, flag, extra):
    """Fail with an actionable message instead of a raw ImportError traceback.

    The heavy ML stack is deliberately optional, so a core install hitting
    --analyze is an expected path, not a bug. Tell the user the one command
    that fixes it rather than dumping a stack trace.
    """
    import importlib
    missing = []
    for mod, pip_name in packages:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"\n{flag} needs packages that aren't installed: "
              f"{', '.join(missing)}")
        print(f"\n  Fix it with:\n      pip install -e \".[{extra}]\"\n")
        print(f"  (or: pip install {' '.join(missing)})")
        return False
    return True


def _artifacts(data_dir, venue_key):
    """Per-venue artifact paths. Each conference owns a separate set of files,
    so `--venues all --analyze` writes neurips_analysis.json, icml_analysis.json,
    ... side by side rather than one pooled blob."""
    return {
        "corpus":   os.path.join(data_dir, corpus_filename(venue_key)),
        "topics":   os.path.join(data_dir, f"{venue_key}_{TOPICS_FILE}"),
        "analysis": os.path.join(data_dir, f"{venue_key}_analysis.json"),
        "briefs":   os.path.join(data_dir, f"{venue_key}_theme_briefs.md"),
    }


def _pooled_paths(data_dir, venue_keys):
    """Paths for the cross-venue reports (--compare, --theme-mix). These need a
    combined view by definition, so they get one shared, clearly-named file."""
    slug = "_".join(sorted(venue_keys))
    return {
        "corpus":   os.path.join(data_dir, f"{slug}_{POOLED_CORPUS}"),
        "analysis": os.path.join(data_dir, f"{slug}_analysis.json"),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_venues:
        print("Known venues:\n" + describe_all())
        return 0

    stages = [args.scrape, args.stats, args.compare, args.topics, args.analyze,
              args.theme_mix, args.llm, args.search, args.like, args.briefs]
    if not any(stages):
        build_parser().print_help()
        return 1

    venues = parse_venue_list(args.venues)
    venue_keys = [v.key for v in venues]
    multi = len(venues) > 1

    os.makedirs(args.data_dir, exist_ok=True)

    multi = len(venues) > 1
    print(f"Venues: {', '.join(venue_keys)}")

    # ---------------------------- scrape ---------------------------------- #
    # Each venue scrapes into its OWN corpus file (<venue>_corpus.jsonl).
    if args.scrape:
        from .scrape_all import scrape_venue
        for v in venues:
            print(f"\n== SCRAPE {v.key.upper()} {args.start}-{args.end} ==")
            print(f"   backend: {v.backend}  |  {v.note}")
            scrape_venue(v, args.start, args.end, args.data_dir,
                         args.limit_per_year)

    # -------- dependency gates (checked once, before any per-venue loop) --- #
    if args.analyze and not _require(
            [("numpy", "numpy"), ("sklearn", "scikit-learn"),
             ("sentence_transformers", "sentence-transformers")],
            "--analyze", "analyze"):
        return 1
    if args.topics and not _require(
            [("sklearn", "scikit-learn"), ("numpy", "numpy"),
             ("sentence_transformers", "sentence-transformers")],
            "--topics", "topics"):
        return 1
    if (args.search or args.like) and not _require(
            [("numpy", "numpy"),
             ("sentence_transformers", "sentence-transformers")],
            "--search/--like", "analyze"):
        return 1

    # ==================================================================== #
    #  PER-VENUE STAGES                                                     #
    #  Every one of these runs once per selected venue and writes that      #
    #  venue's own artifacts. This is the core of "each conference has its   #
    #  own separate json file".                                             #
    # ==================================================================== #
    per_venue_stages = any([args.stats, args.topics, args.analyze, args.llm,
                            args.briefs, args.search, args.like])

    if per_venue_stages:
        for v in venues:
            art = _artifacts(args.data_dir, v.key)
            corpus_path = art["corpus"]
            n = count_records(corpus_path)

            print(f"\n{'#'*64}\n#  {v.key.upper()}  ({os.path.basename(corpus_path)})\n{'#'*64}")
            if n == 0:
                print(f"No papers found for {v.key}.\n"
                      f"  Scrape it first:\n"
                      f"    ml-venue-insights --venues {v.key} --scrape "
                      f"--start {args.start} --end {args.end}")
                continue
            print(f"{n} papers")

            if args.stats:
                from .stats import run_stats
                print(f"\n== STATS: {v.key} ==")
                run_stats(corpus_path, top_n=args.top_n)

            if args.topics:
                from .topics import run_topics
                print(f"\n== TOPICS (legacy): {v.key} ==")
                run_topics(corpus_path, args.n_topics or 20, art["topics"],
                           use_bertopic=not args.no_bertopic)

            if args.analyze:
                from .analyze import run_analyze
                print(f"\n== DEEP ANALYSIS: {v.key} ==")
                run_analyze(corpus_path, args.data_dir, art["analysis"],
                            model_name=args.model_name, n_topics=args.n_topics,
                            reps=args.reps, force_embed=args.force_embed)

            if args.search:
                from .analyze import run_search
                print(f"\n== SEMANTIC SEARCH: {v.key} ==")
                run_search(corpus_path, args.data_dir, args.search,
                           model_name=args.model_name, top_k=args.top_k)

            if args.like:
                from .analyze import run_neighbors
                print(f"\n== PAPERS LIKE THIS: {v.key} ==")
                run_neighbors(corpus_path, args.data_dir, args.like,
                              model_name=args.model_name, top_k=args.top_k)

            if args.briefs:
                if not os.path.exists(art["analysis"]):
                    print(f"\n(--briefs for {v.key}: no {os.path.basename(art['analysis'])} "
                          f"yet; run --analyze first. Skipping.)")
                else:
                    from .briefs import run_theme_briefs
                    print(f"\n== THEME BRIEFS: {v.key} ==")
                    ids = None
                    if args.brief_ids:
                        ids = [int(x) for x in args.brief_ids.split(",") if x.strip()]
                    run_theme_briefs(art["analysis"], corpus_path,
                                     model=args.model, only_ids=ids,
                                     top_n=args.brief_top, out_path=art["briefs"])

            if args.llm:
                print(f"\n== LLM SYNTHESIS: {v.key} ==")
                if os.path.exists(art["analysis"]):
                    from .llm_deep import run_llm_deep
                    run_llm_deep(art["analysis"], model=args.model)
                elif os.path.exists(art["topics"]):
                    from .llm import run_llm
                    print("(using legacy topics.json; run --analyze for richer synthesis)")
                    run_llm(art["topics"], model=args.model)
                else:
                    print(f"No analysis for {v.key}. Run --analyze (preferred) "
                          f"or --topics first.")

    # ==================================================================== #
    #  CROSS-VENUE STAGES                                                   #
    #  These compare venues against each other, so they build a pooled view #
    #  on demand. They never replace the per-venue json files above.        #
    # ==================================================================== #
    if args.compare or args.theme_mix:
        if not multi:
            print("\n(--compare / --theme-mix need 2+ venues; skipping. "
                  "Try --venues all or --venues icml,iclr)")
        else:
            pooled = _pooled_paths(args.data_dir, venue_keys)

            # Build the pooled corpus from the per-venue corpora.
            from .compare import build_pooled_corpus
            print(f"\n{'='*64}\n POOLING {len(venue_keys)} VENUES FOR CROSS-VENUE REPORTS\n{'='*64}")
            missing = [k for k in venue_keys
                       if count_records(os.path.join(args.data_dir,
                                                     corpus_filename(k))) == 0]
            if missing:
                print(f"  ! no corpus for: {', '.join(missing)} "
                      f"(scrape them for a complete comparison)")
            build_pooled_corpus(venue_keys, args.data_dir,
                                out_name=os.path.basename(pooled["corpus"]))
            if count_records(pooled["corpus"]) == 0:
                print("  nothing pooled; scrape some venues first.")
                return 1

            if args.compare:
                from .compare import run_compare
                print("\n== CROSS-VENUE COMPARISON ==")
                run_compare(pooled["corpus"], top_n=args.top_n // 2 or 20)

            if args.theme_mix:
                # theme-mix needs a pooled analysis.json (clusters spanning all
                # venues). Build it if absent, so the user doesn't have to know
                # to run a separate pooled --analyze.
                if not _require([("numpy", "numpy"), ("sklearn", "scikit-learn"),
                                 ("sentence_transformers", "sentence-transformers")],
                                "--theme-mix", "analyze"):
                    return 1
                if not os.path.exists(pooled["analysis"]) or args.analyze:
                    from .analyze import run_analyze
                    print("\n== POOLED ANALYSIS (for theme mix) ==")
                    run_analyze(pooled["corpus"], args.data_dir,
                                pooled["analysis"], model_name=args.model_name,
                                n_topics=args.n_topics, reps=args.reps,
                                force_embed=args.force_embed)
                from .compare import run_theme_mix
                print("\n== THEME x VENUE MIX ==")
                run_theme_mix(pooled["analysis"], pooled["corpus"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
