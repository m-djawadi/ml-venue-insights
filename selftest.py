#!/usr/bin/env python3
"""Offline self-test: proves the install works before you spend hours scraping.

Runs the whole pipeline on a small synthetic corpus with a stubbed embedder, so
it needs no network and no model download. If this passes, your install is good
and any later failure is a data/network issue, not a broken setup.

    python selftest.py

Exit code 0 = everything that can be checked offline works.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

# Make this script runnable even when the editable install is broken — which
# happens on Windows when the project lives in a path with spaces (e.g. inside
# "OneDrive - Company Name"). The package folder sits right next to this file,
# so putting this file's directory on sys.path lets `import ml_venue_insights`
# succeed regardless of whether `pip install -e .` wired up correctly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}\n          {type(e).__name__}: {e}")
        FAILED.append(name)


def main() -> int:
    print("=" * 62)
    print(" ML Venue Insights — offline self-test")
    print("=" * 62)

    # ---- 1. core imports -------------------------------------------------
    print("\n[1] Core install (no optional deps needed)")

    def _imports():
        import ml_venue_insights
        from ml_venue_insights import cli, config, corpus, venues
        from ml_venue_insights import scrape, scrape_pmlr, scrape_openreview
        from ml_venue_insights import scrape_all, compare, stats
        assert ml_venue_insights.__version__
    check("all core modules import", _imports)

    def _registry():
        from ml_venue_insights.venues import VENUES, parse_venue_list
        assert set(VENUES) == {"neurips", "icml", "iclr", "rlc", "colt"}, VENUES
        assert len(parse_venue_list("all")) == 5
        assert [v.key for v in parse_venue_list("icml,colt")] == ["icml", "colt"]
        for v in VENUES.values():
            assert v.available_years(), f"{v.key} has no years"
    check("venue registry (5 venues, parsing, year tables)", _registry)

    def _scrub():
        from ml_venue_insights.corpus import doc_text
        r = {"title": "Scaling Laws at ICML",
             "abstract": "Work from NeurIPS, ICLR, COLT and RLC on agents."}
        out = doc_text(r).lower()
        for bad in ("icml", "neurips", "iclr", "colt", "rlc"):
            assert bad not in out, f"venue token '{bad}' leaked: {out}"
        assert "scaling laws" in out and "agents" in out
    check("venue-token scrubbing (prevents cluster leakage)", _scrub)

    def _parsers():
        from bs4 import BeautifulSoup
        from ml_venue_insights import scrape_pmlr as P, scrape_openreview as O
        html = ('<div class="paper"><p class="title">A Title</p>'
                '<span class="authors">X Y</span><p class="links">'
                '<a href="/v267/a25a.html">abs</a></p></div>')
        soup = BeautifulSoup(html, "html.parser")
        div = soup.find("div", class_="paper")
        assert div.find("p", class_="title").get_text(strip=True) == "A Title"
        meta = ('<meta name="citation_abstract" content="'
                + "word " * 20 + '">')
        assert P._extract_abstract(BeautifulSoup(meta, "html.parser"))
        rec = O.note_to_record(
            {"id": "i1", "content": {"title": {"value": "T"},
                                     "abstract": {"value": "A"},
                                     "authors": {"value": ["N"]}}}, 2025)
        assert rec["title"] == "T" and rec["year"] == 2025
        rec1 = O.note_to_record(
            {"id": "i2", "content": {"title": "T1", "abstract": "A1",
                                     "authors": ["N"]}}, 2021)
        assert rec1["title"] == "T1", "API v1 fallback broken"
    check("PMLR + OpenReview parsers (v1 and v2 shapes)", _parsers)

    # ---- 2. pipeline on synthetic data -----------------------------------
    print("\n[2] Pipeline on synthetic data")
    tmp = tempfile.mkdtemp(prefix="mlvi_selftest_")
    try:
        import random
        random.seed(0)
        THEMES = {"rl": ["policy", "reward", "bellman", "exploration"],
                  "gen": ["diffusion", "generative", "latent", "sampler"],
                  "thy": ["regret", "minimax", "bound", "oracle"]}
        W = {"iclr": "gen", "colt": "thy", "rlc": "rl"}
        for v, main_t in W.items():
            with open(os.path.join(tmp, f"{v}_corpus.jsonl"), "w") as f:
                for y in (2023, 2024):
                    for i in range(40):
                        t = main_t if random.random() < 0.8 else \
                            random.choice(list(THEMES))
                        ws = random.sample(THEMES[t], 3)
                        f.write(json.dumps({
                            "title": " ".join(w.capitalize() for w in ws) + f" {i}",
                            "abstract": ("We propose a method for " + ws[0] +
                                         " using " + ws[1] + ". Results on " +
                                         ws[2] + " show consistent gains."),
                            "authors": ["A"], "url": f"http://x/{v}/{y}/{i}",
                            "year": y, "venue": v}) + "\n")

        def _pool():
            from ml_venue_insights.compare import build_pooled_corpus
            p = build_pooled_corpus(list(W), tmp, out_name="pool.jsonl")
            n = sum(1 for _ in open(p))
            assert n == 240, n
        check("pool corpora across venues", _pool)

        def _stats():
            import io, contextlib
            from ml_venue_insights.stats import run_stats
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_stats(os.path.join(tmp, "pool.jsonl"), top_n=5)
            assert "papers" in buf.getvalue()
        check("--stats", _stats)

        def _compare():
            import io, contextlib
            from ml_venue_insights.compare import run_compare
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_compare(os.path.join(tmp, "pool.jsonl"), top_n=5)
            out = buf.getvalue()
            assert "VENUE SIGNATURES" in out
            # each venue's planted vocabulary should surface
            assert "diffusion" in out or "generative" in out or "latent" in out
        check("--compare (cross-venue log-odds signatures)", _compare)

        # ---- 3. optional heavy path --------------------------------------
        print("\n[3] Deep analysis (needs the [analyze] extra)")
        try:
            import numpy, sklearn  # noqa
            have_ml = True
        except ImportError:
            have_ml = False

        if not have_ml:
            print("  SKIP  numpy/scikit-learn not installed")
            print('        install with:  pip install -e ".[analyze]"')
        else:
            def _analyze():
                import numpy as np
                from sklearn.feature_extraction.text import TfidfVectorizer
                import ml_venue_insights.embeddings as E
                import ml_venue_insights.analyze as A
                import io, contextlib

                def fake(cp, dd, model_name="auto", batch_size=64, force=False):
                    docs, yrs, ti, ur = E.load_corpus_arrays(cp)
                    X = TfidfVectorizer(max_features=64,
                                        stop_words="english"
                                        ).fit_transform(docs).toarray().astype(np.float32)
                    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
                    return X, docs, yrs, ti, ur

                orig = A.embed_corpus
                A.embed_corpus = fake
                try:
                    pool = os.path.join(tmp, "pool.jsonl")
                    ap = os.path.join(tmp, "analysis.json")
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        res = A.run_analyze(pool, tmp, ap, n_topics=3, reps=4)
                    assert res["clusters"], "no clusters"
                    assert "venues" in res, "pooled run must record venues"
                    assert all("venue_mix" in c for c in res["clusters"])
                    from ml_venue_insights.compare import run_theme_mix
                    buf2 = io.StringIO()
                    with contextlib.redirect_stdout(buf2):
                        run_theme_mix(ap, pool)
                    assert "THEME" in buf2.getvalue()
                finally:
                    A.embed_corpus = orig
            check("--analyze + --theme-mix (stubbed embedder)", _analyze)
            print("       (real runs download a sentence-transformers model)")

        # ---- 4. CLI wiring ----------------------------------------------
        print("\n[4] CLI wiring")

        def _cli():
            from ml_venue_insights.cli import main as cli_main
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(["--list-venues"])
            assert rc == 0 and "neurips" in buf.getvalue()
            # guard: a missing corpus prints guidance and continues (a missing
            # venue shouldn't crash a multi-venue run), so rc==0 with a hint.
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                rc2 = cli_main(["--venues", "icml", "--data-dir",
                                os.path.join(tmp, "nope"), "--stats"])
            assert rc2 == 0 and "--scrape" in buf2.getvalue(), buf2.getvalue()
        check("CLI entry point + missing-corpus guard", _cli)

        # ---- 5. per-venue separate JSON files ---------------------------
        print("\n[5] Separate JSON per conference")
        try:
            import numpy, sklearn  # noqa
            have_ml2 = True
        except ImportError:
            have_ml2 = False

        if not have_ml2:
            print("  SKIP  needs numpy/scikit-learn for --analyze")
        else:
            def _per_venue_json():
                import numpy as np, io, contextlib
                from sklearn.feature_extraction.text import TfidfVectorizer
                import ml_venue_insights.analyze as A
                import ml_venue_insights.embeddings as E
                from ml_venue_insights.cli import main as cli_main

                def fake(cp, dd, model_name="auto", batch_size=64, force=False):
                    docs, yrs, ti, ur = E.load_corpus_arrays(cp)
                    X = TfidfVectorizer(max_features=64, stop_words="english"
                                        ).fit_transform(docs).toarray().astype(np.float32)
                    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
                    return X, docs, yrs, ti, ur
                import ml_venue_insights.cli as C
                orig = A.embed_corpus
                orig_req = C._require
                A.embed_corpus = fake
                # The stub replaces the model, so we don't need the real
                # sentence-transformers install just to test analyze wiring.
                C._require = lambda *a, **k: True
                try:
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        rc = cli_main(["--venues", "iclr,colt,rlc",
                                       "--data-dir", tmp, "--analyze",
                                       "--n-topics", "3"])
                    assert rc == 0, "analyze exited nonzero: " + buf.getvalue()[-300:]
                    # each conference must have written its OWN file
                    for vk in ("iclr", "colt", "rlc"):
                        fp = os.path.join(tmp, f"{vk}_analysis.json")
                        assert os.path.exists(fp), f"missing {vk}_analysis.json"
                        with open(fp) as fh:
                            j = json.load(fh)
                        assert j["clusters"], f"{vk} analysis has no clusters"
                    # and there must be NO pooled/combined analysis for a
                    # per-venue --analyze run
                    combined = [f for f in os.listdir(tmp)
                                if f.endswith("_analysis.json")
                                and "_" in f.replace("_analysis.json", "")]
                    # names like 'colt_iclr_rlc_analysis.json' would indicate pooling
                    pooled_like = [f for f in combined
                                   if f.count("_") >= 2 and "analysis" in f
                                   and not f.split("_")[0] in ("iclr","colt","rlc",
                                                               "icml","neurips")]
                    assert not pooled_like, f"unexpected pooled file: {pooled_like}"
                finally:
                    A.embed_corpus = orig
                    C._require = orig_req
            check("--venues iclr,colt,rlc --analyze -> 3 separate json", _per_venue_json)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 62)
    if FAILED:
        print(f" {len(FAILED)} CHECK(S) FAILED: {', '.join(FAILED)}")
        print("=" * 62)
        return 1
    print(" ALL CHECKS PASSED — your install works.")
    print("\n Next: a real 2-minute scrape (ICLR is an API, so it's fast):")
    print("   ml-venue-insights --venues iclr --scrape --start 2024 --end 2024 \\")
    print("       --limit-per-year 50")
    print("   ml-venue-insights --venues iclr --stats")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
