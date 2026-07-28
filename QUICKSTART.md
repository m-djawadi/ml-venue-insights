# Quick Start (Windows / VS Code)

⚠️ Your project is inside OneDrive, whose folder name has spaces
("OneDrive - FASKEN MARTINEAU DUMOULIN LLP"). That breaks pip's short
`ml-venue-insights` command. **Use `python run.py ...` instead — it always
works.** Everything below uses it.

## 1. One-time setup

Open the project folder in VS Code, open a terminal (`` Ctrl+` ``), then:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[analyze]"
```

If activation errors with "running scripts is disabled", run this once, retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Your prompt should now start with `(.venv)`. You need `.venv\Scripts\activate`
every time you open a new terminal.

## 2. Check it works (no internet needed)

```powershell
python run.py --help
python selftest.py
```

`selftest.py` should print "ALL CHECKS PASSED".

## 3. First real run — ICLR is an API, so it's fast

```powershell
python run.py --venues iclr --scrape --start 2024 --end 2024 --limit-per-year 50
python run.py --venues iclr --stats
```

A `data\` folder appears with `iclr_corpus.jsonl` inside.

## 4. Analyze (writes data\iclr_analysis.json)

```powershell
python run.py --venues iclr --analyze
```

First analyze downloads an embedding model once; later runs are cached and fast.

## 5. All five conferences — one JSON each

```powershell
python run.py --venues all --scrape --start 2020 --end 2025 --limit-per-year 25
python run.py --venues all --analyze
```

Writes neurips_analysis.json, icml_analysis.json, iclr_analysis.json,
rlc_analysis.json, colt_analysis.json in data\.

Drop `--limit-per-year 25` for the full corpus — but NeurIPS/ICML/COLT are slow
HTML scrapes (an overnight job). ICLR and RLC are fast APIs.

## 6. Compare venues (no ML libraries needed)

```powershell
python run.py --venues all --compare
```

## 7. Name themes with a local LLM (optional)

Install Ollama from https://ollama.com, then:

```powershell
ollama pull llama3
python run.py --venues iclr --analyze --llm
```

---

## Want the short `ml-venue-insights` command to work?

The only real fix is to get the project out of OneDrive. Move the whole folder
to a simple path like `C:\Projects\ml-venue-insights`, then:

```powershell
cd C:\Projects\ml-venue-insights
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[analyze]"
ml-venue-insights --venues iclr --stats
```

This also stops OneDrive from syncing your multi-GB `.venv`, which it shouldn't.
