.PHONY: help install install-analyze install-all check smoke \
        scrape-fast scrape-all stats compare analyze theme-mix llm briefs clean

VENUES ?= all
START   ?= 2020
END     ?= 2025
DATA    ?= data
CLI      = ml-venue-insights

help:
	@echo "ML Venue Insights — NeurIPS / ICML / ICLR / RLC / COLT"
	@echo
	@echo "  make install          core only (scrape + stats + compare)"
	@echo "  make install-analyze  + embeddings/clustering  <- recommended"
	@echo "  make install-all      everything incl. BERTopic"
	@echo
	@echo "  make check            offline self-test — run this first"
	@echo "  make smoke            tiny end-to-end run (~50 papers, needs network)"
	@echo
	@echo "  make scrape-fast      ICLR+RLC only (API, seconds)"
	@echo "  make scrape-all       all five venues (HTML venues are slow!)"
	@echo "  make stats compare analyze theme-mix llm briefs"
	@echo
	@echo "  Override: VENUES=icml,colt START=2021 END=2024 make analyze"

install:
	pip install -e .

install-analyze:
	pip install -e ".[analyze]"

install-all:
	pip install -e ".[all]"

# Verifies imports, CLI wiring and the venue registry. No network needed.
check:
	python selftest.py

# Tiny end-to-end check: 2 venues, 1 year, 25 papers each.
smoke:
	$(CLI) --venues iclr,colt --scrape --start 2024 --end 2024 \
		--limit-per-year 25 --data-dir $(DATA)
	$(CLI) --venues iclr,colt --stats --compare --data-dir $(DATA)

# ICLR + RLC are APIs — fast. Good first real scrape.
scrape-fast:
	$(CLI) --venues iclr,rlc --scrape --start $(START) --end $(END) --data-dir $(DATA)

# NeurIPS/ICML/COLT are rate-limited HTML: this is an overnight job.
scrape-all:
	$(CLI) --venues $(VENUES) --scrape --start $(START) --end $(END) --data-dir $(DATA)

stats:
	$(CLI) --venues $(VENUES) --stats --data-dir $(DATA)

compare:
	$(CLI) --venues $(VENUES) --compare --data-dir $(DATA)

analyze:
	$(CLI) --venues $(VENUES) --analyze --data-dir $(DATA)

theme-mix:
	$(CLI) --venues $(VENUES) --theme-mix --data-dir $(DATA)

llm:
	$(CLI) --venues $(VENUES) --llm --data-dir $(DATA)

briefs:
	$(CLI) --venues $(VENUES) --briefs --data-dir $(DATA)

clean:
	rm -rf data __pycache__ */__pycache__ *.egg-info build dist
