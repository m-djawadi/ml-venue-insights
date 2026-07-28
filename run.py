#!/usr/bin/env python3
"""Convenience launcher — always works, even when the editable install's
console command is broken by spaces in the path (e.g. a OneDrive folder).

Usage is identical to the CLI, just prefixed with `python run.py`:

    python run.py --list-venues
    python run.py --venues iclr --scrape --start 2024 --end 2024 --limit-per-year 50
    python run.py --venues iclr --stats
    python run.py --venues all --analyze

This puts the project folder on sys.path first, so `ml_venue_insights` imports
from the code sitting right here regardless of pip's install state.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ml_venue_insights.cli import main

if __name__ == "__main__":
    sys.exit(main())
