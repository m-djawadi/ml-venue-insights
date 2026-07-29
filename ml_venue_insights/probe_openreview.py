#!/usr/bin/env python3
"""One-shot probe: logs in and tries several query forms for ICLR 2024,
printing exactly what OpenReview returns for each. This tells us which query
shape actually yields papers when authenticated.

Run (with your creds set):
    $env:OPENREVIEW_USERNAME = "you@email.com"
    $env:OPENREVIEW_PASSWORD = "yourpassword"
    python probe_openreview.py
"""
import os, sys, json
try:
    import requests
except ImportError:
    print("activate your venv first"); sys.exit(1)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
s = requests.Session()
s.headers.update({"User-Agent": UA, "Accept": "application/json"})

user = os.environ.get("OPENREVIEW_USERNAME")
pw = os.environ.get("OPENREVIEW_PASSWORD")
if not (user and pw):
    print("Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD first."); sys.exit(1)

print(f"Logging in as {user} ...")
r = s.post("https://api2.openreview.net/login", json={"id": user, "password": pw}, timeout=30)
print("  login status:", r.status_code)
if r.status_code == 200 and r.json().get("token"):
    s.headers["Authorization"] = "Bearer " + r.json()["token"]
    print("  token acquired.")
else:
    print("  login body:", str(r.text)[:300]); sys.exit(2)

VID = "ICLR.cc/2024/Conference"
attempts = [
    ("content.venueid + api2", "https://api2.openreview.net/notes",
     {"content.venueid": VID, "limit": 3}),
    ("venueid (plain) + api2", "https://api2.openreview.net/notes",
     {"venueid": VID, "limit": 3}),
    ("content.venue + api2", "https://api2.openreview.net/notes",
     {"content.venue": "ICLR 2024 poster", "limit": 3}),
    ("domain + api2", "https://api2.openreview.net/notes",
     {"domain": VID, "limit": 3}),
    ("content.venueid + api1", "https://api.openreview.net/notes",
     {"content.venueid": VID, "limit": 3}),
]
for name, url, params in attempts:
    try:
        rr = s.get(url, params=params, timeout=30)
        body = rr.json() if rr.headers.get("content-type","").startswith("application/json") else {}
        n = len(body.get("notes", []))
        cnt = body.get("count")
        print(f"\n[{name}]  http={rr.status_code}  count={cnt}  notes={n}")
        if n:
            t = body["notes"][0].get("content", {}).get("title")
            print(f"    FIRST TITLE: {t}")
            print(f"    >>> THIS QUERY WORKS <<<")
    except Exception as e:
        print(f"\n[{name}]  ERROR: {e}")

print("\nWhichever line shows notes>0 is the fix. Paste this whole output.")
