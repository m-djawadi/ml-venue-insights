#!/usr/bin/env python3
"""Diagnose why OpenReview (ICLR/RLC) requests fail.

Run:  python check_openreview.py

Tells you in a few seconds whether the problem is your network (corporate
proxy / VPN) or OpenReview itself, and what to do about it.
"""
import sys

URL = "https://api2.openreview.net/notes"
PARAMS = {"content.venueid": "ICLR.cc/2024/Conference", "limit": 3}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

print("Checking OpenReview reachability ...\n")
try:
    import requests
except ImportError:
    print("requests not installed — activate your venv first.")
    sys.exit(1)

try:
    r = requests.get(URL, params=PARAMS, headers={"User-Agent": UA,
                     "Accept": "application/json"}, timeout=20)
except requests.exceptions.SSLError as e:
    print("RESULT: SSL error — a corporate proxy is intercepting HTTPS.")
    print("        This is a network block, not a tool bug.")
    print(f"        ({e})")
    print("\n  Fix: disconnect from the work VPN, or run from a home network.")
    sys.exit(2)
except requests.exceptions.RequestException as e:
    print(f"RESULT: could not connect ({type(e).__name__}).")
    print("        Likely a firewall/proxy blocking the host.")
    print(f"        ({e})")
    sys.exit(2)

print(f"HTTP status: {r.status_code}")
srv = r.headers.get("Server", "")
if r.status_code == 200:
    try:
        n = len(r.json().get("notes", []))
        print(f"RESULT: WORKS — got {n} papers back.")
        print("        Re-run your scrape; ICLR/RLC will download fine now.")
        sys.exit(0)
    except Exception:
        print("RESULT: got 200 but the body wasn't JSON — likely a proxy login")
        print("        page. That's a corporate network intercept.")
        sys.exit(2)
elif r.status_code == 403:
    body = r.text[:200].lower()
    if "allowlist" in body or "egress" in body:
        print("RESULT: your network's egress filter is blocking the host.")
    else:
        print("RESULT: 403 Forbidden — a proxy/firewall is blocking the API,")
        print("        or OpenReview is refusing this network.")
    print("\n  Fixes, in order of ease:")
    print("   1. Disconnect from the Fasken VPN and re-run this check.")
    print("   2. Try from a personal/home network.")
    print("   3. Open this in your browser — if it shows JSON, it's the proxy:")
    print(f"      {URL}?content.venueid=ICLR.cc%2F2024%2FConference&limit=3")
    sys.exit(2)
elif r.status_code == 429:
    print("RESULT: rate-limited (429). Wait a minute and re-run the scrape —")
    print("        the tool now retries with backoff automatically.")
    sys.exit(0)
else:
    print(f"RESULT: unexpected status {r.status_code}. Server: {srv}")
    print(f"  Body: {r.text[:200]}")
    sys.exit(2)
