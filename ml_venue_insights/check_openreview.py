#!/usr/bin/env python3
"""Diagnose OpenReview (ICLR/RLC) access and test your login.

Run:  python check_openreview.py

OpenReview now hides its API behind Cloudflare, so scripted access needs a free
login. This script tells you whether anonymous access works, and if you've set
OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD, whether your login works.
"""
import os
import sys

NOTES = "https://api2.openreview.net/notes"
LOGIN = "https://api2.openreview.net/login"
PARAMS = {"content.venueid": "ICLR.cc/2024/Conference", "limit": 3}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

try:
    import requests
except ImportError:
    print("requests not installed — activate your venv first.")
    sys.exit(1)

s = requests.Session()
s.headers.update({"User-Agent": UA, "Accept": "application/json"})

user = os.environ.get("OPENREVIEW_USERNAME")
pw = os.environ.get("OPENREVIEW_PASSWORD")

print("=" * 60)
print(" OpenReview access check")
print("=" * 60)

# 1. Try login if creds are present
if user and pw:
    print(f"\n[1] Logging in as {user} ...")
    try:
        r = s.post(LOGIN, json={"id": user, "password": pw}, timeout=30)
        if r.status_code == 200 and r.json().get("token"):
            s.headers["Authorization"] = "Bearer " + r.json()["token"]
            print("    OK — logged in, token acquired.")
        else:
            detail = ""
            try:
                b = r.json(); detail = (b.get("message") or b.get("name") or str(b))[:300]
            except Exception:
                detail = r.text[:300]
            print(f"    FAIL — login returned HTTP {r.status_code}: {detail}")
            if r.status_code == 400:
                print("    A 400 usually means the account isn't ACTIVATED yet.")
                print("    Check your email for an OpenReview activation link,")
                print("    click it, wait a few minutes, then re-run this check.")
            elif r.status_code in (401, 403):
                print("    Wrong EMAIL or password (the login field is your email).")
                print("    Reset at https://openreview.net if unsure.")
            sys.exit(2)
    except requests.exceptions.RequestException as e:
        print(f"    Could not reach login endpoint: {e}")
        print("    Likely a corporate proxy/VPN. Try a home network.")
        sys.exit(2)
else:
    print("\n[1] No OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD set — testing")
    print("    anonymous access only. If it fails, set credentials (below).")

# 2. Try fetching notes
print("\n[2] Fetching 3 ICLR 2024 papers ...")
try:
    r = s.get(NOTES, params=PARAMS, timeout=30)
except requests.exceptions.RequestException as e:
    print(f"    Could not connect: {e}")
    print("    Likely a corporate proxy/VPN blocking the host.")
    sys.exit(2)

if r.status_code == 200:
    try:
        n = len(r.json().get("notes", []))
        print(f"    OK — got {n} papers.")
        print("\n" + "=" * 60)
        print(" SUCCESS — you can scrape ICLR/RLC now:")
        print("   python run.py --venues iclr,rlc --scrape --start 2020 --end 2025")
        print("=" * 60)
        sys.exit(0)
    except Exception:
        print("    Got 200 but body wasn't JSON — a proxy login page.")
        sys.exit(2)
elif r.status_code in (401, 403):
    if user and pw:
        print(f"    Still blocked ({r.status_code}) even logged in — unusual.")
        print("    Could be a proxy stripping the auth header. Try home network.")
    else:
        print(f"    Blocked ({r.status_code}) — Cloudflare wants a login.")
        print("\n  FIX (one-time, ~2 min):")
        print("   1. Free account: https://openreview.net  (Sign Up)")
        print("   2. In PowerShell, set and re-run this check:")
        print('        $env:OPENREVIEW_USERNAME = "you@email.com"')
        print('        $env:OPENREVIEW_PASSWORD = "yourpassword"')
        print("        python check_openreview.py")
    sys.exit(2)
else:
    print(f"    Unexpected HTTP {r.status_code}.")
    sys.exit(2)
