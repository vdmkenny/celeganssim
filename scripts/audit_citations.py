"""Check every DOI this repository cites for peer review and retractions.

The model is only as trustworthy as the papers it is built on, and papers
change after publication. Crossref has held the Retraction Watch database
since 2023, so retractions, corrigenda and errata all surface there as
`updated-by` entries on the original work. This script walks every DOI in
the repository and reports three things:

  peer review    journal-article means it went through review; posted-content
                 means a preprint, which is a weaker basis for a claim and
                 should be labelled as such wherever it is cited
  corrections    anything with an `updated-by` entry, since a correction can
                 invalidate the specific figure a parameter came from even
                 when the paper as a whole stands
  reachability   a DOI that does not resolve is a citation nobody can check

It is a check, not a gate: a correction is a reason to read the notice and
say what it means where the citation is used, not a reason to drop a source.
See docs/citations.md for the findings and what was done about them.

Usage:
    .venv/bin/python scripts/audit_citations.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = "celeganssim-citation-audit/1.0 (+https://github.com/vdmkenny/celeganssim)"
API = "https://api.crossref.org/works/"

# DOIs may contain parentheses (the old Elsevier style, e.g. Byerly's
# 1976 Developmental Biology DOI), so the pattern has to allow them and then trim a trailing bracket
# that belongs to the surrounding markdown rather than the DOI.
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\},;\"']+")


def cited_dois() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    files = (list(ROOT.rglob("*.md")) + list((ROOT / "worm").rglob("*.py"))
             + list((ROOT / "scripts").rglob("*.py")))
    for f in files:
        if ".git" in f.parts:
            continue
        for m in DOI_RE.findall(f.read_text(errors="ignore")):
            d = m.rstrip(".,;")
            if d.endswith(")") and d.count("(") < d.count(")"):
                d = d[:-1]
            out.setdefault(d, set()).add(str(f.relative_to(ROOT)))
    return out


def lookup(doi: str) -> dict:
    url = API + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full result here")
    a = ap.parse_args()

    dois = cited_dois()
    rows = []
    for i, (doi, where) in enumerate(sorted(dois.items())):
        row = {"doi": doi, "cited_in": sorted(where)}
        try:
            m = lookup(doi)
            row.update(
                type=m.get("type", "?"),
                year=(m.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0],
                venue=(m.get("container-title") or [""])[0],
                title=(m.get("title") or [""])[0],
                updated_by=[{"type": u.get("type"), "doi": u.get("DOI")}
                            for u in m.get("updated-by", [])],
            )
        except urllib.error.HTTPError as e:
            row.update(type=f"unresolved (HTTP {e.code})", updated_by=[])
        except Exception as e:                      # noqa: BLE001
            row.update(type=f"unresolved ({type(e).__name__})", updated_by=[])
        rows.append(row)
        time.sleep(0.35)                            # be polite to Crossref
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(dois)}", file=sys.stderr, flush=True)

    reviewed = [r for r in rows if r["type"] == "journal-article"]
    preprints = [r for r in rows if r["type"] == "posted-content"]
    corrected = [r for r in rows if r.get("updated_by")]
    broken = [r for r in rows if str(r["type"]).startswith("unresolved")]

    print(f"{len(rows)} DOIs cited across the repository")
    print(f"  peer-reviewed journal articles : {len(reviewed)}")
    print(f"  preprints (not peer reviewed)  : {len(preprints)}")
    print(f"  carrying a correction notice   : {len(corrected)}")
    print(f"  unresolved                     : {len(broken)}")

    for label, group in (("PREPRINT", preprints), ("CORRECTED", corrected),
                         ("UNRESOLVED", broken)):
        for r in group:
            kinds = ", ".join(sorted({u["type"] for u in r.get("updated_by", [])}))
            print(f"\n[{label}] {r['doi']}  {r.get('venue', '')} "
                  f"{r.get('year', '')}")
            if kinds:
                print(f"    notices: {kinds}")
            print(f"    {r.get('title', '')[:96]}")
            print(f"    cited in: {', '.join(r['cited_in'])}")

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
        print(f"\nfull result -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
