"""Download the primary datasets this simulator is built on.

Nothing here is redistributed with the repository -- each file is fetched from
whoever published it, so the provenance stays intact and the licences stay with
their owners.

    python scripts/fetch_data.py            # download, then build
    python scripts/fetch_data.py --no-build # download only
    python scripts/fetch_data.py --force    # re-download even if present

Total download is roughly 43 MB, dominated by the genome assembly.
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

NCBI = ("https://ftp.ncbi.nlm.nih.gov/genomes/refseq/invertebrate/"
        "Caenorhabditis_elegans/latest_assembly_versions/GCF_000002985.6_WBcel235")
CTB = "https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data"
C302 = "https://raw.githubusercontent.com/openworm/c302/master/c302/data"

# (local filename, url, description, gzip?)
FILES: list[tuple[str, str, str, bool]] = [
    ("celegans_genome.fna.gz",
     f"{NCBI}/GCF_000002985.6_WBcel235_genomic.fna.gz",
     "WBcel235 genome assembly (NCBI RefSeq GCF_000002985.6)", True),
    ("celegans_annotation.gff.gz",
     f"{NCBI}/GCF_000002985.6_WBcel235_genomic.gff.gz",
     "matching RefSeq gene annotation", True),
    ("celegans_features.txt.gz",
     f"{NCBI}/GCF_000002985.6_WBcel235_feature_table.txt.gz",
     "RefSeq feature table", True),
    ("herm_full_edgelist.csv",
     f"{CTB}/herm_full_edgelist.csv",
     "Cook et al. 2019 hermaphrodite connectome (via OpenWorm)", False),
    ("all_cell_info.csv",
     f"{CTB}/all_cell_info.csv",
     "WormAtlas cell classifications", False),
    ("IndividualNeurons.csv",
     f"{CTB}/IndividualNeurons.csv",
     "WormAtlas neuron names and lineages", False),
    ("aconnectome_white_1986_whole.csv",
     f"{CTB}/aconnectome_white_1986_whole.csv",
     "White et al. 1986 connectome (reference/comparison)", False),
    ("owmeta_cache.json",
     f"{C302}/owmeta_cache.json",
     "OpenWorm owmeta neuron and muscle metadata", False),
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def verify(path: Path, is_gzip: bool) -> bool:
    """A truncated download is worse than a missing one -- check before trusting."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    if is_gzip:
        try:
            with gzip.open(path, "rb") as fh:
                fh.read(4096)
        except (OSError, EOFError, gzip.BadGzipFile):
            return False
    return True


def download(name: str, url: str, desc: str, is_gzip: bool, force: bool) -> bool:
    dest = RAW / name
    if not force and verify(dest, is_gzip):
        print(f"  [have] {name}  ({human(dest.stat().st_size)})")
        return True
    print(f"  [get ] {name}  <- {desc}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "celeganssim/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while chunk := r.read(1 << 16):
                out.write(chunk)
                got += len(chunk)
                if total:
                    pct = got * 100 // total
                    print(f"\r         {pct:3d}%  {human(got)} / {human(total)}",
                          end="", flush=True)
            if total:
                print()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"\n         FAILED: {exc}")
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(dest)
    if not verify(dest, is_gzip):
        print(f"         FAILED: {name} downloaded but is corrupt")
        dest.unlink(missing_ok=True)
        return False
    print(f"         ok  ({human(dest.stat().st_size)})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--no-build", action="store_true",
                    help="skip running scripts/build_data.py afterwards")
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Fetching primary datasets into {RAW}\n")
    ok = [download(*f, a.force) for f in FILES]
    n_ok = sum(ok)
    print(f"\n{n_ok}/{len(FILES)} files ready")
    if n_ok < len(FILES):
        print("Some downloads failed. Re-run to retry just the missing ones.")
        return 1

    if a.no_build:
        print("Skipping build. Run scripts/build_data.py when ready.")
        return 0

    print("\nBuilding processed data ...\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_data.py")])
    if r.returncode != 0:
        print("Build failed.")
        return r.returncode
    print("\nReady. Try:  python -m worm serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
