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
import hashlib
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
#
# The connectome the simulator runs on is cook_2020_adjacency.xlsx, Cook et
# al.'s corrected July 2020 release, read by scripts/xlsx.py. The 2019
# edgelist and the White 1986 file below are NOT used to build anything: they
# are kept because the corrected release has to be checkable against what it
# corrects, and because the White files are what a reader reaches for when
# asking whether another dataset covers the posterior body (it does not, see
# worm/connectome.py). scripts/build_data.py lists exactly what it reads.
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
     "Cook et al. 2019 edgelist: superseded, kept for comparison", False),
    ("all_cell_info.csv",
     f"{CTB}/all_cell_info.csv",
     "WormAtlas cell classifications", False),
    ("IndividualNeurons.csv",
     f"{CTB}/IndividualNeurons.csv",
     "WormAtlas neuron names and lineages", False),
    ("cook_2020_adjacency.xlsx",
     f"{CTB}/SI%205%20Connectome%20adjacency%20matrices,%20corrected%20July%202020.xlsx",
     "Cook et al. adjacency matrices, corrected July 2020 (THE connectome)", False),
    ("aconnectome_white_1986_whole.csv",
     f"{CTB}/aconnectome_white_1986_whole.csv",
     "White et al. 1986 connectome: comparison only, has no muscle edges", False),
    ("edgelist_MA.csv",
     f"{CTB}/edgelist_MA.csv",
     "Bentley et al. 2016 monoaminergic edges, one row per "
     "source, target, ligand, receptor. The aggregate matrix in the same "
     "repository sums the four monoamines into one weight, which cannot "
     "separate a dopamine edge from a serotonin one, so this is the file "
     "that gets built", False),
    ("edgelist_NP.csv",
     f"{CTB}/edgelist_NP.csv",
     "Bentley et al. 2016 neuropeptide edges, same four columns. Fetched "
     "and pinned but NOT yet built: the dense peptidergic layer is the "
     "second half of issue #13", False),
    ("owmeta_cache.json",
     f"{C302}/owmeta_cache.json",
     "OpenWorm owmeta neuron and muscle metadata", False),
]


# SHA256 of every raw file, pinned so upstream drift is a loud error rather
# than a silent change to every downstream result. The ConnectomeToolbox
# serves files from a moving branch and the NCBI path tracks
# latest_assembly_versions, so the same URL is not a version pin; these are.
# On mismatch the file is re-downloaded once, and a second mismatch stops the
# build: either upstream changed (investigate, then update the pin alongside
# whatever recalibration the new data needs) or the download is corrupt.
PINNED_SHA256: dict[str, str] = {
    "edgelist_MA.csv":
        "5a206a6743479f4ae11eb82456378051c5d9aeb56ea5c1f82b121671c7fa2aa3",
    "edgelist_NP.csv":
        "bec5f0c9526db1523ed73fb6a8b386de95d48a2fd5eae75a9a4889758102389b",
    "celegans_genome.fna.gz":
        "d62fb938c408acd0df3126aa38fd126d4a411f070e1a41ffa4f4a1056984cdd4",
    "celegans_annotation.gff.gz":
        "054c0970f5210aa1c580922561456fdce89c66b7b371df1e5b737dff8a94b549",
    "celegans_features.txt.gz":
        "e8924464982a5d626c97b2c0d0a21caad6ade32afc2e732f35095587a0d18e7c",
    "herm_full_edgelist.csv":
        "142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca",
    "all_cell_info.csv":
        "e467c065342cafe8be7df2b6d781756fe1cfbae5fe7d74352a36682dea6b5fc9",
    "cook_2020_adjacency.xlsx":
        "1f4fdbf84746b69b49a8da0816f52787860ce349b638dce37924ba80f90c70c9",
    "IndividualNeurons.csv":
        "b69c0e994270493535994a69a47210dda04319592697eec2f4a65a9df8886195",
    "aconnectome_white_1986_whole.csv":
        "c8aac78756b71f6337629951e5f4211448e85d148f6db9b367b2cd0450bb403a",
    "owmeta_cache.json":
        "064374a0fba18a6f0661a005032144ce91f46973722e1f07bccabaf2f56e6ed0",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    pinned = PINNED_SHA256.get(path.name)
    if pinned is not None and sha256_of(path) != pinned:
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
