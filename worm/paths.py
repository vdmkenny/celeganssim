"""Where the data and the viewer live.

The package has to work in three situations, and hardcoding one path breaks the
other two:

  * a git checkout, where data/ and viewer/ sit beside the worm/ package
  * an installed package, where they are bundled inside it
  * a user pointing somewhere else entirely, via CELEGANSSIM_DATA

Resolution is lazy and cached, and a failure explains how to fix it rather than
raising FileNotFoundError from somewhere deep in a JSON load.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent


def _candidates(kind: str) -> list[Path]:
    env = os.environ.get("CELEGANSSIM_DATA")
    out: list[Path] = []
    if env:
        out.append(Path(env).expanduser() / kind)
    out += [
        _REPO / kind,        # git checkout
        _PKG / kind,         # bundled inside the installed package
        Path.cwd() / kind,   # run from a directory that has it
    ]
    return out


@lru_cache(maxsize=4)
def data_dir() -> Path:
    """Directory holding the processed JSON datasets."""
    for base in _candidates("data"):
        p = base / "processed"
        if (p / "connectome.json").exists():
            return p
    tried = "\n  ".join(str(b / "processed") for b in _candidates("data"))
    raise FileNotFoundError(
        "Could not find the processed datasets. Looked in:\n  " + tried
        + "\n\nRun `python scripts/fetch_data.py` from a checkout to download and "
          "build them, or set CELEGANSSIM_DATA to the directory containing "
          "data/processed."
    )


@lru_cache(maxsize=4)
def viewer_html() -> Path:
    for base in (_REPO, _PKG, Path.cwd()):
        p = base / "viewer" / "index.html"
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find viewer/index.html. Run the viewer from a checkout, or "
        "set CELEGANSSIM_DATA to a directory containing viewer/."
    )
