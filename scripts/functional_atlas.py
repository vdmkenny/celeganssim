"""Compare the wiring diagram against measured functional influence.

Randi et al. 2023 (Nature 623:406) stimulated single neurons optogenetically
and recorded the whole brain, giving a matrix of which neuron actually
influences which. That is a different object from the connectome, and the
difference is the point: this model, like every connectome-based model,
assumes an anatomical edge means influence and no edge means none.

The atlas ships inside the `wormneuroatlas` package (the same dependency
`build_expression.py` uses for CeNGEN), as `funatlas.h5`. Its convention is
[i, j] = response of i when j is stimulated, per `NeuroAtlas.get_kernel`.

This is a scout, not a validation check, and the distinction matters. Asking
whether a measured functional pair has a DIRECT anatomical edge is unfair to
the model, because the model reaches those pairs by propagation through
intermediate cells. What the comparison establishes is the size of the gap
that propagation and the extrasynaptic layers have to explain. Turning it
into a real check means computing the model's own propagation matrix and
comparing that, which is issue #29.

Usage:
    .venv/bin/python scripts/functional_atlas.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Q_THRESHOLD = 0.05          # Randi et al.'s own significance threshold
# The atlas splits AWC into its ON and OFF states, which are the two AWC
# cells in a given animal; we carry them as left and right.
ALIAS = {"AWCON": "AWCR", "AWCOF": "AWCL", "AWCOFF": "AWCL"}


def main() -> int:
    try:
        import h5py
        import numpy as np
        import wormneuroatlas as wa
    except ImportError as e:
        print(f"needs wormneuroatlas and h5py ({e}).\n"
              "  pip install wormneuroatlas h5py")
        return 1

    from worm.connectome import Connectome

    path = Path(os.path.dirname(wa.__file__)) / "data" / "funatlas.h5"
    if not path.exists():
        print(f"no funatlas.h5 at {path}")
        return 1

    with h5py.File(path, "r") as f:
        ids = [n.decode() for n in f["neuron_ids"][:]]
        out = {}
        for strain in ("wt", "unc31"):
            out[strain] = {"q": f[strain]["q"][:], "occ": f[strain]["occ1"][:],
                           "dFF": f[strain]["dFF"][:]}

    c = Connectome.load()
    idx = {k: c.index[ALIAS.get(n, n)] for k, n in enumerate(ids)
           if ALIAS.get(n, n) in c.index}
    print(f"atlas covers {len(ids)} neurons, {len(idx)} of which this model has")

    mono = json.loads((ROOT / "data/processed/monoamines.json").read_text())
    pep = json.loads((ROOT / "data/processed/peptides.json").read_text())
    extras = ({(e["pre"], e["post"]) for e in mono["edges"]}
              | {(e["pre"], e["post"]) for e in pep["edges"]})

    for strain in ("wt", "unc31"):
        q, occ = out[strain]["q"], out[strain]["occ"]
        meas = np.isfinite(q) & (occ > 0)
        sig = meas & (q < Q_THRESHOLD)
        both = silent = unwired = neither = unwired_extra = 0
        for a in idx:
            for b in idx:
                if a == b or not meas[a, b]:
                    continue
                anat = bool(c.Gs[idx[a], idx[b]] > 0 or c.Gg[idx[a], idx[b]] > 0)
                func = bool(sig[a, b])
                both += anat and func
                silent += anat and not func
                neither += (not anat) and (not func)
                if func and not anat:
                    unwired += 1
                    if (c.names[idx[b]], c.names[idx[a]]) in extras:
                        unwired_extra += 1
        wired = both + silent
        functional = both + unwired
        print(f"\n--- {strain} ---")
        print(f"  measured pairs among modelled cells : {both+silent+unwired+neither}")
        print(f"  wired and functional                : {both} "
              f"({100*both/max(wired,1):.0f}% of wired pairs carry influence)")
        print(f"  wired but silent                    : {silent}")
        print(f"  functional with no direct edge      : {unwired} "
              f"({100*unwired/max(functional,1):.0f}% of functional pairs)")
        print(f"     of those, reachable extrasynaptically: {unwired_extra} "
              f"({100*unwired_extra/max(unwired,1):.0f}%)")
        print(f"  neither                             : {neither}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
