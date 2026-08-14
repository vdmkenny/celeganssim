"""Does this model propagate signals where the animal does?

Randi et al. 2023 measured it: stimulate one neuron optogenetically, record
the whole brain, and record which pairs show a response. Every check in the
suite measures behaviour, which is several layers downstream; this measures
the layer in between, where a connectome-based model is most likely to be
wrong and least likely to be caught.

The comparison the earlier scout (scripts/functional_atlas.py) could not
make is this one. Asking whether a measured functional pair has a DIRECT
anatomical edge is unfair to any network model, because the model reaches
most pairs through intermediate cells. So the model is asked the same
question the experiment asked: drive one cell, see who moves.

Method, deliberately close to the experiment:

  * hold the network at its resting equilibrium, no body, no noise
  * inject a sustained depolarising current into one cell
  * integrate to steady state and record every cell's deflection from rest
  * repeat for each cell the atlas also stimulated

That gives a model propagation matrix in the atlas's own convention,
[i, j] = response of i when j is driven. What is then compared is not the
absolute size of the response, which depends on an arbitrary injection
current, but WHICH pairs respond: the model's ranking against the measured
significant set.

Usage:
    .venv/bin/python scripts/propagation.py [--current 15] [--seconds 4]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Q_THRESHOLD = 0.05
ALIAS = {"AWCON": "AWCR", "AWCOF": "AWCL", "AWCOFF": "AWCL"}


def model_propagation(current_pa: float, seconds: float, sources: list[int]):
    """Deflection of every cell when each source is driven, in mV."""
    import numpy as np

    from worm.connectome import Connectome
    from worm.genome import Genome
    from worm.nervous_system import NervousSystem

    conn = Connectome.load()
    ns = NervousSystem(conn, Genome.load(), seed=0)
    dt_ms = 1.0
    steps = int(seconds * 1000.0 / dt_ms)

    ns.reset()
    zero = np.zeros(conn.n)
    for _ in range(steps):
        ns.step(dt_ms, zero, noise=0.0)
    rest = ns.V.copy()

    out = np.zeros((conn.n, len(sources)))
    for k, j in enumerate(sources):
        ns.reset()
        I = np.zeros(conn.n)
        I[j] = current_pa
        for _ in range(steps):
            ns.step(dt_ms, I, noise=0.0)
        out[:, k] = ns.V - rest
    return conn, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=float, default=15.0, help="pA injected")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="stimulate only N cells")
    a = ap.parse_args()

    try:
        import h5py
        import numpy as np
        import wormneuroatlas as wa
    except ImportError as e:
        print(f"needs wormneuroatlas and h5py ({e})")
        return 1

    path = Path(os.path.dirname(wa.__file__)) / "data" / "funatlas.h5"
    with h5py.File(path, "r") as f:
        ids = [n.decode() for n in f["neuron_ids"][:]]
        q = f["wt"]["q"][:]
        occ = f["wt"]["occ1"][:]

    from worm.connectome import Connectome
    conn0 = Connectome.load()
    atlas_to_ours = {k: conn0.index[ALIAS.get(n, n)] for k, n in enumerate(ids)
                     if ALIAS.get(n, n) in conn0.index}
    cols = sorted(atlas_to_ours)
    if a.limit:
        cols = cols[:a.limit]
    print(f"driving {len(cols)} cells at {a.current} pA for {a.seconds} s each")

    conn, prop = model_propagation(a.current, a.seconds,
                                   [atlas_to_ours[k] for k in cols])
    resp = np.abs(prop)

    meas = np.isfinite(q) & (occ > 0)
    sig = meas & (q < Q_THRESHOLD)

    hit, miss, fa, tn = [], [], [], []
    for cj, kj in enumerate(cols):
        for ki, ri in atlas_to_ours.items():
            if ki == kj or not meas[ki, kj]:
                continue
            r = resp[ri, cj]
            (hit if sig[ki, kj] else fa).append(r)
    hit = np.array(hit)
    fa = np.array(fa)
    if not len(hit) or not len(fa):
        print("no overlapping measured pairs")
        return 1

    # Does the model respond MORE on pairs the animal says are real? A
    # rank comparison, since absolute size depends on the injected current.
    from itertools import islice
    import random
    random.seed(0)
    pairs = [(random.choice(hit), random.choice(fa)) for _ in range(20000)]
    auc = sum(1.0 if h > f else 0.5 if h == f else 0.0 for h, f in pairs) / len(pairs)

    print(f"\nmeasured-significant pairs : {len(hit)}, model response "
          f"median {np.median(hit):.4f} mV")
    print(f"measured-silent pairs      : {len(fa)}, model response "
          f"median {np.median(fa):.4f} mV")
    print(f"\nAUC (chance = 0.50)        : {auc:.3f}")
    print("  the probability that the model responds more on a pair the "
          "animal\n  calls real than on one it calls silent")

    for frac in (0.01, 0.05, 0.10):
        thr = np.quantile(np.concatenate([hit, fa]), 1 - frac)
        tp = int((hit >= thr).sum())
        precision = tp / max(int((hit >= thr).sum() + (fa >= thr).sum()), 1)
        base = len(hit) / (len(hit) + len(fa))
        print(f"  top {frac:.0%} of model responses: {precision:.1%} are "
              f"measured-significant (base rate {base:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
