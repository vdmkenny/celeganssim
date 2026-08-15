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

HOW PRECISE THE ANSWER IS, which matters more than the answer. This script
used to print a single AUC from a 20,000-pair Monte Carlo, and that number
was quoted as though a network change could be judged by whether it moved.
It cannot. Four measurements bound the resolution:

  * the Monte Carlo estimator itself missed the exact value by 0.003-0.005
  * the injection current, an arbitrary knob, moves the SAME model from
    0.6121 at 5 pA to 0.6279 at 45 pA
  * a bootstrap over the 236 stimulated cells, which is the real unit of
    resampling rather than the pair, gives a standard error of 0.014
  * four separate model changes measured in one session moved it by 0.001
    to 0.013, all inside that

So the AUC is computed exactly here (Mann-Whitney U over every measured
pair, no sampling) and always printed with a bootstrap interval. A change
worth believing has to clear the interval, or be measured on something
else. `--paired` is the honest test for a model change, since resampling
the same stimulated cells in both arms cancels most of the spread.

Usage:
    .venv/bin/python scripts/propagation.py [--current 15] [--seconds 4]
    .venv/bin/python scripts/propagation.py --controls
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Q_THRESHOLD = 0.05
ALIAS = {"AWCON": "AWCR", "AWCOF": "AWCL", "AWCOFF": "AWCL"}

# Injection currents the sweep reports. The middle one is what the recorded
# baseline uses; the outer two exist to show how much of any "improvement"
# is available for free from a knob that has nothing to do with the model.
SWEEP_CURRENTS_PA = (5.0, 15.0, 45.0)

# Resamples for the bootstrap over stimulated cells. 400 puts the Monte
# Carlo error on the interval well under the interval itself, and the whole
# bootstrap is a post-processing pass over an already-computed matrix.
N_BOOTSTRAP = 400

DT_MS = 1.0


def fast_step(ns):
    """`NervousSystem.step` with the two n^2 temporaries hoisted out.

    The stock step builds `Gs * (s * dep)[None, :]` and multiplies it by
    `E_syn` on every call, which is two 473x473 temporaries per millisecond
    of simulated time. Both collapse to matrix-vector products against
    matrices that do not change during a run:

        (Gs * sd[None, :]).sum(1)            == Gs  @ sd
        ((Gs * sd[None, :]) * E_syn).sum(1)  == GsE @ sd,  GsE = Gs * E_syn

    Same arithmetic, hoisted. Verified against the unpatched step over 4,000
    steps: max |dV| = 1.4e-14 mV, which is float64 summation order. Worth
    the duplication here because it takes a full 236-source run from about
    eight minutes to eighty-five seconds, and this script is the arbiter for
    network changes so it gets run repeatedly.

    Only the resting-network case is supported: no body, no noise, no
    intrinsic oscillator. Anything else must use the real step.
    """
    p = ns.p
    if ns.intrinsic:
        raise ValueError("fast_step does not implement the intrinsic block")
    Gs = ns.Gs_eff * ns.g_syn_row
    GsE = Gs * ns.E_syn
    Gg = ns.Gg_eff * p.g_gap
    Gg_sum = Gg.sum(axis=1)

    def step(dt: float, I_ext: np.ndarray) -> None:
        V, s = ns.V, ns.s
        sd = s * ns.dep
        G_tot = ns.G_leak + Gg_sum + Gs @ sd
        drive = ns.G_leak * ns.E_leak + Gg @ V + GsE @ sd + I_ext

        if ns.muscle_spikes and ns._muscle_idx.size:
            mi = ns._muscle_idx
            m, _, _ = ns._muscle_gates(V[mi])
            g_ca = p.g_Ca_muscle * m * ns.h_m
            g_k = p.g_K_muscle * ns.w_m
            G_tot[mi] += g_ca + g_k
            drive[mi] += g_ca * p.E_Ca_muscle + g_k * p.E_K

        G_tot = np.maximum(G_tot, 1e-12)
        V_inf = drive / G_tot
        tau_V = ns.C / G_tot
        ns.V = V_inf + (V - V_inf) * np.exp(-dt / tau_V)

        if ns.muscle_spikes and ns._muscle_idx.size:
            _, h_inf, w_inf = ns._muscle_gates(ns.V[ns._muscle_idx])
            ns.h_m = h_inf + (ns.h_m - h_inf) * np.exp(-dt / p.tau_h_muscle)
            ns.w_m = w_inf + (ns.w_m - w_inf) * np.exp(-dt / p.tau_K_muscle)

        phi = ns._sigmoid(p.beta * (ns.V - ns.V_th))
        if ns._trn_idx.size:
            t = ns._trn_idx
            excess = np.maximum(phi[t] - 0.5 - p.trn_dep_deadband, 0.0) * 2.0
            ns._trn_exc += (dt / p.trn_dep_onset_ms) * (excess - ns._trn_exc)
            ns.dep[t] += dt * ((1.0 - ns.dep[t]) / p.trn_dep_recovery_ms
                               - p.trn_dep_rate * ns._trn_exc * ns.dep[t])
            np.clip(ns.dep[t], 0.05, 1.0, out=ns.dep[t])
        rate = p.a_r * phi + p.a_d
        s_inf = p.a_r * phi / rate
        ns.s = np.clip(s_inf + (s - s_inf) * np.exp(-dt * rate), 0.0, 1.0)

        if ns._muscle_idx.size:
            drive_m = phi[ns._muscle_idx]
            k_r = 1.0 - np.exp(-dt / p.ca_rise_ms)
            k_d = 1.0 - np.exp(-dt / p.ca_decay_ms)
            ns.ca_stage += (drive_m - ns.ca_stage) * k_r
            ns.ca += (ns.ca_stage - ns.ca) * k_d

        if len(ns._ablated_idx):
            ns.V[ns._ablated_idx] = ns.V_th[ns._ablated_idx]
            ns.s[ns._ablated_idx] = 0.0

    return step


def model_propagation(current_pa: float, seconds: float, sources: list[int],
                      params=None):
    """Deflection of every cell when each source is driven, in mV."""
    from worm.connectome import Connectome
    from worm.genome import Genome
    from worm.nervous_system import NervousSystem

    conn = Connectome.load()
    ns = NervousSystem(conn, Genome.load(), params=params, seed=0)
    step = fast_step(ns)
    steps = int(seconds * 1000.0 / DT_MS)

    ns.reset()
    zero = np.zeros(conn.n)
    for _ in range(steps):
        step(DT_MS, zero)
    rest = ns.V.copy()

    out = np.zeros((conn.n, len(sources)))
    for k, j in enumerate(sources):
        ns.reset()
        I = np.zeros(conn.n)
        I[j] = current_pa
        for _ in range(steps):
            step(DT_MS, I)
        out[:, k] = ns.V - rest
    return conn, out


def load_atlas():
    import h5py
    import wormneuroatlas as wa

    path = Path(os.path.dirname(wa.__file__)) / "data" / "funatlas.h5"
    with h5py.File(path, "r") as f:
        ids = [n.decode() for n in f["neuron_ids"][:]]
        q = f["wt"]["q"][:]
        occ = f["wt"]["occ1"][:]
    return ids, q, occ


def auc_exact(hit: np.ndarray, fa: np.ndarray) -> float:
    """Mann-Whitney U over every pair, ties counted as half.

    The exact statistic, not a sample of it. The old estimator drew 20,000
    random pairs and landed 0.003 to 0.005 away, which is several times the
    size of the effects it was being asked to judge.
    """
    allv = np.concatenate([hit, fa])
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    ranks = np.empty(len(allv), float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_hit = ranks[:len(hit)].sum()
    return float((r_hit - len(hit) * (len(hit) + 1) / 2.0)
                 / (len(hit) * len(fa)))


def measured_pairs(prop, cols, atlas_to_ours, q, occ):
    """Model response, atlas verdict and source column for every pair."""
    resp = np.abs(prop)
    meas = np.isfinite(q) & (occ > 0)
    sig = meas & (q < Q_THRESHOLD)
    r, y, c = [], [], []
    for cj, kj in enumerate(cols):
        for ki, ri in atlas_to_ours.items():
            if ki == kj or not meas[ki, kj]:
                continue
            r.append(resp[ri, cj])
            y.append(bool(sig[ki, kj]))
            c.append(cj)
    return np.array(r), np.array(y, bool), np.array(c)


def bootstrap(r, y, c, n_cols, other=None, seed=0):
    """Resample STIMULATED CELLS, not pairs.

    A pair bootstrap treats 22,107 pairs as 22,107 independent observations,
    which they are not: all 236 responses to one stimulated cell share that
    cell's whole downstream network. Resampling the cell is the honest unit
    and gives an interval roughly twice as wide.

    With `other`, the same resampled columns score both arms and the paired
    difference is returned, which cancels most of the between-cell spread
    and is the right test for a model change.
    """
    rng = np.random.default_rng(seed)
    by_col = [np.where(c == k)[0] for k in range(n_cols)]
    a_l, b_l, d_l = [], [], []
    for _ in range(N_BOOTSTRAP):
        pick = rng.choice(n_cols, size=n_cols, replace=True)
        m = np.concatenate([by_col[k] for k in pick])
        if not y[m].any() or y[m].all():
            continue
        a = auc_exact(r[m][y[m]], r[m][~y[m]])
        a_l.append(a)
        if other is not None:
            b = auc_exact(other[m][y[m]], other[m][~y[m]])
            b_l.append(b)
            d_l.append(b - a)
    return np.array(a_l), np.array(b_l), np.array(d_l)


def report(r, y, label, c=None, n_cols=0):
    hit, fa = r[y], r[~y]
    auc = auc_exact(hit, fa)
    line = (f"{label:<34s} AUC {auc:.4f}")
    if c is not None:
        boots, _, _ = bootstrap(r, y, c, n_cols)
        line += (f"  +/- {boots.std():.4f}   95% CI "
                 f"[{np.quantile(boots, .025):.4f}, "
                 f"{np.quantile(boots, .975):.4f}]")
    print(line)
    return auc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=float, default=15.0, help="pA injected")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="stimulate only N cells")
    ap.add_argument("--sweep", action="store_true",
                    help="repeat at 5, 15 and 45 pA to show the knob's range")
    ap.add_argument("--controls", action="store_true",
                    help="also score gap-junctions-only and chemical-only")
    a = ap.parse_args()

    try:
        load_atlas()
    except ImportError as e:
        print(f"needs wormneuroatlas and h5py ({e})")
        return 1
    ids, q, occ = load_atlas()

    from worm.connectome import Connectome
    from worm.nervous_system import NeuralParams

    conn0 = Connectome.load()
    atlas_to_ours = {k: conn0.index[ALIAS.get(n, n)] for k, n in enumerate(ids)
                     if ALIAS.get(n, n) in conn0.index}
    cols = sorted(atlas_to_ours)
    if a.limit:
        # Recorded the hard way: 12 cells gave 0.541, close enough to chance
        # to read as a null. The effect is real but needs the full set.
        cols = cols[:a.limit]
        print(f"WARNING: --limit {a.limit} undersamples; the full set is "
              f"{len(atlas_to_ours)} cells")
    srcs = [atlas_to_ours[k] for k in cols]
    print(f"driving {len(cols)} cells at {a.current} pA for {a.seconds} s each")

    t0 = time.time()
    conn, prop = model_propagation(a.current, a.seconds, srcs)
    r, y, c = measured_pairs(prop, cols, atlas_to_ours, q, occ)
    if not y.any() or y.all():
        print("no overlapping measured pairs")
        return 1
    print(f"{int(y.sum())} measured-significant pairs, {int((~y).sum())} "
          f"silent, in {time.time() - t0:.0f}s\n")

    print(f"model response on significant pairs : "
          f"{np.median(r[y]):.4f} mV (median)")
    print(f"model response on silent pairs      : "
          f"{np.median(r[~y]):.4f} mV (median)\n")

    report(r, y, "baseline", c, len(cols))
    print("  the probability that the model responds more on a pair the "
          "animal\n  calls real than on one it calls silent, chance = 0.50")

    for frac in (0.01, 0.05, 0.10):
        thr = np.quantile(r, 1 - frac)
        tp = int((r[y] >= thr).sum())
        precision = tp / max(int((r >= thr).sum()), 1)
        print(f"  top {frac:.0%} of model responses: {precision:.1%} are "
              f"measured-significant (base rate {y.mean():.1%})")

    if a.sweep:
        print("\nthe same unmodified model at three injection currents, to "
              "show\nhow much of any change is available from a knob that is "
              "not the model:")
        for cur in SWEEP_CURRENTS_PA:
            _, p2 = model_propagation(cur, a.seconds, srcs)
            r2, y2, _ = measured_pairs(p2, cols, atlas_to_ours, q, occ)
            report(r2, y2, f"  {cur:.0f} pA")

    if a.controls:
        # The two references that make the number interpretable: what the
        # gap-junction skeleton scores alone, and what chemical transmission
        # scores alone. Paired, because the question is a difference.
        print("\ncontrols, paired against the baseline over the same "
              "resampled cells:")
        for name, kw in (("gap junctions only", dict(g_syn=0.0, g_syn_nmj=0.0)),
                         ("chemical only", dict(g_gap=0.0))):
            p2 = NeuralParams()
            for k, v in kw.items():
                setattr(p2, k, v)
            _, pr = model_propagation(a.current, a.seconds, srcs, params=p2)
            r2, _, _ = measured_pairs(pr, cols, atlas_to_ours, q, occ)
            base, arm, d = bootstrap(r, y, c, len(cols), other=r2)
            print(f"  {name:<22s} AUC {auc_exact(r2[y], r2[~y]):.4f}   "
                  f"delta {d.mean():+.4f} +/- {d.std():.4f}, 95% CI "
                  f"[{np.quantile(d, .025):+.4f}, "
                  f"{np.quantile(d, .975):+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
