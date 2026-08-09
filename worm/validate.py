"""Check the simulator's behaviour against published phenotypes.

Two kinds of checks live here, reported separately because they are not the
same claim:

  * "behaviour" checks run the animal and measure something -- these are the
    validation suite proper.
  * "consistency" checks pin a model parameter or a data-derived invariant.
    They catch editing accidents; they do not validate the model against the
    animal.

A check can be registered as expected-to-fail (xfail) when the gap between
model and literature is known and tracked. XFAILs are reported, not hidden,
and a check that starts passing unexpectedly (XPASS) is flagged so the gap
issue can be closed. Tolerances are wide because the model is coarse: what
the behavioural checks catch is sign errors -- a mutant getting faster when
the animal gets slower, touch working when it should be abolished.
"""

from __future__ import annotations

import numpy as np

from .environment import Environment
from .simulation import SimConfig, WormSimulation


def _sim(knockouts=(), seed=0, **env_kw) -> WormSimulation:
    env = Environment(width=44.0, height=32.0, **env_kw)
    s = WormSimulation(env=env, config=SimConfig(seed=seed))
    for g in knockouts:
        s.knock_out(g)
    return s


def gait(knockouts=(), seconds=45.0, seed=0, settle=10.0) -> dict:
    """Run and measure speed, undulation amplitude and body length."""
    s = _sim(knockouts, seed=seed)
    n_settle = int(settle / s.cfg.dt)
    n = int(seconds / s.cfg.dt)
    for _ in range(n_settle):
        s.step()
    d0 = s.state.distance
    amps, lens, speeds = [], [], []
    for i in range(n):
        s.step()
        if i % 5 == 0:
            nodes = s.body.world_nodes()
            ax = nodes[-1] - nodes[0]
            L = np.linalg.norm(ax)
            if L > 1e-6:
                ax = ax / L
                perp = np.array([-ax[1], ax[0]])
                d = (nodes - nodes[0]) @ perp
                # Normalise by the animal's ACTUAL length -- adults keep growing
                # past 1 mm, so a fixed divisor understates the amplitude.
                amps.append(float(d.max() - d.min()) / max(s.body.body_length, 1e-6))
            lens.append(float(s.body.length_scale))
            speeds.append(float(s.body.speed))
    return {
        "speed": (s.state.distance - d0) / seconds,
        "inst_speed": float(np.mean(speeds)),
        "amplitude": float(np.mean(amps)) if amps else 0.0,
        "length_scale": float(np.mean(lens)),
        "reversals": s.state.reversal_count,
        "omegas": s.state.omega_count,
    }


def touch_response(region: str, knockouts=(), seed=0) -> dict:
    """Poke the animal and report whether it reversed and how fast it then moved."""
    s = _sim(knockouts, seed=seed)
    for _ in range(500):
        s.step()
    r0, d0 = s.state.reversal_count, s.state.distance
    s.env.poke(region, 1.0, duration=0.4)
    n = 150
    for _ in range(n):
        s.step()
    return {
        "reversed": s.state.reversal_count > r0,
        "n_reversals": s.state.reversal_count - r0,
        "speed_after": (s.state.distance - d0) / (n * s.cfg.dt),
    }


CHECKS = []


def check(name, expectation, source, section="behaviour", xfail=None):
    """Register a check.

    section: "behaviour" (runs the animal, measures) or "consistency" (pins a
        parameter or data invariant).
    xfail: reason string for a check expected to fail -- a known, tracked gap
        between the model and the literature. XPASS (unexpected pass) is
        reported so the gap can be closed.
    """
    def deco(fn):
        CHECKS.append((name, expectation, source, section, xfail, fn))
        return fn
    return deco


# --------------------------------------------------------------------------
@check("wild-type gait",
       "speed 0.15-0.40 mm/s, undulation amplitude 12-30% of body length",
       "Cronin et al. 2005 Genetics: N2 0.20+/-0.04 mm/s, amplitude 19.3% BL")
def _wt():
    g = gait()
    ok = 0.15 <= g["speed"] <= 0.40 and 0.12 <= g["amplitude"] <= 0.30
    return ok, f"speed {g['speed']:.3f} mm/s, amplitude {g['amplitude']*100:.1f}% BL"


@check("an imposed bend propagates posteriorly, as in the channel experiment",
       "holding a middle body region at an imposed curvature should bend the "
       "free region behind it in the same direction and in proportion, with a "
       "slope near 0.62. This is the experiment the proprioceptive coupling "
       "was measured with, run against the model",
       "Wen et al. 2012 Neuron 76:750 Fig 4C: posterior curvature rises "
       "linearly with imposed anterior curvature, slope 0.62 +/- 0.03; the "
       "channel's posterior limit sits at body position 0.7",
       xfail="the model propagates the bend in the right direction and "
             "linearly, at slope 0.364 against the measured 0.62. Giving "
             "muscle its calcium action potential and putting the "
             "neuromuscular conductance on the measured junction moved this "
             "from 0.117, so the remaining factor is under two. What is still "
             "missing is that muscles do not reach the -10 mV spike threshold "
             "from synaptic input alone: crossing it needs about 2.4 nS of "
             "extra excitatory conductance, close to the 2.26 nS the measured "
             "trigger current of 67.9 pA implies, but the release variable "
             "cannot exceed a_r/(a_r + a_d) so one junction spans only 1.83x "
             "from rest to saturation. Propagation here is therefore still "
             "subthreshold and graded rather than spike-mediated")
def _bend_propagation():
    from .body import N_SEG
    from .simulation import SimConfig
    clamp = slice(int(0.35 * N_SEG), int(0.70 * N_SEG))
    free = slice(int(0.75 * N_SEG), int(0.95 * N_SEG))
    xs, ys = [], []
    for k in (-8.0, -4.0, 4.0, 8.0):
        s = WormSimulation(config=SimConfig(seed=0, muscle_pacemaker_pa=0.0,
                                            propr_gain=30.0))
        n = int(10.0 / s.cfg.dt)
        got = []
        for i in range(n):
            s.body.curvature[clamp] = k       # the microfluidic channel
            s.step()
            s.body.curvature[clamp] = k
            if i > n * 0.6:
                got.append(float(s.body.curvature[free].mean()))
        xs.append(k)
        ys.append(float(np.mean(got)))
    slope = float(np.polyfit(xs, ys, 1)[0])
    return abs(slope - 0.62) <= 0.15, \
        f"posterior follows imposed curvature with slope {slope:.3f} (want 0.62)"


@check("ablating muscle bends the animal, in the mode where muscles are real",
       "killing the mid-body dorsal muscles must bend that region ventrally "
       "and cost amplitude, because the surviving ventral muscles pull "
       "unopposed. Only the paced mode can show this: in the default scripted "
       "mode the body is prescribed and ablating 14 muscles changes nothing, "
       "which is the measured reason muscle experiments belong in paced mode",
       "Sulston & White 1980 Dev Biol 78:577 (muscle ablation causes local "
       "paralysis and body-shape defects); paced-mode geometry as in the "
       "paced-gait check")
def _muscle_ablation():
    from .simulation import SimConfig
    from .kinematics import record
    got = {}
    for label, ablate in (("intact", ()), ("ablated", None)):
        s = WormSimulation(env=Environment(width=44.0, height=32.0),
                           config=SimConfig(seed=0))
        if ablate is None:
            for k in range(10, 17):
                s.ablate(f"dBWML{k}")
                s.ablate(f"dBWMR{k}")
        for _ in range(int(12.0 / s.cfg.dt)):
            s.step()
        rec = record(s, seconds=20.0, settle=0.0)
        got[label] = float(rec.curvature[:, 10:16].mean())
    ok = got["intact"] > 0.0 and got["ablated"] < -0.3
    return ok, (f"mid-body dorsoventral curvature bias: intact "
                f"{got['intact']:+.3f}, dorsal muscles ablated "
                f"{got['ablated']:+.3f} (ventral collapse)")


@check("the animal locomotes on connectome drive alone",
       "with the body driven by the muscle cells the connectome actually "
       "drives, rather than by the scripted oscillator, the animal should "
       "still crawl at a wild-type speed. This is the end-to-end version of "
       "the drive check: it measures displacement, not activation",
       "Cronin et al. 2005 Genetics: N2 0.20 +/- 0.04 mm/s",
       xfail="the animal is effectively stationary in this mode, covering "
             "0.005 mm in 40 s against 16.4 mm on the scripted oscillator, "
             "because the connectome supplies no undulatory rhythm to bend it "
             "with. The mechanical path is complete and shared with the "
             "scripted path, so what is missing is upstream: proprioceptive "
             "feedback to make motor neuron output oscillate (issue #10)")
def _emergent_locomotion():
    from .simulation import SimConfig
    from .environment import Environment
    env = Environment(width=44.0, height=32.0)
    s = WormSimulation(env=env, config=SimConfig(seed=0, muscle_pacemaker_pa=0.0))
    for _ in range(int(10.0 / s.cfg.dt)):
        s.step()
    d0 = s.state.distance
    seconds = 45.0
    for _ in range(int(seconds / s.cfg.dt)):
        s.step()
    speed = (s.state.distance - d0) / seconds
    return 0.15 <= speed <= 0.40, f"speed {speed:.4f} mm/s on connectome drive"


@check("muscle activation leads curvature by the measured phase",
       "peak muscle activation should sit about 45 degrees, an eighth of a "
       "cycle, ahead of peak midline curvature",
       "Butler et al. 2015 J R Soc Interface 12:20140963: ~45 deg raw and "
       "49.9 deg (95% CI 46.5-53.3) corrected for indicator kinetics, held "
       "across a threefold change in wavelength from crawling to swimming",
       xfail="the model gives 8 degrees against a measured 45. The phase here "
             "is set by one mechanical time constant, body.curvature_tau_s at "
             "60 ms, which at 0.47 Hz can only produce arctan(2*pi*f*tau) = 8 "
             "degrees. Raising that constant to 340 ms would hit 45 degrees at "
             "this frequency and then miss everywhere else, because a fixed "
             "time constant makes phase scale with frequency: it would read 33 "
             "degrees at the 0.30 Hz crawl and 75 at the 1.76 Hz swim. The "
             "measurement is that the phase barely moves across that range, so "
             "a constant phase cannot come from a constant delay. Activation "
             "timing has to scale with the cycle, which is what curvature "
             "feedback does and a clock does not, so this is a second "
             "acceptance test for the proprioceptive loop (issue #10)")
def _activation_phase():
    from .kinematics import analyse, record
    got = analyse(record(_sim(), seconds=60.0, settle=10.0))
    phase = got["phase_drive_to_curvature_deg"]
    return 30.0 <= phase <= 65.0, \
        (f"activation leads curvature by {phase:+.1f} deg (want ~45) at "
         f"{got['undulation_hz']:.2f} Hz")


@check("posterior muscle innervation is thinner than the animal's",
       "each body-wall muscle receives chemical input from about 10 neurons. "
       "This check records how far the posterior body falls below that in the "
       "dataset, because it is the region the undulatory wave has to travel "
       "through and the region the reconstruction is thinnest in",
       "Cook et al. 2019 Nature 571:63: 9.5 neurons per dorsal muscle (range "
       "5-12.5) and 10.5 per ventral (range 4-18.5). The same paper reports "
       "'a gap still remains in a region of the posterior body where there are "
       "no high-power EM series from either sex', and that gaps leaving cells "
       "without innervation 'precisely line up with the unreconstructed region "
       "and thus are unquestionably artefactual'",
       section="consistency",
       xfail="posterior body muscles average 6.3 presynaptic partners against "
             "the ~10 reported, a 37% deficit, while head and neck sit at 10.0 "
             "and 12.8. Sublateral input to posterior muscle is absent "
             "entirely. This is a limit of the available reconstruction rather "
             "than of the model: the authors call the gaps artefactual, and "
             "45% of neuron-muscle edges are neuromuscular junctions, half of "
             "them involving sublateral motor neurons whose connectivity was "
             "extrapolated rather than traced. No wiring-derived model can "
             "propagate a wave through a region the wiring is missing from, "
             "and no other dataset fills it: the White 1986 releases carry no "
             "neuron-to-muscle edges at all, all eight Witvliet 2021 specimens "
             "are nerve ring only, and Cook's own corrected July 2020 matrices "
             "keep the same gradient at 10.2, 13.1 and 6.8 partners")
def _posterior_innervation():
    from .connectome import Connectome
    c = Connectome.load()
    mus = [n for n in c.names if c.cell_info[n]["kind"] == "muscle"]
    mx = max(c.cell_info[n]["row"] for n in mus)
    got = {}
    for label, lo, hi in (("head", 1, 8), ("neck", 9, 16), ("body", 17, mx)):
        k = [int((c.Gs[c.idx(n)] > 0).sum()) for n in mus
             if lo <= c.cell_info[n]["row"] <= hi]
        got[label] = float(np.mean(k)) if k else 0.0
    ok = got["body"] >= 8.0
    return ok, ("presynaptic partners per muscle: "
                + ", ".join(f"{k} {v:.1f}" for k, v in got.items())
                + " (published ~10)")


@check("leak reversals stay inside the range an ion could supply",
       "E_leak is a battery made of real ionic gradients, so it cannot sit "
       "below the potassium equilibrium (about -80 mV) or above the sodium "
       "one. A solver that drives it past those is not describing a cell",
       "Gao & Zhen 2011 PNAS 108:2557 (muscle E_Cl about -30 mV, E_K about "
       "-80 mV); Liu, Chen & Wang 2014 Nat Commun 5:5155 (motor neuron "
       "resting potentials); Liu, Hollopeter & Jorgensen 2009 PNAS 106:10823 "
       "(motor neurons release tonically at rest, so tonic synaptic "
       "conductance is real and has to be accounted for, not cancelled)",
       section="consistency",
       xfail="calibrate_rest solves E_leak to put each measured cell at its "
             "measured resting potential, with no bound on the answer. For "
             "muscle it returns a median of -135 mV and a minimum of -402 mV, "
             "because tonic synaptic conductance is 5.38 nS against a 1 nS "
             "leak, 83% of the cell's resting conductance, all of it pulling "
             "toward 0 mV. Holding -25 mV against that needs a battery no ion "
             "provides. It is also what caps the muscle: with rest pinned by "
             "an ever more negative leak, the driven potential is bounded by "
             "E_muscle * s_eq/s_max = -13.6 mV and cannot reach the -10 mV "
             "spike threshold. The measured whole-cell input resistance is "
             "1 GOhm, so the tonic conductance should be a fraction of the "
             "leak rather than five times it, which points at the release "
             "variable resting at 54.5% of its maximum")
def _leak_bounds():
    from .connectome import Connectome
    from .genome import Genome
    from .nervous_system import NervousSystem
    c = Connectome.load()
    ns = NervousSystem(c, Genome.load())
    m = c.is_muscle
    lo, hi = -85.0, 60.0          # E_K to E_Na, generously
    bad = int(((ns.E_leak < lo) | (ns.E_leak > hi)).sum())
    return bad == 0, (
        f"{bad} of {ns.n} cells outside {lo:.0f}..{hi:.0f} mV; "
        f"muscle median {float(np.median(ns.E_leak[m])):.0f} mV "
        f"(min {ns.E_leak[m].min():.0f}), "
        f"neuron median {float(np.median(ns.E_leak[~m])):.0f} mV")


@check("every parameter carries provenance, and measured ones cite a source",
       "each parameter must be tagged from the closed vocabulary, and any "
       "tagged measured or published must have a citation near where it is "
       "defined. A number tagged measured with no source is the failure mode "
       "the registry exists to prevent, since it reads as anchored to the "
       "animal while being anchored to nothing",
       "self-audit of worm/parameters.py against the defining modules",
       section="consistency")
def _provenance():
    import re
    from pathlib import Path

    from . import parameters
    allowed = {"measured", "published", "assay", "tuned", "scripted"}
    params = parameters.PARAMETERS

    bad_tag = sorted(p.name for p in params.values()
                     if p.provenance not in allowed)

    # A citation is a four-digit year in the comment block above the
    # definition, which is how every source in this codebase is written.
    src: dict[str, list[str]] = {}
    uncited, unfound = [], []
    for p in params.values():
        if p.provenance not in ("measured", "published"):
            continue
        path = Path(p.where)
        if not path.exists():
            continue
        lines = src.setdefault(p.where, path.read_text().splitlines())
        short = p.name.split(".", 1)[1]
        pat = re.compile(rf"^\s*{re.escape(short)}\s*[:=]")
        idx = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
        if idx is None:
            unfound.append(p.name)
            continue
        window = "\n".join(lines[max(0, idx - 30):idx + 1])
        if not re.search(r"\b(19|20)\d{2}\b", window):
            uncited.append(p.name)

    by_tag: dict[str, int] = {}
    for p in params.values():
        by_tag[p.provenance] = by_tag.get(p.provenance, 0) + 1
    ok = not bad_tag and not uncited and not unfound
    detail = (", ".join(f"{k} {v}" for k, v in sorted(by_tag.items()))
              + f"; {len(params)} total")
    if bad_tag:
        detail += f"; UNKNOWN TAG: {bad_tag}"
    if uncited:
        detail += f"; NO CITATION: {uncited}"
    if unfound:
        detail += f"; DEFINITION NOT FOUND: {unfound}"
    return ok, detail


@check("muscle calcium reproduces the measured transient",
       "force follows calcium, so the calcium stage sets when a muscle pulls. "
       "Its impulse response must peak where the measured transient peaks",
       "Butler et al. 2015 J R Soc Interface 12:20140963 Fig 3b: electrically "
       "evoked GCaMP3 in dissected body-wall muscle, exp(-t/0.88 s) - "
       "exp(-t/0.25 s), peak 0.44 s after excitation",
       section="consistency")
def _muscle_calcium():
    from .connectome import Connectome
    from .genome import Genome
    from .nervous_system import NervousSystem
    ns = NervousSystem(Connectome.load(), Genome.load())
    p = ns.p
    t_r, t_d = p.ca_rise_ms / 1000.0, p.ca_decay_ms / 1000.0
    # Peak of exp(-t/t_d) - exp(-t/t_r), the response of the two-stage cascade.
    peak = np.log(t_d / t_r) / (1.0 / t_r - 1.0 / t_d)
    ca = ns.muscle_calcium()
    ok = (0.42 <= peak <= 0.46 and ca.size == int(ns.conn.is_muscle.sum())
          and bool(np.all(np.isfinite(ca))))
    return ok, (f"rise {t_r*1000:.0f} ms, decay {t_d*1000:.0f} ms, "
                f"peak {peak:.3f} s (measured 0.44); {ca.size} muscles tracked")


@check("the network drives the muscles, not a script",
       "the dorsoventral difference in real muscle activation should itself "
       "oscillate: a clean spectral peak in the 0.2-1.2 Hz undulation band, "
       "with a swing big enough to bend the body. This is the acceptance test "
       "for a connectome-generated rhythm, and it reads only the neuromuscular "
       "output, so no scripted quantity can satisfy it",
       "Wen et al. 2012 Neuron 76:750 (proprioceptive coupling propagates the "
       "wave); Gao & Zhen 2011 PNAS 108:2557 (motor neuron input sets muscle "
       "activity); Cronin et al. 2005 Genetics (0.47 Hz wild-type undulation)",
       xfail="the connectome delivers no rhythm to the muscles at all. The "
             "dorsoventral drive difference has sd 0.003 against the scripted "
             "oscillator's 0.599, a factor of 200, and a peak-to-median power "
             "ratio of 8 where a clean tone is orders of magnitude. Motor "
             "neurons sit at their solved resting equilibrium, so activation "
             "stays at 0.5 by construction and nothing pushes it off: there is "
             "no oscillation in the network to phase-lock to. Closing this "
             "needs muscle force, the body reading real muscle output, and "
             "proprioceptive feedback (issue #10)")
def _emergent_drive():
    from .kinematics import dominant_frequency, muscle_drive, rhythmicity
    from .simulation import SimConfig
    quiet = WormSimulation(env=Environment(width=44.0, height=32.0),
                           config=SimConfig(seed=0, muscle_pacemaker_pa=0.0))
    dor, ven, dt = muscle_drive(quiet, seconds=60.0, settle=10.0)
    mid = dor.shape[1] // 2
    diff = (dor - ven)[:, mid]
    swing = float(np.std(diff))
    peak = rhythmicity(diff, dt)
    freq = dominant_frequency(diff, dt)
    ok = swing > 0.05 and peak > 50.0 and 0.2 <= freq <= 1.2
    return ok, (f"dorsoventral drive sd {swing:.4f} (want >0.05), "
                f"spectral peak-to-median {peak:.1f} (want >50), "
                f"dominant {freq:.2f} Hz (want 0.2-1.2)")


@check("passive properties match patch-clamp measurements",
       "neuron: input resistance 1.6-8 GOhm, membrane time constant 3-10 ms, "
       "a gap junction pair coupled at tens of pS not nanosiemens. Muscle is a "
       "different cell type and must not inherit those: 1.0 GOhm and ~70 ms",
       "Goodman, Hall, Avery & Lockery 1998 Neuron 20:763; Shindou et al. 2019 "
       "Sci Rep 9:3430; Liu, Chen & Wang 2020 Nat Commun 11:5076 (AVAL-AVAR "
       "56 pS); Jospin et al. 2002 J Cell Biol 159:337 (muscle 1.0 +/- 0.08 "
       "GOhm, n=10); Richmond WormBook doi:10.1895/wormbook.1.112.1 (~70 pF)")
def _passive():
    from .connectome import Connectome
    from .genome import Genome
    from .nervous_system import NervousSystem
    c = Connectome.load()
    ns = NervousSystem(c, Genome.load())
    p = ns.p
    r_in = 1.0 / p.G_leak            # GOhm, since nS
    tau = p.C / p.G_leak             # ms, since pF/nS
    ava = c.Gg[c.idx("AVAR"), c.idx("AVAL")] * p.g_gap * 1000.0   # pS
    r_mus = 1.0 / p.G_leak_muscle
    tau_mus = p.C_muscle / p.G_leak_muscle
    ok = (1.5 <= r_in <= 8.5 and 3.0 <= tau <= 10.0 and 20 <= ava <= 200
          and 0.9 <= r_mus <= 1.1 and 60.0 <= tau_mus <= 80.0)
    return ok, (f"neuron R_in {r_in:.1f} GOhm, tau_m {tau:.1f} ms, "
                f"AVAL-AVAR coupling {ava:.0f} pS; "
                f"muscle R_in {r_mus:.2f} GOhm, tau_m {tau_mus:.0f} ms")


@check("cells rest where they were measured to rest",
       "VA5 -71.7 mV, VB6 -53.2 mV, VD5 -45.8 mV, with A and B class 19 mV "
       "apart; body-wall muscle far depolarised of all of them at -25 mV; "
       "whole network inside the -70 to -20 mV operating range",
       "Liu, Chen & Wang 2014 Nat Commun 5:5155 Table 1 (the only ventral cord "
       "motor neurons ever patched); Gao & Zhen 2011 PNAS 108:2557 (muscle "
       "-25.0 +/- 1.0 mV, n=27); Jospin et al. 2002 (muscle -19.7 +/- 1.8 mV, "
       "n=12); Goodman et al. 2012 WormBook")
def _resting():
    import numpy as np

    from .connectome import Connectome
    from .genome import Genome
    from .nervous_system import NervousSystem
    c = Connectome.load()
    ns = NervousSystem(c, Genome.load())
    got = {n: float(ns.V_th[c.idx(n)]) for n in ("VA05", "VB06", "VD05")}
    want = {"VA05": -71.7, "VB06": -53.2, "VD05": -45.8}
    err = max(abs(got[k] - want[k]) for k in want)
    spread = got["VB06"] - got["VA05"]
    in_range = ns.V_th.min() > -85.0 and ns.V_th.max() < 5.0
    # Muscle must land on its own measurement, not on the neuronal leak. The
    # two independent reports bracket -25 to -19.7 mV.
    mus = ns.V_th[c.is_muscle]
    mus_ok = abs(float(np.median(mus)) - ns.p.E_muscle) < 1.0
    ok = err < 1.0 and 15.0 < spread < 23.0 and in_range and mus_ok
    return ok, (", ".join(f"{k} {got[k]:.1f}" for k in want)
                + f"; A-to-B spread {spread:.1f} mV; muscle "
                  f"{float(np.median(mus)):.1f} mV; network "
                  f"{ns.V_th.min():.1f} to {ns.V_th.max():.1f} mV")


@check("anterior touch triggers reversal",
       "gentle anterior touch drives backward locomotion",
       "Chalfie et al. 1985; WormBook Mechanosensation")
def _ant():
    r = touch_response("anterior")
    return r["reversed"], f"reversed={r['reversed']}"


@check("posterior touch does not reverse",
       "posterior touch accelerates forward instead of reversing",
       "Chalfie et al. 1985: PLM gap-junctions to the forward command neuron PVC")
def _post():
    r = touch_response("posterior")
    base = gait(seconds=6.0, settle=10.0)["speed"]
    faster = r["speed_after"] >= base * 0.95
    return (not r["reversed"]) and faster, \
        f"reversed={r['reversed']}, speed {r['speed_after']:.3f} vs baseline {base:.3f}"


@check("touch is positional, not bucketed",
       "the ALM and PLM receptive fields cross over near mid-body, so anterior "
       "touch drives the anterior cells and posterior touch the posterior ones",
       "Chalfie & Sulston 1981; Chalfie et al. 1985; WormAtlas TRNs",
       section="consistency")
def _fields():
    from .sensory import TOUCH_FIELDS, _field_coverage
    alm = next(f for f in TOUCH_FIELDS if f["cells"][0] == "ALML")
    plm = next(f for f in TOUCH_FIELDS if f["cells"][0] == "PLML")

    def cov(f, u):
        return _field_coverage(u, f["start"], f["end"], f["soft"])

    avm = next(f for f in TOUCH_FIELDS if f["cells"][0] == "AVM")

    def ant(u):   # the anterior touch system is ALM together with AVM
        return max(cov(alm, u), cov(avm, u))

    # Locate where anterior and posterior coverage actually cross over.
    us = [i / 200 for i in range(201)]
    crossover = min(us, key=lambda u: abs(ant(u) - cov(plm, u)) if u > 0.2 else 9)

    ok = (ant(0.2) > 0.8 and cov(plm, 0.2) < 0.1          # head: anterior only
          and cov(plm, 0.85) > 0.8 and ant(0.85) < 0.1    # tail: posterior only
          and 0.45 <= crossover <= 0.65)                  # they meet mid-body
    return ok, (f"crossover at u={crossover:.2f}; "
                f"u=0.20 ant {ant(0.2):.2f}/post {cov(plm,0.2):.2f}; "
                f"u=0.85 ant {ant(0.85):.2f}/post {cov(plm,0.85):.2f}")


@check("mid-body touch is the ambiguous zone",
       "a mid-body stroke lands where the anterior and posterior fields meet, "
       "so it does not reliably drive a reversal the way a head stroke does",
       "Chalfie et al. 1985: ALM and PLM processes overlap around mid-body")
def _midbody():
    head = sum(touch_response("anterior", seed=s)["reversed"] for s in range(3))
    mid = sum(touch_response("midbody", seed=s)["reversed"] for s in range(3))
    tail = sum(touch_response("posterior", seed=s)["reversed"] for s in range(3))
    return head > mid >= tail or (head == 3 and mid < 3 and tail == 0), \
        f"reversals over 3 seeds: head {head}/3, mid-body {mid}/3, tail {tail}/3"


@check("mec-4 is touch insensitive",
       "mec-4 nulls do not respond to gentle body touch",
       "Chalfie & Sulston 1981; Arnadottir et al. 2011 J Neurosci")
def _mec4():
    r = touch_response("anterior", knockouts=["mec-4"])
    return not r["reversed"], f"reversed={r['reversed']} (expected False)"


@check("mec-10 is only partially touch insensitive",
       "a true mec-10 deletion responds to a FRACTION of gentle touches, "
       "between wild type (always) and mec-4 (never)",
       "Arnadottir et al. 2011: classic full-Mec mec-10 alleles are gain-of-function")
def _mec10():
    # Behavioural: touch-evoked reversal counts over seeds per genotype.
    wt = sum(touch_response("anterior", seed=s)["reversed"] for s in range(6))
    m10 = sum(touch_response("anterior", knockouts=["mec-10"], seed=s)["reversed"]
              for s in range(6))
    m4 = sum(touch_response("anterior", knockouts=["mec-4"], seed=s)["reversed"]
             for s in range(6))
    return m4 < m10 < wt, \
        f"reversals over 6 seeds: wild type {wt}/6, mec-10 {m10}/6, mec-4 {m4}/6"


@check("harsh touch survives mec-4",
       "harsh prodding is MEC-4 independent, so mec-4 nulls respond to "
       "anterior harsh touch at wild-type levels (via FLP)",
       "Li, Kang, Piggott, Feng & Xu 2011 Nat Commun 2:315")
def _harsh():
    def harsh_ant(kos=()):
        n = 0
        for seed in range(3):
            s = _sim(kos, seed=seed)
            for _ in range(500):
                s.step()
            r0 = s.state.reversal_count
            s.env.poke(0.15, strength=2.2, duration=0.4, harsh=True)
            for _ in range(150):
                s.step()
            n += s.state.reversal_count - r0
        return n
    wt_ant, ko_ant = harsh_ant(), harsh_ant(("mec-4",))
    return wt_ant >= 2 and ko_ant >= 2, \
        f"anterior harsh: wild type {wt_ant}/3, mec-4 {ko_ant}/3"


@check("posterior harsh touch drives forward escape, not reversal",
       "PVD photoactivation drives forward acceleration; removing PVC flips "
       "it to reverse, so the functional bias is forward",
       "Husson, Steuer Costa et al. 2012 Curr Biol 22:743")
def _harsh_forward():
    # PVD wiring favours the forward pool (107:81 contacts), and with the
    # calibrated reversal threshold the PVD-driven deviation (~0.0022) stays
    # below the trigger: the animal accelerates rather than reversing.
    n = 0
    for seed in range(3):
        s = _sim((), seed=seed)
        for _ in range(500):
            s.step()
        r0 = s.state.reversal_count
        s.env.poke(0.85, strength=2.2, duration=0.4, harsh=True)
        for _ in range(150):
            s.step()
        n += s.state.reversal_count - r0
    return n == 0, f"posterior harsh reversals over 3 seeds: {n}/3 (want 0)"


@check("unc-25 is a shrinker",
       "GABA loss: body shortens and bending amplitude falls, but not paralysed",
       "Jin et al. 1999; Deng et al. 2021 eNeuro (reduced speed and amplitude)")
def _unc25():
    wt, ko = gait(), gait(knockouts=["unc-25"])
    shorter = ko["length_scale"] < wt["length_scale"] - 0.01
    shallower = ko["amplitude"] < wt["amplitude"] * 0.95
    moving = ko["speed"] > 0.01
    return shorter and shallower and moving, (
        f"length {ko['length_scale']:.2f} vs {wt['length_scale']:.2f}, "
        f"amplitude {ko['amplitude']*100:.1f}% vs {wt['amplitude']*100:.1f}%, "
        f"speed {ko['speed']:.3f} mm/s (still moving)")


@check("unc-47 and unc-49 shrink like unc-25",
       "all three GABA-pathway genes give the same shrinker class",
       "WormBook GABA chapter: canonical shrinkers are unc-25/30/46/47/49")
def _gaba_class():
    base = gait()["length_scale"]
    out = {g: gait(knockouts=[g])["length_scale"] for g in ("unc-25", "unc-47", "unc-49")}
    ok = all(v < base - 0.01 for v in out.values())
    return ok, ", ".join(f"{k} {v:.2f}" for k, v in out.items()) + f" vs wt {base:.2f}"


@check("unc-13 is near-paralysed",
       "blocking synaptic vesicle priming leaves almost no locomotion",
       "Richmond et al. 1999; WormBook Synaptic Function")
def _unc13():
    wt, ko = gait(), gait(knockouts=["unc-13"])
    return ko["speed"] < wt["speed"] * 0.5, \
        f"speed {ko['speed']:.3f} vs wild type {wt['speed']:.3f} mm/s"


@check("egl-30 loss of function is lethargic",
       "Gq loss reduces bend depth and speed",
       "Brundage et al. 1996; Cronin et al. 2005")
def _egl30():
    wt, ko = gait(), gait(knockouts=["egl-30"])
    return ko["speed"] < wt["speed"] and ko["amplitude"] < wt["amplitude"], \
        f"speed {ko['speed']:.3f} vs {wt['speed']:.3f}, " \
        f"amplitude {ko['amplitude']*100:.1f}% vs {wt['amplitude']*100:.1f}%"


@check("goa-1 loss of function is loopy and hyperactive",
       "Go loss deepens bends and raises speed - opposite sign to egl-30(lf)",
       "Segalat et al. 1995; Cronin et al. 2005: 0.29 vs 0.20 mm/s, flex 1.3 vs 1.0",
       xfail="deeper bends and a faster rhythm are both right (0.71 Hz "
             "against 0.47 Hz wild type), but NET SPEED is low because the "
             "animal reverses and turns constantly: goa-1 does ~11 reversals "
             "and ~7 omega turns in a 45 s run against wild type's zero, and "
             "reversals subtract displacement. The undulation is hyperactive; "
             "the trajectory is not. Closing this needs the spontaneous "
             "reversal rate to be right (issue #6), not the CPG (issue #10)")
def _goa1():
    wt, ko = gait(), gait(knockouts=["goa-1"])
    return ko["amplitude"] > wt["amplitude"] and ko["speed"] > wt["speed"] * 0.95, \
        f"speed {ko['speed']:.3f} vs {wt['speed']:.3f}, " \
        f"amplitude {ko['amplitude']*100:.1f}% vs {wt['amplitude']*100:.1f}%; " \
        f"reversals {ko['reversals']} vs {wt['reversals']}, " \
        f"omegas {ko['omegas']} vs {wt['omegas']}"


@check("the kinematics harness recovers a known gait",
       "measuring the scripted oscillator must return the frequency, "
       "wavelength, direction and bend amplitude it was given; a harness that "
       "cannot recover hard-coded inputs makes every gait result "
       "uninterpretable",
       "self-consistency against body.BodyParams",
       section="consistency")
def _kinematics():
    from .body import BodyParams
    from .kinematics import analyse, record
    p = BodyParams()
    got = analyse(record(_sim(), seconds=60.0, settle=10.0))
    ok = (abs(got["undulation_hz"] - p.freq_hz) < 0.03
          and abs(got["wavelength_bl"] - p.wavelength_bl) < 0.05
          and got["wave_direction"] == "forward"
          and 0.12 <= got["bend_amplitude_bl"] <= 0.30)
    return ok, (f"{got['undulation_hz']:.3f} Hz (want {p.freq_hz}), "
                f"{got['wavelength_bl']:.3f} BL (want {p.wavelength_bl}), "
                f"{got['wave_direction']}, "
                f"amplitude {got['bend_amplitude_bl']*100:.1f}% BL")


@check("che-1 removes salt sensing but leaves odour intact",
       "che-1 is the ASE terminal selector, so only gustation is lost; the "
       "behavioural consequence is covered by the chemotaxis discrimination check",
       "Uchida et al. 2003; WormBook Chemosensation",
       section="consistency")
def _che1():
    from .genome import Genome
    g = Genome.load()
    g.knock_out("che-1")
    return g.sensory_scale("salt") == 0.0 and g.sensory_scale("odor") == 1.0, \
        f"salt gain {g.sensory_scale('salt')}, odour gain {g.sensory_scale('odor')}"


@check("expression cache matches documented ground truth",
       "CeNGEN-derived cell sets reproduce the textbook cases: mec-4 in the "
       "six touch receptors, che-1 only in ASE, glc-3 in AIY, glr-1 in "
       "command interneurons, unc-25 in the GABAergic cells, cat-2 in the "
       "eight dopaminergic neurons",
       "Chalfie & Sulston 1981; Uchida 2003; Chalasani 2007; Maricq 1995; "
       "Gendrel 2016; Sulston 1975",
       section="consistency")
def _expression_truth():
    import json
    from .paths import data_dir
    exp = json.loads((data_dir() / "expression.json").read_text())["genes"]

    def cells(gene):
        return set(exp.get(gene, {}).get("cells", []))

    trns = {"ALML", "ALMR", "AVM", "PLML", "PLMR", "PVM"}
    gaba = {"DD01", "VD01", "RMEL", "AVL", "DVB", "RIS"}
    da = {"ADEL", "ADER", "CEPDL", "CEPDR", "CEPVL", "CEPVR", "PDEL", "PDER"}
    aiy_ok = {"AIYL", "AIYR"} <= cells("glc-3")
    cmd = cells("glr-1") & {"AVAL", "AVAR", "AVDL", "AVDR", "PVCL", "PVCR"}
    ok = (cells("mec-4") == trns
          and cells("che-1") == {"ASEL", "ASER"}
          and aiy_ok
          and {"AVAL", "AVAR", "AVDL", "AVDR", "PVCL", "PVCR"} <= cells("glr-1")
          and gaba <= cells("unc-25")
          and cells("cat-2") == da)
    return ok, (f"mec-4 -> {sorted(cells('mec-4'))}; glc-3 covers AIY: {aiy_ok}; "
                f"glr-1 covers command cells {sorted(cmd)}")


@check("receptor-derived signs match documented synapses",
       "AWC->AIY is inhibitory (glutamate-gated chloride); the neuromuscular "
       "junction is acetylcholine-excitatory and GABA-inhibitory, the latter "
       "at the muscle chloride reversal rather than the neuronal one",
       "Chalasani et al. 2007 Nature 450:63; Jospin et al. 2002; "
       "Bamber et al. 1999",
       section="consistency")
def _sign_truth():
    from .connectome import E_EXC, E_INH, E_INH_MUSCLE, Connectome
    c = Connectome.load()
    awc = all(c.E_syn[c.index[post], c.index[pre]] == E_INH
              for pre in ("AWCL", "AWCR") for post in ("AIYL", "AIYR")
              if c.Gs[c.index[post], c.index[pre]] > 0)
    nmj_ok = True
    posts, pres = np.nonzero(c.Gs)   # Gs is [post, pre]
    for j, i in zip(posts, pres):
        if not c.is_muscle[j]:
            continue
        nts = c.pre_nt[i]
        if "Acetylcholine" in nts and c.E_syn[j, i] != E_EXC:
            nmj_ok = False
        if "GABA" in nts and c.E_syn[j, i] != E_INH_MUSCLE:
            nmj_ok = False
    n_derived = c.sign_provenance.get("receptor_expression", 0)
    return awc and nmj_ok and n_derived > 0, \
        (f"AWC->AIY inhibitory: {awc}; NMJ ACh-exc/GABA-inh: {nmj_ok}; "
         f"{n_derived} edges derived from expression, "
         f"{len(c.sign_flips)} flips vs the transmitter heuristic")


@check("neuromuscular conductance matches the measured junction",
       "the ACHIEVABLE whole-cell cholinergic conductance of a body-wall "
       "muscle should come out near the patch-clamp value. Achievable, not "
       "nominal: the synaptic release variable cannot exceed a_r/(a_r + a_d), "
       "so a synapse delivers at most a sixth of its per-contact conductance, "
       "and comparing the nominal figure overstates junction strength about "
       "sixfold. Also that the junction's reversal potentials are the measured "
       "ones, a non-selective cation channel near 0 mV for acetylcholine and "
       "the chloride equilibrium near -30 mV for GABA, not the neuronal "
       "-48 mV",
       "Richmond & Jorgensen 1999 Nat Neurosci 2:791: acetylcholine 774 +/- 79 "
       "pA (n=22) at -80 mV holding, E_rev +11 mV -> 8.5 nS; GABA receptor "
       "chloride-permeant. Gao & Zhen 2011 PNAS 108:2557: E_Cl about -30 mV",
       section="consistency")
def _nmj():
    import numpy as np

    from .connectome import E_INH_MUSCLE, Connectome
    from .genome import Genome
    from .nervous_system import NervousSystem
    c = Connectome.load()
    ns = NervousSystem(c, Genome.load())
    m = c.is_muscle
    Gs = c.Gs * ns.g_syn_row
    E = c.E_syn
    # Pressure-ejected acetylcholine activates the cholinergic receptors only,
    # so the comparable quantity is the excitatory conductance, and it has to
    # be scaled by the ceiling on the release variable to be what the junction
    # can actually deliver.
    s_max = ns.p.a_r / (ns.p.a_r + ns.p.a_d)
    chol = np.median((Gs * (E > E_INH_MUSCLE / 2.0))[m].sum(axis=1)) * s_max
    measured = 774.0 / (80.0 + 11.0)            # pA / mV -> nS
    ratio = chol / measured
    e_on_muscle = E[m][Gs[m] > 0]
    ok = (0.3 <= ratio <= 3.0
          and abs(e_on_muscle.max() - 0.0) < 1.0
          and abs(e_on_muscle.min() - E_INH_MUSCLE) < 1.0)
    return ok, (f"achievable cholinergic {chol:.2f} nS per muscle against "
                f"{measured:.2f} measured ({ratio:.2f}x); release ceiling "
                f"{s_max:.3f}; junction reversals "
                f"{e_on_muscle.min():.0f} to {e_on_muscle.max():.0f} mV")


@check("tdc-1 impairs the omega turn",
       "tyramine loss leaves reversals but degrades the ventral omega turn",
       "Donnelly et al. 2013 PLoS Biol: 32% complete the omega vs wild type")
def _tdc1():
    # Behavioural: poke, then count reversals and omega turns per genotype.
    def run(kos):
        s = _sim(kos)
        for _ in range(500):
            s.step()
        for _ in range(5):
            s.env.poke("anterior", 1.0, duration=0.4)
            for _ in range(int(10.0 / s.cfg.dt)):
                s.step()
        return s.state.reversal_count, s.state.omega_count

    wt_rev, wt_om = run(())
    ko_rev, ko_om = run(("tdc-1",))
    ok = wt_om >= 2 and ko_om == 0 and ko_rev >= wt_rev - 1
    return ok, (f"wild type {wt_rev} reversals/{wt_om} omegas; "
                f"tdc-1 {ko_rev} reversals/{ko_om} omegas")


@check("gait adapts to the medium: swimming is faster than crawling",
       "lowering the drag ratio from agar to water raises undulation frequency "
       "and speed in the animal; the scripted oscillator cannot do this",
       "Berri et al. 2009; Fang-Yen et al. 2010 gait adaptation",
       xfail="frequency is fixed at 0.47 Hz regardless of medium; gait "
             "adaptation must emerge from the proprioceptive CPG (issue #10)")
def _swim():
    crawl = gait()
    s = _sim(); s.env.drag_ratio = 1.6
    for _ in range(int(10.0 / s.cfg.dt)):
        s.step()
    d0 = s.state.distance
    for _ in range(int(30.0 / s.cfg.dt)):
        s.step()
    swim = (s.state.distance - d0) / 30.0
    return swim > crawl["speed"], \
        f"water {swim:.3f} vs agar {crawl['speed']:.3f} mm/s " \
        f"(fixed gait: lower drag ratio gives less thrust per stroke)"


@check("chemotaxis discriminates salt-blind mutants",
       "wild type chemotaxes up a salt gradient; che-1 (no ASE) does not",
       "Ward 1973; Bargmann & Horvitz 1991; Pierce-Shimomura et al. 1999",
       xfail="no spontaneous reversals (issue #6) means no pirouettes, so the "
             "biased random walk cannot exist and wild type scores no better "
             "than che-1 from a random heading")
def _chemo():
    from .assays import run_assay
    wt = run_assay("chemotaxis", knockouts=(), seed=0,
                   minutes=2.0, replicates=3)["result"]
    ko = run_assay("chemotaxis", knockouts=("che-1",), seed=0,
                   minutes=2.0, replicates=3)["result"]
    # The discrimination must clear both bars: wild type approaches, and the
    # mutant does measurably worse. Thresholds are set just above what pure
    # geometry scores, so passing requires genuine gradient-guided behaviour.
    wt_min_ci, min_gap = 0.30, 0.25
    ok = (wt["chemotaxis_index"] >= wt_min_ci
          and wt["chemotaxis_index"] - ko["chemotaxis_index"] >= min_gap)
    return ok, (f"wild type CI {wt['chemotaxis_index']} "
                f"({wt['reversals_mean']} reversals/run), "
                f"che-1 CI {ko['chemotaxis_index']}")


@check("ablating the command interneurons reproduces their ablation phenotypes",
       "killing AVB+PVC abolishes forward locomotion while leaving reversals; "
       "killing AVA+AVD+AVE abolishes the touch-evoked reversal; killing "
       "ALM+AVM removes anterior touch sensitivity",
       "Chalfie et al. 1985 J Neurosci 5:956; Zheng et al. 1999; "
       "Zhen & Samuel 2015 review. Note (issue #7, external review): the "
       "single-class serial structure is NOT yet checked here -- AVD is the "
       "input stage, AVA the output stage (ALM/AVM make zero connections "
       "onto AVA), so AVD-only and AVA-only ablations have finer Chalfie "
       "Table III phenotypes than this pool-level test asserts")
def _ablation():
    import numpy as np

    def drives(cells):
        s = _sim(seed=0)
        for c in cells:
            s.ablate(c)
        for _ in range(500):
            s.step()
        f = []
        for _ in range(400):
            t = s.step()
            f.append(t["forward_drive"])
        return float(np.mean(f))

    def touch_after(cells):
        s = _sim(seed=0)
        for c in cells:
            s.ablate(c)
        for _ in range(500):
            s.step()
        r0 = s.state.reversal_count
        s.env.poke("anterior", 1.0, duration=0.4)
        for _ in range(150):
            s.step()
        return s.state.reversal_count > r0

    intact_f = drives(())
    no_fwd_f = drives(("AVBL", "AVBR", "PVCL", "PVCR"))
    no_bwd_touch = touch_after(("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"))
    no_trn_touch = touch_after(("ALML", "ALMR", "AVM"))
    intact_touch = touch_after(())

    ok = (intact_f > 0.5 and no_fwd_f < 0.05
          and intact_touch and not no_bwd_touch and not no_trn_touch)
    return ok, (f"forward drive {intact_f:.2f} intact -> {no_fwd_f:.2f} without "
                f"AVB/PVC; head-touch reversal: intact {intact_touch}, "
                f"no AVA/AVD/AVE {no_bwd_touch}, no ALM/AVM {no_trn_touch}")


def _run_life(longevity: float = 1.0, temp: float = 20.0, dt: float = 60.0,
              seed: int = 0):
    """Run one animal from fertilised egg to death, without the neural sim."""
    from .lifecycle import Lifecycle
    L = Lifecycle(seed=seed)
    marks: dict = {}
    t = 0.0
    while L.alive and t < 90 * 86400:
        ev = L.step(dt, food=1.0, temp_c=temp, pheromone=0.0,
                    longevity_scale=longevity)
        t += dt
        for k in ("hatch", "adult", "sperm_exhausted", "death"):
            if k in ev and k not in marks:
                marks[k] = L.age_s / 3600.0
    return L, marks


@check("development timing egg to adult",
       "embryo ~14.2 h, then ~50.7 h from hatch to adult at 20 C",
       "Sulston et al. 1983 (800 min lineage clock); Faerberg, Gurarie & "
       "Ruvinsky 2022 BMC Biol 20:87 (50.67 +/- 1.95 h hatch to adult)",
       section="consistency")
def _dev():
    L, m = _run_life()
    embryo = m.get("hatch", 0.0)
    larval = m.get("adult", 0.0) - embryo
    ok = 13.0 <= embryo <= 15.5 and 46.0 <= larval <= 55.0
    return ok, f"embryo {embryo:.1f} h, hatch to adult {larval:.1f} h"


@check("brood size is sperm-limited",
       "~300 self progeny over a ~5 day reproductive period, ending when the "
       "fixed store of self-sperm runs out rather than when oocytes do",
       "Hodgkin & Barnes 1991 (mean 327); Ward & Carrel 1979; Huang et al. "
       "2004 (5.8 +/- 2.0 d fertile period)",
       section="consistency")
def _brood():
    L, m = _run_life()
    period_d = (m.get("sperm_exhausted", 0.0) - m.get("adult", 0.0)) / 24.0
    ok = 250 <= L.eggs_laid <= 360 and 3.5 <= period_d <= 7.0 and L.self_sperm == 0
    return ok, (f"{L.eggs_laid} eggs over {period_d:.1f} d, "
                f"{L.self_sperm} sperm left")


@check("the animal ages and dies, on a distribution",
       "mean adult lifespan ~15 d at 20 C with real spread across a cohort, "
       "preceded by decline in pumping and locomotion class",
       "Huang, Xiong & Kornfeld 2004 PNAS (15.2 +/- 3.6 d); Herndon et al. "
       "2002 Nature (movement classes A/B/C)")
def _death():
    # A cohort, not one animal: the check is on the distribution now that
    # individual lifespans are drawn from the measured mean and SD.
    days, causes = [], set()
    for seed in range(6):
        L, _ = _run_life(seed=seed)
        days.append(L.adult_day)
        causes.add(L.cause_of_death)
    mean, sd = float(np.mean(days)), float(np.std(days))
    ok = (causes == {"senescence"} and 13.0 <= mean <= 17.5 and sd > 0.5)
    return ok, (f"cohort lifespans {[round(d, 1) for d in days]} d: "
                f"mean {mean:.1f} d, sd {sd:.1f} d, causes {sorted(causes)}")


@check("daf-2 longevity requires daf-16",
       "daf-2 loss roughly doubles lifespan, and removing daf-16 as well "
       "abolishes the extension completely; eat-2 does not need daf-16",
       "Kenyon et al. 1993 Nature 366:461; Lakowski & Hekimi 1998 PNAS",
       section="consistency")
def _daf():
    from .genome import Genome

    def scale(*kos):
        g = Genome.load()
        for k in kos:
            g.knock_out(k)
        return g.longevity_scale()

    d2, d2_16 = scale("daf-2"), scale("daf-2", "daf-16")
    e2, e2_16 = scale("eat-2"), scale("eat-2", "daf-16")
    ok = d2 >= 1.8 and d2_16 < 1.05 and e2 > 1.2 and e2_16 > 1.1
    return ok, (f"daf-2 x{d2:.2f} -> daf-2;daf-16 x{d2_16:.2f} (abolished); "
                f"eat-2 x{e2:.2f} -> eat-2;daf-16 x{e2_16:.2f} (retained)")


def _run_one(entry):
    """Execute one check. Module-level so process-pool workers can pickle it."""
    name, expectation, source, sect, xfail, fn = entry
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"raised {type(exc).__name__}: {exc}"
    return ok, detail


def main(verbose: bool = True, jobs: int = 1, match: str | None = None) -> int:
    checks = CHECKS
    if match:
        checks = [c for c in CHECKS if match.lower() in c[0].lower()]
        if not checks:
            print(f"no checks match {match!r}")
            return 1
    # The checks are independent simulations, so they parallelise cleanly.
    # Results come back in registration order regardless of finish order.
    if jobs > 1 and len(checks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_run_one, checks))
    else:
        results = [_run_one(c) for c in checks]

    passed = failed = xfailed = xpassed = 0
    section = None
    for (name, expectation, source, sect, xfail, fn), (ok, detail) in \
            zip(checks, results):
        if sect != section:
            section = sect
            if verbose:
                header = ("BEHAVIOURAL -- the animal is run and measured"
                          if sect == "behaviour" else
                          "CONSISTENCY -- parameters and data invariants, not validation")
                print(f"\n== {header} ==")
        if xfail is not None:
            if ok:
                xpassed += 1
                mark = "XPASS"
            else:
                xfailed += 1
                mark = "XFAIL"
        else:
            passed += ok
            failed += not ok
            mark = "PASS" if ok else "FAIL"
        if verbose:
            print(f"[{mark}] {name}")
            print(f"       expect: {expectation}")
            print(f"       got:    {detail}")
            if xfail is not None:
                print(f"       xfail:  {xfail}")
            print(f"       source: {source}\n")
    print(f"{passed} passed, {failed} failed, "
          f"{xfailed} expected failures, {xpassed} unexpected passes, "
          f"{len(checks)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
