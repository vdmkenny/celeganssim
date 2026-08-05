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


@check("passive properties match patch-clamp measurements",
       "input resistance 1.6-8 GOhm and membrane time constant 3-10 ms; a gap "
       "junction pair coupled at tens of pS, not nanosiemens",
       "Goodman, Hall, Avery & Lockery 1998 Neuron 20:763; Shindou et al. 2019 "
       "Sci Rep 9:3430; Liu, Chen & Wang 2020 Nat Commun 11:5076 (AVAL-AVAR "
       "56 pS)")
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
    ok = 1.5 <= r_in <= 8.5 and 3.0 <= tau <= 10.0 and 20 <= ava <= 200
    return ok, (f"R_in {r_in:.1f} GOhm, tau_m {tau:.1f} ms, "
                f"AVAL-AVAR coupling {ava:.0f} pS")


@check("motor neurons rest where they were measured to rest",
       "VA5 -71.7 mV, VB6 -53.2 mV, VD5 -45.8 mV, with A and B class 19 mV "
       "apart; whole network inside the -70 to -20 mV operating range",
       "Liu, Chen & Wang 2014 Nat Commun 5:5155 Table 1 (the only ventral cord "
       "motor neurons ever patched); Goodman et al. 2012 WormBook")
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
    ok = err < 1.0 and 15.0 < spread < 23.0 and in_range
    return ok, (", ".join(f"{k} {got[k]:.1f}" for k in want)
                + f"; A-to-B spread {spread:.1f} mV; network "
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
       "a true mec-10 deletion loses only part of the touch response, unlike mec-4",
       "Arnadottir et al. 2011: classic full-Mec mec-10 alleles are gain-of-function")
def _mec10():
    # Behavioural: count touch-evoked reversals over seeds for each genotype.
    m4 = sum(touch_response("anterior", knockouts=["mec-4"], seed=s)["reversed"]
             for s in range(4))
    m10 = sum(touch_response("anterior", knockouts=["mec-10"], seed=s)["reversed"]
              for s in range(4))
    return m10 > m4, f"anterior-touch reversals over 4 seeds: mec-10 {m10}/4, mec-4 {m4}/4"


@check("harsh touch survives mec-4 and keeps its direction",
       "harsh prodding is MEC-4 independent, so mec-4 nulls respond at "
       "wild-type levels; anterior harsh touch still reverses (via FLP) and "
       "posterior harsh touch still drives forward (via PVD)",
       "Li, Kang, Piggott, Feng & Xu 2011 Nat Commun 2:315; Husson et al. "
       "2012 Curr Biol 22:743")
def _harsh():
    def harsh(u, kos=()):
        n = 0
        for seed in range(3):
            s = _sim(kos, seed=seed)
            for _ in range(500):
                s.step()
            r0 = s.state.reversal_count
            s.env.poke(u, strength=2.2, duration=0.4, harsh=True)
            for _ in range(150):
                s.step()
            n += s.state.reversal_count - r0
        return n
    wt_ant, ko_ant = harsh(0.15), harsh(0.15, ("mec-4",))
    wt_post = harsh(0.85)
    ok = wt_ant >= 2 and ko_ant >= 2 and wt_post == 0
    return ok, (f"anterior harsh: wild type {wt_ant}/3, mec-4 {ko_ant}/3; "
                f"posterior harsh: {wt_post}/3 reversals (forward escape)")


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
       "Segalat et al. 1995; Cronin et al. 2005: 0.29 vs 0.20 mm/s, flex 1.3 vs 1.0")
def _goa1():
    wt, ko = gait(), gait(knockouts=["goa-1"])
    return ko["amplitude"] > wt["amplitude"] and ko["speed"] > wt["speed"] * 0.95, \
        f"speed {ko['speed']:.3f} vs {wt['speed']:.3f}, " \
        f"amplitude {ko['amplitude']*100:.1f}% vs {wt['amplitude']*100:.1f}%"


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
       "Zhen & Samuel 2015 review")
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


def main(verbose: bool = True, jobs: int = 1) -> int:
    # The checks are independent simulations, so they parallelise cleanly.
    # Results come back in registration order regardless of finish order.
    if jobs > 1 and len(CHECKS) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_run_one, CHECKS))
    else:
        results = [_run_one(c) for c in CHECKS]

    passed = failed = xfailed = xpassed = 0
    section = None
    for (name, expectation, source, sect, xfail, fn), (ok, detail) in \
            zip(CHECKS, results):
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
          f"{len(CHECKS)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
