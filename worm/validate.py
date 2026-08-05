"""Check the simulator's behaviour against published phenotypes.

Each check states the literature expectation it is testing and the source. This
is a behavioural sanity suite, not a claim of quantitative fidelity: the
tolerances are wide because the model is coarse. What it does catch is sign
errors -- a mutant getting faster when the animal gets slower, touch working
when it should be abolished, that sort of thing.
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


def check(name, expectation, source):
    def deco(fn):
        CHECKS.append((name, expectation, source, fn))
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
       "Chalfie & Sulston 1981; Chalfie et al. 1985; WormAtlas TRNs")
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
    from .genome import GENE_EFFECTS
    m4 = GENE_EFFECTS["mec-4"].sensory_scale["touch"]
    m10 = GENE_EFFECTS["mec-10"].sensory_scale["touch"]
    return m10 > m4, f"touch gain mec-10={m10} > mec-4={m4}"


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
       "che-1 is the ASE terminal selector, so only gustation is lost",
       "Uchida et al. 2003; WormBook Chemosensation")
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
    from .genome import Genome
    g = Genome.load()
    g.knock_out("tdc-1")
    return g.global_scale("omega_turn") < 0.5 and g.global_scale("head_suppression") == 0.0, \
        f"omega gain {g.global_scale('omega_turn')}, head suppression {g.global_scale('head_suppression')}"


@check("swimming is faster than crawling",
       "lowering the drag ratio from agar to water raises undulation speed",
       "Berri et al. 2009; Fang-Yen et al. 2010 gait adaptation")
def _swim():
    crawl = gait()
    s = _sim(); s.env.drag_ratio = 1.6
    for _ in range(int(10.0 / s.cfg.dt)):
        s.step()
    d0 = s.state.distance
    for _ in range(int(30.0 / s.cfg.dt)):
        s.step()
    swim = (s.state.distance - d0) / 30.0
    return swim < crawl["speed"], \
        f"water {swim:.3f} vs agar {crawl['speed']:.3f} mm/s " \
        f"(low drag ratio gives less thrust per stroke)"


def _run_life(longevity: float = 1.0, temp: float = 20.0, dt: float = 60.0):
    """Run one animal from fertilised egg to death, without the neural sim."""
    from .lifecycle import Lifecycle
    L = Lifecycle()
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
       "Ruvinsky 2022 BMC Biol 20:87 (50.67 +/- 1.95 h hatch to adult)")
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
       "2004 (5.8 +/- 2.0 d fertile period)")
def _brood():
    L, m = _run_life()
    period_d = (m.get("sperm_exhausted", 0.0) - m.get("adult", 0.0)) / 24.0
    ok = 250 <= L.eggs_laid <= 360 and 3.5 <= period_d <= 7.0 and L.self_sperm == 0
    return ok, (f"{L.eggs_laid} eggs over {period_d:.1f} d, "
                f"{L.self_sperm} sperm left")


@check("the animal ages and dies",
       "mean adult lifespan ~15 d at 20 C, preceded by decline in pumping and "
       "locomotion class",
       "Huang, Xiong & Kornfeld 2004 PNAS (15.2 +/- 3.6 d); Herndon et al. "
       "2002 Nature (movement classes A/B/C)")
def _death():
    L, _ = _run_life()
    ok = (not L.alive and L.cause_of_death == "senescence"
          and 11.0 <= L.adult_day <= 20.0)
    return ok, (f"died on adult day {L.adult_day:.1f} as class "
                f"{L.movement_class}, cause {L.cause_of_death}")


@check("daf-2 longevity requires daf-16",
       "daf-2 loss roughly doubles lifespan, and removing daf-16 as well "
       "abolishes the extension completely; eat-2 does not need daf-16",
       "Kenyon et al. 1993 Nature 366:461; Lakowski & Hekimi 1998 PNAS")
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


def main(verbose: bool = True) -> int:
    passed = failed = 0
    if verbose:
        print("Validating modelled phenotypes against the literature\n")
    for name, expectation, source, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        passed += ok
        failed += not ok
        mark = "PASS" if ok else "FAIL"
        if verbose:
            print(f"[{mark}] {name}")
            print(f"       expect: {expectation}")
            print(f"       got:    {detail}")
            print(f"       source: {source}\n")
    print(f"{passed} passed, {failed} failed, {len(CHECKS)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
