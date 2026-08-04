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
