"""Standard C. elegans behavioural assays, scored the way the field scores them.

Each assay returns the metric a paper would actually report, so results can be
put next to published numbers rather than being simulator-specific. Every assay
records its own reference value in `expected`.

    python -m worm assay chemotaxis
    python -m worm assay all --knockout che-1
    python -m worm assay thermotaxis --json
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .environment import Environment
from .simulation import SimConfig, WormSimulation

ASSAYS: dict[str, "Assay"] = {}


def _torus_dist(a: np.ndarray, b: np.ndarray, width: float, height: float) -> float:
    """Minimum-image distance on the toroidal arena.

    The arena wraps, so the naive Euclidean distance is wrong for any pair
    separated by more than half a dimension: a worm that crosses a boundary
    is close to its target on the torus while being far from it on the page.
    """
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    d[0] = min(d[0], width - d[0])
    d[1] = min(d[1], height - d[1])
    return float(np.hypot(d[0], d[1]))


# Chemotaxis assay geometry. The classic assay is a population count in scoring
# regions of a 10 cm plate (Ward 1973; Bargmann & Horvitz 1991); the
# single-animal version here is a time-fraction analogue, so the geometry is a
# modelling choice -- but a named one, expressed relative to the arena so it
# cannot silently drift from the physics.
# The arena is toroidal (Environment.wrap) and scoring uses torus distance,
# so a wrap is measured correctly rather than read as a jump. What the size
# buys is the SCORING GEOMETRY: on the old 60x44 plate the near band covered
# 30% of the surface, so a blind ballistic walker occupied it by chance and
# scored CI 0.6-0.7, which is what an earlier chemotaxis pass was made of.
# Here the near band is under 3%, and a six-minute run covers about 90 mm of
# path against a 75 mm half-width, so wrapping is occasional rather than the
# dominant behaviour it was on the small plate.
#
# The geometry is Ward's, not ours. Concentric bands around a single peak
# were an invention, and they interacted badly with everything: the near
# band's share of the arena set a blind animal's score (which is how a
# chemotaxis pass survived on the old small plate), and the start ring sat
# OUTSIDE the far band, so every animal began already scored as avoiding.
# The real assay puts an attractant spot and a control spot at opposite
# ends, releases the animals in the middle, and counts how many are at each
# spot at scoring time. Both regions are the same size and equally far, so
# an animal with no chemosensation scores zero by construction rather than
# by the arena happening to be the right size.
CHEMO_ARENA = (150.0, 110.0)
# Distances are SCALED DOWN from a real plate, and the reason is simulation
# time rather than biology. Ward's plate puts the spots about 40 mm apart
# and scores after an hour; the animal here crawls at the measured speed, so
# an hour of assay is an hour of compute per animal and a plate of sixteen
# is a day. At 25 mm apart over eight minutes, fourteen of sixteen animals
# were still in the middle at scoring time and the index was pure noise.
# Halving the separation and sharpening the gradient keeps the structure
# that matters, two symmetric spots with the animals released between them,
# at a scale the walk can actually resolve.
CHEMO_SPOT_DX_MM = 13.0             # attractant at +dx, control at -dx
CHEMO_SOURCE_XY = (75.0 + 13.0, 55.0)
CHEMO_CONTROL_XY = (75.0 - 13.0, 55.0)
CHEMO_SOURCE_SIGMA_MM = 10.0        # localised, so the control spot is not
                                    # sitting inside the attractant's skirt
CHEMO_START_XY = (75.0, 55.0)       # released midway between the spots
CHEMO_START_JITTER_MM = 2.0         # a released drop is not a point
CHEMO_SCORE_RADIUS_MM = 7.0         # counted as "at" a spot inside this
CHEMO_RUN_MINUTES = 5.0             # per replicate
CHEMO_REPLICATES = 6                # randomized starts/headings per condition

# Provenance tags for the parameter registry (worm/parameters.py).
CONST_PROVENANCE = {
    "CHEMO_ARENA": "assay",
    "CHEMO_SOURCE_XY": "assay",
    "CHEMO_SOURCE_SIGMA_MM": "assay",
    "CHEMO_START_JITTER_MM": "assay",
    "CHEMO_SCORE_RADIUS_MM": "assay",
    "CHEMO_SPOT_DX_MM": "assay",
    "CHEMO_RUN_MINUTES": "assay",
    "CHEMO_REPLICATES": "assay",
}


@dataclass
class Assay:
    name: str
    metric: str
    expected: str
    source: str
    run: object
    description: str = ""


def assay(name, metric, expected, source, description=""):
    def deco(fn):
        ASSAYS[name] = Assay(name, metric, expected, source, fn, description)
        return fn
    return deco


def _build(knockouts=(), ablations=(), seed=0, **env_kw) -> WormSimulation:
    env = Environment(**env_kw)
    sim = WormSimulation(env=env, config=SimConfig(seed=seed))
    for g in knockouts:
        sim.knock_out(g)
    for c in ablations:
        sim.ablate(c)
    return sim


# --------------------------------------------------------------------------
def _chemo_run(task):
    """One chemotaxis replicate: random ring start, random heading.

    Module-level so process-pool workers can pickle it. Start geometry is
    drawn by the caller (seeded there) so results do not depend on how many
    workers happen to run.
    """
    knockouts, ablations, seed, minutes, start_xy, heading, theta = task
    sim = _build(knockouts, ablations, seed,
                 width=CHEMO_ARENA[0], height=CHEMO_ARENA[1])
    peak = np.array(CHEMO_SOURCE_XY)
    ctrl = np.array(CHEMO_CONTROL_XY)
    sim.env.add_source(peak[0], peak[1], kind="salt", strength=1.0,
                       sigma=CHEMO_SOURCE_SIGMA_MM)
    sim.reset(x=float(start_xy[0]), y=float(start_xy[1]), heading=heading)

    steps = int(minutes * 60.0 / sim.cfg.dt)
    near = far = 0
    d0 = _torus_dist(sim.body.X, peak, sim.env.width, sim.env.height)
    for _ in range(steps):
        sim.step()
        d = _torus_dist(sim.body.X, peak, sim.env.width, sim.env.height)
        dc = _torus_dist(sim.body.X, ctrl, sim.env.width, sim.env.height)
        # Dwell time at each spot, the finer-grained per-animal readout.
        if d < CHEMO_SCORE_RADIUS_MM:
            near += 1
        elif dc < CHEMO_SCORE_RADIUS_MM:
            far += 1
    d1 = _torus_dist(sim.body.X, peak, sim.env.width, sim.env.height)
    dc1 = _torus_dist(sim.body.X, ctrl, sim.env.width, sim.env.height)
    ci = (near - far) / max(near + far, 1)
    # Where this animal ENDS is what a plate assay actually scores: the
    # index counts animals in a region at scoring time, not the time one
    # animal spent there. Both are returned; see the assay docstring.
    where = ("near" if d1 < CHEMO_SCORE_RADIUS_MM
             else "far" if dc1 < CHEMO_SCORE_RADIUS_MM else "middle")
    return {"ci": round(ci, 3), "reversals": sim.state.reversal_count,
            "approached_mm": round(d0 - d1, 2), "scored": where,
            "start_deg": round(float(np.degrees(theta)), 1)}


@assay("chemotaxis",
       metric="chemotaxis index, (N_near - N_far) / N_total, mean +/- sd over "
             "replicates with randomized starts and headings",
       expected="wild type strongly positive (~0.5 or above) to NaCl; che-1 "
                "and tax-4 near zero",
       source="Ward 1973; Bargmann & Horvitz 1991; Pierce-Shimomura et al. 1999",
       description="Place the animal in a salt gradient and score where it "
                   "spends its time relative to the peak.")
def _chemotaxis(knockouts=(), ablations=(), seed=0, minutes=CHEMO_RUN_MINUTES,
                replicates=CHEMO_REPLICATES, workers=1) -> dict:
    """The plate assay, scored the way plates are scored.

    Ward's index is (N_near - N_far) / N_total over the ANIMALS on a plate
    at scoring time. Running one animal and taking the fraction of its run
    spent near the peak is a different statistic that happens to share the
    name, and it is not robust: a blind animal's dwell fraction depends on
    how much of the arena the scoring band covers, which is how an earlier
    version of this assay reported a strong index for a salt-blind mutant.

    So both are reported. `chemotaxis_index` counts where the replicates
    END, which is the population statistic and the one to quote.
    `dwell_index` is the per-animal time fraction, kept because it is a
    finer-grained readout of a single animal's behaviour and because the
    old number should stay comparable.

    Replicates are independent animals: no interaction is modelled, so this
    is a plate of animals that ignore each other. Aggregation and social
    feeding need real interaction (issue #24).

    Each replicate starts on a ring around the peak at a random angle with a
    random heading. An earlier version started the animal pointing directly
    at the peak, which returns a near-perfect index even for a salt-blind
    mutant -- the assay must be able to fail, or it measures nothing.
    Distances are torus-aware because the arena wraps.
    """
    tasks = []
    for r in range(replicates):
        rng = np.random.default_rng(1000 + seed * 100 + r)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        jitter = rng.uniform(0.0, CHEMO_START_JITTER_MM)
        start = (np.array(CHEMO_START_XY)
                 + jitter * np.array([np.cos(theta), np.sin(theta)]))
        heading = rng.uniform(0.0, 2.0 * np.pi)
        tasks.append((tuple(knockouts), tuple(ablations), seed + r, minutes,
                      tuple(start), heading, theta))
    if workers != 1 and replicates > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers or None) as pool:
            runs = list(pool.map(_chemo_run, tasks))
    else:
        runs = [_chemo_run(t) for t in tasks]
    cis = [r["ci"] for r in runs]
    revs = [r["reversals"] for r in runs]
    approaches = [r["approached_mm"] for r in runs]
    n_near = sum(1 for r in runs if r["scored"] == "near")
    n_far = sum(1 for r in runs if r["scored"] == "far")
    pop_ci = (n_near - n_far) / max(n_near + n_far, 1)
    # Binomial standard error on the population index, so a reader can see
    # when N is too small for the number to mean anything.
    n_scored = n_near + n_far
    pop_se = (2.0 * np.sqrt(max(n_near * n_far, 1)) / n_scored ** 1.5
              if n_scored else float("nan"))
    return {"chemotaxis_index": round(float(pop_ci), 3),
            "chemotaxis_index_se": round(float(pop_se), 3),
            "n_near": n_near, "n_far": n_far,
            "n_middle": len(runs) - n_scored,
            "dwell_index": round(float(np.mean(cis)), 3),
            "dwell_index_sd": round(float(np.std(cis)), 3),
            "animals": replicates,
            "replicates": replicates,
            "per_run": runs,
            "reversals_mean": round(float(np.mean(revs)), 2),
            "approached_mm_mean": round(float(np.mean(approaches)), 2),
            "reversal_note": ("0 reversals means no pirouettes: the biased "
                              "random walk is absent, so any positive index "
                              "here came from geometry or steering, not the "
                              "Pierce-Shimomura mechanism")}


@assay("thermotaxis",
       metric="net drift along the thermal gradient, mm",
       expected="above the cultivation temperature the animal is cryophilic "
                "and drifts down-gradient; tax-4 is athermotactic",
       source="Hedgecock & Russell 1975; Mori & Ohshima 1995; WormBook "
              "Thermotaxis",
       description="Put the animal on a linear thermal gradient warmer than "
                   "the temperature it remembers.")
def _thermotaxis(knockouts=(), ablations=(), seed=0, minutes=8.0) -> dict:
    sim = _build(knockouts, ablations, seed, width=60.0, height=44.0)
    sim.env.temp_low, sim.env.temp_high = 17.0, 25.0
    sim.env.cultivation_temp = 18.0        # animal is warmer than it likes
    sim.reset(x=10.0, y=0.0, heading=0.0)  # start on the warm side
    x0 = float(sim.body.X[0])
    for _ in range(int(minutes * 60.0 / sim.cfg.dt)):
        sim.step()
    x1 = float(sim.body.X[0])
    return {"drift_mm": round(x1 - x0, 2),
            "cryophilic": bool(x1 < x0),
            "start_temp_c": round(sim.env.temperature(np.array([x0, 0.0])), 2),
            "end_temp_c": round(sim.env.temperature(np.array([x1, 0.0])), 2),
            "reversals": sim.state.reversal_count}


# Plate taps in the behavioural literature are strong, standardized stimuli
# (a solenoid striking the plate), well above a single gentle stroke. In this
# model the tap drives both antagonistic fields at once and their competition
# cancels most of a strength-1 stimulus (naive deviation 0.0029 against a
# reversal threshold of 0.0034), so a naive strength-1 tap never reverses; at
# strength 2 the naive tap is reliably suprathreshold (0.0107, 3 of 3
# seeds), which matches the assay's design point of starting from a robust
# response and watching it wane (Rankin, Beck & Chiba 1990).
TAP_STRENGTH = 2.0


@assay("touch-habituation",
       metric="evoked command deviation across repeated plate taps, then "
              "after rest",
       expected="response magnitude wanes with repeated stimulation and "
                "recovers with rest; the canonical non-associative learning "
                "paradigm. The stimulus is a plate TAP, both receptive "
                "fields at once, and that matters: the tap response is the "
                "difference of two antagonistic reflexes, the anterior "
                "field's reversal against the posterior field's "
                "acceleration, so depression of both sides collapses the "
                "difference far faster than either reflex alone. A "
                "localised anterior poke habituates only weakly here, "
                "because that response is carried largely by gap junctions "
                "depression cannot touch and its chemical component is "
                "sign-mixed",
       source="Rankin, Beck & Chiba 1990 Behav Brain Res 37:89 (decrement "
              "and recovery); Wicks & Rankin 1995 J Neurosci 15:2434 (tap "
              "response integrates antagonistic ALM and PLM reflexes); "
              "Wicks & Rankin 1997 Behav Neurosci 111:342 (depression at "
              "the touch cell output synapses)",
       description="Deliver repeated plate taps (both fields), measure each "
                   "response's peak command deviation, rest, then probe for "
                   "recovery. The life clock is held: at the default 400x "
                   "development speedup a five-minute protocol is a day and "
                   "a half unfed, and starvation is what a decayed probe "
                   "would otherwise measure.")
def _habituation(knockouts=(), ablations=(), seed=0, taps=12,
                 isi_s=10.0, rest_s=180.0, probe_taps=2,
                 tap_strength=TAP_STRENGTH) -> dict:
    sim = _build(knockouts, ablations, seed, width=60.0, height=44.0)
    sim.cfg.life_speedup = 0.0
    # Held for the same reason as the life clock: the tap-locked windows
    # measure the EVOKED peak, and a spontaneous reversal landing in a
    # late window masquerades as a failure to habituate. See validate._quiet.
    sim.cfg.spont_rev_per_min_off_food = 0.0
    sim.cfg.spont_rev_per_min_on_food = 0.0
    for _ in range(500):
        sim.step()

    def tap() -> dict:
        r0 = sim.state.reversal_count
        sim.env.poke("anterior", tap_strength, duration=0.35)
        sim.env.poke("posterior", tap_strength, duration=0.35)
        peak = 0.0
        # Peak is read tap-locked, in the two seconds after the tap, not
        # across the whole interstimulus interval: the animal now reverses
        # spontaneously (Gray et al. 2005) and an upstate landing late in the
        # interval is not a tap response.
        for k in range(int(isi_s / sim.cfg.dt)):
            sim.step()
            if k * sim.cfg.dt < 2.0:
                peak = max(peak,
                           float(getattr(sim, "last_cmd_deviation", 0.0)))
        return {"reversed": int(sim.state.reversal_count > r0),
                "peak_deviation": round(peak, 5)}

    train = [tap() for _ in range(taps)]
    for _ in range(int(rest_s / sim.cfg.dt)):
        sim.step()
    probe = [tap() for _ in range(probe_taps)]

    def mag(rows):
        return float(np.mean([r["peak_deviation"] for r in rows])) if rows else 0.0

    early, late, rec = mag(train[:3]), mag(train[-3:]), mag(probe)
    return {
        "train": train,
        "probe_after_rest": probe,
        "early_mag": round(early, 5),
        "late_mag": round(late, 5),
        "decrement": round(1.0 - late / early, 3) if early > 0 else None,
        "recovery_mag": round(rec, 5),
        "recovered": bool(early > 0 and rec > late + 0.25 * (early - late)),
        "early_response_rate": round(
            sum(r["reversed"] for r in train[:6]) / 6.0, 3),
        "late_response_rate": round(
            sum(r["reversed"] for r in train[6:]) / max(len(train) - 6, 1), 3),
    }


@assay("basal-slowing",
       metric="fractional speed drop on encountering a bacterial lawn",
       expected="wild type slows on food; cat-2 (no dopamine) does not",
       source="Sawin, Ranganathan & Horvitz 2000 Neuron 26:619",
       description="Compare crawling speed off food with speed on a lawn, in "
                   "a well-fed animal.")
def _basal_slowing(knockouts=(), ablations=(), seed=0, minutes=2.0) -> dict:
    def speed(on_food: bool) -> float:
        sim = _build(knockouts, ablations, seed, width=60.0, height=44.0)
        if on_food:
            sim.env.add_source(0.0, 0.0, kind="food", strength=1.0, sigma=30.0)
        for _ in range(500):
            sim.step()
        d0 = sim.state.distance
        n = int(minutes * 60.0 / sim.cfg.dt)
        for _ in range(n):
            sim.step()
        return (sim.state.distance - d0) / (n * sim.cfg.dt)

    off, on = speed(False), speed(True)
    drop = (off - on) / off if off > 0 else 0.0
    return {"speed_off_food_mm_s": round(off, 4),
            "speed_on_food_mm_s": round(on, 4),
            "slowing_fraction": round(drop, 3),
            "slows_on_food": bool(drop > 0.05)}


@assay("touch-response",
       metric="reversal probability by body position",
       expected="anterior touch reverses, posterior touch does not; mec-4 "
                "nulls do neither",
       source="Chalfie & Sulston 1981; Chalfie et al. 1985",
       description="Score the response to a gentle stroke at several points "
                   "along the body.")
def _touch(knockouts=(), ablations=(), seed=0, repeats=3) -> dict:
    out = {}
    for u, label in [(0.0, "nose"), (0.2, "anterior"), (0.5, "mid_body"),
                     (0.85, "posterior")]:
        hits = 0
        for k in range(repeats):
            sim = _build(knockouts, ablations, seed + k, width=60.0, height=44.0)
            for _ in range(500):
                sim.step()
            r0 = sim.state.reversal_count
            sim.env.poke(u, 1.0, duration=0.4)
            for _ in range(150):
                sim.step()
            hits += int(sim.state.reversal_count > r0)
        out[label] = round(hits / repeats, 2)
    return {"reversal_probability": out}


@assay("lifespan",
       metric="adult lifespan in days, and brood size",
       expected="wild type ~15 d and ~300 progeny at 20 C; daf-2 roughly "
                "doubles lifespan unless daf-16 is also removed",
       source="Huang, Xiong & Kornfeld 2004 PNAS; Kenyon et al. 1993 Nature; "
              "Hodgkin & Barnes 1991",
       description="Run one animal from fertilised egg to death.")
def _lifespan(knockouts=(), ablations=(), seed=0, temp_c=20.0) -> dict:
    from .genome import Genome
    from .lifecycle import Lifecycle
    g = Genome.load()
    for k in knockouts:
        g.knock_out(k)
    L = Lifecycle()
    marks: dict = {}
    t = 0.0
    while L.alive and t < 120 * 86400:
        ev = L.step(60.0, food=1.0, temp_c=temp_c, pheromone=0.0,
                    longevity_scale=g.longevity_scale())
        t += 60.0
        for k in ("hatch", "adult", "sperm_exhausted", "death"):
            if k in ev and k not in marks:
                marks[k] = L.age_s / 3600.0
    return {"embryo_h": round(marks.get("hatch", 0.0), 1),
            "hatch_to_adult_h": round(marks.get("adult", 0.0)
                                      - marks.get("hatch", 0.0), 1),
            "adult_lifespan_d": round(L.adult_day, 2),
            "total_age_d": round(L.age_s / 86400.0, 2),
            "brood_size": L.eggs_laid,
            "reproductive_period_d": round((marks.get("sperm_exhausted", 0.0)
                                            - marks.get("adult", 0.0)) / 24.0, 2),
            "cause_of_death": L.cause_of_death}


def run_assay(name: str, **kw) -> dict:
    if name not in ASSAYS:
        raise KeyError(f"unknown assay {name!r}; have {sorted(ASSAYS)}")
    a = ASSAYS[name]
    # Forward only the kwargs this assay accepts, so shared options (workers,
    # seed) can be passed without every assay having to take them.
    import inspect
    sig = inspect.signature(a.run)
    kw = {k: v for k, v in kw.items() if k in sig.parameters}
    result = a.run(**kw)
    return {"assay": a.name, "metric": a.metric, "expected": a.expected,
            "source": a.source, "result": result}
