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
@assay("chemotaxis",
       metric="chemotaxis index, (N_near - N_far) / N_total",
       expected="wild type strongly positive (~0.5 or above) to NaCl; che-1 "
                "and tax-4 near zero",
       source="Ward 1973; Bargmann & Horvitz 1991; Pierce-Shimomura et al. 1999",
       description="Place the animal in a salt gradient and score where it "
                   "spends its time relative to the peak.")
def _chemotaxis(knockouts=(), ablations=(), seed=0, minutes=8.0) -> dict:
    """Classic population assay run on one animal over time.

    The chemotaxis index is normally counts of animals in a scoring region;
    with one animal the time-equivalent is the fraction of the run spent in
    the near half versus the far half of the gradient.
    """
    sim = _build(knockouts, ablations, seed, width=60.0, height=44.0)
    peak = np.array([16.0, 0.0])
    sim.env.add_source(peak[0], peak[1], kind="salt", strength=1.0, sigma=14.0)
    sim.reset(x=-16.0, y=0.0, heading=0.0)

    steps = int(minutes * 60.0 / sim.cfg.dt)
    near = far = 0
    d0 = float(np.linalg.norm(sim.body.X - peak))
    track = []
    for i in range(steps):
        sim.step()
        d = float(np.linalg.norm(sim.body.X - peak))
        if i % 25 == 0:
            track.append(round(d, 2))
        if d < 16.0:
            near += 1
        elif d > 32.0:
            far += 1
    total = max(near + far, 1)
    ci = (near - far) / total
    d1 = float(np.linalg.norm(sim.body.X - peak))
    return {"chemotaxis_index": round(ci, 3),
            "start_distance_mm": round(d0, 2),
            "end_distance_mm": round(d1, 2),
            "approached_mm": round(d0 - d1, 2),
            "reversals": sim.state.reversal_count,
            "omega_turns": sim.state.omega_count,
            "distance_travelled_mm": round(sim.state.distance, 1),
            "distance_track": track[:60]}


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


@assay("touch-habituation",
       metric="response probability across repeated taps",
       expected="responses wane with repeated stimulation and recover with "
                "rest; the canonical non-associative learning paradigm",
       source="Rankin, Beck & Chiba 1990 Behav Brain Res 37:89; Wicks & "
              "Rankin 1996",
       description="Deliver repeated anterior taps and count how often each "
                   "one still produces a reversal.")
def _habituation(knockouts=(), ablations=(), seed=0, taps=12,
                 isi_s=10.0) -> dict:
    sim = _build(knockouts, ablations, seed, width=60.0, height=44.0)
    for _ in range(500):
        sim.step()
    responses = []
    for _ in range(taps):
        r0 = sim.state.reversal_count
        sim.env.poke("anterior", 1.0, duration=0.35)
        for _ in range(int(3.0 / sim.cfg.dt)):
            sim.step()
        responses.append(int(sim.state.reversal_count > r0))
        for _ in range(int(max(isi_s - 3.0, 0.0) / sim.cfg.dt)):
            sim.step()
    half = max(len(responses) // 2, 1)
    return {"responses": responses,
            "response_rate": round(sum(responses) / len(responses), 3),
            "first_half_rate": round(sum(responses[:half]) / half, 3),
            "second_half_rate": round(sum(responses[half:]) /
                                      max(len(responses) - half, 1), 3),
            "habituated": bool(sum(responses[half:]) < sum(responses[:half]))}


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
    result = a.run(**kw)
    return {"assay": a.name, "metric": a.metric, "expected": a.expected,
            "source": a.source, "result": result}
