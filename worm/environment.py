"""The world the worm lives in: gradients, food, temperature and pokes.

Everything is expressed in millimetres, matching the 1 mm body. Fields are
sampled pointwise at whatever position the relevant sensory ending sits at,
which is what makes the temporal-derivative sensors (ASE, AWC) behave the way
they do in a real gradient assay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Source:
    """A radially symmetric Gaussian source of some quantity."""

    x: float
    y: float
    strength: float = 1.0
    sigma: float = 12.0
    kind: str = "salt"  # salt | odor | repellent | food

    def at(self, p: np.ndarray) -> float:
        d2 = (p[0] - self.x) ** 2 + (p[1] - self.y) ** 2
        return self.strength * math.exp(-d2 / (2.0 * self.sigma ** 2))


@dataclass
class Poke:
    """A transient mechanical stimulus applied to part of the body."""

    region: str          # "anterior" | "posterior" | "nose"
    strength: float = 1.0
    remaining: float = 0.25   # seconds


@dataclass
class Environment:
    width: float = 120.0
    height: float = 90.0
    sources: list[Source] = field(default_factory=list)
    pokes: list[Poke] = field(default_factory=list)

    # Linear thermal gradient across the arena, plus the worm's remembered
    # cultivation temperature.
    temp_low: float = 17.0
    temp_high: float = 25.0
    cultivation_temp: float = 20.0

    # Ambient oxygen as a percentage. C. elegans prefers roughly 5-12%.
    oxygen: float = 21.0

    # Uniform ascaroside background, i.e. how crowded the plate is. Drives the
    # dauer decision together with food and temperature.
    background_pheromone: float = 0.0

    # How fast one animal grazes down a lawn. A single 1 mm worm against a
    # bacterial lawn is a very small mouth against a lot of food, so this is
    # deliberately small: grazing is visible over hours, not seconds.
    depletion_rate: float = 0.10

    # Viscous medium: normal/tangential drag ratio. ~32 on agar, ~1.5 in water.
    drag_ratio: float = 32.0

    def add_source(self, x, y, kind="salt", strength=1.0, sigma=12.0) -> Source:
        s = Source(x=x, y=y, strength=strength, sigma=sigma, kind=kind)
        self.sources.append(s)
        return s

    def clear_sources(self, kind: str | None = None) -> None:
        if kind is None:
            self.sources.clear()
        else:
            self.sources = [s for s in self.sources if s.kind != kind]

    def poke(self, region: str, strength: float = 1.0, duration: float = 0.25) -> None:
        self.pokes.append(Poke(region=region, strength=strength, remaining=duration))

    def concentration(self, p: np.ndarray, kind: str) -> float:
        return sum(s.at(p) for s in self.sources if s.kind == kind)

    def temperature(self, p: np.ndarray) -> float:
        u = float(np.clip(p[0] / max(self.width, 1e-6) + 0.5, 0.0, 1.0))
        return self.temp_low + u * (self.temp_high - self.temp_low)

    def on_food(self, p: np.ndarray) -> float:
        return float(np.clip(self.concentration(p, "food"), 0.0, 1.0))

    def pheromone_at(self, p: np.ndarray) -> float:
        """Ascaroside density. Stands in for population crowding."""
        return float(np.clip(self.concentration(p, "pheromone")
                             + self.background_pheromone, 0.0, 2.0))

    def consume(self, p: np.ndarray, amount: float) -> float:
        """Eat bacteria at p, depleting the lawn. Returns what was actually taken.

        A real worm grazing a lawn leaves a cleared track behind it, which is
        what eventually forces it to move on.
        """
        if amount <= 0:
            return 0.0
        lawns = [s for s in self.sources if s.kind == "food"]
        if not lawns:
            return 0.0
        weights = [s.at(p) for s in lawns]
        total = sum(weights)
        if total <= 1e-9:
            return 0.0
        taken = 0.0
        for src, w in zip(lawns, weights):
            share = amount * (w / total) * self.depletion_rate
            removed = min(src.strength, share)
            src.strength -= removed
            taken += removed
        self.sources = [s for s in self.sources
                        if s.kind != "food" or s.strength > 0.005]
        return taken

    def step(self, dt: float) -> None:
        for pk in self.pokes:
            pk.remaining -= dt
        self.pokes = [p for p in self.pokes if p.remaining > 0]

    def active_poke(self, region: str) -> float:
        return sum(p.strength for p in self.pokes if p.region == region)

    def wrap(self, p: np.ndarray) -> np.ndarray:
        """Toroidal arena, so the worm never gets stuck at a wall."""
        w, h = self.width, self.height
        return np.array([
            (p[0] + w / 2) % w - w / 2,
            (p[1] + h / 2) % h - h / 2,
        ])
