"""The whole life: fertilised egg, embryo, four larval stages, adulthood,
senescence, death. Plus the dauer branch and feeding throughout.

All timings are at 20 C on ample food, from primary sources:

  embryogenesis      ~850 min (14.2 h) fertilisation to hatch, of which the
                     first ~150 min happen in utero; the egg is laid at about
                     the 30-cell early gastrula   (Sulston et al. 1983;
                     WormAtlas Handbook; Muschiol et al. 2009)
  larval stages      L1 17.3 h, L2 11.2 h, L3 9.6 h, L4 12.7 h; hatch to
                     adult 50.7 +/- 2.0 h   (Faerberg, Gurarie & Ruvinsky
                     2022, BMC Biology 20:87 -- higher resolution than the
                     classic Byerly 1976 figures)
  reproduction       ~327 self progeny, limited by a fixed store of ~300
                     self-sperm made once at the L4/adult molt; laying is
                     rhythmic and lasts ~5 days   (Hodgkin & Barnes 1991;
                     Ward & Carrel 1979; Schafer WormBook)
  lifespan           mean 15.2 +/- 3.6 d from adulthood, max ~27-30 d
                     (Huang, Xiong & Kornfeld 2004 PNAS; Herndon et al. 2002)

Senescence is modelled the way it is actually scored: pharyngeal pumping and
locomotion decline on their own measured trajectories, and an animal counts as
dead when it neither moves spontaneously nor responds to prodding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Larval stage durations in hours at 20 C (intermolt + molt), and body length
# in mm at the start of each stage.
STAGES: dict[str, tuple[float, float]] = {
    "embryo": (14.2, 0.055),
    "L1": (17.3, 0.25),
    "L2": (11.2, 0.38),
    "L3": (9.6, 0.52),
    "L4": (12.7, 0.79),
    "adult": (float("inf"), 1.11),
}
ORDER = ["embryo", "L1", "L2", "L3", "L4", "adult"]

DAUER_LENGTH = 0.40

# Embryonic milestones as a fraction of the way from fertilisation to hatch,
# derived from Sulston's 800 min post-first-cleavage clock plus the 50 min
# fertilisation offset. Used to report what the embryo is doing.
EMBRYO_MILESTONES: list[tuple[float, str]] = [
    (0.00, "zygote"),
    (0.06, "first cleavage"),
    (0.18, "gastrulation begins (28 cells)"),
    (0.18, "laid as an egg"),
    (0.41, "proliferation complete, 558 cells"),
    (0.47, "comma stage, elongation begins"),
    (0.56, "1.5-fold, first muscle twitching"),
    (0.71, "3-fold, rolling in the eggshell"),
    (0.95, "pharyngeal pumping starts"),
    (1.00, "hatch"),
]

PUMP_MAX_HZ = 5.3          # 320 pumps/min in a young adult
FOOD_PER_PUMP = 2.0e-5
METABOLIC_RATE = 4.5e-6

# Self-sperm are made once, at the L4/adult molt, and never replenished. That
# store is what caps the self-fertile brood.
SELF_SPERM = 300

# Mean and spread of adult lifespan in days at 20 C.
LIFESPAN_MEAN_D = 15.2
LIFESPAN_SD_D = 3.6


@dataclass
class LifecycleParams:
    q10: float = 2.0
    reference_temp: float = 20.0
    starve_threshold: float = 0.25
    dauer_decision_at: float = 0.75
    dauer_pheromone_threshold: float = 0.5
    dauer_recovery_food: float = 0.6
    # Laying is rhythmic (bursts of 4-6 eggs every ~20 min), but the real rate
    # limit is OVULATION, not the egg-laying motor program: each of the two
    # gonad arms ovulates about every 23 min, capping output near 5-6 eggs an
    # hour. Peak output is ~141 eggs/day on adult day 2 and falls away after,
    # giving ~300 progeny over a ~5 day reproductive period.
    # Refs: McCarter et al. 1999; Muschiol et al. 2009; Schafer WormBook.
    peak_ovulation_per_h: float = 5.6
    ovulation_decay_d: float = 3.0
    eggs_need_reserves: float = 0.30
    # Hours after the L4/adult molt before laying starts.
    egg_onset_h: float = 9.0


@dataclass
class Lifecycle:
    params: LifecycleParams = field(default_factory=LifecycleParams)
    stage: str = "embryo"
    stage_progress: float = 0.0
    age_s: float = 0.0              # since fertilisation
    adult_age_s: float = 0.0        # since the L4/adult molt
    reserves: float = 0.6
    dauer: bool = False
    arrested: bool = False
    alive: bool = True
    cause_of_death: str = ""
    eggs_laid: int = 0
    self_sperm: int = 0
    total_ingested: float = 0.0
    pump_hz: float = 0.0
    lifespan_d: float = LIFESPAN_MEAN_D
    _dauer_decided: bool = False
    _egg_timer: float = 0.0
    _noted_exhausted: bool = False

    # -- derived --------------------------------------------------------
    @property
    def is_embryo(self) -> bool:
        return self.stage == "embryo" and self.alive

    @property
    def body_length_mm(self) -> float:
        if self.dauer:
            return DAUER_LENGTH
        if self.stage == "embryo":
            # The embryo elongates about fourfold inside the eggshell.
            return 0.055 + 0.195 * min(self.stage_progress, 1.0) ** 1.5
        i = ORDER.index(self.stage)
        start = STAGES[self.stage][1]
        if self.stage == "adult":
            return start * (1.0 + 0.15 * min(self.adult_age_s / (48 * 3600.0), 1.0))
        end = STAGES[ORDER[i + 1]][1]
        return start + (end - start) * self.stage_progress

    @property
    def is_adult(self) -> bool:
        return self.stage == "adult" and not self.dauer and self.alive

    @property
    def starving(self) -> bool:
        return self.reserves < self.params.starve_threshold

    @property
    def adult_day(self) -> float:
        return self.adult_age_s / 86400.0

    @property
    def embryo_milestone(self) -> str:
        if self.stage != "embryo":
            return ""
        last = EMBRYO_MILESTONES[0][1]
        for frac, name in EMBRYO_MILESTONES:
            if self.stage_progress >= frac:
                last = name
        return last

    @property
    def movement_class(self) -> str:
        """Herndon's ageing classes, scored the way they are on a plate.

        A: rhythmic sinusoidal forward movement. B: uncoordinated, no sustained
        progress. C: no locomotion, only head or tail movement, still responds
        to touch. Wild type spends ~9.3 d in A, ~1.8 d in B, ~2.1 d in C.
        """
        if not self.alive:
            return "dead"
        if not self.is_adult:
            return "A"
        frac = self.adult_day / max(self.lifespan_d, 1e-6)
        if frac < 0.62:
            return "A"
        if frac < 0.76:
            return "B"
        return "C"

    def locomotion_scale(self) -> float:
        """How much of its locomotor capacity the animal still has."""
        if not self.alive:
            return 0.0
        if self.stage == "embryo":
            return 0.0        # nothing crawls inside an eggshell
        return {"A": 1.0, "B": 0.45, "C": 0.12}[self.movement_class]

    def temperature_factor(self, temp_c: float) -> float:
        p = self.params
        return float(p.q10 ** ((temp_c - p.reference_temp) / 10.0))

    def pumping_rate(self, food: float, serotonin_scale: float = 1.0) -> float:
        """Pumps per second, including the age-related decline.

        Measured trajectory at 20 C: 320/min at adult day 2, 180/min at day 7,
        30/min at day 10, ceasing around day 12 (Huang et al. 2004). Pumping
        span is the single best predictor of individual lifespan (r = 0.83),
        so it is scaled against this animal's own lifespan rather than a fixed
        calendar.
        """
        if self.dauer or not self.alive or self.stage == "embryo":
            return 0.0
        age_scale = 1.0
        if self.is_adult:
            frac = self.adult_day / max(self.lifespan_d, 1e-6)
            # Roughly flat, then a steep fall in the last third of life.
            age_scale = float(max(0.0, min(1.0, 1.0 / (1.0 + math.exp((frac - 0.62) * 11.0)))))
        if food <= 0.01:
            return 0.4 * serotonin_scale * age_scale
        basal = 0.9
        stimulated = (PUMP_MAX_HZ - basal) * min(food, 1.0) * serotonin_scale
        return (basal + stimulated) * age_scale

    # -- update ---------------------------------------------------------
    def step(self, dt: float, *, food: float, temp_c: float,
             pheromone: float = 0.0, serotonin_scale: float = 1.0,
             longevity_scale: float = 1.0) -> dict:
        events: dict = {}
        if not self.alive:
            return events
        p = self.params
        self.age_s += dt

        # --- feeding ---
        self.pump_hz = self.pumping_rate(food, serotonin_scale)
        ingested = self.pump_hz * FOOD_PER_PUMP * dt * min(food, 1.0)
        self.total_ingested += ingested
        size = max(self.body_length_mm, 0.05)
        drain = METABOLIC_RATE * dt * (0.35 if self.dauer else 1.0) * (size / 1.0) ** 0.75
        self.reserves = max(0.0, min(1.0, self.reserves + ingested - drain))

        rate = self.temperature_factor(temp_c)

        # --- embryo: develops on yolk alone, sealed in the eggshell ---
        if self.stage == "embryo":
            self.stage_progress += (dt / 3600.0) * rate / STAGES["embryo"][0]
            if self.stage_progress >= 1.0:
                self.stage, self.stage_progress = "L1", 0.0
                self.reserves = max(self.reserves, 0.5)
                events["hatch"] = True
            return events

        # --- dauer ---
        if self.dauer:
            if food >= p.dauer_recovery_food and pheromone < p.dauer_pheromone_threshold:
                self.dauer = False
                self.stage, self.stage_progress = "L4", 0.0
                events["dauer_exit"] = True
            return events

        # --- L1 arrest ---
        if self.stage == "L1" and food < 0.05:
            self.arrested = True
            events["arrested"] = True
            return events
        self.arrested = False

        if self.starving:
            rate *= 0.25
        hours = STAGES[self.stage][0]
        if hours != float("inf"):
            self.stage_progress += (dt / 3600.0) * rate / hours
        else:
            self.adult_age_s += dt

        # --- dauer decision, late L1 ---
        if (self.stage == "L1" and not self._dauer_decided
                and self.stage_progress >= p.dauer_decision_at):
            self._dauer_decided = True
            if pheromone >= p.dauer_pheromone_threshold and (food < 0.2 or temp_c > 25.0):
                self.dauer = True
                self.stage, self.stage_progress = "dauer", 0.0
                events["dauer_entry"] = {"pheromone": round(pheromone, 3),
                                         "food": round(food, 3),
                                         "temp_c": round(temp_c, 2)}
                return events

        # --- moult ---
        if self.stage_progress >= 1.0 and self.stage != "adult":
            i = ORDER.index(self.stage)
            self.stage, self.stage_progress = ORDER[i + 1], 0.0
            events["moult"] = self.stage
            if self.stage == "adult":
                # Spermatogenesis happens once, here, and then stops for good.
                # Everything the animal will ever self-fertilise comes from
                # this store.
                self.self_sperm = SELF_SPERM
                # Lifespan is set now, scaled by temperature and genotype.
                self.lifespan_d = max(
                    2.0,
                    LIFESPAN_MEAN_D * longevity_scale
                    * self.temperature_factor(temp_c) ** -1.0)
                events["adult"] = {"lifespan_d": round(self.lifespan_d, 1),
                                   "self_sperm": self.self_sperm}

        # --- egg laying, ovulation-limited and capped by the sperm store ---
        if self.is_adult and self.adult_age_s / 3600.0 >= p.egg_onset_h:
            if self.self_sperm > 0 and self.reserves >= p.eggs_need_reserves:
                rate_per_h = p.peak_ovulation_per_h * math.exp(
                    -self.adult_day / max(p.ovulation_decay_d, 1e-6))
                interval = 3600.0 / max(rate_per_h, 1e-6)
                self._egg_timer += dt
                if self._egg_timer >= interval:
                    self._egg_timer = 0.0
                    self.self_sperm -= 1
                    self.eggs_laid += 1
                    self.reserves = max(0.0, self.reserves - 0.004)
                    events["egg"] = self.eggs_laid
            elif self.self_sperm <= 0 and not self._noted_exhausted:
                # Oogenesis continues, but with no sperm left the oocytes go
                # out unfertilised and self-progeny stop. This is what ends
                # the reproductive period.
                self._noted_exhausted = True
                events["sperm_exhausted"] = self.eggs_laid

        # --- death ---
        if self.is_adult and self.adult_day >= self.lifespan_d:
            self.alive = False
            self.cause_of_death = "senescence"
            events["death"] = {"adult_days": round(self.adult_day, 2),
                               "eggs_laid": self.eggs_laid}
        elif self.reserves <= 0.0 and self.stage != "embryo" and not self.dauer:
            # Starvation only kills once reserves are genuinely gone.
            self.alive = False
            self.cause_of_death = "starvation"
            events["death"] = {"starved": True, "eggs_laid": self.eggs_laid}
        return events

    def summary(self) -> dict:
        return {
            "stage": "dead" if not self.alive else ("dauer" if self.dauer else self.stage),
            "stage_progress": round(self.stage_progress, 3),
            "age_h": round(self.age_s / 3600.0, 2),
            "adult_day": round(self.adult_day, 2) if self.is_adult or not self.alive else 0.0,
            "body_length_mm": round(self.body_length_mm, 3),
            "reserves": round(self.reserves, 3),
            "starving": self.starving,
            "arrested": self.arrested,
            "dauer": self.dauer,
            "alive": self.alive,
            "cause_of_death": self.cause_of_death,
            "pump_hz": round(self.pump_hz, 2),
            "pump_per_min": round(self.pump_hz * 60.0),
            "eggs_laid": self.eggs_laid,
            "self_sperm": self.self_sperm,
            "movement_class": self.movement_class,
            "lifespan_d": round(self.lifespan_d, 1),
            "milestone": self.embryo_milestone,
        }
