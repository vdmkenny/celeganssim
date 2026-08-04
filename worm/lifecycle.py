"""Development, feeding and the dauer decision.

C. elegans runs a fixed post-embryonic programme at 20 C on plentiful food:

    L1  ~12 h   ->  L2  ~8 h  ->  L3  ~8 h  ->  L4  ~10 h  ->  adult
    hatch to egg-laying adult is roughly 45-50 h at 20 C (Byerly et al. 1976)

Development is temperature-dependent and food-dependent. Two branch points are
modelled because both are behaviourally visible:

  * **L1 arrest.** Hatching with no food halts development at L1 until food
    appears; the animal survives but does not grow.
  * **Dauer.** Late L1 assesses crowding (ascaroside pheromone), food and
    temperature. A bad verdict sends it into dauer, an alternative stress-
    resistant L3 that does not feed and can persist for months.

Body size scales through the stages, which feeds straight into the biomechanics
(a 0.25 mm L1 is not a 1.0 mm adult), and feeding rate is set by pharyngeal
pumping, which is itself under serotonergic control.

References: Byerly, Cassada & Russell 1976 Dev Biol 51:23 (growth timing);
Cassada & Russell 1975 Dev Biol 46:326 (dauer); Golden & Riddle 1984 Dev Biol
102:368 (dauer pheromone/food/temperature); Avery & Horvitz 1990 J Exp Zool 253:263
(pumping and feeding); Fang-Yen et al. 2009 PNAS 106:20093 (pumping vs food).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Stage -> (hours at 20 C on food, body length in mm at the START of the stage)
# Lengths from WormAtlas growth data.
STAGES: dict[str, tuple[float, float]] = {
    "L1": (12.0, 0.25),
    "L2": (8.0, 0.38),
    "L3": (8.0, 0.52),
    "L4": (10.0, 0.79),
    "adult": (float("inf"), 1.11),
}
ORDER = ["L1", "L2", "L3", "L4", "adult"]

# Dauer is an alternative L3. It is thinner and shorter than a normal L3 and
# seals its mouth, so it does not feed at all.
DAUER_LENGTH = 0.40

# Pharyngeal pumping, pumps per second. Well-fed adults run 4-5 Hz; pumping
# collapses off food and is potentiated by serotonin.
PUMP_MAX_HZ = 4.5
# Reserves are a normalised 0..1 store, not a real mass. The two constants
# below are calibrated against measured timescales rather than invented:
#   * A well-fed animal should refill an empty store in a few hours.
#   * A starved adult should survive on reserves for days, not minutes.
#     L1 larvae survive starvation for 1-2 weeks and adults roughly 10-15 days
#     (Angelo & Van Gilst 2009 Science 326:954; Baugh 2013 Genetics 194:539).
# With METABOLIC_RATE below, a full store drains in ~2.5 days off food, and a
# 4.5 Hz pumping adult refills it in ~3 h.
FOOD_PER_PUMP = 2.0e-5
METABOLIC_RATE = 4.5e-6


@dataclass
class LifecycleParams:
    # Q10-style temperature scaling of developmental rate. Development roughly
    # halves in duration going from 15 C to 25 C.
    q10: float = 2.0
    reference_temp: float = 20.0
    # Reserves below this fraction count as starving.
    starve_threshold: float = 0.25
    # Dauer entry is assessed once, late in L1.
    dauer_decision_at: float = 0.75      # fraction through L1
    dauer_pheromone_threshold: float = 0.5
    dauer_recovery_food: float = 0.6
    # Egg laying (adults only).
    egg_interval_s: float = 1200.0       # ~1 egg per 20 min in the model
    eggs_need_reserves: float = 0.45


@dataclass
class Lifecycle:
    """Developmental state of one animal."""

    params: LifecycleParams = field(default_factory=LifecycleParams)
    stage: str = "L1"
    stage_progress: float = 0.0    # 0..1 through the current stage
    age_s: float = 0.0
    reserves: float = 0.6          # 0..1 nutritional stores
    dauer: bool = False
    arrested: bool = False         # L1 arrest for lack of food
    eggs_laid: int = 0
    total_ingested: float = 0.0
    pump_hz: float = 0.0
    _dauer_decided: bool = False
    _egg_timer: float = 0.0

    # -- derived quantities ---------------------------------------------
    @property
    def body_length_mm(self) -> float:
        if self.dauer:
            return DAUER_LENGTH
        i = ORDER.index(self.stage)
        start = STAGES[self.stage][1]
        if self.stage == "adult":
            # Adults keep growing slowly for the first day or so.
            return start * (1.0 + 0.15 * min(self.stage_progress, 1.0))
        end = STAGES[ORDER[i + 1]][1]
        return start + (end - start) * self.stage_progress

    @property
    def is_adult(self) -> bool:
        return self.stage == "adult" and not self.dauer

    @property
    def starving(self) -> bool:
        return self.reserves < self.params.starve_threshold

    def temperature_factor(self, temp_c: float) -> float:
        """Developmental rate multiplier at a given temperature."""
        p = self.params
        return float(p.q10 ** ((temp_c - p.reference_temp) / 10.0))

    def pumping_rate(self, food: float, serotonin_scale: float = 1.0) -> float:
        """Pumps per second given local bacterial density.

        Dauers have a sealed buccal cavity and never pump. Off food, pumping
        drops to a low basal rate. Serotonin potentiates food-stimulated
        pumping, which is why tph-1 mutants eat less.
        """
        if self.dauer:
            return 0.0
        if food <= 0.01:
            return 0.4 * serotonin_scale
        basal = 0.9
        stimulated = (PUMP_MAX_HZ - basal) * min(food, 1.0) * serotonin_scale
        return basal + stimulated

    # -- update ---------------------------------------------------------
    def step(self, dt: float, *, food: float, temp_c: float,
             pheromone: float = 0.0, serotonin_scale: float = 1.0) -> dict:
        """Advance development by dt seconds. Returns any events that fired."""
        p = self.params
        events: dict = {}
        self.age_s += dt

        # --- feeding ---
        self.pump_hz = self.pumping_rate(food, serotonin_scale)
        ingested = self.pump_hz * FOOD_PER_PUMP * dt * min(food, 1.0)
        self.total_ingested += ingested
        size = max(self.body_length_mm, 0.05)
        drain = METABOLIC_RATE * dt * (0.35 if self.dauer else 1.0) * (size / 1.0) ** 0.75
        self.reserves = max(0.0, min(1.0, self.reserves + ingested - drain))

        # --- dauer exit ---
        if self.dauer:
            if food >= p.dauer_recovery_food and pheromone < p.dauer_pheromone_threshold:
                self.dauer = False
                self.stage = "L4"
                self.stage_progress = 0.0
                events["dauer_exit"] = True
            return events

        # --- L1 arrest: hatching without food halts development outright ---
        # This is a food-sensing decision, not a reserves one: an L1 that
        # hatches onto an empty plate arrests immediately and can hold there
        # for one to two weeks (Baugh 2013 Genetics 194:539).
        if self.stage == "L1" and food < 0.05:
            self.arrested = True
            events["arrested"] = True
            return events
        self.arrested = False

        # --- developmental clock ---
        rate = self.temperature_factor(temp_c)
        if self.starving:
            rate *= 0.25   # starvation slows but does not fully stop growth
        hours, _ = STAGES[self.stage]
        if hours != float("inf"):
            self.stage_progress += (dt / 3600.0) * rate / hours
        else:
            self.stage_progress += (dt / 3600.0) * rate / 24.0

        # --- dauer decision, late L1 ---
        if (self.stage == "L1" and not self._dauer_decided
                and self.stage_progress >= p.dauer_decision_at):
            self._dauer_decided = True
            crowded = pheromone >= p.dauer_pheromone_threshold
            hungry = food < 0.2
            hot = temp_c > 25.0
            if crowded and (hungry or hot):
                self.dauer = True
                self.stage = "dauer"
                self.stage_progress = 0.0
                events["dauer_entry"] = {
                    "pheromone": round(pheromone, 3),
                    "food": round(food, 3),
                    "temp_c": round(temp_c, 2),
                }
                return events

        # --- moult ---
        if self.stage_progress >= 1.0 and self.stage != "adult":
            i = ORDER.index(self.stage)
            self.stage = ORDER[i + 1]
            self.stage_progress = 0.0
            events["moult"] = self.stage

        # --- egg laying ---
        if self.is_adult and self.reserves >= p.eggs_need_reserves:
            self._egg_timer += dt
            if self._egg_timer >= p.egg_interval_s:
                self._egg_timer = 0.0
                self.eggs_laid += 1
                self.reserves = max(0.0, self.reserves - 0.02)
                events["egg"] = self.eggs_laid
        return events

    def summary(self) -> dict:
        return {
            "stage": "dauer" if self.dauer else self.stage,
            "stage_progress": round(self.stage_progress, 3),
            "age_h": round(self.age_s / 3600.0, 2),
            "body_length_mm": round(self.body_length_mm, 3),
            "reserves": round(self.reserves, 3),
            "starving": self.starving,
            "arrested": self.arrested,
            "dauer": self.dauer,
            "pump_hz": round(self.pump_hz, 2),
            "eggs_laid": self.eggs_laid,
        }
