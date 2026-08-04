"""Sensory transduction: world state -> injected current in named neurons.

Each modality writes into the specific neurons that carry it in the real animal,
and each is scaled by the genes that implement its transduction machinery, so
`mec-4` really does silence gentle touch and `che-1` really does remove salt
sensing without touching odour.

Polarities worth stating explicitly, because the names mislead:
  * ASEL is the salt-ON cell (responds to concentration UPsteps), ASER is the
    salt-OFF cell (DOWNsteps). Together they drive the biased random walk.
  * AWC is an OFF cell despite the AWC-ON/AWC-OFF naming, which refers to a
    developmental identity. It is tonically active and *suppressed* by odour,
    so odour removal excites it.
  * URX/AQR/PQR are tonic HIGH-oxygen sensors; BAG responds to oxygen DOWNsteps
    and to CO2.
"""

from __future__ import annotations

import numpy as np

from .connectome import Connectome
from .environment import Environment
from .genome import Genome

# Which cells carry which modality.
SENSORS = {
    "salt_on": ["ASEL"],
    "salt_off": ["ASER"],
    "odor_attract": ["AWAL", "AWAR", "AWCL", "AWCR"],
    "odor_avoid": ["AWBL", "AWBR"],
    "nociception": ["ASHL", "ASHR"],
    "thermo": ["AFDL", "AFDR"],
    "touch_anterior": ["ALML", "ALMR", "AVM"],
    "touch_posterior": ["PLML", "PLMR"],
    "nose_touch": ["ASHL", "ASHR", "FLPL", "FLPR", "OLQDL", "OLQDR",
                   "OLQVL", "OLQVR"],
    "oxygen_high": ["URXL", "URXR", "AQR", "PQR"],
    "oxygen_low": ["BAGL", "BAGR"],
    "food_mech": ["CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER"],
}

# Which gene-level knob gates each modality.
GATE = {
    "salt_on": ("salt", "chemotaxis"),
    "salt_off": ("salt", "chemotaxis"),
    "odor_attract": ("odor", "chemotaxis"),
    "odor_avoid": ("odor", "chemotaxis"),
    "nociception": ("nociception",),
    "thermo": ("thermotaxis",),
    "touch_anterior": ("touch",),
    "touch_posterior": ("touch",),
    "nose_touch": ("nociception",),
    "oxygen_high": (),
    "oxygen_low": (),
    "food_mech": (),
}


class SensorySystem:
    def __init__(self, conn: Connectome, genome: Genome) -> None:
        self.conn = conn
        self.genome = genome
        self.idx = {
            k: conn.indices([n for n in names if n in conn.index])
            for k, names in SENSORS.items()
        }
        self.missing = {
            k: [n for n in names if n not in conn.index]
            for k, names in SENSORS.items()
        }
        # Adaptive state for the derivative-taking sensors.
        self._salt_prev: float | None = None
        self._odor_prev: float | None = None
        self._temp_prev: float | None = None
        self.last: dict[str, float] = {}

    def _gain(self, modality: str) -> float:
        g = 1.0
        for key in GATE.get(modality, ()):
            g *= self.genome.sensory_scale(key)
        return g

    def compute(self, env: Environment, head: np.ndarray, tail: np.ndarray,
                dt: float, amplitude: float = 55.0) -> np.ndarray:
        """Return the external-current vector to inject this step."""
        I = np.zeros(self.conn.n)
        drive: dict[str, float] = {}

        # --- chemosensation: ASE reads the time derivative of salt ---
        salt = env.concentration(head, "salt")
        d_salt = 0.0 if self._salt_prev is None else (salt - self._salt_prev) / max(dt, 1e-6)
        self._salt_prev = salt
        drive["salt_on"] = np.clip(d_salt * 8.0, 0.0, 1.5)
        drive["salt_off"] = np.clip(-d_salt * 8.0, 0.0, 1.5)

        # --- olfaction: AWC is tonically active and suppressed by odour ---
        odor = env.concentration(head, "odor")
        d_odor = 0.0 if self._odor_prev is None else (odor - self._odor_prev) / max(dt, 1e-6)
        self._odor_prev = odor
        # Tonic baseline minus current odour, plus a kick on removal.
        drive["odor_attract"] = float(np.clip(0.45 - odor * 0.9 - d_odor * 6.0, 0.0, 1.5))
        drive["odor_avoid"] = float(np.clip(env.concentration(head, "repellent") * 1.2,
                                            0.0, 1.5))

        # --- nociception ---
        repel = env.concentration(head, "repellent")
        drive["nociception"] = float(np.clip(repel * 1.6 + env.active_poke("nose"),
                                             0.0, 2.0))
        drive["nose_touch"] = float(np.clip(env.active_poke("nose"), 0.0, 2.0))

        # --- thermosensation: AFD is warming-activated above the remembered
        # cultivation temperature, and silent below it ---
        temp = env.temperature(head)
        d_temp = 0.0 if self._temp_prev is None else (temp - self._temp_prev) / max(dt, 1e-6)
        self._temp_prev = temp
        above = temp - env.cultivation_temp
        drive["thermo"] = float(np.clip((0.4 + d_temp * 3.0) * (1.0 if above > -0.5 else 0.1),
                                        0.0, 1.5))

        # --- mechanosensation ---
        drive["touch_anterior"] = float(np.clip(env.active_poke("anterior"), 0.0, 2.0))
        drive["touch_posterior"] = float(np.clip(env.active_poke("posterior"), 0.0, 2.0))

        # --- gas sensing ---
        o2 = env.oxygen
        drive["oxygen_high"] = float(np.clip((o2 - 12.0) / 12.0, 0.0, 1.0))
        drive["oxygen_low"] = float(np.clip((10.0 - o2) / 10.0, 0.0, 1.0))

        # --- food, sensed mechanically by the dopaminergic neurons ---
        drive["food_mech"] = env.on_food(head)

        for modality, value in drive.items():
            gain = self._gain(modality)
            v = value * gain
            drive[modality] = v
            if v != 0.0 and len(self.idx[modality]):
                I[self.idx[modality]] += v * amplitude

        self.last = drive
        return I
