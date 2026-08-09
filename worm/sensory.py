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
    # Touch is handled positionally through TOUCH_FIELDS below rather than as
    # flat modality buckets. These entries exist so the gene gating and the
    # telemetry readout still have named handles.
    "touch_anterior": ["ALML", "ALMR", "AVM"],
    "touch_posterior": ["PLML", "PLMR"],
    "nose_touch": ["ASHL", "ASHR", "FLPL", "FLPR", "OLQDL", "OLQDR",
                   "OLQVL", "OLQVR"],
    "harsh_touch": ["PVDL", "PVDR"],
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
    # Harsh touch runs through MEC-10/DEGT-1 in PVD, not the MEC-4 channel, so
    # it survives mec-4 loss. That dissociation is a real diagnostic: mec-4
    # nulls ignore a gentle stroke but still respond to hard prodding.
    "harsh_touch": ("harsh_touch",),
    "oxygen_high": (),
    "oxygen_low": (),
    "food_mech": (),
}


# Receptive fields of the mechanosensory neurons along the body, expressed as
# the stretch of body each one's sensory process actually runs along. Position
# u is 0 at the nose and 1 at the tail tip.
#
# The touch receptor neurons are not point sensors: each extends a long
# undifferentiated process embedded in the cuticle, and it is that process
# which transduces. ALM cell bodies sit near the middle and project ANTERIORLY
# to the head; PLM cell bodies sit in the tail and project anteriorly to about
# the vulva. The two therefore overlap around mid-body, which is why a
# mid-body stroke drives both weakly and gives the least reliable response,
# while head and tail strokes drive opposing circuits cleanly.
#
# Refs: Chalfie & Sulston 1981 Dev Biol 82:358; Chalfie et al. 1985 J Neurosci
# 5:956; WormAtlas Touch Receptor Neurons; Goodman 2006 WormBook
# Mechanosensation. PVD tiles almost the whole body with a menorah-like arbor
# and is high-threshold (Albeg et al. 2011; Chatzigeorgiou et al. 2010).
TOUCH_FIELDS: list[dict] = [
    # (cells, start, end, edge softness, modality this counts toward)
    # ASH/FLP/OLQ endings sit at the nose tip; the taper must die out within
    # a few percent of body length or a mid-head poke leaks into the
    # nociceptors (measured: soft=0.035 left 1.8% coverage at u=0.20, which
    # produced a reversal-biased drive in a mec-4 null that should have had
    # none).
    {"cells": ["ASHL", "ASHR", "FLPL", "FLPR", "OLQDL", "OLQDR", "OLQVL", "OLQVR"],
     "start": -0.02, "end": 0.06, "soft": 0.012, "modality": "nose_touch"},
    {"cells": ["ALML", "ALMR"],
     "start": 0.02, "end": 0.48, "soft": 0.075, "modality": "touch_anterior"},
    # AVM is born post-embryonically, sits ventrally and slightly posterior to
    # ALM, and projects anteriorly. Functionally an anterior touch cell.
    {"cells": ["AVM"],
     "start": 0.12, "end": 0.55, "soft": 0.075, "modality": "touch_anterior"},
    {"cells": ["PLML", "PLMR"],
     "start": 0.55, "end": 1.02, "soft": 0.075, "modality": "touch_posterior"},
    # PVM is a touch receptor neuron anatomically but has no demonstrated
    # touch-withdrawal role, so it is given a field and left behaviourally
    # weak rather than wired as a driver.
    {"cells": ["PVM"],
     "start": 0.60, "end": 0.95, "soft": 0.08, "modality": "touch_posterior",
     "weight": 0.15},
    # High-threshold harsh touch (~100-200 uN, versus 1-10 uN for a gentle
    # eyelash stroke) is carried by a separate pair of nociceptors, and they
    # split the body between them with OPPOSITE behavioural signs:
    #
    #   FLP covers the head and drives REVERSAL. Its wiring is reversal-biased
    #   (FLP->AVA/AVD/AVE greatly outweighs FLP->AVB/PVC).
    #   PVD tiles the rest of the body and drives FORWARD escape. Its synapse
    #   counts onto PVC and AVA are nearly equal, yet PVC wins functionally:
    #   removing PVC flips PVD photoactivation from forward into reverse.
    #
    # This is why harsh touch keeps the same anterior-reverses /
    # posterior-advances logic as gentle touch while being MEC-4 independent,
    # so it survives intact in mec-4 nulls.
    # Refs: Li, Kang, Piggott, Feng & Xu 2011 Nat Commun 2:315; Husson, Steuer
    # Costa et al. 2012 Curr Biol 22:743; Chatzigeorgiou et al. 2010.
    {"cells": ["FLPL", "FLPR"],
     "start": -0.02, "end": 0.22, "soft": 0.06, "modality": "harsh_touch",
     "harsh_only": True},
    {"cells": ["PVDL", "PVDR"],
     "start": 0.18, "end": 0.95, "soft": 0.09, "modality": "harsh_touch",
     "harsh_only": True},
]


# Mechanoreceptor currents are PHASIC: force evokes a rapidly activating
# current that decays during a sustained press, and REMOVAL of the force
# evokes a second current of the same sign, so the receptor reports change in
# both directions rather than load (O'Hagan, Chalfie & Goodman 2005 Nat
# Neurosci 8:43: MRCs at onset and offset, Na+-carried, amiloride-blocked,
# adapting within a few tens of milliseconds; threshold ~100 nN, saturating
# at 1-2 uN). Modelled as the rectified difference between the stimulus and
# its own lowpass: |s - lowpass(s)| peaks at onset and offset and dies during
# the hold, which is the measured waveform shape.
TOUCH_ADAPT_TAU_S = 0.04   # a few tens of ms, O'Hagan et al. 2005


def _field_coverage(u: float, start: float, end: float, soft: float) -> float:
    """Smooth window over the body: full inside [start, end], tapering at the
    edges rather than switching on and off, because a sensory process fades
    out over a distance rather than stopping at a line."""
    s = max(soft, 1e-4)
    a = 1.0 / (1.0 + np.exp(-(u - start) / s))
    b = 1.0 / (1.0 + np.exp(-(end - u) / s))
    return float(a * b)


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
        # Resolve each mechanosensory receptive field to cell indices once.
        self.fields = []
        for f in TOUCH_FIELDS:
            present = [c for c in f["cells"] if c in conn.index]
            self.fields.append({**f, "idx": conn.indices(present),
                                "missing": [c for c in f["cells"]
                                            if c not in conn.index]})
        # Phasic mechanotransduction state: one lowpass per receptive field.
        self._touch_lp = [0.0] * len(self.fields)
        # Adaptive state for the derivative-taking sensors.
        self._salt_prev: float | None = None
        self._odor_prev: float | None = None
        self._temp_prev: float | None = None
        self.last: dict[str, float] = {}

    def _gain(self, modality: str) -> float:
        cells = SENSORS.get(modality, ())
        g = 1.0
        for key in GATE.get(modality, ()):
            g *= self.genome.sensory_scale_in_cells(key, cells)
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

        # --- nociception: chemical only here, the mechanical part of ASH is
        # driven by its nose receptive field below ---
        repel = env.concentration(head, "repellent")
        drive["nociception"] = float(np.clip(repel * 1.6, 0.0, 2.0))

        # --- thermosensation: AFD is warming-activated above the remembered
        # cultivation temperature, and silent below it ---
        temp = env.temperature(head)
        d_temp = 0.0 if self._temp_prev is None else (temp - self._temp_prev) / max(dt, 1e-6)
        self._temp_prev = temp
        above = temp - env.cultivation_temp
        drive["thermo"] = float(np.clip((0.4 + d_temp * 3.0) * (1.0 if above > -0.5 else 0.1),
                                        0.0, 1.5))

        # Mechanosensation is positional and handled after this loop, since
        # cells within one modality take different weights depending on where
        # along the body the stimulus lands.

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

        drive.update(self._apply_touch(env, I, amplitude, dt))
        self.last = {k: v for k, v in drive.items() if v}
        return I

    def _apply_touch(self, env: Environment, I: np.ndarray,
                     amplitude: float, dt: float) -> dict[str, float]:
        """Inject mechanosensory current according to where the animal was touched.

        Each receptive field contributes in proportion to how much of it the
        poke lands inside. This only decides which cells hear the stimulus; the
        behavioural consequence comes out of the connectome, which is why a head
        poke reverses and a tail poke accelerates.
        """
        per_modality: dict[str, float] = {}
        pokes = env.active_pokes()
        alpha = 1.0 - np.exp(-dt / TOUCH_ADAPT_TAU_S)

        for k, f in enumerate(self.fields):
            total = 0.0
            if pokes and len(f["idx"]):
                for p in pokes:
                    # PVD is high-threshold: gentle stroking never reaches it.
                    if f.get("harsh_only") and not p.harsh:
                        continue
                    cov = _field_coverage(p.u, f["start"], f["end"], f["soft"])
                    total += p.strength * cov * f.get("weight", 1.0)
            # Phasic transduction: the receptor reports |change|, exciting at
            # both onset and offset and adapting out during a hold.
            lp = self._touch_lp[k]
            self._touch_lp[k] = lp + alpha * (total - lp)
            phasic = abs(total - lp)
            if phasic <= 0.0 or len(f["idx"]) == 0:
                continue
            gain = self._gain(f["modality"])
            if gain <= 0.0:
                continue
            v = float(np.clip(phasic * gain, 0.0, 2.5))
            I[f["idx"]] += v * amplitude
            per_modality[f["modality"]] = per_modality.get(f["modality"], 0.0) + v
        return per_modality
