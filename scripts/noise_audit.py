"""Which sensory modality sets the command-balance noise floor?

Blocks issue #12. Calibrating sensory injection to measured receptor
potentials (mV x G_rest) collapsed all six touch discrimination bands onto a
single value near 0.006: with the poke contributing nothing distinguishable,
the "response" being measured was background. The floor scaled with the
sensory overdrive, so it is sensory-driven fluctuation rather than intrinsic
membrane noise, but it was NOT explained by resting-conductance spread (only
2.4x across sensory neurons) and every tonic drive was LOWER under the new
scheme than under the old flat 55 pA constant. Some modality's fluctuating
component dominates, and this names it.

Method: run poke-free, measure the command-balance fluctuation the reversal
threshold actually sees, then knock out one modality at a time by zeroing its
drive and re-measure. The modality whose removal drops the floor most is the
culprit. Two controls guard the interpretation: silencing ALL sensory input
bounds how much of the floor is sensory at all, and the mec-4 null floor is
the quantity the touch bands are actually compared against.

The knockout is applied to the DRIVE dict inside SensorySystem.compute by
monkeypatching, so nothing in the shipped model changes and the audit can run
against main unmodified.

First result, from a short smoke run against main unmodified (12 s, 1 seed,
so indicative rather than final):

    intact baseline      peak 0.00195
    ALL sensory silenced peak 0.00131   (-0.00065)
    oxygen_high          peak 0.00163   (-0.00033)  <- largest single source
    odor_attract         peak 0.00169   (-0.00027)
    thermo               peak 0.00181   (-0.00014)
    every other modality             no measurable contribution

Two things follow. About a third of the floor is sensory and two thirds is
intrinsic, so silencing sensation cannot flatten it entirely. And the largest
sensory contributor is the TONIC one: URX/AQR/PQR at 21% oxygen, not the
derivative-taking sensors that were the prime suspects.

That is a candidate mechanism for the collapse. URX has the HIGHEST resting
conductance of any sensory neuron here (0.93 nS against 0.39-0.63 for the
touch cells), so switching from a flat current to mV x G_rest roughly doubles
oxygen drive RELATIVE to touch even though its absolute current falls. A
tonic, always-on modality gaining relative weight would raise the floor
against which every touch band is measured, which is what was observed.

The two-arm comparison was then run and the diagnosis is COMPLETE. Under the
mV scheme (20 s, 2 seeds): baseline floor 0.00652 against flat's 0.00195,
reproducing the collapse; silencing everything leaves 0.00133; and silencing
OXYGEN ALONE leaves 0.00144, i.e. oxygen is 98% of the sensory floor
(+0.00508 of +0.00519). Odour contributes +0.00065, thermo +0.00026, nothing
else registers.

The mechanism is not current magnitude, and both magnitude hypotheses died on
the numbers: gas-sensor conductances are ordinary (URX 0.93-0.98 nS, AQR
0.89, PQR 0.61), and the mV scheme delivers LESS current to them (about 8 pA
against the flat scheme's 41) while producing FIFTEEN TIMES the floor. The
only account consistent with that inversion: the flat 41 pA SATURATED the
tonic gas sensors, parking them at their rails where their output is
constant and transmits nothing; the physiological ~8 pA lands them mid-range,
at maximum gain, where they amplify intrinsic network noise into the command
balance. Less drive, more noise, because saturation was doing the silencing.

Partial confirmation by intervention: oxygen at 3 mV (about 3 pA, partially
engaged) cuts the floor to 0.00285, and with it touch separation returns
directionally under the mV scheme (anterior wild type 0.0033-0.0048 against
mec-4 0.0024-0.0033) but the bands still overlap: signal-to-floor is ~1.5x
against the flat scheme's 4x.

What issue #12 therefore needs, precisely: (a) a deliberate treatment of
TONIC modalities under physiological drive, either adaptation of the tonic
component (the command baseline already models exactly this pattern) or an
explicit, documented decision that tonic gas sensing operates in saturation;
and (b) the touch targets at the upper end of O'Hagan's measured range, or a
matched reduction of neural_noise, to restore at least 3x band separation.
Both are bounded changes now that the floor has one named owner.

Usage:
    .venv/bin/python scripts/noise_audit.py [seconds] [seeds]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worm.environment import Environment           # noqa: E402
from worm.sensory import SENSORS, SensorySystem    # noqa: E402
from worm.simulation import SimConfig, WormSimulation  # noqa: E402

MODALITIES = list(SENSORS)

# Arm B: the receptor-potential scheme that collapsed the touch bands,
# reconstructed here so the audit can compare both schemes without touching
# the shipped model. Values are the reverted design's: two measured anchors
# (thermo 14 mV, Ramot et al. 2008; odour 17 mV, the Liu et al. 2018 10-25 mV
# midpoint) and the rest labelled guesses inside the measured 5-25 mV
# envelope. MV_OVERDRIVE is the scale at which the collapse was measured.
SENSORY_MV = {
    "salt_on": 15.0, "salt_off": 15.0,
    "odor_attract": 17.0, "odor_avoid": 15.0,
    "nociception": 18.0, "thermo": 14.0,
    "touch_anterior": 16.0, "touch_posterior": 16.0,
    "nose_touch": 16.0, "harsh_touch": 18.0,
    "oxygen_high": 8.0, "oxygen_low": 8.0, "food_mech": 8.0,
}
MV_OVERDRIVE = 1.5
# Derivative-taking sensors are the prime suspects: they read d/dt of a field
# the animal is moving through, so their drive fluctuates at gait frequency
# even in a uniform environment.
DERIVATIVE = {"salt_on", "salt_off", "odor_attract", "thermo"}


def _patched_compute(silence: set[str], scheme: str = "flat",
                     g_rest=None, mv_override=None):
    """SensorySystem.compute wrapped for the audit.

    scheme "flat" is the shipped model. scheme "mv" reconstructs the
    receptor-potential injection from the modality drives the original call
    just computed: injection is linear per modality over that modality's
    cells, so rebuilding I as drive x mV x G_rest x overdrive reproduces the
    reverted scheme exactly for everything except mid-poke touch (whose
    per-field split self.last does not carry), which a poke-free floor never
    exercises.
    """
    original = SensorySystem.compute
    mv = dict(SENSORY_MV)
    if mv_override:
        mv.update(mv_override)

    def compute(self, env, head, tail, dt, *a, **kw):
        I = original(self, env, head, tail, dt, *a, **kw)
        if scheme == "mv":
            I = np.zeros_like(I)
            for m, v in self.last.items():
                idx = self.idx.get(m)
                if idx is None or not len(idx) or m not in mv:
                    continue
                I[idx] += v * mv[m] * MV_OVERDRIVE * g_rest[idx]
        for m in silence:
            idx = self.idx.get(m)
            if idx is not None and len(idx):
                # Re-zero the cells this modality writes to. Exact for the
                # top contributors, whose cells belong to one modality each.
                I[idx] = 0.0
        return I

    return compute


def floor(silence=frozenset(), knockouts=(), seconds=40.0, seeds=(0, 1, 2),
          scheme="flat", mv_override=None):
    """Poke-free command-balance fluctuation, as the state machine sees it."""
    original = SensorySystem.compute
    g_rest = None
    if scheme == "mv":
        probe = WormSimulation(env=Environment(width=44.0, height=32.0),
                               config=SimConfig(seed=0))
        g_rest = probe._g_rest if hasattr(probe, "_g_rest") else (
            probe.ns.G_leak
            + (probe.ns.Gg_eff * probe.ns.p.g_gap).sum(axis=1)
            + probe.ns.s_eq
            * (probe.ns.Gs_eff * probe.ns.g_syn_row).sum(axis=1))
    SensorySystem.compute = _patched_compute(set(silence), scheme, g_rest,
                                             mv_override)
    try:
        out = []
        for sd in seeds:
            sim = WormSimulation(env=Environment(width=44.0, height=32.0),
                                 config=SimConfig(seed=sd))
            for g in knockouts:
                sim.knock_out(g)
            for _ in range(500):
                sim.step()
            peak = 0.0
            vals = []
            for _ in range(int(seconds / sim.cfg.dt)):
                sim.step()
                d = float(getattr(sim, "last_cmd_deviation", 0.0))
                vals.append(d)
                peak = max(peak, d)
            out.append((peak, float(np.std(vals))))
        return (float(np.mean([p for p, _ in out])),
                float(np.mean([s for _, s in out])))
    finally:
        SensorySystem.compute = original


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    scheme = sys.argv[3] if len(sys.argv) > 3 else "flat"
    fix = {"oxygen_high": 3.0, "oxygen_low": 3.0} \
        if len(sys.argv) > 4 and sys.argv[4] == "oxyfix" else None
    seeds = tuple(range(n_seeds))
    kw = {"scheme": scheme, "mv_override": fix}
    print(f"scheme={scheme}" + (f" with oxygen mv 8->3" if fix else ""))

    base_peak, base_sd = floor(seconds=seconds, seeds=seeds, **kw)
    print(f"{'condition':>26}{'peak':>10}{'sd':>10}{'peak drop':>11}")
    print(f"{'intact (baseline)':>26}{base_peak:>10.5f}{base_sd:>10.5f}"
          f"{'':>11}")

    allm = floor(silence=set(MODALITIES), seconds=seconds, seeds=seeds, **kw)
    print(f"{'ALL sensory silenced':>26}{allm[0]:>10.5f}{allm[1]:>10.5f}"
          f"{base_peak - allm[0]:>+11.5f}")

    rows = []
    for m in MODALITIES:
        pk, sd = floor(silence={m}, seconds=seconds, seeds=seeds, **kw)
        rows.append((base_peak - pk, m, pk, sd))
    rows.sort(reverse=True)
    for drop, m, pk, sd in rows:
        tag = " (derivative)" if m in DERIVATIVE else ""
        print(f"{m:>26}{pk:>10.5f}{sd:>10.5f}{drop:>+11.5f}{tag}")

    mec4 = floor(knockouts=("mec-4",), seconds=seconds, seeds=seeds, **kw)
    print(f"\n{'mec-4 null, poke-free':>26}{mec4[0]:>10.5f}{mec4[1]:>10.5f}")
    print("\nReference bands on main (strength-1 poke, peak deviation):")
    print("  anterior wild type 0.0061-0.0065   mec-10 0.0033-0.0037")
    print("  mec-4 null 0.0015-0.0018           reversal threshold 0.0034")
    print("\nRead: the modality with the largest peak drop sets the floor.")
    print("If ALL-silenced is still near baseline, the floor is not sensory")
    print("and the mV calibration is not what needs fixing.")


if __name__ == "__main__":
    main()
