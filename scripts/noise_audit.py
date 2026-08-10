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

The next attempt should test that directly: re-run this audit under the mV x
G_rest scheme and check whether oxygen_high's contribution grows. If it does,
the fix is a lower receptor-potential target for the tonic gas sensors (the
guess of 8 mV was already the lowest assigned, and may need to be lower
still, or the tonic component should be adapted out the way the command
baseline is), not a change to the touch values.

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
# Derivative-taking sensors are the prime suspects: they read d/dt of a field
# the animal is moving through, so their drive fluctuates at gait frequency
# even in a uniform environment.
DERIVATIVE = {"salt_on", "salt_off", "odor_attract", "thermo"}


def _patched_compute(silence: set[str]):
    """SensorySystem.compute with the named modalities forced to zero drive."""
    original = SensorySystem.compute

    def compute(self, env, head, tail, dt, *a, **kw):
        I = original(self, env, head, tail, dt, *a, **kw)
        if silence:
            for m in silence:
                idx = self.idx.get(m)
                if idx is not None and len(idx):
                    # Re-zero the cells this modality writes to. Touch fields
                    # share cells with modalities, so this is applied after
                    # the fact rather than by skipping the term.
                    I[idx] = 0.0
        return I

    return compute


def floor(silence=frozenset(), knockouts=(), seconds=40.0, seeds=(0, 1, 2)):
    """Poke-free command-balance fluctuation, as the state machine sees it."""
    original = SensorySystem.compute
    SensorySystem.compute = _patched_compute(set(silence))
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
    seeds = tuple(range(n_seeds))

    base_peak, base_sd = floor(seconds=seconds, seeds=seeds)
    print(f"{'condition':>26}{'peak':>10}{'sd':>10}{'peak drop':>11}")
    print(f"{'intact (baseline)':>26}{base_peak:>10.5f}{base_sd:>10.5f}"
          f"{'':>11}")

    allm = floor(silence=set(MODALITIES), seconds=seconds, seeds=seeds)
    print(f"{'ALL sensory silenced':>26}{allm[0]:>10.5f}{allm[1]:>10.5f}"
          f"{base_peak - allm[0]:>+11.5f}")

    rows = []
    for m in MODALITIES:
        pk, sd = floor(silence={m}, seconds=seconds, seeds=seeds)
        rows.append((base_peak - pk, m, pk, sd))
    rows.sort(reverse=True)
    for drop, m, pk, sd in rows:
        tag = " (derivative)" if m in DERIVATIVE else ""
        print(f"{m:>26}{pk:>10.5f}{sd:>10.5f}{drop:>+11.5f}{tag}")

    mec4 = floor(knockouts=("mec-4",), seconds=seconds, seeds=seeds)
    print(f"\n{'mec-4 null, poke-free':>26}{mec4[0]:>10.5f}{mec4[1]:>10.5f}")
    print("\nReference bands on main (strength-1 poke, peak deviation):")
    print("  anterior wild type 0.0061-0.0065   mec-10 0.0033-0.0037")
    print("  mec-4 null 0.0015-0.0018           reversal threshold 0.0034")
    print("\nRead: the modality with the largest peak drop sets the floor.")
    print("If ALL-silenced is still near baseline, the floor is not sensory")
    print("and the mV calibration is not what needs fixing.")


if __name__ == "__main__":
    main()
