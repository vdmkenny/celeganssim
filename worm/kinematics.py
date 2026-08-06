"""Gait measurement: frequency, wavelength, wave direction, phase.

The validation suite measures speed and bend envelope, which cannot distinguish
a travelling wave from a standing one, nor tell which way it runs. Every
acceptance test for a connectome-driven rhythm needs those distinctions, so
they live here rather than inside any one check.

Everything is computed from what a tracking microscope would see: the body
centreline over time, plus the muscle activations driving it. Numpy only; the
spectral estimates use a windowed periodogram with parabolic interpolation
around the peak, which resolves frequency far below the FFT bin spacing and
avoids a scipy dependency for the core.

Conventions
-----------
Segment 0 is the head, segment N_SEG-1 the tail. A FORWARD wave travels head to
tail, so posterior segments lag anterior ones, which is a phase that DECREASES
along the body: the fitted phase gradient is negative for forward and positive
for backward. Reversal shows up as that gradient changing sign, which is what a
tracking paper actually scores, and it does not depend on the simulator's
internal behavioural state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .body import N_SEG

# Mid-body reference segment for the spectral estimates. Far enough from both
# ends to avoid the head oscillator and the tail taper.
REF_SEG = N_SEG // 2

# Segments used for the wavelength fit: skip the head, which in the animal is
# driven by nerve-ring motor neurons rather than the ventral cord, and skip the
# last segment where the taper makes curvature noisy.
FIT_SEGMENTS = slice(6, N_SEG - 1)


@dataclass
class GaitRecording:
    """Raw time series from a run. All arrays are [time, segment]."""

    dt: float
    curvature: np.ndarray
    dorsal: np.ndarray
    ventral: np.ndarray
    position: np.ndarray          # [time, 2] centroid
    nodes: np.ndarray             # [time, node, 2] centreline
    body_length_mm: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.curvature) * self.dt


def record(sim, seconds: float = 60.0, settle: float = 10.0,
           every: int = 1) -> GaitRecording:
    """Run a simulation and capture its kinematics."""
    n_settle = int(settle / sim.cfg.dt)
    for _ in range(n_settle):
        sim.step()

    n = int(seconds / sim.cfg.dt)
    kap, dor, ven, pos, nod = [], [], [], [], []
    for i in range(n):
        sim.step()
        if i % every:
            continue
        kap.append(sim.body.curvature.copy())
        dor.append(sim.body.dorsal.copy())
        ven.append(sim.body.ventral.copy())
        pos.append(sim.body.X.copy())
        nod.append(sim.body.world_nodes())
    return GaitRecording(
        dt=sim.cfg.dt * every,
        curvature=np.asarray(kap),
        dorsal=np.asarray(dor),
        ventral=np.asarray(ven),
        position=np.asarray(pos),
        nodes=np.asarray(nod),
        body_length_mm=float(sim.body.body_length),
    )


# -- spectral helpers -------------------------------------------------------
def _interpolated_peak(power: np.ndarray, idx: int) -> float:
    """Sub-bin peak offset by fitting a parabola to the three points at idx.

    A 60 s record gives 0.017 Hz bins, which is not enough to assert a
    frequency to +/-0.03 Hz on the bin grid alone. The peak of a windowed
    periodogram is locally quadratic in log power, so three points locate it
    to a small fraction of a bin.
    """
    if idx <= 0 or idx >= len(power) - 1:
        return 0.0
    a, b, c = power[idx - 1], power[idx], power[idx + 1]
    denom = a - 2.0 * b + c
    return 0.0 if denom == 0 else float(0.5 * (a - c) / denom)


def dominant_frequency(x: np.ndarray, dt: float,
                       fmin: float = 0.05, fmax: float = 4.0) -> float:
    """Frequency of the strongest oscillation in a band, in Hz."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if len(x) < 8 or not np.any(np.abs(x) > 1e-12):
        return float("nan")
    win = np.hanning(len(x))
    power = np.abs(np.fft.rfft(x * win)) ** 2
    freq = np.fft.rfftfreq(len(x), dt)
    band = (freq >= fmin) & (freq <= fmax)
    if not band.any():
        return float("nan")
    idx = int(np.argmax(np.where(band, power, -np.inf)))
    df = freq[1] - freq[0]
    return float(freq[idx] + _interpolated_peak(power, idx) * df)


def segment_phases(kap: np.ndarray, dt: float, freq: float) -> np.ndarray:
    """Phase of each segment's oscillation at `freq`, unwrapped along the body.

    Taken from the Fourier coefficient at the dominant frequency rather than
    from a cross-correlation peak. A correlation lag is only defined modulo the
    period, and the body spans more than one wavelength, so lags computed that
    way alias and the fitted slope collapses. Phase unwrapping along the
    anterior-posterior axis has no such ambiguity.
    """
    n_t = kap.shape[0]
    win = np.hanning(n_t)
    spec = np.fft.rfft((kap - kap.mean(axis=0)) * win[:, None], axis=0)
    k = int(round(freq * n_t * dt))
    k = max(1, min(k, spec.shape[0] - 1))
    return np.unwrap(np.angle(spec[k]))


def _lag_samples(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Lag of `a` relative to `b`, in samples, positive when `a` comes later."""
    a = a - a.mean()
    b = b - b.mean()
    if not (np.any(np.abs(a) > 1e-12) and np.any(np.abs(b) > 1e-12)):
        return float("nan")
    n = len(a)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)),
                        size)
    corr = np.concatenate([corr[-max_lag:], corr[:max_lag + 1]])
    idx = int(np.argmax(corr))
    return float(idx - max_lag + _interpolated_peak(corr, idx))


# -- gait metrics -----------------------------------------------------------
def analyse(rec: GaitRecording) -> dict:
    """Frequency, wavelength, direction, phase and amplitude from a recording."""
    kap = rec.curvature
    n_t, n_seg = kap.shape
    ref = kap[:, REF_SEG]

    freq = dominant_frequency(ref, rec.dt)

    # Wavelength from the phase gradient along the body: radians per segment,
    # fitted across segments so no single noisy segment sets the answer.
    max_lag = int(min(n_t // 3, (1.5 / freq / rec.dt) if freq > 0 else n_t // 3))
    segs = np.arange(n_seg)[FIT_SEGMENTS]
    slope = float("nan")
    wavelength_bl = float("nan")
    direction = "none"
    if np.isfinite(freq) and freq > 0:
        ph = segment_phases(kap, rec.dt, freq)
        slope = float(np.polyfit(segs, ph[FIT_SEGMENTS], 1)[0])  # rad/segment
        if abs(slope) > 1e-9:
            wavelength_bl = (2.0 * np.pi / abs(slope)) / n_seg
            # Posterior segments lag anterior ones in a forward wave, which is
            # a phase that DECREASES going backward along the body.
            direction = "forward" if slope < 0 else "backward"

    # Phase from muscle drive to curvature: the Butler et al. 2015 observable,
    # measured viscosity-independent in the animal.
    drive = rec.dorsal - rec.ventral
    phases = []
    for s in segs:
        lag = _lag_samples(kap[:, s], drive[:, s], max_lag)
        if np.isfinite(lag) and freq > 0:
            phases.append(lag * rec.dt * freq * 360.0)
    phase_deg = float(np.mean(phases)) if phases else float("nan")

    # Bend envelope: perpendicular spread of the centreline about its own
    # head-tail axis, normalised by body length. Same definition the validation
    # suite uses, so the two stay comparable.
    amps = []
    for nd in rec.nodes:
        ax = nd[-1] - nd[0]
        L = np.linalg.norm(ax)
        if L <= 1e-9:
            continue
        perp = np.array([-ax[1] / L, ax[0] / L])
        d = (nd - nd[0]) @ perp
        amps.append((d.max() - d.min()) / max(rec.body_length_mm, 1e-9))
    amp = float(np.mean(amps)) if amps else float("nan")

    return {
        "undulation_hz": freq,
        "wavelength_bl": wavelength_bl,
        "wave_direction": direction,
        "phase_rad_per_segment": slope,
        "phase_drive_to_curvature_deg": phase_deg,
        "curvature_amplitude": float(np.mean(np.abs(kap).max(axis=1))),
        "bend_amplitude_bl": amp,
        "duration_s": rec.duration,
    }


def reversal_events(rec: GaitRecording, window_s: float = 3.0,
                    hop_s: float = 0.5, min_hold_s: float = 0.5) -> dict:
    """Count direction changes from the sign of the wave's travel.

    A reversal is the lag-versus-segment slope changing sign and holding, which
    is what a tracking experiment scores. It makes no reference to the
    simulator's behavioural state machine, so it stays valid once that is gone.
    """
    kap = rec.curvature
    n_t, n_seg = kap.shape
    w = max(int(window_s / rec.dt), 16)
    hop = max(int(hop_s / rec.dt), 1)
    segs = np.arange(n_seg)[FIT_SEGMENTS]

    times, signs = [], []
    for start in range(0, max(n_t - w, 1), hop):
        seg_win = kap[start:start + w]
        if seg_win.shape[0] < w:
            break
        f = dominant_frequency(seg_win[:, REF_SEG], rec.dt)
        if not np.isfinite(f) or f <= 0:
            continue
        ph = segment_phases(seg_win, rec.dt, f)
        slope = float(np.polyfit(segs, ph[FIT_SEGMENTS], 1)[0])
        times.append((start + w / 2) * rec.dt)
        # Negative phase gradient is a forward (head to tail) wave.
        signs.append(-np.sign(slope))

    if not signs:
        return {"n_reversals": 0, "events": [], "forward_fraction": float("nan")}

    signs = np.array(signs)
    times = np.array(times)
    events, current, held_from = [], signs[0], times[0]
    for t, sg in zip(times[1:], signs[1:]):
        if sg == current:
            continue
        if t - held_from >= min_hold_s:
            events.append({"t": round(float(t), 2),
                           "to": "forward" if sg > 0 else "backward"})
            current, held_from = sg, t
    return {
        "n_reversals": sum(1 for e in events if e["to"] == "backward"),
        "n_direction_changes": len(events),
        "events": events,
        "forward_fraction": float(np.mean(signs > 0)),
    }


def rhythmicity(x: np.ndarray, dt: float,
                fmin: float = 0.05, fmax: float = 4.0) -> float:
    """Ratio of peak to median spectral power in a band.

    Separates an oscillation from a drifting or noisy signal, which a dominant
    frequency alone cannot do: `dominant_frequency` always returns the largest
    bin even when the spectrum is flat. White noise sits near 1, a clean tone
    runs to many orders of magnitude.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if len(x) < 8 or not np.any(np.abs(x) > 1e-15):
        return 0.0
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), dt)
    band = (freq >= fmin) & (freq <= fmax)
    if not band.any():
        return 0.0
    med = float(np.median(power[band]))
    return float(power[band].max() / med) if med > 0 else 0.0


def muscle_drive(sim, seconds: float = 60.0, settle: float = 10.0):
    """Per-segment dorsal and ventral activation of the REAL muscle cells.

    Reads the connectome's own neuromuscular output rather than whatever the
    body model was told to do, so the two can be compared directly. Returns
    (dorsal, ventral, dt) with arrays shaped [time, segment].
    """
    for _ in range(int(settle / sim.cfg.dt)):
        sim.step()
    n = int(seconds / sim.cfg.dt)
    dor = np.zeros((n, N_SEG))
    ven = np.zeros((n, N_SEG))
    for t in range(n):
        sim.step()
        act = sim.ns.activation()
        for s in range(N_SEG):
            dor[t, s] = np.mean(act[sim.row_d[s]]) if sim.row_d[s] else 0.0
            ven[t, s] = np.mean(act[sim.row_v[s]]) if sim.row_v[s] else 0.0
    return dor, ven, sim.cfg.dt


def measure(sim, seconds: float = 60.0, settle: float = 10.0) -> dict:
    """Convenience: run, analyse, and include reversal detection."""
    rec = record(sim, seconds=seconds, settle=settle)
    out = analyse(rec)
    out.update({f"rev_{k}": v for k, v in reversal_events(rec).items()
                if k != "events"})
    return out
