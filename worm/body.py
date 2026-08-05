"""Biomechanics: muscle activation -> body shape -> movement through a medium.

Two things happen here.

1. A ventral-cord oscillator chain converts motor-neuron drive into a travelling
   wave of dorsal/ventral muscle activation. The rhythm is modelled explicitly
   rather than emerging from the connectome. That is a deliberate, standard
   choice (cf. Boyle & Cohen 2012; Olivares et al. 2021): the wiring diagram
   alone does not pin down the ionic conductances needed to make a 302-cell
   network oscillate, so published whole-animal models drive an explicit
   locomotor rhythm whose amplitude, direction, frequency and dorsoventral
   balance are set by the network. The connectome still does the real work of
   turning sensory input into a motor decision.

2. The resulting shape is pushed through a viscous medium using resistive force
   theory. At C. elegans scale the Reynolds number is ~1e-3, so inertia is
   irrelevant and the body's translation and rotation each instant are whatever
   makes net external force and torque vanish.
"""

from __future__ import annotations

import numpy as np

N_SEG = 24          # body segments == body-wall muscle rows
N_NODE = N_SEG + 1
BODY_LENGTH_MM = 1.0


class BodyParams:
    # Wild-type crawling references (Cronin et al. 2005, automated tracking):
    # 0.47 Hz rapid mode (84% of the time), wavelength 62% of body length,
    # amplitude ~19% BL, speed 0.20 mm/s.
    freq_hz = 0.47
    wavelength_bl = 0.62
    # How fast a segment's actual curvature chases its commanded curvature.
    curvature_tau_s = 0.06
    # Commanded curvature per unit of dorsal-ventral drive difference (1/mm).
    curvature_gain = 11.0
    max_curvature = 14.0
    # Resistive force theory: normal/tangential drag ratio. ~40 on agar.
    drag_ratio = 32.0
    drag_tangential = 1.0
    # Body shortening when dorsal and ventral contract together (the
    # "shrinker" axis that GABA mutants fall down).
    max_shortening = 0.28


# Provenance tags for the parameter registry (worm/parameters.py). Citations
# are in the comments above; the tag says how much to trust the number.
PROVENANCE = {
    "freq_hz": "measured",         # Cronin et al. 2005
    "wavelength_bl": "measured",   # Cronin et al. 2005
    "curvature_tau_s": "tuned",    # muscle low-pass, not measured
    "curvature_gain": "tuned",     # fit to ~19% BL amplitude
    "max_curvature": "tuned",
    "drag_ratio": "published",     # RFT on agar, Gray & Lissmann line of work
    "drag_tangential": "published",
    "max_shortening": "tuned",     # fit to shrinker phenotype
}


class Body:
    def __init__(self, params: BodyParams | None = None, seed: int = 0) -> None:
        self.p = params or BodyParams()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> None:
        self.X = np.array([x, y], dtype=float)
        self.phi = float(heading)
        self.phase = np.zeros(N_SEG)        # oscillator phase per segment
        self.curvature = np.zeros(N_SEG)    # actual curvature, 1/mm
        self.dorsal = np.zeros(N_SEG)       # muscle activation 0..1
        self.ventral = np.zeros(N_SEG)
        self.length_scale = 1.0
        self.speed = 0.0
        self.angular_speed = 0.0
        self.body_length = getattr(self, "body_length", BODY_LENGTH_MM)
        self._seg_len = self.body_length / N_SEG
        # Radius profile: C. elegans is a tapered cylinder, fattest just past
        # the middle. Weights drag along the body.
        self._set_radius()
        self._update_shape()

    def _set_radius(self) -> None:
        u = np.linspace(0.0, 1.0, N_NODE)
        girth = 0.04 * (self.body_length / BODY_LENGTH_MM)
        self.radius = girth * np.sqrt(np.clip(4 * u * (1 - u), 0.0, None)) \
            + 0.008 * (self.body_length / BODY_LENGTH_MM)

    def set_length(self, length_mm: float) -> None:
        """Resize the animal as it develops. Curvature is per mm, so the
        commanded curvature has to scale inversely with body length for a
        larva to hold the same shape as an adult."""
        length_mm = max(float(length_mm), 0.05)
        if abs(length_mm - self.body_length) < 1e-6:
            return
        self.body_length = length_mm
        self._seg_len = length_mm / N_SEG
        self._set_radius()

    # -- neural drive ---------------------------------------------------
    def step_oscillator(self, dt: float, *, forward_drive: float,
                        backward_drive: float, gaba_scale: float,
                        head_bias: float = 0.0, arousal: float = 1.0) -> None:
        """Advance the ventral-cord oscillator chain by dt seconds.

        forward_drive / backward_drive are 0..1 activations of the B-type and
        A-type motor neuron pools. Their difference sets travel direction; their
        sum sets amplitude. gaba_scale is the strength of GABAergic
        cross-inhibition surviving whatever genetics are in play.
        """
        p = self.p
        net = forward_drive - backward_drive
        total = min(forward_drive + backward_drive, 1.5)
        direction = 1.0 if net >= 0 else -1.0

        omega = 2 * np.pi * p.freq_hz * arousal * (0.35 + 0.65 * min(total, 1.0))
        # Phase lag per segment giving the observed wavelength.
        dphi = 2 * np.pi / (p.wavelength_bl * N_SEG)

        # The head is the pacemaker; each segment chases its neighbour on the
        # side the wave is coming from. Forward waves travel head -> tail, so
        # segment r follows r-1; backward waves reverse the dependency.
        target = self.phase.copy()
        if direction > 0:
            target[1:] = self.phase[:-1] - dphi
            target[0] = self.phase[0]
        else:
            target[:-1] = self.phase[1:] - dphi
            target[-1] = self.phase[-1]

        coupling = 6.0
        err = np.angle(np.exp(1j * (target - self.phase)))
        self.phase = self.phase + dt * (omega + coupling * err)
        self.phase = np.mod(self.phase, 2 * np.pi)

        # Cholinergic excitation is phasic and one-sided; GABAergic inhibition
        # is contralateral. With GABA intact the two sides alternate. Remove
        # GABA and both sides get only the excitatory term, so the difference
        # (bending) halves while the sum (contraction) climbs -- the shrinker.
        wave = np.sin(self.phase)
        exc = 0.85 * min(total, 1.0)
        inh = 0.85 * min(total, 1.0) * gaba_scale

        d_exc = exc * (0.5 + 0.5 * wave)
        v_exc = exc * (0.5 - 0.5 * wave)
        d_inh = inh * (0.5 - 0.5 * wave)
        v_inh = inh * (0.5 + 0.5 * wave)

        bias = np.zeros(N_SEG)
        n_head = 8
        bias[:n_head] = head_bias * np.linspace(1.0, 0.2, n_head)

        self.dorsal = np.clip(d_exc - d_inh + bias, 0.0, 1.0)
        self.ventral = np.clip(v_exc - v_inh - bias, 0.0, 1.0)

        # Co-contraction shortens the animal.
        co = float(np.mean(np.minimum(self.dorsal, self.ventral)))
        self.length_scale = 1.0 - p.max_shortening * np.clip(co * 2.2, 0.0, 1.0)

        target_kappa = np.clip(
            p.curvature_gain * (self.dorsal - self.ventral),
            -p.max_curvature, p.max_curvature,
        )
        alpha = 1.0 - np.exp(-dt / max(p.curvature_tau_s, 1e-6))
        self.curvature += alpha * (target_kappa - self.curvature)

    # -- shape ----------------------------------------------------------
    def _local_nodes(self, curvature: np.ndarray) -> np.ndarray:
        """Body-frame node positions from segment curvatures."""
        seg = self._seg_len * self.length_scale
        angles = np.cumsum(curvature * seg)
        angles = angles - angles.mean()
        pts = np.zeros((N_NODE, 2))
        pts[1:, 0] = np.cumsum(np.cos(angles) * seg)
        pts[1:, 1] = np.cumsum(np.sin(angles) * seg)
        return pts - pts.mean(axis=0)

    def _update_shape(self) -> None:
        self.local = self._local_nodes(self.curvature)

    # -- locomotion -----------------------------------------------------
    def step_motion(self, dt: float, drag_ratio: float | None = None) -> None:
        """Solve rigid-body translation+rotation that balances viscous drag.

        The shape change is prescribed by the muscles; the only unknowns are the
        body's overall velocity U and angular velocity Omega. At zero Reynolds
        number the external force and torque must both vanish, giving a 3x3
        linear system.
        """
        p = self.p
        cn_ct = p.drag_ratio if drag_ratio is None else drag_ratio

        prev_local = self.local
        self._update_shape()
        new_local = self.local

        # Shape velocity in the body frame, rotated into the lab frame.
        c, s = np.cos(self.phi), np.sin(self.phi)
        R = np.array([[c, -s], [s, c]])
        shape_vel = ((new_local - prev_local) / dt) @ R.T
        pos = new_local @ R.T  # node offsets from body centre, lab frame

        # Segment tangents (per node, averaged from adjacent segments).
        d = np.diff(pos, axis=0)
        seg_t = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        tang = np.zeros_like(pos)
        tang[:-1] += seg_t
        tang[1:] += seg_t
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12

        # Per-node drag tensor D = Ct*t t^T + Cn*n n^T, weighted by girth.
        w = self.radius / self.radius.mean()
        Ct = p.drag_tangential * w
        Cn = p.drag_tangential * cn_ct * w
        tx, ty = tang[:, 0], tang[:, 1]
        # n n^T = I - t t^T
        Dxx = Ct * tx * tx + Cn * (1 - tx * tx)
        Dyy = Ct * ty * ty + Cn * (1 - ty * ty)
        Dxy = Ct * tx * ty - Cn * tx * ty

        def drag(v: np.ndarray) -> np.ndarray:
            return np.stack(
                [Dxx * v[:, 0] + Dxy * v[:, 1], Dxy * v[:, 0] + Dyy * v[:, 1]],
                axis=1,
            )

        # a_i = z_hat x p_i, the velocity field of unit angular velocity.
        a = np.stack([-pos[:, 1], pos[:, 0]], axis=1)
        basis = [
            np.tile(np.array([1.0, 0.0]), (N_NODE, 1)),
            np.tile(np.array([0.0, 1.0]), (N_NODE, 1)),
            a,
        ]
        M = np.zeros((3, 3))
        for k, v in enumerate(basis):
            f = drag(v)
            M[0, k] = f[:, 0].sum()
            M[1, k] = f[:, 1].sum()
            M[2, k] = (pos[:, 0] * f[:, 1] - pos[:, 1] * f[:, 0]).sum()

        fs = drag(shape_vel)
        rhs = -np.array([
            fs[:, 0].sum(),
            fs[:, 1].sum(),
            (pos[:, 0] * fs[:, 1] - pos[:, 1] * fs[:, 0]).sum(),
        ])

        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.zeros(3)
        if not np.all(np.isfinite(sol)):
            sol = np.zeros(3)

        U = sol[:2]
        Omega = float(sol[2])
        self.X += U * dt
        self.phi = float(np.mod(self.phi + Omega * dt, 2 * np.pi))
        self.speed = float(np.linalg.norm(U))
        self.angular_speed = Omega

    # -- output ---------------------------------------------------------
    def world_nodes(self) -> np.ndarray:
        c, s = np.cos(self.phi), np.sin(self.phi)
        R = np.array([[c, -s], [s, c]])
        return self.local @ R.T + self.X

    def head(self) -> np.ndarray:
        return self.world_nodes()[0]

    def tail(self) -> np.ndarray:
        return self.world_nodes()[-1]

    def head_direction(self) -> np.ndarray:
        n = self.world_nodes()
        d = n[0] - n[2]
        return d / (np.linalg.norm(d) + 1e-12)
