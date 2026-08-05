"""Graded-potential network dynamics over the real connectome.

C. elegans neurons do not fire action potentials. They are isopotential,
graded-release cells, so the right model is a leaky integrator with sigmoidal
synaptic release rather than anything integrate-and-fire. This implements the
standard formulation used by Wicks, Roehrig & Rankin (1996) and by Kunert,
Shlizerman & Kutz (2014):

    C dV_i/dt = -G_c (V_i - E_cell) - I_gap_i - I_syn_i + I_ext_i

    I_gap_i = sum_j  g_gap * Gg[i,j] * (V_i - V_j)
    I_syn_i = sum_j  g_syn * Gs[i,j] * s_j * (V_i - E_j)

    ds_j/dt = a_r * phi(V_j) * (1 - s_j) - a_d * s_j
    phi(V_j) = 1 / (1 + exp(-beta (V_j - V_th_j)))

V_th is not a free parameter: it is chosen per cell so the network sits at
equilibrium with every synapse half-activated, which is solved as a linear
system (see `_solve_thresholds`). This is what keeps a 448-cell network with
real, wildly heterogeneous connectivity from saturating.
"""

from __future__ import annotations

import numpy as np

from .connectome import Connectome
from .genome import Genome


class NeuralParams:
    """Units: mV, nS, pF, ms.

    Values follow the Kunert/Shlizerman reference implementation
    (github.com/shlizee/C-elegans-Neural-Interactome), which is Kunert et al.
    2014 with a uniform 1.5x time rescale so the membrane time constant lands
    on the 150 ms that Wicks et al. 1996 derived from cell geometry.

    Two easy unit traps, both of which bite hard:
      * a_r and a_d are published per SECOND. Read as per-millisecond they run
        synapses a thousand times too fast and the network loses its dynamics.
      * one synaptic contact should be about 10x the leak conductance
        (100 pS vs 10 pS). Making them equal guts the network's recurrence.
    """

    C = 1.5           # membrane capacitance, pF
    G_leak = 0.01     # leak conductance, nS  -> tau_m = C/G = 150 ms
    E_cell = -35.0    # leak reversal / resting potential, mV
    g_gap = 0.1       # conductance of one gap-junction contact, nS (100 pS)
    g_syn = 0.1       # conductance of one chemical synaptic contact, nS
    E_exc = 0.0       # excitatory reversal potential, mV
    E_inh = -48.0     # inhibitory reversal potential, mV (Wicks Table 1)
    beta = 0.125      # sigmoid steepness, 1/mV
    a_r = (1.0 / 1.5) / 1000.0   # synaptic rise rate, 1/ms  (0.667 s^-1)
    a_d = (5.0 / 1.5) / 1000.0   # synaptic decay rate, 1/ms (3.33 s^-1)


class NervousSystem:
    def __init__(self, conn: Connectome, genome: Genome,
                 params: NeuralParams | None = None, seed: int = 0) -> None:
        self.conn = conn
        self.genome = genome
        self.p = params or NeuralParams()
        self.rng = np.random.default_rng(seed)
        self.n = conn.n

        # Per-edge reversal potentials, E_syn[post, pre]. Mostly determined by
        # the presynaptic transmitter, but glutamate is target-dependent in this
        # animal, so the connectome carries explicit overrides.
        self.E_syn = conn.E_syn

        half = 0.5 * self.p.a_r
        self.s_eq = half / (half + self.p.a_d)
        self.ablated: set[str] = set()
        self._ablated_idx = np.array([], dtype=int)

        self._apply_genetics()
        self.reset()

    # -- ablation -------------------------------------------------------
    def ablate(self, name: str) -> None:
        """Kill a cell, the way a laser microbeam does.

        The cell stops sending and stops receiving: its rows and columns are
        zeroed in both the chemical and electrical matrices, and it is clamped
        at its resting potential so it cannot be driven. Mapping circuits by
        killing cells one at a time is how the touch circuit, the command
        interneurons and the escape response were all worked out in the first
        place (Chalfie et al. 1985; Gray, Hill & Bargmann 2005).
        """
        self.ablated.add(name)
        self._apply_genetics()

    def restore_cell(self, name: str) -> None:
        self.ablated.discard(name)
        self._apply_genetics()

    def clear_ablations(self) -> None:
        self.ablated.clear()
        self._apply_genetics()

    # -- genetics -------------------------------------------------------
    def _apply_genetics(self) -> None:
        """Scale synaptic weights by neurotransmitter, honouring knockouts."""
        g = self.genome
        scale = np.ones(self.n)
        for i, nts in enumerate(self.conn.pre_nt):
            if not nts:
                continue
            scale[i] = float(np.mean([g.nt_scale(nt) for nt in nts]))
        global_syn = g.global_scale("chemical_synapse")

        # Gs[post, pre] -> scaling is per presynaptic cell, so scale columns.
        self.Gs_eff = self.conn.Gs * scale[np.newaxis, :] * global_syn
        self.Gg_eff = self.conn.Gg.copy()
        self.nt_scale_vec = scale

        # Ablated cells neither send nor receive.
        if self.ablated:
            idx = [self.conn.index[n] for n in self.ablated if n in self.conn.index]
            if idx:
                self.Gs_eff[idx, :] = 0.0
                self.Gs_eff[:, idx] = 0.0
                self.Gg_eff[idx, :] = 0.0
                self.Gg_eff[:, idx] = 0.0
        self._ablated_idx = np.array(
            [self.conn.index[n] for n in self.ablated if n in self.conn.index],
            dtype=int)
        # Recompute thresholds so the surviving network still rests at
        # equilibrium after cells are removed.
        self.V_th = self._solve_thresholds()

    def refresh_genetics(self) -> None:
        """Recompute weights and thresholds after a knockout changes."""
        self._apply_genetics()
        self.V_th = self._solve_thresholds()

    # -- initialisation -------------------------------------------------
    def _solve_thresholds(self) -> np.ndarray:
        """Pick V_th per cell so the resting network is at equilibrium.

        At rest every synapse sits at s_eq = a_r/(a_r + a_d). Setting dV/dt = 0
        and solving for V gives the operating point; using that as V_th centres
        each sigmoid on its own resting potential, so every cell has headroom to
        respond in both directions.
        """
        p = self.p
        # At V = V_th the sigmoid reads exactly 0.5, so the synaptic variable
        # settles at a_r*0.5 / (a_r*0.5 + a_d) -- not a_r/(a_r+a_d). Using the
        # wrong constant here leaves the network drifting off its own threshold.
        s_eq = self.s_eq

        Gg = self.Gg_eff * p.g_gap
        Gs = self.Gs_eff * p.g_syn

        diag = p.G_leak + Gg.sum(axis=1) + s_eq * Gs.sum(axis=1)
        A = np.diag(diag) - Gg
        b = p.G_leak * p.E_cell + s_eq * (Gs * self.E_syn).sum(axis=1)

        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def reset(self) -> None:
        self.V_th = self._solve_thresholds()
        self.V = self.V_th.copy()
        self.s = np.full(self.n, self.s_eq)

    # -- dynamics -------------------------------------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def derivatives(self, V: np.ndarray, s: np.ndarray, I_ext: np.ndarray):
        """Explicit derivatives. Kept for testing against the stable stepper."""
        p = self.p
        Gg = self.Gg_eff * p.g_gap
        Gs = self.Gs_eff * p.g_syn

        I_gap = V * Gg.sum(axis=1) - Gg @ V
        gs_active = Gs * s[np.newaxis, :]
        I_syn = V * gs_active.sum(axis=1) - (gs_active * self.E_syn).sum(axis=1)

        dV = (-p.G_leak * (V - p.E_cell) - I_gap - I_syn + I_ext) / p.C
        phi = self._sigmoid(p.beta * (V - self.V_th))
        ds = p.a_r * phi * (1.0 - s) - p.a_d * s
        return dV, ds

    def step(self, dt: float, I_ext: np.ndarray, noise: float = 0.0) -> None:
        """Advance by dt ms using exponential Euler.

        Both equations are linear in their own state variable once the coupling
        terms are frozen over the step, so each can be advanced exactly:
        x(t+dt) = x_inf + (x - x_inf) exp(-dt/tau). Heavily wired cells here have
        membrane time constants well under 1 ms, which makes explicit stepping
        blow up at any timestep fast enough to be useful; this form is stable
        for any dt.
        """
        p = self.p
        V, s = self.V, self.s
        Gg = self.Gg_eff * p.g_gap
        Gs = self.Gs_eff * p.g_syn

        gs_active = Gs * s[np.newaxis, :]
        # Coefficient of V_i in its own current balance, and everything else.
        G_tot = p.G_leak + Gg.sum(axis=1) + gs_active.sum(axis=1)
        drive = (p.G_leak * p.E_cell + Gg @ V
                 + (gs_active * self.E_syn).sum(axis=1) + I_ext)

        G_tot = np.maximum(G_tot, 1e-12)
        V_inf = drive / G_tot
        tau_V = p.C / G_tot
        self.V = V_inf + (V - V_inf) * np.exp(-dt / tau_V)

        phi = self._sigmoid(p.beta * (self.V - self.V_th))
        rate = p.a_r * phi + p.a_d
        s_inf = p.a_r * phi / rate
        self.s = np.clip(s_inf + (s - s_inf) * np.exp(-dt * rate), 0.0, 1.0)

        if noise > 0:
            self.V += self.rng.normal(0.0, noise, self.n)
        if len(self._ablated_idx):
            self.V[self._ablated_idx] = self.V_th[self._ablated_idx]
            self.s[self._ablated_idx] = 0.0

    # -- readout --------------------------------------------------------
    def activation(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Normalised 0..1 drive of each cell relative to its own threshold."""
        v = self.V if idx is None else self.V[idx]
        th = self.V_th if idx is None else self.V_th[idx]
        return 1.0 / (1.0 + np.exp(-self.p.beta * (v - th)))

    def potential(self, name: str) -> float:
        return float(self.V[self.conn.idx(name)])
