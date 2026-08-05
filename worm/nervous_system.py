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


# Measured resting potentials of ventral cord motor neurons, whole-cell current
# clamp in situ (Liu, Chen & Wang 2014 Nat Commun 5:5155, Table 1). VA5, VB6 and
# VD5 are the ONLY ventral cord motor neurons ever patched; DA, DB, DD and AS
# have never been recorded, so they inherit their functional partner's value and
# that is an assumption.
MEASURED_REST: dict[str, float] = {
    "VA": -71.7, "DA": -71.7,     # A class, backward
    "VB": -53.2, "DB": -53.2,     # B class, forward (19 mV depolarised of A)
    "VD": -45.8, "DD": -45.8,     # D class, GABAergic
}


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

    # Passive properties, corrected against measurement. The Kunert values
    # (G_leak 0.01 nS, E_cell -35 mV) imply a 150 GOhm input resistance; real
    # identified C. elegans neurons measure 1.6-8 GOhm, giving a leak nearer
    # 0.25 nS. E_leak follows the AWA fit, which is the best-constrained set in
    # the literature.
    # Refs: Goodman, Hall, Avery & Lockery 1998 Neuron 20:763 (R_in 2-8 GOhm,
    # C_m < 4 pF); Shindou et al. 2019 Sci Rep 9:3430 (R_in 1.6-2.2 GOhm);
    # Liu, Kidd, Dobosiewicz & Bargmann 2018 Cell 175:57 Table S4 (g_L 0.25 nS,
    # E_L -65 mV, C 1.5 pF).
    C = 1.5           # membrane capacitance, pF
    G_leak = 0.25     # leak conductance, nS  -> tau_m = C/G = 6 ms
    E_cell = -65.0    # leak reversal potential, mV
    # Kunert uses 100 pS per anatomical CONTACT, but measured whole-cell
    # coupling between a pair is 56 pS (AVAL-AVAR) to 60-95 pS one-way
    # (AVA-VA5) across ALL their contacts together. Since AVAL and AVAR share
    # 18 contacts, 100 pS each would give 1.8 nS, roughly 30x the measurement.
    # Refs: Liu, Chen & Wang 2020 Nat Commun 11:5076; Liu et al. 2017 Nat
    # Commun 8:14818.
    g_gap = 0.005     # conductance of one gap-junction contact, nS (5 pS)
    g_syn = 0.1       # conductance of one chemical synaptic contact, nS
    E_exc = 0.0       # excitatory reversal potential, mV
    E_inh = -48.0     # inhibitory reversal potential, mV (Wicks Table 1)
    beta = 0.125      # sigmoid steepness, 1/mV
    a_r = (1.0 / 1.5) / 1000.0   # synaptic rise rate, 1/ms  (0.667 s^-1)
    a_d = (5.0 / 1.5) / 1000.0   # synaptic decay rate, 1/ms (3.33 s^-1)

    # --- intrinsic oscillator currents (motor neurons only) ---
    #
    # The leak-plus-synapse model above is a pure relaxation system: it can only
    # settle to fixed points, never oscillate. But A-class motor neurons ARE
    # intrinsic oscillators, and are sufficient to drive backward locomotion
    # with the premotor interneurons removed entirely (Gao et al. 2018 eLife
    # 7:e29915). That oscillation needs the P/Q/N-type calcium channel UNC-2.
    #
    # So motor neurons get a Morris-Lecar style pair: a fast regenerative
    # calcium conductance for the upstroke and a slow potassium conductance for
    # the downstroke. Below threshold the cell is quiescent; depolarise it (as
    # AVB does through its gap junctions) and it oscillates. That gating is the
    # point -- the premotor interneurons set state, they do not generate rhythm.
    #
    # Refs: Gao et al. 2018 (A-class oscillators, UNC-2 dependent); Liu et al.
    # 2018 Cell 175:57 and Jiang et al. 2022 (regenerative calcium currents and
    # action-potential-like events in C. elegans neurons); Morris & Lecar 1981.
    g_Ca = 0.28       # regenerative calcium conductance, nS
    # +120 mV is the mammalian textbook figure and appears in the C. elegans
    # literature only as a FITTED parameter (Liu et al. 2018 AWA model). The
    # measured reversal is far lower: +50 to +59 mV in body-wall muscle at 6 mM
    # external Ca (Jospin et al. 2002 J Cell Biol 159:337) and +21 mV in ASER at
    # 1 mM (Goodman et al. 1998). Using +120 overstates the driving force at
    # -40 mV by 1.7x, which a regenerative conductance turns into runaway
    # depolarisation.
    E_Ca = 60.0       # calcium reversal potential, mV
    V_Ca = -25.0      # half-activation of the calcium conductance, mV
    k_Ca = 7.0        # its steepness, mV
    g_K = 0.55        # slow potassium conductance, nS
    E_K = -80.0       # potassium reversal potential, mV
    V_K = -18.0       # half-activation of the slow current, mV
    k_K = 9.0         # its steepness, mV
    tau_w = 850.0     # slow-current time constant, ms -> ~0.5 Hz rhythm


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

        # Which cells carry the intrinsic oscillator currents. Only the ventral
        # cord motor neurons: they are the cells shown to oscillate, and giving
        # the whole network regenerative calcium would be inventing physiology
        # nobody has measured.
        self.osc_mask = np.zeros(self.n, dtype=bool)
        for name in conn.names:
            if conn.cell_info[name].get("vnc_class") in ("DA", "DB", "VA", "VB", "AS"):
                self.osc_mask[conn.index[name]] = True
        # OFF by default: the parameters above do not yet produce a
        # physiological oscillation (a parameter search over g_Ca, g_K, their
        # half-activations and tau_w found no regime with a realistic voltage
        # swing), so this stays disabled until the biophysics is pinned down
        # against measured values. See docs/emergent-cpg.md.
        self.intrinsic = False

        # Per-cell leak reversal. A and B class motor neurons rest 19 mV apart,
        # so treating them as interchangeable is wrong. These are the only
        # ventral cord motor neurons ever patched; every other cell keeps the
        # generic value and that is an inference, not a measurement.
        # Ref: Liu, Chen & Wang 2014 Nat Commun 5:5155, Table 1.
        self.E_leak = np.full(self.n, self.p.E_cell)
        self._rest_targets = {}
        for name in conn.names:
            cls = conn.cell_info[name].get("vnc_class")
            if cls in MEASURED_REST:
                self._rest_targets[conn.index[name]] = MEASURED_REST[cls]

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
            # Per-cell, so a knockout only touches cells that express the gene.
            cell = self.conn.names[i]
            scale[i] = float(np.mean([g.nt_scale_in_cell(nt, cell) for nt in nts]))
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
        b = p.G_leak * self.E_leak + s_eq * (Gs * self.E_syn).sum(axis=1)

        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def calibrate_rest(self, iterations: int = 40) -> None:
        """Set E_leak so measured cells rest where they were measured to rest.

        The published resting potential of VA5 is the potential of an intact
        animal, synaptic input included, so simply assigning it as E_leak
        overshoots: the network then pulls the cell somewhere else entirely.
        The right question is the inverse one, what leak reversal puts the
        SOLVED equilibrium on the measured value, and because the equilibrium
        is linear in E_leak a few fixed-point steps converge on it.

        Cells without a measurement are untouched.
        """
        if not self._rest_targets:
            return
        idx = np.array(sorted(self._rest_targets))
        target = np.array([self._rest_targets[i] for i in idx])
        for _ in range(iterations):
            V = self._solve_thresholds()
            err = target - V[idx]
            if np.abs(err).max() < 0.05:
                break
            # dV/dE_leak is G_leak/G_total; step by the inverse of that.
            Gg = self.Gg_eff * self.p.g_gap
            Gs = self.Gs_eff * self.p.g_syn
            G_tot = (self.p.G_leak + Gg.sum(axis=1)
                     + self.s_eq * Gs.sum(axis=1))[idx]
            self.E_leak[idx] += err * (G_tot / self.p.G_leak)
        self.V_th = self._solve_thresholds()

    def reset(self) -> None:
        self.calibrate_rest()
        self.V_th = self._solve_thresholds()
        self.V = self.V_th.copy()
        self.s = np.full(self.n, self.s_eq)
        # Slow recovery variable of the intrinsic oscillator, at its steady
        # state for the resting potential.
        self.w = self._w_inf(self.V)

    def _w_inf(self, V):
        p = self.p
        return self._sigmoid((V - p.V_K) / p.k_K)

    def _m_inf(self, V):
        p = self.p
        return self._sigmoid((V - p.V_Ca) / p.k_Ca)

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

        dV = (-p.G_leak * (V - self.E_leak) - I_gap - I_syn + I_ext) / p.C
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
        drive = (p.G_leak * self.E_leak + Gg @ V
                 + (gs_active * self.E_syn).sum(axis=1) + I_ext)

        if self.intrinsic:
            # Fast regenerative calcium (instantaneous gate) plus slow
            # potassium. Together these turn a motor neuron from a relaxation
            # element into a relaxation OSCILLATOR, but only once it is
            # depolarised past threshold -- which is what premotor drive does.
            m = self._m_inf(V) * self.osc_mask
            gK = p.g_K * self.w * self.osc_mask
            gCa = p.g_Ca * m
            G_tot = G_tot + gCa + gK
            drive = drive + gCa * p.E_Ca + gK * p.E_K

        G_tot = np.maximum(G_tot, 1e-12)
        V_inf = drive / G_tot
        tau_V = p.C / G_tot
        self.V = V_inf + (V - V_inf) * np.exp(-dt / tau_V)

        if self.intrinsic:
            w_inf = self._w_inf(self.V)
            self.w = w_inf + (self.w - w_inf) * np.exp(-dt / p.tau_w)

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
