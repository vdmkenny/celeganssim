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


# The six touch receptor neurons. Habituation of the tap-withdrawal response
# lives at their OUTPUT synapses, not in the receptor current: the decrement
# is synaptic depression at the sensory-to-interneuron connections (Wicks &
# Rankin 1997 Behav Neurosci 111:342; Rankin, Beck & Chiba 1990 Behav Brain
# Res 37:89 for the behavioural decrement and its recovery over minutes).
# Scoping depression to these cells leaves every other synapse static, so
# single-poke calibrations are untouched and gait cannot be collaterally
# damaged, which is the documented failure of applying depression
# network-wide (docs/emergent-cpg.md, release-cooperativity branch).
TOUCH_RECEPTOR_NEURONS = ("ALML", "ALMR", "AVM", "PLML", "PLMR", "PVM")


class NeuralParams:
    """Units: mV, nS, pF, ms.

    Synaptic form and rate constants follow Kunert et al. 2014; passive
    properties and reversal potentials follow direct patch-clamp measurement
    in C. elegans, cited per parameter below.

    Note a_r and a_d are published per second and are written here per
    millisecond to match the timebase.
    """

    # Passive properties. Identified C. elegans neurons have an input
    # resistance of 1.6-8 GOhm and a capacitance under 4 pF, giving a membrane
    # time constant of a few milliseconds.
    # Refs: Goodman, Hall, Avery & Lockery 1998 Neuron 20:763 (R_in 2-8 GOhm,
    # C_m < 4 pF); Shindou et al. 2019 Sci Rep 9:3430 (R_in 1.6-2.2 GOhm);
    # Liu, Kidd, Dobosiewicz & Bargmann 2018 Cell 175:57 Table S4 (g_L 0.25 nS,
    # E_L -65 mV, C 1.5 pF).
    C = 1.5           # membrane capacitance, pF
    G_leak = 0.25     # leak conductance, nS  -> tau_m = C/G = 6 ms
    E_cell = -65.0    # leak reversal potential, mV
    # Per anatomical contact. Measured coupling is reported whole-cell across
    # all the contacts a pair shares: 56 pS for AVAL-AVAR (18 contacts) and
    # 60-95 pS one-way for AVA-VA5. At 3 pS per contact the AVAL-AVAR pair
    # comes out at 54 pS against the measured 56 pS.
    # Refs: Liu, Chen & Wang 2020 Nat Commun 11:5076; Liu et al. 2017 Nat
    # Commun 8:14818.
    g_gap = 0.003     # conductance of one gap-junction contact, nS (3 pS)
    g_syn = 0.1       # conductance of one chemical synaptic contact, nS
    # Neuromuscular contacts are calibrated to the junction rather than
    # inheriting the neuron-to-neuron value. The synaptic release variable in
    # this formulation cannot exceed a_r/(a_r + a_d) = 0.167, so a synapse
    # delivers at most a sixth of its nominal conductance and spans only 1.83x
    # from rest to saturation. At the generic 0.1 nS a muscle could reach
    # 0.95 nS against the 8.5 nS measured by whole-cell recording (774 +/- 79
    # pA of acetylcholine current at -80 mV with a +11 mV reversal, Richmond &
    # Jorgensen 1999 Nat Neurosci 2:791), which is 11% of the real junction and
    # far too weak to reach the muscle spike threshold. This value puts the
    # achievable whole-cell conductance on the measurement.
    g_syn_nmj = 1.02  # conductance of one neuromuscular contact, nS
    E_exc = 0.0       # excitatory reversal potential, mV
    E_inh = -48.0     # inhibitory reversal potential, mV (Wicks Table 1)

    # --- body-wall muscle passive properties ---
    #
    # Muscle is not a small neuron, and the 95 body-wall muscle cells carried
    # in the connectome are not neurons. Measured against the values above, a
    # muscle cell rests 40 mV depolarised, is ~47x larger and integrates ~12x
    # slower, so the neuronal parameters misstate all three.
    #
    # Resting potential, whole-cell current clamp:
    #   -25.0 +/- 1.0 mV (n = 27)  Gao & Zhen 2011 PNAS 108:2557
    #   -19.7 +/- 1.8 mV (n = 12)  Jospin et al. 2002 J Cell Biol 159:337
    # Two labs and two preparations agree within 5 mV; the larger sample is
    # used. This rest is far depolarised of the potassium equilibrium and is
    # attributed to a high chloride permeability (Gao & Zhen 2011), which is
    # why it cannot be recovered from a neuron's -65 mV leak.
    E_muscle = -25.0        # resting potential, mV
    # Input resistance 1.0 +/- 0.08 GOhm (n = 10), Jospin et al. 2002 -> 1 nS.
    G_leak_muscle = 1.0     # leak conductance, nS
    # ~70 pF, Richmond, "Electrophysiological recordings from the neuromuscular
    # junction of C. elegans", WormBook doi:10.1895/wormbook.1.112.1.
    C_muscle = 70.0         # membrane capacitance, pF -> tau_m = 70 ms

    # Muscle calcium. Force follows calcium, not membrane potential, and the
    # calcium transient is an order of magnitude slower than the membrane, so
    # it is the stage that sets when a muscle actually pulls. Contractile
    # calcium comes from SR release through the ryanodine receptor UNC-68
    # gated by EGL-19: transients are abolished by nemadipine-A and absent in
    # unc-68 nulls (Liu et al. 2011 J Physiol 589:101 Fig 10).
    #
    # Kinetics are the electrically evoked GCaMP3 transient in dissected
    # body-wall muscle, fitted as exp(-t/0.88 s) - exp(-t/0.25 s) with its peak
    # 0.44 s after excitation (Butler et al. 2015 J R Soc Interface 12:20140963
    # Fig 3b). A cascade of two first-order filters with these constants
    # reproduces that impulse response, and its peak time
    # ln(t_d/t_r) / (1/t_r - 1/t_d) comes out at 0.439 s independently.
    #
    # This is indicator fluorescence, so it is an upper bound on the true
    # transient: GCaMP3 kinetics are convolved into it and cannot be removed.
    # Absolute free calcium has never been measured in this animal, in any
    # units, so calcium here is normalised drive rather than a concentration.
    ca_rise_ms = 250.0      # calcium rise time constant, ms
    ca_decay_ms = 880.0     # calcium decay time constant, ms

    # --- muscle action potential ---
    #
    # Body-wall muscle fires all-or-none calcium action potentials, and those
    # spikes are what drive contraction: Gao & Zhen 2011 PNAS 108:2557 titled
    # the result "Action potentials drive body wall muscle contractions". A
    # postsynaptic current of only -67.9 +/- 7.5 pA (n=8) triggers one, which
    # is the step that converts graded neuromuscular input into force. Without
    # it, synaptic input moves a 1 GOhm cell a few millivolts and nothing
    # contracts.
    #
    # INWARD, measured. Jospin et al. 2002 J Cell Biol 159:337 give the peak
    # (transient) EGL-19 component as G_max 199 S/F, V_0.5 +0.6 mV, k 4.7 mV,
    # threshold -20 mV, reversal +50 mV. At the measured 70 pF that is 13.9 nS,
    # and the same number falls out of the measured maximum upstroke rate:
    # 1.38 V/s x 70 pF is 97 pA of net inward current, which at -5 mV needs
    # 9.2 nS against the 8.9 nS the steady-state density gives. Two
    # independent routes to the same conductance.
    #
    # Inactivation is PARTIAL, and the residual is measured too: the maintained
    # component is 127 S/F of the 199 S/F peak, so 0.638 survives.
    g_Ca_muscle = 13.93     # peak calcium conductance, nS (199 S/F x 70 pF)
    E_Ca_muscle = 50.0      # calcium reversal, mV (Jospin peak component)
    V_Ca_muscle = 0.6       # half-activation, mV
    k_Ca_muscle = 4.7       # activation steepness, mV
    h_min_muscle = 127.0 / 199.0   # residual after inactivation, measured
    #
    # OUTWARD, fitted. SHK-1 and SLO-2 carry repolarisation (shk-1 lengthens
    # the spike more than 30-fold) but neither has a published body-wall muscle
    # I-V, so these four are fitted to the measured waveform rather than
    # measured: amplitude 45.1 +/- 0.7 mV (Liu et al. 2011, n=25) to 52.6 +/-
    # 1.8 mV (Gao & Zhen 2011), half-width 15.5 +/- 0.9 ms, threshold about
    # -10 mV. The fit reproduces 45.6 mV and 18.7 ms, and is all-or-none: a
    # sustained 5 pA does not fire and 30 pA does.
    g_K_muscle = 6.0        # repolarising potassium conductance, nS
    V_K_muscle = 0.0        # half-activation, mV
    k_K_muscle = 6.0        # steepness, mV
    tau_K_muscle = 4.0      # activation time constant, ms
    V_h_muscle = -5.0       # calcium inactivation half-point, mV
    k_h_muscle = 7.0        # its steepness, mV
    tau_h_muscle = 10.0     # its time constant, ms

    # Touch-cell release depression. Rate is fitted to the classic decrement,
    # about half the response gone after ten taps at a 10 s interstimulus
    # interval (Rankin et al. 1990); recovery runs over minutes.
    trn_dep_rate = 2.3e-4     # depression per ms of above-rest release
    trn_dep_recovery_ms = 90000.0
    beta = 0.125      # sigmoid steepness, 1/mV
    a_r = (1.0 / 1.5) / 1000.0   # synaptic rise rate, 1/ms  (0.667 s^-1)
    a_d = (5.0 / 1.5) / 1000.0   # synaptic decay rate, 1/ms (3.33 s^-1)

    # --- intrinsic oscillator currents (motor neurons only) ---
    #
    # A-class motor neurons are intrinsic oscillators, sufficient to drive
    # backward locomotion with the premotor interneurons removed, and that
    # oscillation depends on the P/Q/N-type calcium channel UNC-2 (Gao et al.
    # 2018 eLife 7:e29915). These are a Morris-Lecar pair modelling it: a fast
    # regenerative calcium conductance and a slow potassium conductance, so a
    # cell is quiescent until premotor drive depolarises it.
    #
    # Values are not measured and the currents are disabled by default; see
    # NervousSystem.intrinsic and docs/emergent-cpg.md.
    # Refs: Liu et al. 2018 Cell 175:57 and Jiang et al. 2022 (regenerative
    # calcium in C. elegans neurons); Morris & Lecar 1981.
    g_Ca = 0.28       # regenerative calcium conductance, nS
    # Measured calcium reversal in C. elegans: +50 to +59 mV in body-wall
    # muscle at 6 mM external Ca (Jospin et al. 2002 J Cell Biol 159:337) and
    # +21 mV in ASER at 1 mM (Goodman et al. 1998). The +120 mV textbook value
    # does not apply to this animal.
    E_Ca = 60.0       # calcium reversal potential, mV
    V_Ca = -25.0      # half-activation of the calcium conductance, mV
    k_Ca = 7.0        # its steepness, mV
    g_K = 0.55        # slow potassium conductance, nS
    E_K = -80.0       # potassium reversal potential, mV
    V_K = -18.0       # half-activation of the slow current, mV
    k_K = 9.0         # its steepness, mV
    tau_w = 850.0     # slow-current time constant, ms -> ~0.5 Hz rhythm


# Provenance tags for the parameter registry (worm/parameters.py). The
# intrinsic-oscillator block is disabled by default; its tags reflect that
# only E_Ca and E_K are anchored to measurement.
PROVENANCE = {
    "C": "measured",        # Goodman et al. 1998; Liu et al. 2018 Table S4
    "G_leak": "measured",   # R_in 1.6-8 GOhm: Goodman 1998; Shindou 2019
    "E_cell": "measured",   # AWA fit, Liu et al. 2018
    "g_gap": "measured",    # Liu et al. 2017/2020 pair coupling (within 1.6x)
    "g_syn": "published",   # Kunert et al. 2014 contact conductance
    "g_syn_nmj": "measured",  # calibrated to Richmond & Jorgensen 1999
    "E_exc": "published",
    "E_inh": "published",   # Wicks et al. 1996 Table 1
    "E_muscle": "measured",      # Gao & Zhen 2011; Jospin et al. 2002
    "G_leak_muscle": "measured", # R_in 1.0 GOhm, Jospin et al. 2002
    "C_muscle": "measured",      # ~70 pF, Richmond WormBook
    "ca_rise_ms": "measured",    # Butler et al. 2015 Fig 3b (GCaMP3)
    "ca_decay_ms": "measured",   # same fit; peak 0.44 s after excitation
    "g_Ca_muscle": "measured",   # Jospin 199 S/F x 70 pF; also from max dV/dt
    "E_Ca_muscle": "measured",   # Jospin et al. 2002, +50 mV peak component
    "V_Ca_muscle": "measured",   # Jospin Boltzmann fit
    "k_Ca_muscle": "measured",
    "h_min_muscle": "measured",  # maintained/peak conductance ratio, Jospin
    "g_K_muscle": "tuned",       # no muscle I-V for SHK-1 or SLO-2
    "V_K_muscle": "tuned",
    "k_K_muscle": "tuned",
    "tau_K_muscle": "tuned",
    "V_h_muscle": "tuned",
    "k_h_muscle": "tuned",
    "tau_h_muscle": "tuned",
    "trn_dep_rate": "tuned",         # fit to Rankin et al. 1990 decrement
    "trn_dep_recovery_ms": "tuned",  # recovery over minutes, same source
    "beta": "published",    # Kunert et al. 2014
    "a_r": "published",     # Wicks/Kunert, 1.5x time rescale
    "a_d": "published",
    "g_Ca": "tuned",        # intrinsic oscillator: no measured conductance
    "E_Ca": "measured",     # Jospin et al. 2002 (+50..+59 mV in muscle)
    "V_Ca": "tuned",
    "k_Ca": "tuned",
    "g_K": "tuned",
    "E_K": "measured",      # -80..-84.5 mV, Liu et al. 2014 solutions
    "V_K": "tuned",
    "k_K": "tuned",
    "tau_w": "tuned",
}


class NervousSystem:
    def __init__(self, conn: Connectome, genome: Genome,
                 params: NeuralParams | None = None, seed: int = 0) -> None:
        self.conn = conn
        self.genome = genome
        self.p = params or NeuralParams()
        self.rng = np.random.default_rng(seed)
        self.n = conn.n

        # Per-edge reversal potentials, E_syn[post, pre]. Derived from the
        # postsynaptic cell's measured receptor expression where possible
        # (glutamate is target-dependent in this animal), with a
        # transmitter-level heuristic as the tagged fallback.
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
        # Disabled: no parameter regime for these currents produces a
        # physiological voltage swing, so the locomotor rhythm comes from the
        # ventral cord oscillator in body.py. See docs/emergent-cpg.md for the
        # measured values a working version needs.
        self.intrinsic = False
        # Body-wall muscle fires calcium action potentials, and they are what
        # drives contraction. Unlike the neuronal oscillator above, the inward
        # conductance here is measured, so this is on.
        self.muscle_spikes = True

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

        # Per-cell passive properties. Muscle differs from neuron in all three,
        # so they are arrays rather than scalars; see NeuralParams for the
        # measurements. Muscle rest is measured in 27 and 12 cells respectively
        # and is a target for calibrate_rest like any other measured cell.
        self._muscle_idx = np.where(conn.is_muscle)[0]
        # Per-postsynaptic-row synaptic conductance: the junction is measured,
        # the neuron-to-neuron value is a published generic.
        self.g_syn_row = np.where(conn.is_muscle, self.p.g_syn_nmj,
                                  self.p.g_syn)[:, np.newaxis]
        self.C = np.where(conn.is_muscle, self.p.C_muscle, self.p.C)
        self.G_leak = np.where(conn.is_muscle, self.p.G_leak_muscle,
                               self.p.G_leak)
        for i in np.where(conn.is_muscle)[0]:
            self._rest_targets[int(i)] = self.p.E_muscle

        self._trn_idx = np.array([conn.index[n] for n in TOUCH_RECEPTOR_NEURONS
                                  if n in conn.index], dtype=int)
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
    def _muscle_gates(self, V: np.ndarray):
        """Activation and inactivation of the muscle spike currents at V."""
        p = self.p
        m = self._sigmoid((V - p.V_Ca_muscle) / p.k_Ca_muscle)
        h = p.h_min_muscle + (1.0 - p.h_min_muscle) * self._sigmoid(
            -(V - p.V_h_muscle) / p.k_h_muscle)
        w = self._sigmoid((V - p.V_K_muscle) / p.k_K_muscle)
        return m, h, w

    def _muscle_rest_conductance(self):
        """Standing spike-current conductance at the target resting potential.

        The threshold solve is linear, so the calcium and potassium currents
        enter it as the fixed conductances they carry at rest. Leaving them out
        would put the solved equilibrium tens of millivolts away from where the
        network actually settles.
        """
        g_ca = np.zeros(self.n)
        g_k = np.zeros(self.n)
        if not self.muscle_spikes or not self._muscle_idx.size:
            return g_ca, g_k
        p = self.p
        V0 = np.full(self._muscle_idx.size, p.E_muscle)
        m, h, w = self._muscle_gates(V0)
        g_ca[self._muscle_idx] = p.g_Ca_muscle * m * h
        g_k[self._muscle_idx] = p.g_K_muscle * w
        return g_ca, g_k

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
        Gs = self.Gs_eff * self.g_syn_row

        g_ca, g_k = self._muscle_rest_conductance()
        diag = (self.G_leak + g_ca + g_k + Gg.sum(axis=1)
                + s_eq * Gs.sum(axis=1))
        A = np.diag(diag) - Gg
        b = (self.G_leak * self.E_leak + g_ca * p.E_Ca_muscle + g_k * p.E_K
             + s_eq * (Gs * self.E_syn).sum(axis=1))

        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def calibrate_rest(self, iterations: int = 40) -> None:
        """Set E_leak so measured cells rest where they were measured to rest.

        A published resting potential is that of an intact animal with its
        synaptic input intact, so the quantity to match is the solved network
        equilibrium rather than the leak reversal itself. Since that
        equilibrium is linear in E_leak, a few fixed-point steps converge.

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
            Gs = self.Gs_eff * self.g_syn_row
            g_ca, g_k = self._muscle_rest_conductance()
            G_tot = (self.G_leak + g_ca + g_k + Gg.sum(axis=1)
                     + self.s_eq * Gs.sum(axis=1))[idx]
            self.E_leak[idx] += err * (G_tot / self.G_leak[idx])
        self.V_th = self._solve_thresholds()

    def reset(self) -> None:
        self.calibrate_rest()
        self.V_th = self._solve_thresholds()
        self.V = self.V_th.copy()
        self.s = np.full(self.n, self.s_eq)
        # Touch-cell output depression, 1 = fresh. See TOUCH_RECEPTOR_NEURONS.
        self.dep = np.ones(self.n)
        # Muscle calcium, as a two-stage cascade so the impulse response is
        # the measured biexponential. Starts at the resting activation, which
        # is 0.5 because each threshold is solved to its own resting potential.
        self.ca_stage = np.full(self._muscle_idx.size, 0.5)
        self.ca = np.full(self._muscle_idx.size, 0.5)
        # Spike gating, at its steady state for the muscle resting potential.
        _, h0, w0 = self._muscle_gates(
            np.full(self._muscle_idx.size, self.p.E_muscle))
        self.h_m, self.w_m = h0, w0
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
        Gs = self.Gs_eff * self.g_syn_row

        I_gap = V * Gg.sum(axis=1) - Gg @ V
        gs_active = Gs * s[np.newaxis, :]
        I_syn = V * gs_active.sum(axis=1) - (gs_active * self.E_syn).sum(axis=1)

        dV = (-self.G_leak * (V - self.E_leak) - I_gap - I_syn + I_ext) / self.C
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
        Gs = self.Gs_eff * self.g_syn_row

        gs_active = Gs * (s * self.dep)[np.newaxis, :]
        # Coefficient of V_i in its own current balance, and everything else.
        G_tot = self.G_leak + Gg.sum(axis=1) + gs_active.sum(axis=1)
        drive = (self.G_leak * self.E_leak + Gg @ V
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

        if self.muscle_spikes and self._muscle_idx.size:
            # Regenerative calcium against a slower potassium current. This is
            # the step that turns graded synaptic input into contraction.
            mi = self._muscle_idx
            m, _, _ = self._muscle_gates(V[mi])
            g_ca = p.g_Ca_muscle * m * self.h_m
            g_k = p.g_K_muscle * self.w_m
            G_tot[mi] += g_ca + g_k
            drive[mi] += g_ca * p.E_Ca_muscle + g_k * p.E_K

        G_tot = np.maximum(G_tot, 1e-12)
        V_inf = drive / G_tot
        tau_V = self.C / G_tot
        self.V = V_inf + (V - V_inf) * np.exp(-dt / tau_V)

        if self.intrinsic:
            w_inf = self._w_inf(self.V)
            self.w = w_inf + (self.w - w_inf) * np.exp(-dt / p.tau_w)

        if self.muscle_spikes and self._muscle_idx.size:
            _, h_inf, w_inf = self._muscle_gates(self.V[self._muscle_idx])
            self.h_m = h_inf + (self.h_m - h_inf) * np.exp(-dt / p.tau_h_muscle)
            self.w_m = w_inf + (self.w_m - w_inf) * np.exp(-dt / p.tau_K_muscle)

        phi = self._sigmoid(p.beta * (self.V - self.V_th))
        if self._trn_idx.size:
            t = self._trn_idx
            excess = np.maximum(phi[t] - 0.5, 0.0) * 2.0
            self.dep[t] += dt * ((1.0 - self.dep[t]) / p.trn_dep_recovery_ms
                                 - p.trn_dep_rate * excess * self.dep[t])
            np.clip(self.dep[t], 0.05, 1.0, out=self.dep[t])
        rate = p.a_r * phi + p.a_d
        s_inf = p.a_r * phi / rate
        self.s = np.clip(s_inf + (s - s_inf) * np.exp(-dt * rate), 0.0, 1.0)

        # Muscle calcium follows activation through two first-order stages,
        # which together give the measured exp(-t/t_d) - exp(-t/t_r) transient.
        if self._muscle_idx.size:
            drive_m = phi[self._muscle_idx]
            k_r = 1.0 - np.exp(-dt / p.ca_rise_ms)
            k_d = 1.0 - np.exp(-dt / p.ca_decay_ms)
            self.ca_stage += (drive_m - self.ca_stage) * k_r
            self.ca += (self.ca_stage - self.ca) * k_d

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
        return self._sigmoid(self.p.beta * (v - th))

    def muscle_calcium(self) -> np.ndarray:
        """Calcium of each body-wall muscle, in the order of self._muscle_idx.

        Normalised drive rather than a concentration: no C. elegans body-wall
        muscle calcium measurement has ever been calibrated to nM or uM, so
        every published transient is a fluorescence ratio.
        """
        return self.ca

    def potential(self, name: str) -> float:
        return float(self.V[self.conn.idx(name)])
