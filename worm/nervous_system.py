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

import math

import numpy as np

from .connectome import E_INH, E_INH_MUSCLE, Connectome
from .genome import Genome


# Chloride homeostasis: two extruders against one importer.
#
# Bellemer et al. 2011 EMBO J 30:1852 measured the logic. Either extruder
# single mutant is mildly affected, body length falling from 1212 um to 1145
# (kcc-2) or 1064 (abts-1), and already reverses the muscimol response. The
# DOUBLE mutant is paralysed at 510 um with body bends almost absent, because
# chloride flow reverses and inhibitory transmitters start to excite. So the
# two extruders are largely redundant, neither alone carries the gradient, and
# what opposes them is the importer NKCC-1.
#
# The shares are a GUESS constrained by those body lengths, with abts-1 given
# the larger one because its single mutant is the more affected of the two.
# What the shares have to reproduce is the ORDERING, which they do: wild type
# holds E_Cl below rest, either single lands near neutral, and the double puts
# it above rest so that GABA depolarises.
CHLORIDE_EXTRUDERS: dict[str, float] = {"kcc-2": 0.45, "abts-1": 0.55}
CHLORIDE_IMPORTER = "nkcc-1"

# Passes of the chloride fixed point, and the residual that ends it. Each pass
# moves rest by at most the inhibitory share of the conductance, about 16%, so
# a handful is plenty; the tolerance is far below any voltage that matters.
# Floor on the chloride reversal, set below the most hyperpolarised cell ever
# measured in this animal (VA5, -71.7 mV) so no measurement is affected. GUESS.
CL_POLE_FLOOR_MV = -80.0
CL_SOLVE_ITERATIONS = 24
CL_SOLVE_TOL_MV = 1e-6


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
    # How far below its own resting potential a cell holds its chloride
    # reversal. Read off the muscle pair above: rest -25 mV against a -30 mV
    # chloride reversal, so the extruders win by 5 mV and GABA there is
    # mostly a shunt. GUESS for neurons, which have never had a chloride
    # reversal measured; see _apply_chloride_gradient for why this value
    # rather than the one that scores best.
    cl_extrusion_mv = 5.0   # E_Cl below rest, mV
    # How far ABOVE rest the importer NKCC-1 drives the chloride reversal once
    # the extruders are gone. GUESS, taken symmetric with the extruded case
    # because nothing measures it; what constrains it is that Bellemer's
    # extruder double mutant must come out clearly reversed rather than merely
    # shunting, since that mutant is paralysed and hypercontracted.
    cl_import_mv = 5.0      # E_Cl above rest with no extrusion, mV
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

    # Touch-cell release depression. Rate is fitted to the decrement the
    # mechanism genuinely produces, and its limits are findings, not tuning
    # targets. A localised anterior poke barely habituates: that response is
    # carried largely by gap junctions depression cannot touch, and its
    # chemical component is sign-mixed (ALM's chemical output targets the
    # forward driver PVC). A near-threshold plate tap habituates well,
    # through the DIFFERENCE of two antagonistic reflexes (Wicks & Rankin
    # 1995 J Neurosci 15:2434). A strong, reliably suprathreshold tap does
    # NOT habituate here, and driving the rate harder makes it SENSITISE:
    # PLM's output onto the backward pool is inhibitory, so deep depression
    # disinhibits AVA faster than the anterior side decrements (measured: at
    # 3.3x this rate the tap response grows 10% over twelve taps). Closing
    # that gap needs per-class depression rates, which were never measured,
    # or plasticity downstream of the sensory synapses; it is tracked as an
    # expected failure in the validation suite. Recovery runs over minutes
    # (Rankin, Beck & Chiba 1990).
    trn_dep_rate = 5.5e-4     # depression per ms of above-rest release
    # Depression must not be driven by noise. Release above rest is rectified
    # before it drives depression, and rectified zero-mean noise has a
    # positive mean, so without a deadband the ordinary membrane jitter
    # depresses the touch cells tonically and recovery can never complete
    # (measured: a 180 s rest recovered essentially nothing). The deadband
    # sits several standard deviations above the resting release noise; a
    # real touch drives release to saturation and clears it by an order of
    # magnitude.
    trn_dep_deadband = 0.05   # release excess ignored as noise
    # Depression onset is slow relative to one stimulus. The measured
    # decrement is ACROSS taps at a 10 s interstimulus interval (Rankin et
    # al. 1990), and letting release depress instantaneously lets a single
    # 350 ms poke disinhibit its own late response: PLM's inhibition onto AVA
    # weakened mid-poke, and gentle-posterior and harsh-posterior touch both
    # began driving reversals through the rebound (measured: the posterior
    # band rose from 0.0017-0.0021 to 0.0032-0.0037 and harsh posterior to
    # 0.0043-0.0048, past the reversal threshold). The depression drive is
    # therefore low-passed with a time constant that sits between one
    # response window (about 3 s) and the interstimulus interval (10 s), so
    # depression expresses BETWEEN stimuli, which is where the measurement
    # lives, and most of each tap's depression charge still lands before the
    # next tap. At 2 s the filter only delayed the rebound into the window's
    # tail; at 8 s the in-window expression is a few tenths of a percent.
    trn_dep_onset_ms = 8000.0
    trn_dep_recovery_ms = 90000.0
    beta = 0.125      # sigmoid steepness, 1/mV
    # How far above its own resting operating point each cell's release
    # sigmoid sits. Zero is the Wicks/Kunert convention, which centres the
    # sigmoid on rest and therefore leaves every synapse releasing at 54.5%
    # of maximum while the animal is doing nothing: that is what drives
    # tonic synaptic conductance to 5x the leak and forces unphysical leak
    # reversals (issue #17). Raising it moves release into the sigmoid's
    # foot, where a resting synapse is quiet and has most of its range
    # still ahead of it.
    #
    # Held at zero because the trade-off was measured and it is a wall, not
    # a tuning problem. Raising the offset improves every static quantity
    # monotonically: at 40 mV tonic release falls from 54.5% of maximum to
    # 0.8%, muscle whole-cell input resistance rises from 0.15 to 0.85 GOhm
    # against Jospin's measured 1.0, unphysical leak reversals fall from 111
    # cells to 59, and the ratio of evoked to resting release improves
    # tenfold. Behaviour does not survive it: at 20 mV the touch response is
    # already gone (no poke-evoked reversal at all) and crawling drops from
    # 0.230 to 0.099 mm/s, at 40 mV to 0.066 with 25 cells driven far from
    # threshold. Everything downstream was calibrated against a network
    # whose synapses sit half-saturated at rest, so closing issue #17 means
    # re-deriving the reversal threshold, the junction strength and the gait
    # constants together, not moving this number.
    v_th_offset_mv = 0.0
    # Wicks et al. 1996 J Neurosci 16:4017, rescaled 1.5x as in Kunert
    # et al. 2014 PLoS Comput Biol 10:e1003472.
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
    # Read off the measured muscle pair (rest -25, E_Cl -30) and carried to
    # neurons, where no chloride reversal has ever been measured because the
    # gramicidin method fails on these membranes (Bellemer et al. 2011).
    "cl_extrusion_mv": "inferred",
    "cl_import_mv": "inferred",  # symmetric with the above; nothing measures it
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
    "trn_dep_deadband": "tuned",     # several sigma above resting noise
    "trn_dep_onset_ms": "tuned",     # long against one tap, short against ISI
    "beta": "published",    # Kunert et al. 2014
    "v_th_offset_mv": "tuned",  # issue #17; 0.0 is the published convention
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
        #
        # Copied, not aliased: _apply_chloride_gradient rewrites the chloride
        # end of every edge against this cell's own resting potential, and a
        # Connectome can be shared between several NervousSystem instances.
        self.E_syn = conn.E_syn.copy()
        # The connectome's nominal chloride pole per postsynaptic cell, which
        # is what the receptor-derived signs were expressed against.
        self._nominal_cl_pole = np.where(conn.is_muscle, E_INH_MUSCLE, E_INH)
        self._e_frac = None
        # How far below rest the chloride reversal sits, in mV. Positive
        # means inhibition inhibits. Set by the genome; see
        # CHLORIDE_EXTRUDERS and _apply_chloride_gradient.
        self.cl_offset_mv = self.p.cl_extrusion_mv

        # Resting release follows the sigmoid's own foot. With the offset
        # at zero this is phi(rest) = 0.5 and s_eq is the Wicks value; with
        # the sigmoid moved up, resting release falls with it, and the
        # threshold solver has to use the SAME number the dynamics will,
        # or the solved operating point and the real one disagree and
        # calibrate_rest drives E_leak to extremes chasing the difference.
        phi_rest = 1.0 / (1.0 + math.exp(self.p.beta * self.p.v_th_offset_mv))
        self.s_eq = (self.p.a_r * phi_rest) / (self.p.a_r * phi_rest
                                               + self.p.a_d)
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
        # Chloride homeostasis, as the balance of extrusion against import.
        # Applied here rather than inside the chloride solve so ablation and
        # knockout share one entry point.
        #
        #   offset = extrusion_mv * E  -  import_mv * (1 - E) * I
        #
        # with E the surviving extrusion capacity and I the importer. A
        # positive offset holds E_Cl below rest and inhibition inhibits; a
        # negative one puts it above rest and inhibition excites, which is
        # the measured double-mutant condition. The importer only bites once
        # extrusion is compromised, which is why the wild type is unaffected
        # by nkcc-1 and the extruder double mutant is not.
        extrusion = 1.0
        for gene, share in CHLORIDE_EXTRUDERS.items():
            resolved = g.resolve(gene)
            if resolved is not None and resolved in g.knockouts:
                extrusion -= share
        extrusion = float(np.clip(extrusion, 0.0, 1.0))
        importer = g.resolve(CHLORIDE_IMPORTER)
        imports = 0.0 if (importer is not None
                          and importer in g.knockouts) else 1.0
        self.cl_offset_mv = float(
            self.p.cl_extrusion_mv * extrusion
            - self.p.cl_import_mv * (1.0 - extrusion) * imports)

        # Recompute thresholds so the surviving network still rests at
        # equilibrium after cells are removed.
        self.V_th = self._solve_thresholds()
        self._apply_chloride_gradient()

    def _apply_chloride_gradient(self) -> None:
        """Hold each chloride reversal below its OWN cell's resting potential.

        E_Cl is not a constant of the animal, it is a quantity every excitable
        cell maintains. C. elegans neurons and muscles extrude chloride
        through KCC-2 and the Na-driven Cl-HCO3 exchanger ABTS-1; lose one and
        the animal is mildly impaired, lose both and chloride flow REVERSES,
        inhibitory transmitters excite, and the animal is paralysed
        (Bellemer et al. 2011 EMBO J 30:1852).

        Modelling it as a single number was the bug this fixes. The
        connectome assigns a chloride-mediated synapse the fixed neuronal
        reversal of -48 mV, but neurons here rest anywhere from -64 to -23 mV,
        so 1,029 of the 1,712 chloride edges were DEPOLARISING their target:
        the double-mutant condition, network-wide, in an animal that is
        supposed to be wild type. See docs/dynamic-range.md.

        The offset is anchored on the one chloride reversal in this animal
        with a measured basis. Body-wall muscle rests at -25 mV (E_muscle,
        Gao & Zhen 2011) and its chloride reversal is -30 mV (E_INH_MUSCLE,
        derived from Richmond & Jorgensen 1999), so the extruders hold E_Cl
        exactly 5 mV below rest and inhibition there acts mainly by shunting.
        Applying that same offset to neurons reproduces the muscle value to
        0.08 mV, which is the consistency check `_chloride_gradient` makes.

        The offset for NEURONS is a GUESS in the strict sense: no C. elegans
        neuron has a measured chloride reversal, and Bellemer says why, the
        standard gramicidin method fails to perforate these membranes. Two
        things bound how much that matters. It is the least invented value
        available, being read off the one cell type that was measured. And
        muscle is arguably the worst available model for a neuron, since its
        depolarised rest is itself attributed to unusually high chloride
        permeability, so a neuron may well hold a steeper gradient. Measured
        sensitivity, on the propagation AUC against the functional atlas:
        5 mV gives +0.0003 +/- 0.0036, 10 mV gives +0.0089 +/- 0.0036 and
        15 mV gives +0.0052 +/- 0.0039. The best-scoring offset is therefore
        NOT the one used here, deliberately: picking 10 because it scores
        better is fitting the constant to the metric, and this metric cannot
        resolve that difference honestly (docs/citations.md).

        Circular by construction, since rest depends on E_syn and E_syn now
        depends on rest, so it is iterated. calibrate_rest runs INSIDE that
        loop rather than after it: it pins the cells with a measured resting
        potential by moving E_leak, which moves rest, which moves the pole.
        Solving the two separately leaves muscle 12 mV off its measured
        chloride reversal, which was the first version of this and is what
        `_chloride_gradient` now guards.

        What this does NOT reach are the edges the receptor pass could not
        resolve. Those carry a continuous blend between the chloride pole and
        the cation reversal, so an edge that is only marginally chloride sits
        most of the way to 0 mV and still depolarises. That is a real and
        separate bias, since it means "sign unknown" is modelled as "mildly
        excitatory", and it is issue #36 rather than something to fix here.
        """
        from .connectome import E_EXC

        # Excitatory fraction per edge: 1 = pure cation, 0 = pure chloride.
        # Recovered from the connectome's nominal poles so the continuous
        # blend it uses for edges expression could not resolve is preserved
        # and only the chloride END of each edge moves.
        if self._e_frac is None:
            pole = self._nominal_cl_pole[:, np.newaxis]
            self._e_frac = (self.conn.E_syn - pole) / (E_EXC - pole)

        for _ in range(CL_SOLVE_ITERATIONS):
            self.calibrate_rest()
            V_op = self._solve_thresholds() - self.p.v_th_offset_mv
            # A chloride gradient is bounded: a cell cannot hold E_Cl below
            # what its achievable internal chloride allows, and letting the
            # pole track an arbitrarily hyperpolarised cell is what drove
            # the network past its -85 mV operating floor. Clamped below
            # VA5's measured -71.7 mV so no measured cell is affected.
            pole = np.clip(V_op - self.cl_offset_mv,
                           CL_POLE_FLOOR_MV, None)[:, np.newaxis]
            new = pole + self._e_frac * (E_EXC - pole)
            moved = float(np.abs(new - self.E_syn).max())
            self.E_syn = new
            if moved < CL_SOLVE_TOL_MV:
                break
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
            V_op = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            V_op = np.linalg.lstsq(A, b, rcond=None)[0]
        return V_op + p.v_th_offset_mv

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
        self._trn_exc = np.zeros(self._trn_idx.size)
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
            excess = np.maximum(phi[t] - 0.5 - p.trn_dep_deadband,
                                0.0) * 2.0
            self._trn_exc += (dt / p.trn_dep_onset_ms) \
                * (excess - self._trn_exc)
            self.dep[t] += dt * ((1.0 - self.dep[t]) / p.trn_dep_recovery_ms
                                 - p.trn_dep_rate * self._trn_exc
                                 * self.dep[t])
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
