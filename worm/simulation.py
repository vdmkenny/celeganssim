"""Closed loop: world -> sensory neurons -> connectome -> muscles -> movement.

The escape response is modelled as an explicit state machine because the
literature is clear that reversal and the omega turn have separable final motor
pathways (SMD/RIV are required for the turn but not the reversal). Which state
the animal enters, and when, is decided by the real network: the command
interneurons AVA/AVD/AVE and AVB/PVC are read out directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np

from .body import Body, BodyParams, N_SEG
from .connectome import Connectome
from .environment import Environment
from .genome import Genome
from .nervous_system import NervousSystem

FORWARD_CMD = ["AVBL", "AVBR", "PVCL", "PVCR"]
BACKWARD_CMD = ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"]
TURN_CELLS = ["RIVL", "RIVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR"]
RIM = ["RIML", "RIMR"]
HEAD_DORSAL = ["SMDDL", "SMDDR", "RMDDL", "RMDDR"]
HEAD_VENTRAL = ["SMDVL", "SMDVR", "RMDVL", "RMDVR"]


# Activation of an undriven muscle. Each cell's threshold is solved to its own
# resting potential, so a muscle at rest reads exactly 0.5 rather than 0, and
# that is where it produces no force.
MUSCLE_REST_ACTIVATION = 0.5

# Wave-algebra constants shared by the two script levels (the body oscillator
# and the muscle pacer), so both express the same wave and the GABA couplings
# keep their validated magnitudes whichever level carries the script.
# PACE_EXC is the excitation scale fitted long ago to the ~19% BL bend
# amplitude of Cronin et al. 2005; PACE_TOTAL_MAX is the drive ceiling above
# which the oscillator's amplitude term saturates.
PACE_EXC = 0.85
PACE_TOTAL_MAX = 1.5
# Steering reaches the head and neck rows only: the anterior 16 muscles are
# innervated by the nerve ring alone and the next 16 by ring and cord
# together (White et al. 1986), which is rows 1-8 of 24, and head steering
# through SMD/RMD acts there. As a fraction of body length:
HEAD_STEER_EXTENT_BL = 8.0 / 24.0
# Time-mean of the rectified wave clip(sin)+ is 1/pi. Subtracting exc times
# this from each muscle's target makes the GABA-intact pattern inject zero
# net current, so a wild-type animal neither shortens nor tonically
# contracts; when GABA is lost the pattern's mean rises above this baseline
# and the excess is genuine co-contraction, which the mechanics turn into
# the shrinker's shortening (McIntire, Jorgensen, Kaplan & Horvitz 1993
# Nature 364:337).
RECTIFIED_WAVE_MEAN = 1.0 / np.pi
# Neural substeps to settle the tonic calcium baseline at startup: covers
# several muscle calcium time constants (rise 250 ms, decay 880 ms, Butler et
# al. 2015), measured once on the pristine network.
CA_BASELINE_SETTLE_STEPS = 4000

# Proprioceptive coupling length, as a fraction of body length. Motor activity
# in a posterior region requires the active bending of an anterior region
# extending ~200 um, measured on a ~1 mm adult in a microfluidic channel that
# clamps the curvature of a middle segment (Wen et al. 2012 Neuron 76:750
# Fig 3D). Held as a fraction rather than an absolute length so a larva keeps
# the same body-relative reach.
#
# The kernel shape is NOT measured. Only three channel lengths were tested,
# 100 um (no effect), 200 and 300 um (both reduced posterior bending), which
# brackets the reach without saying whether sensitivity falls off as a step,
# an exponential or a ramp. A uniform window over the reach is the least
# committed reading of that bracket.
PROPRIO_LENGTH_BL = 0.2

# Head pacemaker cells. The head is a separate oscillator from the body in this
# animal: clamping a middle region straight abolishes bending behind it while
# the region in front keeps undulating (Wen et al. 2012 Neuron 76:750), and the
# head rhythm survives when the ventral cord is cut (Xu et al. 2018 PNAS
# 115:E4493). RMD and SMD are the nerve-ring motor neurons innervating head
# muscle, and RMD is one of the few C. elegans neurons with a measured
# regenerative response, generating all-or-none plateau potentials (Mellem,
# Brockie, Madsen & Maricq 2008 Nat Neurosci 11:865).
HEAD_PACEMAKER_D = ["RMDDL", "RMDDR", "SMDDL", "SMDDR"]
HEAD_PACEMAKER_V = ["RMDVL", "RMDVR", "SMDVL", "SMDVR"]

# Spontaneous reversal generator. The animal reverses without any stimulus,
# and the cellular correlate is measured: AVA activity rises precede
# spontaneous reversals (Kato et al. 2015 Cell 163:656; Gordus et al. 2015),
# and A-class upstates recur one to three times a minute in the only cells
# ever patched (Liu, Chen & Wang 2014 Nat Commun 5:5155: VA5 upstates,
# 1-3/min). The behavioural rate is measured too: roughly two reversals a
# minute during local search just off food, falling toward dispersal, and
# far fewer on food (Gray, Hill & Bargmann 2005 PNAS 102:3184). What is NOT
# measured is the conductance basis of the upstate (UNC-2 by genetics, Gao
# et al. 2018, no parameters), so the GENERATOR is scripted at the measured
# statistics, a seeded Poisson process pulsing the backward command pool,
# while everything downstream, the reversal itself, the omega coupling, the
# gait consequences, runs through the real machinery.
SPONT_PULSE_PA = 12.0        # enough to lift AVA/AVE past the calibrated
                             # reversal threshold; scripted stand-in scale
SPONT_PULSE_S = 1.2          # order of the behavioural bout trigger, well
# Down/up pirouette-rate ratio ~3 (Pierce-Shimomura 1999, Fig. 6). The gain
# is derived from a saturation point, not free. The field is in arbitrary
# units, so the dC/dt at which modulation saturates (GUESS) sits at the
# lower end of the model's own descent rates (cruise down a sigma-15
# Gaussian spans roughly 0.1-0.4 units/min): Pierce-Shimomura's Fig. 6
# saturates at small |dC/dt|, and a higher setting leaves the multiplier
# in its linear region for most headings (measured 0.55x/1.2x instead of
# the 3x asymmetry). Floor and ceiling bound the multiplier to the measured 3x.
# Area-restricted search (Gray, Hill & Bargmann 2005 PNAS 102:3184, Fig. 1):
# the reversal rate is highest in the minutes just after leaving food (the
# cfg.spont_rev_per_min_off_food value IS that local-search rate) and decays
# to a low dispersal rate by 30-40 min. The dispersal endpoint is measured
# (about 0.3/min); the exponential form and time constant are a GUESS drawn
# through Gray's time course, which has most of the decay done by ~15 min.
# Dopamine carries the food memory: cat-2 animals never elevate above
# dispersal while their baseline stays normal (Hills, Brockie & Maricq 2004
# J Neurosci 24:1217), wired via the genome scale "area_restricted_search".
# Omega coupling reads reversal length, not a coin. Gray, Hill & Bargmann
# 2005 (PNAS 102:3184, Fig. 2): the probability that a reversal terminates
# in an omega turn rises with the number of backward body bends, from about
# 0.1-0.2 after a single bend toward near-certainty for long reversals.
# The piecewise-linear curve below is read off their figure, so the values
# are approximate to the plot and tagged as such. A backward bend lasts
# half an undulation period at the pacing frequency.
# Graded escape drive (see the reversal branch of the motor-pool logic):
# at threshold the backward drive starts at BASE and gains PER_THRESHOLD
# for each additional threshold-worth of command excess, capped at MAX.
# The omega ventral sweep. Turn rate was measured against both knobs
# (scratch sweep, recorded here because the result is counter-intuitive):
# turning comes from a traveling wave running along an already-curved
# body, NOT from a deep static bend. A hard anterior bias saturates the
# muscle targets, kills the wave and the thrust with it, and the animal
# holds a static curl at 4-9 deg/s. A GENTLE bias spread over the WHOLE
# body curves the swimming path instead, reaching 65-80 deg/s, which is
# the 140 deg turn Gray, Hill & Bargmann 2005 and Broekmans et al. 2016
# report for a real omega in about two seconds. Both constants are tuned
# to that measured reorientation; beyond extent ~1.3 BL the body folds
# onto itself and the turn direction becomes unstable.
OMEGA_HEAD_BIAS = 0.18
OMEGA_COIL_EXTENT_BL = 1.0
BWD_DRIVE_BASE = 0.80
BWD_DRIVE_PER_THRESHOLD = 0.15
BWD_DRIVE_MAX = 1.0
OMEGA_P_ONE_BEND = 0.15
OMEGA_P_PER_BEND = 0.3
OMEGA_P_MAX = 0.9
ARS_DISPERSAL_PER_MIN = 0.3
ARS_TAU_S = 600.0
KLINO_SAT_DCDT_PER_MIN = 0.15
KLINO_TAU_S = 4.0
KLINO_GAIN = math.log(3.0) / KLINO_SAT_DCDT_PER_MIN
KLINO_RATE_CEIL_X = 3.0
KLINO_RATE_FLOOR_X = 1.0 / 3.0
                             # under the 14.9 s dissected-prep upstate


@dataclass
class SimConfig:
    dt: float = 0.02              # body/world timestep, seconds
    neural_substeps: int = 20     # neural steps per body step (1 ms each)
    neural_noise: float = 0.02    # mV per neural step
    # DO NOT lower this without the issue #12 readout work. It is not merely
    # uncalibrated, it is load-bearing through SATURATION: at 55 pA the tonic
    # gas sensors (URX, AQR, PQR at 21% oxygen) sit parked at their rails,
    # where constant output transmits nothing. At physiological amplitudes
    # they land mid-range at maximum gain and amplify membrane noise into the
    # command balance, becoming 98% of the sensory noise floor and burying
    # touch discrimination (scripts/noise_audit.py, both arms measured).
    # Physiological receptor potentials were implemented twice and reverted
    # twice; the terminal constraint is that the single-threshold command
    # readout cannot fit mec-10 partiality, posterior silence and harsh
    # silence inside a 1.7x signal-to-floor ratio. The fix is a
    # pathway-discriminating readout, which belongs to the same redesign as
    # issue #7's escape machinery; until then this constant stays.
    sensory_amplitude: float = 55.0
    # Reversal fires when AVA/AVD/AVE rise this far above their own recent
    # average, not above the naive resting value. Tonic sensory input (21%
    # oxygen, AWC's baseline activity, ambient temperature) holds the command
    # neurons well off rest, so a fixed threshold either never fires or fires
    # constantly. Adapting the baseline is also what the real animal does.
    # Calibrated against the corrected 2020 wiring with the documented PLM
    # sign override and PHASIC mechanotransduction (O'Hagan et al. 2005: the
    # receptor current fires at stimulus onset and offset and adapts within
    # tens of ms, so the command neurons integrate a transient, not a hold).
    # Peak command-balance deviation to a strength-1 poke, 3 seeds, min to
    # max, measured WITH touch-cell depression and its 8 s onset filter in
    # place: anterior wild type 0.0061-0.0065; anterior mec-10 0.0033-0.0037;
    # anterior mec-4 null 0.0015-0.0018; mid-body 0.0019-0.0023; gentle
    # posterior 0.0020-0.0024; harsh posterior 0.0024-0.0029. The threshold
    # sits INSIDE the mec-10 band and above every other, so wild type always
    # responds, mec-10 responds to a fraction of pokes (the partial-loss
    # phenotype, Arnadottir et al. 2011 report ~50% residual touch current),
    # and nulls, mid-body, gentle posterior and harsh posterior stay below.
    # The onset filter matters here: with instantaneous depression a single
    # poke disinhibited its own late response through PLM and pushed the
    # posterior bands past this threshold.
    reversal_threshold: float = 0.0034
    baseline_tau_s: float = 8.0
    reversal_min_s: float = 0.9
    reversal_max_s: float = 4.0
    omega_s: float = 3.0
    refractory_s: float = 0.6
    tonic_forward: float = 0.62   # baseline AVB drive -> spontaneous forward
    command_gain: float = 9.0
    seed: int = 0
    # Strength of the proprioceptive current, pA per unit normalised curvature.
    # A free parameter: no stretch-evoked current has ever been recorded in a
    # B-type motor neuron, so there is no measured amplitude, reversal
    # potential, threshold or adaptation to anchor it. Off by default.
    propr_gain: float = 0.0
    # Spontaneous reversal rates, per minute (see SPONT_PULSE_PA above).
    # Off food: local-search rate, Gray, Hill & Bargmann 2005 (about 2/min
    # in the first minutes off food). On food: reversals are much rarer;
    # this value is a GUESS consistent with the same paper's on-food
    # behaviour, no precise published number is carried here.
    spont_rev_per_min_off_food: float = 1.9
    spont_rev_per_min_on_food: float = 0.35
    # Muscle-level pacing: the compromise mode. The same wave algebra the
    # body oscillator uses, delivered as per-muscle currents into the 95 real
    # muscle cells instead of as prescribed curvature. The script couples in
    # at the end organ: downstream of every neuron the assays read, so the
    # command balance stays clean, and upstream of the mechanics, so muscles
    # are load-bearing: ablating one removes its force, and the body is
    # driven by real muscle calcium with measured kinetics. Motor-gene knobs
    # (nmj gate, gaba cross-inhibition) keep their validated couplings by
    # gating this current exactly as they gated the oscillator. Amplitude in
    # pA per muscle, fitted to the measured bend amplitude.
    # Prescribed as a VOLTAGE swing, converted per muscle through that
    # cell's own resting conductance (I = mV x G_rest). Injecting equal
    # CURRENT into every muscle turns the animal in circles: ventral muscle
    # carries roughly twice the GABAergic shunt of dorsal (13 VD against 6 DD
    # motor neurons, White et al. 1986), so equal current moves ventral cells
    # less and the standing dorsal excess curls the path (measured: mean
    # force 0.332 dorsal vs 0.274 ventral, 0.65/mm standing curvature, the
    # whole trajectory inside a 2 mm box). Equal voltage cancels the wiring
    # asymmetry structurally.
    muscle_pacemaker_mv: float = 6.0
    # Calcium-to-force gain. The transfer function is unmeasured in this
    # animal (no length-tension, no force-velocity, no calibrated
    # calcium-to-force curve; Butler et al. 2015 say so), so the scale is fit
    # to the measured bend amplitude of ~19-21%% BL (Cronin et al. 2005),
    # with force referenced to each muscle's tonic baseline: 16.0 gives
    # 19.1%% BL at 0.25 mm/s net.
    muscle_force_gain: float = 16.0
    start_adult: bool = True      # False starts as a freshly hatched L1
    # How many seconds of development pass per second of simulated behaviour.
    # Real development takes ~50 h, which nobody wants to watch in real time.
    life_speedup: float = 400.0


# Provenance tags for the parameter registry (worm/parameters.py). Most of
# SimConfig is behavioural glue, honestly tagged: these are the knobs a
# mechanistic replacement (issues #6, #7, #10) is expected to delete.
PROVENANCE = {
    "dt": "tuned",                # numerical, body/world step
    "neural_substeps": "tuned",   # numerical, 1 ms neural step
    "neural_noise": "tuned",
    "sensory_amplitude": "tuned", # too large vs measured receptor potentials (issue #12)
    "reversal_threshold": "tuned",
    "baseline_tau_s": "tuned",
    "reversal_min_s": "tuned",
    "reversal_max_s": "tuned",
    "omega_s": "tuned",
    "refractory_s": "tuned",
    "tonic_forward": "scripted",  # stands in for AVB->B tonic drive
    "command_gain": "tuned",
    "seed": "tuned",
    "muscle_pacemaker_mv": "scripted",  # the wave, delivered at the end organ
    "muscle_force_gain": "tuned",    # fit to measured bend amplitude
    "propr_gain": "tuned",   # no stretch-evoked current ever recorded
    "spont_rev_per_min_off_food": "measured",  # Gray et al. 2005 local search
    "spont_rev_per_min_on_food": "tuned",      # guess, same source qualitative
    "start_adult": "tuned",
    "life_speedup": "tuned",      # display convenience, not biology
}


@dataclass
class SimState:
    t: float = 0.0
    behavior: str = "forward"
    state_time: float = 0.0
    reversal_count: int = 0
    omega_count: int = 0
    reversal_vigor: float = 0.0
    distance: float = 0.0
    trail: list = field(default_factory=list)


class WormSimulation:
    # Per-process memo of the pristine tonic calcium baseline, keyed on the
    # exact tonic input vector (see __init__).
    _ca0_cache: dict = {}

    def __init__(self, env: Environment | None = None,
                 config: SimConfig | None = None,
                 genome: Genome | None = None,
                 connectome: Connectome | None = None) -> None:
        self.cfg = config or SimConfig()
        self.env = env or Environment()
        self.genome = genome or Genome.load()
        self.conn = connectome or Connectome.load()
        self.ns = NervousSystem(self.conn, self.genome, seed=self.cfg.seed)
        from .sensory import SensorySystem  # local import: avoids a cycle
        self.sensory = SensorySystem(self.conn, self.genome)
        self.body = Body(BodyParams(), seed=self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        from .lifecycle import Lifecycle
        self.life = Lifecycle(seed=self.cfg.seed)
        if self.cfg.start_adult:
            # Skip development, but still run what the L4/adult moult does:
            # spermatogenesis happens once, there, and nothing else makes
            # sperm afterwards. Lifespan is drawn as at the moult.
            from .lifecycle import SELF_SPERM
            self.life.stage = "adult"
            self.life.reserves = 0.7
            self.life.self_sperm = SELF_SPERM
            self.life.lifespan_d = self.life.draw_lifespan(
                self.genome.longevity_scale())
        self.body.set_length(self.life.body_length_mm)
        self.events: list[tuple[float, str]] = []

        self.i_fwd = self.conn.indices(FORWARD_CMD)
        self.i_bwd = self.conn.indices(BACKWARD_CMD)
        self.i_turn = self.conn.indices(TURN_CELLS)
        self.i_rim = self.conn.indices(RIM)
        self.i_hd = self.conn.indices(HEAD_DORSAL)
        self.i_hv = self.conn.indices(HEAD_VENTRAL)
        self.i_B = self.conn.indices(
            self.conn.group(vnc_class="DB") + self.conn.group(vnc_class="VB")
            + self.conn.group(vnc_class="AS"))
        self.i_A = self.conn.indices(
            self.conn.group(vnc_class="DA") + self.conn.group(vnc_class="VA"))
        self.i_muscle = np.where(self.conn.is_muscle)[0]
        self._muscle_rows()
        # Positions of each segment's muscles within the calcium array, which
        # is indexed over muscle cells only rather than over the whole network.
        pos = {int(i): k for k, i in enumerate(self.ns._muscle_idx)}
        self.ca_row_d = [np.array([pos[i] for i in r], dtype=int)
                         for r in self.row_d]
        self.ca_row_v = [np.array([pos[i] for i in r], dtype=int)
                         for r in self.row_v]
        self._build_proprioception()
        self.i_head_d = self.conn.indices(HEAD_PACEMAKER_D)
        self.i_head_v = self.conn.indices(HEAD_PACEMAKER_V)
        # Muscle pacing geometry: body position u and side per muscle cell,
        # from the anatomical row (1 anterior to 24 posterior).
        mus = [n for n in self.conn.names
               if self.conn.cell_info[n]["kind"] == "muscle"]
        mx = max(self.conn.cell_info[n]["row"] for n in mus)
        self._mus_idx = np.array([self.conn.idx(n) for n in mus], dtype=int)
        self._mus_u = np.array([(self.conn.cell_info[n]["row"] - 0.5) / mx
                                for n in mus])
        self._mus_dorsal = np.array(
            [self.conn.cell_info[n]["side"] == "dorsal" for n in mus])
        # Resting total conductance per muscle, from the pristine wild-type
        # network, so the mV-to-pA conversion is a fixed anatomical frame:
        # knockouts and ablations shift a cell relative to this baseline
        # rather than silently re-normalising the drive.
        ns = self.ns
        g_tot = (ns.G_leak + (ns.Gg_eff * ns.p.g_gap).sum(axis=1)
                 + ns.s_eq * (ns.Gs_eff * ns.g_syn_row).sum(axis=1))
        self._mus_g0 = g_tot[self._mus_idx].copy()
        self._pace_phase = 0.0
        # Pacing runs on the previous body step's drives, one dt behind.
        self._pace_drive = (0.0, 0.0, 0.0, 1.0)  # fwd, bwd, steer, rate

        self.state = SimState()
        self.reset()
        # Per-muscle tonic operating point. Standing sensory input (ambient
        # oxygen, AWC's tonic activity, temperature) plus the asymmetric
        # GABAergic shunt (13 VD against 6 DD motor neurons, White et al.
        # 1986) hold each muscle's calcium at its own baseline, slightly off
        # the global rest and different dorsal versus ventral. Referencing
        # force to the global 0.5 rectifies that offset into a standing bend
        # (measured: +0.65/mm mean curvature, the path curling into a 2 mm
        # box), so force is referenced to each muscle's own tonic baseline,
        # measured here once on the pristine wild-type network with the pacer
        # silent. Knockouts and ablations are applied afterwards and
        # therefore shift a cell relative to this wild-type frame, which is
        # what keeps the shrinker and ablation phenotypes visible.
        nodes = self.body.world_nodes()
        I0 = self.sensory.compute(self.env, nodes[0], nodes[-1], self.cfg.dt,
                                  amplitude=self.cfg.sensory_amplitude)
        sub_dt = (self.cfg.dt * 1000.0) / self.cfg.neural_substeps
        # The settled baseline is a pure function of the tonic input vector
        # and the pristine network, and every sim in a validation run builds
        # the same one, so it is memoised per process on the exact inputs.
        # 4000 noise-free substeps per construction is the single largest
        # fixed cost in the suite; the cache removes all but the first.
        key = (I0.tobytes(), sub_dt, CA_BASELINE_SETTLE_STEPS)
        cached = WormSimulation._ca0_cache.get(key)
        if cached is not None:
            self._mus_ca0 = cached.copy()
        else:
            for _ in range(CA_BASELINE_SETTLE_STEPS):
                self.ns.step(sub_dt, I0, noise=0.0)
            self._mus_ca0 = self.ns.muscle_calcium().copy()
            WormSimulation._ca0_cache[key] = self._mus_ca0.copy()
        self.ns.reset()
        self.sensory._salt_prev = None
        self.sensory._odor_prev = None
        self.sensory._temp_prev = None

    def _build_proprioception(self) -> None:
        """Wire each B-type motor neuron to the curvature just anterior to it.

        B-type motor neurons sense the bending of the region in front of them,
        which is what carries the undulatory wave from head to tail: clamping a
        middle segment straight abolishes bending behind it while the region in
        front keeps undulating, and imposing a curvature on that segment sets
        the sign and size of the curvature behind it (Wen et al. 2012).

        Direction is anterior to posterior and is established independently by
        vab-7 mutants, in which DB axons project forwards instead of backwards
        and dorsal bends fail to propagate posteriorly while ventral
        propagation through VB is unaffected.
        """
        reach = PROPRIO_LENGTH_BL * N_SEG
        self._propr_idx: list[int] = []
        self._propr_sign: list[float] = []
        self._propr_w: list[np.ndarray] = []
        seg_mid = np.arange(N_SEG) + 0.5
        for cls, sign in (("DB", +1.0), ("VB", -1.0)):
            names = [n for n in self.conn.names
                     if self.conn.cell_info[n].get("vnc_class") == cls]
            if not names:
                continue
            order = sorted(names,
                           key=lambda n: self.conn.cell_info[n]["vnc_index"])
            for k, name in enumerate(order):
                # Spread the class evenly along the body: the cords carry no
                # measured per-cell coordinate, only an anterior-posterior
                # ordering, which this preserves without claiming positions.
                here = (k + 0.5) / len(order) * N_SEG
                w = ((seg_mid < here) & (seg_mid >= here - reach)).astype(float)
                if w.sum() == 0:
                    continue
                self._propr_idx.append(self.conn.idx(name))
                self._propr_sign.append(sign)
                self._propr_w.append(w / w.sum())
        self._propr_idx = np.array(self._propr_idx, dtype=int)
        self._propr_sign = np.array(self._propr_sign)
        self._propr_w = (np.array(self._propr_w) if len(self._propr_w)
                         else np.zeros((0, N_SEG)))

    def muscle_pacemaker_current(self, fwd: float, bwd: float,
                                 gaba: float, head_bias: float) -> np.ndarray:
        """THE scripted surface: the locomotor wave, as per-muscle currents.

        No published model produces this rhythm from the connectome, and no
        ventral cord motor neuron has ever been recorded during locomotion
        (docs/emergent-cpg.md), so the wave is imposed. It is imposed at the
        END ORGAN: downstream of every neuron the assays read, which keeps
        the command balance clean (imposing it on the motor neurons was tried
        and measured out: their gap junctions onto AVA/AVB made mec-4 read as
        wild type), and upstream of the mechanics, so muscles are load
        bearing, ablation has consequences, and the body runs on real muscle
        calcium.

        The algebra per muscle is the validated one: phasic one-sided
        cholinergic excitation, contralateral GABAergic inhibition scaled by
        what genetics leave (Wicks & Rankin-era circuit logic; McIntire et
        al. 1993 Nature 364:337 for the shrinker), a nerve-ring steering bias
        on the head rows (White et al. 1986). Wave direction follows the
        command state, matching B-class activity in forward and A-class in
        backward locomotion (Haspel, O'Donovan & Hart 2010 J Neurosci
        30:11151; Kawano et al. 2011 Neuron 72:572); frequency and wavelength
        are the measured gait (Cronin et al. 2005: 0.47 Hz, 0.62 BL).

        The current is (target - rest) so an undriven muscle sits at its
        measured resting potential, and genetics act exactly as they did on
        the oscillator: unc-13/unc-17 gate the drive, GABA loss turns
        alternation into co-contraction, and the shrinker follows through the
        real co-contraction term in the mechanics.
        """
        cfg = self.cfg
        I = np.zeros(self.conn.n)
        if cfg.muscle_pacemaker_mv <= 0.0:
            return I
        net = fwd - bwd
        total = min(fwd + bwd, PACE_TOTAL_MAX)
        grad = (-1.0 if net >= 0 else 1.0) * 2.0 * np.pi \
            / self.body.p.wavelength_bl
        wave = np.sin(self._pace_phase + grad * self._mus_u)
        exc = PACE_EXC * min(total, 1.0)
        inh = exc * float(np.clip(gaba, 0.0, 1.0))
        d_target = np.clip(exc * (0.5 + 0.5 * wave) - inh * (0.5 - 0.5 * wave),
                           0.0, 1.0)
        v_target = np.clip(exc * (0.5 - 0.5 * wave) - inh * (0.5 + 0.5 * wave),
                           0.0, 1.0)
        target = np.where(self._mus_dorsal, d_target, v_target)
        steer_extent = HEAD_STEER_EXTENT_BL
        if self.state.behavior == "omega":
            # An omega turn is not steering: the whole anterior body coils
            # ventrally (Gray, Hill & Bargmann 2005; Broekmans et al. 2016
            # eLife 5:e17227 quantify the deep body bend). The same bias
            # pathway is widened from the steering head to the coil extent,
            # and the target clip saturates it into the coil. Extent is a
            # GUESS from omega videography (about the anterior half);
            # together with cfg.omega_s it is tuned to the measured
            # post-omega reorientation.
            steer_extent = OMEGA_COIL_EXTENT_BL
        bias = head_bias * np.clip(1.0 - self._mus_u / steer_extent,
                                   0.0, 1.0)
        target = np.clip(target + np.where(self._mus_dorsal, bias, -bias),
                         0.0, 1.0)
        I[self._mus_idx] = cfg.muscle_pacemaker_mv * self._mus_g0 \
            * (target - exc * RECTIFIED_WAVE_MEAN)
        return I

    def set_food_memory_age(self, seconds: float) -> None:
        """Assay preparation: the animal left food this long ago.

        Lets a check measure the long-starved dispersal state without
        simulating thirty empty minutes first.
        """
        self._food_memory_s = self.state.t - seconds

    def _update_gradient_signals(self) -> None:
        """Sample the salt field at the head for klinokinesis.

        Sampled here rather than inside spontaneous_current so the signal
        keeps updating when the reversal generator is silenced, which
        assays do.

        dC/dt is low-passed over seconds the way ASE integrates: the head
        sweeps dorsoventrally at gait frequency, so the raw derivative
        carries a large fast component from the SWEEP riding on the slow
        one from where the body is GOING, and only the slow part says
        whether the animal is climbing.

        The fast part is in principle the weathervane signal of klinotaxis
        (Iino & Yoshida 2009 J Neurosci 29:5370). That was built and
        measured out, and the record is worth keeping. Correlating against
        the COMMANDED pacemaker phase fails outright: the calcium cascade
        delays contraction about 0.44 s (Butler et al. 2015), some 75
        degrees at gait frequency, and a quadrature error that large
        washes the correlation away. Against the muscle activation
        asymmetry, which is downstream of that delay, a steering gain does
        produce drift up the gradient, but it does not survive being
        measured: paired on the same seeds it fell from +6.6 mm at n=16 to
        +3.5 mm at n=32 while t stayed near 1, which is what a null looks
        like as data accumulates. Issue #20 owns the retry.
        """
        c = float(self.env.concentration(self.body.world_nodes()[0], "salt"))
        if self._klino_c_prev is None:
            self._klino_c_prev = c
        inst = (c - self._klino_c_prev) / self.cfg.dt * 60.0
        self._klino_c_prev = c
        a_slow = self.cfg.dt / KLINO_TAU_S
        self._klino_dcdt += a_slow * (inst - self._klino_dcdt)
        # A head-sweep correlation was built here to drive klinotaxis
        # and measured out; see the docstring and issue #20.

    def spontaneous_current(self) -> np.ndarray:
        """Scripted upstate generator: seeded Poisson pulses into AVA/AVE.

        Statistics are the measured ones (see SPONT_PULSE_PA block); the
        genome hook lets goa-1 raise the rate, which is its measured
        phenotype (Segalat, Elkes & Kaplan 1995 Science 267:1648:
        hyperreversal), and food state selects between
        the on-food guess and an off-food rate that decays from local
        search to dispersal as the dopamine-borne food memory ages (see
        ARS_DISPERSAL_PER_MIN).
        """
        I = np.zeros(self.conn.n)
        cfg = self.cfg
        t = self.state.t
        on_food = self.env.on_food(self.body.world_nodes()[0]) > 0.01
        if on_food:
            per_min = cfg.spont_rev_per_min_on_food
            self._food_memory_s = t
        else:
            since = t - self._food_memory_s
            elev = cfg.spont_rev_per_min_off_food - ARS_DISPERSAL_PER_MIN
            elev *= self.genome.global_scale("area_restricted_search")
            per_min = ARS_DISPERSAL_PER_MIN \
                + elev * math.exp(-since / ARS_TAU_S)
        per_min *= self.genome.global_scale("spontaneous_reversal")
        if per_min <= 0.0:
            return I
        # Klinokinesis: the pirouette rate listens to the salt gradient.
        # Pierce-Shimomura et al. 1999 J Neurosci 19:9557 measured pirouette
        # initiation roughly 3x higher while heading down-gradient than up
        # (their Fig. 6), and that asymmetry IS the biased random walk of
        # chemotaxis. The coupling below is the scripted stand-in for the
        # ASE-to-command loop (labelled as such; the emergent route needs
        # issue #7's readout): the generator's rate is multiplied by
        # exp(gain x (down-drive - up-drive)) using the model's own ASER/ASEL
        # outputs, so che-1 and tax-4, which zero those drives through the
        # genome gates, abolish the modulation with no extra wiring. Rate
        # changes rescale the pending waiting time (inhomogeneous Poisson by
        # time rescaling), so modulation acts immediately, not one interval
        # late.
        dcdt_per_min = self._klino_dcdt
        gate = self.sensory._gain("salt_on")
        m = math.exp(-KLINO_GAIN * dcdt_per_min * gate)
        m = min(max(m, KLINO_RATE_FLOOR_X), KLINO_RATE_CEIL_X)
        # Descent-evoked pirouettes do not fade with the food memory:
        # Pierce-Shimomura measured their rates on animals long off food,
        # and chemotaxis assays run for tens of minutes, which the ARS
        # decay would otherwise silence (the full suite caught exactly
        # that). So the boost side is evoked at full local-search scale,
        # while suppression acts on whatever exploratory baseline the food
        # memory currently sets. At m = 1 the branches agree.
        if m > 1.0:
            per_min += (m - 1.0) * cfg.spont_rev_per_min_off_food \
                * self.genome.global_scale("spontaneous_reversal")
        else:
            per_min *= m
        if self._spont_next_s is None:
            self._spont_next_s = t + float(self.rng.exponential(60.0 / per_min))
            self._spont_rate = per_min
        elif per_min != self._spont_rate and self._spont_rate > 0.0:
            self._spont_next_s = t + (self._spont_next_s - t) \
                * (self._spont_rate / per_min)
            self._spont_rate = per_min
        if t >= self._spont_next_s:
            self._spont_until_s = t + SPONT_PULSE_S
            self._spont_next_s = t + float(self.rng.exponential(60.0 / per_min))
        if t < self._spont_until_s:
            I[self.i_bwd] = SPONT_PULSE_PA
        return I

    def proprioceptive_current(self) -> np.ndarray:
        """Current into each cell from the curvature anterior to it, pA.

        Positive curvature is dorsal bending, so it excites the dorsal B-class
        and inhibits the ventral one, which is what makes a bend reproduce
        itself further back. The gain is a free parameter: no stretch-evoked
        current has ever been recorded in a B-type motor neuron, so there is no
        measured amplitude, reversal potential or threshold to anchor it.
        """
        I = np.zeros(self.conn.n)
        if not self._propr_idx.size or self.cfg.propr_gain == 0.0:
            return I
        # Curvature is per mm, so a larva at the same shape reads a larger
        # value; normalising by body length makes the signal shape-based.
        kappa = self.body.curvature * self.body.body_length
        sensed = self._propr_w @ kappa
        I[self._propr_idx] = self.cfg.propr_gain * self._propr_sign * sensed
        return I

    def _muscle_rows(self) -> None:
        """Group the 95 body-wall muscles into dorsal/ventral body segments.

        Approximate by construction. Muscle cells are staggered within each
        quadrant rather than forming transverse rings, and the ventral-left
        quadrant has 23 cells against 24 elsewhere, so no one-to-one alignment
        across quadrants exists. Cell index is mapped proportionally onto
        segments, which preserves anterior-posterior order (verified monotonic
        for every motor class) without claiming a ring.

        Also uniform where the animal is not: the anterior 16 muscles are
        driven by nerve-ring motor neurons only and the next 16 by both the
        ring and the cord, with just the posterior 63 on the cord alone
        (White et al. 1986). The head is a separate oscillator in the animal.
        """
        self.row_d: list[list[int]] = [[] for _ in range(N_SEG)]
        self.row_v: list[list[int]] = [[] for _ in range(N_SEG)]
        rows = [self.conn.cell_info[n].get("row", 1)
                for n in self.conn.names if self.conn.cell_info[n]["kind"] == "muscle"]
        max_row = max(rows) if rows else 24
        for name in self.conn.names:
            info = self.conn.cell_info[name]
            if info["kind"] != "muscle":
                continue
            seg = min(int((info["row"] - 1) / max_row * N_SEG), N_SEG - 1)
            (self.row_d if info["side"] == "dorsal" else self.row_v)[seg].append(
                self.conn.idx(name))

    def muscle_force(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-segment dorsal and ventral force from real muscle calcium.

        Force is taken proportional to calcium above its resting level. That
        proportionality is an assumption, not a measurement: C. elegans muscle
        has no measured length-tension relation, no measured force-velocity
        relation and no calibrated calcium-to-force transfer function. Butler
        et al. 2015 state this and substitute a relation borrowed from a
        neuromechanical model; this is the same placeholder.
        """
        ca = self.ns.muscle_calcium()
        rest = getattr(self, "_mus_ca0", None)
        if rest is None:
            rest = np.full(ca.shape, MUSCLE_REST_ACTIVATION)
        f = np.clip(self.cfg.muscle_force_gain
                    * (ca - rest) / (1.0 - MUSCLE_REST_ACTIVATION), 0.0, 1.0)
        # A dead muscle pulls nothing: ablation clamps the cell at the global
        # rest, which can differ from its tonic baseline, and that residue is
        # not force.
        if self.ns._ablated_idx.size:
            dead = np.isin(self.ns._muscle_idx, self.ns._ablated_idx)
            f = np.where(dead, 0.0, f)
        d = np.array([f[r].mean() if r.size else 0.0 for r in self.ca_row_d])
        v = np.array([f[r].mean() if r.size else 0.0 for r in self.ca_row_v])
        return d, v

    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> None:
        self._pace_phase = 0.0
        self._pace_drive = (0.0, 0.0, 0.0, 1.0)
        self._pace_gated = (0.0, 0.0, 1.0, 0.0)
        self._spont_next_s = None
        self._spont_until_s = 0.0
        self._spont_rate = 0.0
        self._klino_c_prev = None
        # Worms are raised on food, so a fresh animal has just left it:
        # local search from t=0. Assays override with set_food_memory_age.
        self._food_memory_s = 0.0
        self._klino_dcdt = 0.0
        self.ns.reset()
        self.body.reset(x, y, heading)
        self.state = SimState()
        self.state.trail = [self.body.X.copy()]
        # Resting activation is 0.5 for every cell by construction: activation
        # is sigmoid(beta * (V - V_th)) and the threshold solve puts each cell
        # at V == V_th at rest. It is therefore invariant under ablation and
        # knockouts, both of which re-solve V_th and move V with it.
        self._rest_fwd = float(np.mean(self.ns.activation(self.i_fwd)))
        self._rest_bwd = float(np.mean(self.ns.activation(self.i_bwd)))
        self._bwd_baseline: float | None = None

    # -- stimulation ----------------------------------------------------
    def poke_at(self, x: float, y: float, strength: float = 1.0,
                radius: float = 0.22, duration: float = 0.3,
                harsh: bool | None = None) -> dict | None:
        """Poke wherever the user clicked, if that lands on the animal.

        Finds the nearest point on the body centreline and converts it to a
        position along the body, which is what the mechanosensory receptive
        fields are defined over. Returns None if the click missed, so the UI
        can say so instead of silently doing nothing.
        """
        nodes = self.body.world_nodes()
        target = np.array([float(x), float(y)])
        # The arena wraps, so measure to the nearest periodic image.
        d = nodes - target
        d[:, 0] -= np.round(d[:, 0] / self.env.width) * self.env.width
        d[:, 1] -= np.round(d[:, 1] / self.env.height) * self.env.height
        dist = np.linalg.norm(d, axis=1)
        i = int(np.argmin(dist))
        hit_radius = radius * max(self.body.body_length, 0.05)
        if dist[i] > max(hit_radius, float(self.body.radius[i]) * 3.0):
            return None
        u = i / (len(nodes) - 1)
        self.env.poke(u, strength=strength, duration=duration, harsh=harsh)
        return {"u": round(u, 3), "segment": i,
                "distance_mm": round(float(dist[i]), 3),
                "harsh": bool(strength > 1.5 if harsh is None else harsh)}

    def _pool_intact(self, idx: np.ndarray) -> float:
        """Fraction of a named cell pool that has not been ablated."""
        if not len(idx) or not self.ns.ablated:
            return 1.0
        dead = set(self.ns._ablated_idx.tolist())
        alive = sum(1 for i in idx.tolist() if i not in dead)
        return alive / len(idx)

    # -- ablation -------------------------------------------------------
    def ablate(self, name: str) -> dict:
        if name not in self.conn.index:
            raise KeyError(f"{name!r} is not a cell in this connectome")
        self.ns.ablate(name)
        info = self.conn.cell_info[name]
        return {"cell": name, "kind": info["kind"],
                "roles": info.get("roles") or [],
                "ablated": sorted(self.ns.ablated)}

    def restore_cell(self, name: str) -> None:
        self.ns.restore_cell(name)

    def clear_ablations(self) -> None:
        self.ns.clear_ablations()

    # -- genetics -------------------------------------------------------
    def knock_out(self, gene: str) -> dict:
        rec = self.genome.knock_out(gene)
        self.ns.refresh_genetics()
        return rec

    def restore(self, gene: str) -> None:
        self.genome.restore(gene)
        self.ns.refresh_genetics()

    def reset_genome(self) -> None:
        self.genome.reset()
        self.ns.refresh_genetics()

    # -- main loop ------------------------------------------------------
    def step(self) -> dict:
        cfg = self.cfg
        dt = cfg.dt
        nodes = self.body.world_nodes()
        head, tail = nodes[0], nodes[-1]

        # One rhythm clock for cord and head pacing, advanced per body step
        # at the modulated rate set by the previous step's drives.
        self._pace_phase += 2.0 * np.pi * self.body.p.freq_hz \
            * self._pace_drive[3] * dt
        I = self.sensory.compute(self.env, head, tail, dt,
                                 amplitude=cfg.sensory_amplitude)
        self._update_gradient_signals()
        I = I + self.proprioceptive_current() + self.spontaneous_current()
        I = I + self.muscle_pacemaker_current(*self._pace_gated)
        sub_dt = (dt * 1000.0) / cfg.neural_substeps  # ms
        for _ in range(cfg.neural_substeps):
            self.ns.step(sub_dt, I, noise=cfg.neural_noise)

        act = self.ns.activation()
        fwd_cmd = float(np.mean(act[self.i_fwd])) - self._rest_fwd
        bwd_cmd = float(np.mean(act[self.i_bwd])) - self._rest_bwd
        turn_cmd = float(np.mean(act[self.i_turn])) - 0.5

        # Reversal is a competition, not an absolute level. AVA/AVD/AVE and
        # AVB/PVC push against each other, and almost any sensory input raises
        # both -- posterior touch, for instance, drives PVC through gap
        # junctions, and PVC in turn synapses onto AVA. Only the difference
        # says which way the animal actually goes.
        cmd_balance = bwd_cmd - fwd_cmd

        # Slow-adapting baseline. Frozen while reversing so the reversal's own
        # command activity cannot chase the threshold up and cut itself short.
        if self._bwd_baseline is None:
            self._bwd_baseline = cmd_balance
        elif self.state.behavior in ("forward", "refractory"):
            a = 1.0 - np.exp(-dt / max(cfg.baseline_tau_s, 1e-6))
            self._bwd_baseline += a * (cmd_balance - self._bwd_baseline)
        bwd_rel = cmd_balance - self._bwd_baseline

        # Exposed for assays and the viewer: the decision variable the
        # state machine just consumed.
        self.last_cmd_deviation = bwd_rel
        self._update_behavior(dt, bwd_rel)

        g = self.genome
        arousal = g.global_scale("arousal")
        gaba = float(np.mean([g.nt_scale("GABA")]))
        bend = g.global_scale("bend_amplitude")

        # Motor pools: baseline forward drive plus what the network is saying.
        drive_B = float(np.mean(act[self.i_B])) - 0.5
        drive_A = float(np.mean(act[self.i_A])) - 0.5

        # The tonic term stands in for baseline AVB output driving the B-class
        # motor neurons. It has to depend on those cells still existing, or
        # ablating the forward command pair would leave the animal cruising
        # along regardless -- and co-ablating AVB and PVC is precisely the
        # experiment that abolishes forward locomotion (Chalfie et al. 1985).
        fwd_ok = self._pool_intact(self.i_fwd) * self._pool_intact(self.i_B)
        bwd_ok = self._pool_intact(self.i_bwd) * self._pool_intact(self.i_A)

        forward = cfg.tonic_forward + cfg.command_gain * (fwd_cmd + drive_B)
        backward = cfg.command_gain * (bwd_cmd + drive_A)

        if self.state.behavior == "reversal":
            # Escape vigor is graded, not a constant: A-class and AVA
            # activity set backward speed (Kawano et al. 2011 Neuron
            # 72:572; Kato et al. 2015 Cell 163:656), so the backward
            # drive follows how far the command balance sits above the
            # reversal threshold. The relationship is the published one;
            # base, gain and cap are tuned (see BWD_DRIVE_BASE).
            forward = 0.05
            backward = min(BWD_DRIVE_MAX,
                           BWD_DRIVE_BASE
                           + BWD_DRIVE_PER_THRESHOLD * self.state.reversal_vigor)
        elif self.state.behavior == "omega":
            forward, backward = 0.85, 0.05

        # Apply ablation AFTER the state machine, so killing a command pool
        # silences its motor programme outright rather than being overwritten
        # by the state override a line above.
        forward *= fwd_ok
        backward *= bwd_ok

        # Clip to the physiological range FIRST. Weakening the synapses drives
        # the motor pools toward saturation, so scaling before clipping would
        # let a saturated command signal survive the neuromuscular gate below.
        forward = float(np.clip(forward, 0.0, 1.2))
        backward = float(np.clip(backward, 0.0, 1.2))

        # The neuromuscular junction is cholinergic, so anything that breaks
        # acetylcholine or the release machinery it depends on has to reach the
        # muscle -- otherwise unc-13 and unc-17 mutants would keep crawling on
        # the tonic drive alone, which is exactly backwards.
        nmj = float(np.clip(min(g.nt_scale("Acetylcholine"),
                                max(g.global_scale("chemical_synapse"), 0.0)),
                            0.0, 1.3))
        # Pacing bypasses this gate on purpose: in the paced path unc-13 and
        # unc-17 act through the real junction, whose presynaptic release is
        # already scaled per cell by the genome, and gating the current too
        # would count the same lesion twice.
        fwd_pace, bwd_pace = forward, backward
        forward *= nmj
        backward *= nmj

        # Slow down on food (dopaminergic basal + serotonergic enhanced), and
        # a dauer does not crawl around at all if it can help it.
        slow = self.food_slowing(self.env.on_food(head))
        if self.life.dauer:
            slow *= 0.35
        # An embryo cannot crawl, an ageing animal crawls worse, and a dead
        # one does not crawl at all.
        slow *= self.life.locomotion_scale()
        forward *= slow
        backward *= slow
        fwd_pace *= slow
        bwd_pace *= slow
        # Drive alone is not enough: above a total of 1.0 the oscillator's
        # amplitude term saturates, so a 20% cut in drive can vanish entirely.
        # Slowing on food and senescent decline are reductions in locomotion
        # RATE, so they also have to reach the undulation frequency.
        rate_scale = slow

        head_bias = self._head_bias(act, turn_cmd)
        # Rhythm rate for the NEXT step's pacing: the same modulation the
        # scripted oscillator used, so arousal, food slowing and total drive
        # reach the frequency identically in both paths.
        rate = float(np.clip(arousal, 0.3, 2.0)) * max(rate_scale, 0.05) \
            * (0.35 + 0.65 * min(fwd_pace + bwd_pace, 1.0))
        self._pace_drive = (fwd_pace, bwd_pace, head_bias, rate)
        self._pace_gated = (forward, backward,
                            float(np.clip(gaba, 0.0, 1.0)), head_bias)

        self.body.p.curvature_gain = BodyParams.curvature_gain * np.clip(bend, 0.3, 2.0)
        # Shape follows the muscle cells: real calcium, measured kinetics.
        d, v = self.muscle_force()
        self.body.drive_from_muscles(dt, d, v)
        self.muscle_activation = np.zeros(self.conn.n)
        self.muscle_activation[self.ns._muscle_idx] = self.ns.muscle_calcium()
        # A dead muscle displays as dead. Ablation clamps the cell at rest,
        # which would otherwise render at the same mid-brightness as a live
        # idle cell and make the lesion invisible in the viewer.
        if self.ns._ablated_idx.size:
            self.muscle_activation[self.ns._ablated_idx] = 0.0

        prev = self.body.X.copy()
        self.body.step_motion(dt, drag_ratio=self.env.drag_ratio)
        self.body.X = self.env.wrap(self.body.X)
        moved = float(np.linalg.norm(self.body.X - prev))
        if moved < max(self.env.width, self.env.height) / 2:
            self.state.distance += moved

        self._step_lifecycle(dt, head)

        self.env.step(dt)
        self.state.t += dt
        if len(self.state.trail) == 0 or \
                np.linalg.norm(self.body.X - self.state.trail[-1]) > 0.35:
            self.state.trail.append(self.body.X.copy())
            if len(self.state.trail) > 1200:
                self.state.trail.pop(0)

        return self.telemetry(forward, backward, head_bias)

    def _step_lifecycle(self, dt: float, head: np.ndarray) -> None:
        """Feed, grow, and possibly moult, arrest, enter dauer or lay an egg."""
        env = self.env
        food = env.on_food(head)
        temp = env.temperature(head)
        pher = env.pheromone_at(head)
        # Serotonin potentiates food-stimulated pumping, so tph-1 eats less.
        serotonin = float(np.clip(self.genome.nt_scale("Serotonin"), 0.0, 1.5))

        life_dt = dt * self.cfg.life_speedup
        events = self.life.step(life_dt, food=food, temp_c=temp,
                                pheromone=pher, serotonin_scale=serotonin,
                                pump_scale=self.genome.global_scale("pumping"),
                                longevity_scale=self.genome.longevity_scale())
        if food > 0.01:
            from .lifecycle import FOOD_PER_PUMP
            eaten = self.life.pump_hz * FOOD_PER_PUMP * life_dt * min(food, 1.0)
            env.consume(head, eaten)
        self.body.set_length(self.life.body_length_mm)

        for key, val in events.items():
            self.events.append((round(self.state.t, 2), f"{key}:{val}"))
        if len(self.events) > 60:
            self.events = self.events[-60:]

    def food_slowing(self, food: float) -> float:
        """Speed multiplier from the two food-slowing responses.

        These are genetically separable and this is a common place to get the
        sign wrong, so both are modelled explicitly:
          * BASAL slowing is dopaminergic (cat-2) and mechanosensory - a
            well-fed animal slows on contacting bacteria.
          * ENHANCED slowing is serotonergic (tph-1) and only appears when the
            animal has been food-deprived.
        Refs: Sawin, Ranganathan & Horvitz 2000 Neuron 26:619.
        """
        if food <= 0.01:
            return 1.0
        dop = float(np.clip(self.genome.nt_scale("Dopamine"), 0.0, 1.0))
        ser = float(np.clip(self.genome.nt_scale("Serotonin"), 0.0, 1.0))
        basal = 1.0 - 0.22 * dop * min(food, 1.0)
        enhanced = 1.0 - 0.35 * ser * min(food, 1.0) if self.life.starving else 1.0
        return float(basal * enhanced)

    def _update_behavior(self, dt: float, bwd_cmd: float) -> None:
        st = self.state
        st.state_time += dt
        cfg = self.cfg
        if st.behavior == "forward":
            if bwd_cmd > cfg.reversal_threshold:
                st.behavior, st.state_time = "reversal", 0.0
                st.reversal_count += 1
                # Vigor is latched at onset: the size of the AVA transient
                # at reversal entry predicts the speed and extent of the
                # whole retreat (Kato et al. 2015 Cell 163:656), and the
                # instantaneous balance decays during the reversal, which
                # would otherwise read long escapes as weak ones.
                st.reversal_vigor = bwd_cmd / cfg.reversal_threshold - 1.0
        elif st.behavior == "reversal":
            done = st.state_time > cfg.reversal_min_s and bwd_cmd <= cfg.reversal_threshold
            if done or st.state_time > cfg.reversal_max_s:
                # Omega turn only if the SMD/RIV pathway is intact, with a
                # probability set by how far the animal backed up
                # (Gray 2005 Fig. 2; constants at OMEGA_P_ONE_BEND).
                bends = st.state_time * 2.0 * self.body.p.freq_hz
                p_omega = min(OMEGA_P_MAX,
                              OMEGA_P_ONE_BEND
                              + OMEGA_P_PER_BEND * max(bends - 1.0, 0.0))
                if self.genome.global_scale("omega_turn") > 0.5 \
                        and self.rng.random() < p_omega:
                    st.behavior, st.state_time = "omega", 0.0
                    st.omega_count += 1
                else:
                    st.behavior, st.state_time = "refractory", 0.0
        elif st.behavior == "omega":
            if st.state_time > cfg.omega_s:
                st.behavior, st.state_time = "refractory", 0.0
        elif st.behavior == "refractory":
            if st.state_time > cfg.refractory_s:
                st.behavior, st.state_time = "forward", 0.0

    def _head_bias(self, act: np.ndarray, turn_cmd: float) -> float:
        """Dorsoventral bias of the head, i.e. steering.

        Baseline steering is the SMD/RMD dorsal-vs-ventral imbalance, which is
        the klinotaxis (weathervane) pathway. During an omega turn this is
        overridden with a large ventral bias, matching the observed ventral
        preference of real omega turns.
        """
        if self.state.behavior == "omega":
            strength = OMEGA_HEAD_BIAS * self.genome.global_scale("omega_turn")
            return -float(np.clip(strength, 0.0, 1.0))
        d = float(np.mean(act[self.i_hd])) if len(self.i_hd) else 0.5
        v = float(np.mean(act[self.i_hv])) if len(self.i_hv) else 0.5
        bias = np.clip((d - v) * 6.0, -0.45, 0.45)
        # RIM tyramine suppresses head movement during reversals; tdc-1 removes it.
        if self.state.behavior == "reversal":
            bias *= self.genome.global_scale("head_suppression")
        return float(bias)

    # -- reporting ------------------------------------------------------
    def telemetry(self, forward: float, backward: float, head_bias: float) -> dict:
        b = self.body
        return {
            "t": round(self.state.t, 3),
            "behavior": self.state.behavior,
            "x": float(b.X[0]), "y": float(b.X[1]),
            "heading": float(b.phi),
            "speed_mm_s": round(float(b.speed), 4),
            "forward_drive": round(forward, 3),
            "backward_drive": round(backward, 3),
            "head_bias": round(head_bias, 3),
            "length_scale": round(float(b.length_scale), 3),
            "reversals": self.state.reversal_count,
            "omegas": self.state.omega_count,
            "distance_mm": round(self.state.distance, 3),
            "sensory": {k: round(v, 3) for k, v in self.sensory.last.items() if v},
            "knockouts": sorted(self.genome.knockouts),
            "ablated": sorted(self.ns.ablated),
            "life": self.life.summary(),
            "events": self.events[-6:],
        }

    def snapshot(self) -> dict:
        """Everything the viewer needs for one frame."""
        nodes = self.body.world_nodes()
        act = self.ns.activation()
        return {
            "nodes": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in nodes],
            "radius": [round(float(r), 4) for r in self.body.radius],
            "dorsal": [round(float(v), 3) for v in self.body.dorsal],
            "ventral": [round(float(v), 3) for v in self.body.ventral],
            "curvature": [round(float(v), 3) for v in self.body.curvature],
            "trail": [[round(float(p[0]), 2), round(float(p[1]), 2)]
                      for p in self.state.trail[-400:]],
            # Per-cell drive for the network view, in connectome index order.
            # Two decimals keeps 448 values under ~2 kB a frame.
            "activity": [round(float(v), 2) for v in act],
            "neuron_activity": {
                "forward_cmd": round(float(np.mean(act[self.i_fwd])), 4),
                "backward_cmd": round(float(np.mean(act[self.i_bwd])), 4),
                "B_motor": round(float(np.mean(act[self.i_B])), 4),
                "A_motor": round(float(np.mean(act[self.i_A])), 4),
                "turn": round(float(np.mean(act[self.i_turn])), 4),
            },
        }
