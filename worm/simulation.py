"""Closed loop: world -> sensory neurons -> connectome -> muscles -> movement.

The escape response is modelled as an explicit state machine because the
literature is clear that reversal and the omega turn have separable final motor
pathways (SMD/RIV are required for the turn but not the reversal). Which state
the animal enters, and when, is decided by the real network: the command
interneurons AVA/AVD/AVE and AVB/PVC are read out directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class SimConfig:
    dt: float = 0.02              # body/world timestep, seconds
    neural_substeps: int = 20     # neural steps per body step (1 ms each)
    neural_noise: float = 0.02    # mV per neural step
    sensory_amplitude: float = 55.0
    # Reversal fires when AVA/AVD/AVE rise this far above their own recent
    # average, not above the naive resting value. Tonic sensory input (21%
    # oxygen, AWC's baseline activity, ambient temperature) holds the command
    # neurons well off rest, so a fixed threshold either never fires or fires
    # constantly. Adapting the baseline is also what the real animal does.
    # Calibrated against the receptor-derived sign regime. Measured peak
    # command-balance deviation to a strength-1 anterior poke: wild type
    # ~0.007; mec-10 (50% residual touch current, Arnadottir et al. 2011)
    # 0.0028-0.0037; harsh posterior ~0.0022; gentle posterior ~0.001;
    # mec-4/mec-2 nulls and poke-free fluctuation ~0.0005 or less. The
    # threshold sits inside the mec-10 band: wild type always responds,
    # mec-10 responds to a FRACTION of pokes (the partial-loss phenotype),
    # and nulls, gentle-posterior and harsh-posterior touch stay below it.
    reversal_threshold: float = 0.0032
    baseline_tau_s: float = 8.0
    reversal_min_s: float = 0.9
    reversal_max_s: float = 4.0
    omega_s: float = 0.9
    refractory_s: float = 0.6
    tonic_forward: float = 0.62   # baseline AVB drive -> spontaneous forward
    command_gain: float = 9.0
    seed: int = 0
    # Drive the body from the muscle cells the connectome actually drives,
    # instead of from the scripted ventral-cord oscillator. Off by default:
    # the network delivers no undulatory rhythm on its own, so an animal in
    # this mode barely moves until proprioceptive feedback closes the loop.
    # See docs/emergent-cpg.md and the "network drives the muscles" check.
    emergent_muscles: bool = False
    # Strength of the proprioceptive current, pA per unit normalised curvature.
    # A free parameter: no stretch-evoked current has ever been recorded in a
    # B-type motor neuron, so there is no measured amplitude, reversal
    # potential, threshold or adaptation to anchor it. Off by default.
    propr_gain: float = 0.0
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
    "emergent_muscles": "scripted",
    "propr_gain": "tuned",   # no stretch-evoked current ever recorded
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
    distance: float = 0.0
    trail: list = field(default_factory=list)


class WormSimulation:
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

        self.state = SimState()
        self.reset()

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
        rest = MUSCLE_REST_ACTIVATION
        f = np.clip((ca - rest) / (1.0 - rest), 0.0, 1.0)
        d = np.array([f[r].mean() if r.size else 0.0 for r in self.ca_row_d])
        v = np.array([f[r].mean() if r.size else 0.0 for r in self.ca_row_v])
        return d, v

    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> None:
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

        I = self.sensory.compute(self.env, head, tail, dt,
                                 amplitude=cfg.sensory_amplitude)
        I = I + self.proprioceptive_current()
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
            forward, backward = 0.05, 0.95
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
        # Drive alone is not enough: above a total of 1.0 the oscillator's
        # amplitude term saturates, so a 20% cut in drive can vanish entirely.
        # Slowing on food and senescent decline are reductions in locomotion
        # RATE, so they also have to reach the undulation frequency.
        rate_scale = slow

        head_bias = self._head_bias(act, turn_cmd)

        self.body.p.curvature_gain = BodyParams.curvature_gain * np.clip(bend, 0.3, 2.0)
        if self.cfg.emergent_muscles:
            # Shape follows the muscle cells the connectome actually drives.
            d, v = self.muscle_force()
            self.body.drive_from_muscles(dt, d, v)
            self.muscle_activation = np.zeros(self.conn.n)
            self.muscle_activation[self.ns._muscle_idx] = self.ns.muscle_calcium()
        else:
            self.body.step_oscillator(
                dt, forward_drive=forward, backward_drive=backward,
                gaba_scale=float(np.clip(gaba, 0.0, 1.0)),
                head_bias=head_bias,
                arousal=float(np.clip(arousal, 0.3, 2.0) * max(rate_scale, 0.05)))
            self._drive_muscles()

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
        elif st.behavior == "reversal":
            done = st.state_time > cfg.reversal_min_s and bwd_cmd <= cfg.reversal_threshold
            if done or st.state_time > cfg.reversal_max_s:
                # Omega turn only if the SMD/RIV pathway is intact.
                if self.genome.global_scale("omega_turn") > 0.5 and self.rng.random() < 0.75:
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
            strength = 0.85 * self.genome.global_scale("omega_turn")
            return -float(np.clip(strength, 0.0, 1.0))
        d = float(np.mean(act[self.i_hd])) if len(self.i_hd) else 0.5
        v = float(np.mean(act[self.i_hv])) if len(self.i_hv) else 0.5
        bias = np.clip((d - v) * 6.0, -0.45, 0.45)
        # RIM tyramine suppresses head movement during reversals; tdc-1 removes it.
        if self.state.behavior == "reversal":
            bias *= self.genome.global_scale("head_suppression")
        return float(bias)

    def _drive_muscles(self) -> None:
        """Write the commanded activation back onto the real muscle cells.

        The body model runs on 24 segment rows, but the connectome carries all
        95 individually named body-wall muscles, so the visualiser and any
        readout can show genuine per-muscle activity.
        """
        self.muscle_activation = np.zeros(self.conn.n)
        for seg in range(N_SEG):
            for i in self.row_d[seg]:
                self.muscle_activation[i] = self.body.dorsal[seg]
            for i in self.row_v[seg]:
                self.muscle_activation[i] = self.body.ventral[seg]

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
