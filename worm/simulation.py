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
    reversal_threshold: float = 0.018
    baseline_tau_s: float = 8.0
    reversal_min_s: float = 0.9
    reversal_max_s: float = 4.0
    omega_s: float = 0.9
    refractory_s: float = 0.6
    tonic_forward: float = 0.62   # baseline AVB drive -> spontaneous forward
    command_gain: float = 9.0
    seed: int = 0
    start_adult: bool = True      # False starts as a freshly hatched L1
    # How many seconds of development pass per second of simulated behaviour.
    # Real development takes ~50 h, which nobody wants to watch in real time.
    life_speedup: float = 400.0


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
        self.life = Lifecycle()
        if self.cfg.start_adult:
            self.life.stage = "adult"
            self.life.reserves = 0.7
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

        self.state = SimState()
        self.reset()

    def _muscle_rows(self) -> None:
        """Group the 95 body-wall muscles into dorsal/ventral rows."""
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

    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> None:
        self.ns.reset()
        self.body.reset(x, y, heading)
        self.state = SimState()
        self.state.trail = [self.body.X.copy()]
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
        forward = cfg.tonic_forward + cfg.command_gain * (fwd_cmd + drive_B)
        backward = cfg.command_gain * (bwd_cmd + drive_A)

        if self.state.behavior == "reversal":
            forward, backward = 0.05, 0.95
        elif self.state.behavior == "omega":
            forward, backward = 0.85, 0.05

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

        head_bias = self._head_bias(act, turn_cmd)

        self.body.p.curvature_gain = BodyParams.curvature_gain * np.clip(bend, 0.3, 2.0)
        self.body.step_oscillator(
            dt, forward_drive=forward, backward_drive=backward,
            gaba_scale=float(np.clip(gaba, 0.0, 1.0)),
            head_bias=head_bias, arousal=float(np.clip(arousal, 0.3, 2.0)))
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
