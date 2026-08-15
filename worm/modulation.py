"""Monoaminergic modulation, as a slow extrasynaptic layer.

Monoamines do not travel down the wired connectome. A cell that makes
dopamine releases it into the surrounding tissue, and every cell carrying a
cognate receptor responds, whether or not the two are synaptic partners
(Bentley et al. 2016 PLoS Comput Biol 12:e1005283, whose ligand-receptor
edges this layer is built from; only 6% of those pairs are also chemical
synapses). That is why this is a separate layer rather than extra weight on
existing edges.

Three properties follow from the biology and shape the implementation.

VOLUME TRANSMISSION. The transmitter forms a pool, so every target sees the
same concentration and what distinguishes them is their receptor complement.
Release is therefore a scalar per ligand, and each cell's response is that
scalar times the net sign of the receptors it expresses. This is also why
the layer costs one dot product per step rather than a matrix multiply.

ZERO AT BASELINE. Release is measured as the DEVIATION of the releasing
cells from their own slow baseline, so an animal sitting still gets no net
modulation and the resting operating point is untouched. This is not a
convenience: everything downstream in this model, from the reversal
threshold to the gait constants, is calibrated against that operating point,
and layers that move it break the behavioural suite wholesale even when the
biology they add is correct (see issues #12 and #17 for the two measured
cases). Modulation is a response to change, which is what a slow modulatory
system is for.

SLOW. Monoamine signalling runs on seconds to minutes, not milliseconds, so
the baseline follows with a time constant far longer than a gait cycle and
the layer cannot participate in the undulation rhythm.

WHAT THIS LAYER CAN AND CANNOT CARRY, measured. Release is driven by the
activation of the source cells, so it inherits their dynamic range, and in
this model that range depends entirely on how the source is driven. A
SENSORY source works: food contact injects current straight into the
dopaminergic cells, their activation moves, and basal slowing follows with
its knockouts dissociating correctly. A NETWORK-RECRUITED source does not:
RIM activation changes by 0.08% between forward locomotion and reversal,
and even AVA, the cell the reversal drive is injected into, moves 1.57%,
because the graded network sits compressed around 0.5 (the ceiling issue
#17 measures from the biophysics side). Tyramine release therefore never
rises enough for LGC-55 to shunt the head motor neurons, which is why head
suppression during reversal is still the scripted scalar it was: the
pathway is present and correctly signed, 54 edges from RIM onto exactly
the RMD and SMD cells whose dorsal-ventral difference is the head bias
(Pirri et al. 2009 Neuron 62:526), and it carries no signal. Widening the
network's dynamic range is the prerequisite, not a larger gain here.

THE PEPTIDERGIC HALF IS OFF BY DEFAULT, and the reason is measured rather
than cautious. Bentley's 8,931 peptide edges load and gate correctly, but
switching them into the dynamics costs two checks and, worse, confounds the
one result this layer has earned. Summed edge weights reach 198 on a single
cell at that density, the modulation feeds back into the cells whose
activation drives release, and the loop runs away: deviations grow and the
animal slows to a halt over three minutes. Normalising weights to net
polarity in [-1, 1] fixes the instability but rescales everything, and
re-deriving the gain against that scale lands on a cliff: at 3.5 food
slowing is 5.9% and too weak, at 4.5 it is 24.5% but the animal also
reverses 6 times a minute on food where it should reverse less, so most of
that "slowing" is just interrupted forward motion rather than DOP-3
inhibition. Enabling peptides therefore needs the reversal threshold
re-derived alongside the gain, which is issue #17's compressed dynamic
range again. The data, the receptor signs and the unc-31/egl-3 gating are
kept and checked so that work starts from them.

Knockouts act where the biology puts them. A ligand gene (cat-2, tph-1,
tdc-1, tbh-1) removes the transmitter while leaving every receptor in place,
which is what the mutant is; a receptor gene (dop-3, mod-1, lgc-55 and the
rest) removes only the edges through that receptor, which is how the
pharmacology is dissected experimentally.
"""

from __future__ import annotations

import json

import numpy as np

from .paths import data_dir

# TWO time constants, because one is not enough and the measurement says so.
# The releasing cells' activation swings by about 0.5 at gait frequency as
# the animal undulates, so a single baseline simply tracks the mean of that
# swing and the deviation oscillates at full amplitude: the layer would
# then modulate on the undulation rhythm, which is precisely what a slow
# system must not do. The release pool is therefore low-passed first, on
# the timescale over which a monoamine actually accumulates and clears in
# the tissue, and only that smoothed pool is compared against the slower
# adaptation baseline. Same lesson as the gradient signal in
# _update_gradient_signals: separate the self-motion oscillation from the
# state change before using either. Both GUESSES, ordered so that release
# is slow against the gait cycle and the baseline is slow against release.
MOD_RELEASE_TAU_S = 5.0
MOD_TAU_S = 60.0

# Picoamps delivered per unit of release deviation per unit of net receptor
# weight. TUNED against the one behaviour this layer is asked to produce:
# at 0.6 a well-fed animal slows 18.3% on contacting food, inside Sawin,
# Ranganathan & Horvitz 2000's measured basal slowing, and cat-2 slows 0.2%,
# which is their dissociation. The layer is meant to bias a decision the
# fast network is already making, not to make it: at gain 5 the same
# machinery cost the animal three quarters of its speed.
#
# RE-DERIVED for the per-cell chloride reversal (issue #35). Correctly signed
# inhibition changed the activation the dopaminergic cells reach on food, and
# at the old 0.6 slowing fell to 4.4%, outside Sawin's band. Swept against the
# check's own assay, reporting the knockouts at every point because the
# dissociation is the result and the slowing on its own is not:
#
#   gain   wild type   cat-2   tph-1
#   0.5         4.9%   -1.0%    3.9%   too weak
#   0.7        27.5%   -1.9%   27.5%   inside Sawin's band
#   0.9        35.2%   -0.8%   35.8%   past it
#   1.3        47.3%   -1.7%   43.5%
#
# The curve is steep, which is the cliff this layer's docstring warns about,
# but cat-2 stays at about zero throughout, so the pathway is still carrying
# the dissociation rather than the gain manufacturing a slowdown.
#
# This value depends on reversal_threshold and had to be swept twice. At the
# intermediate threshold of 0.0026 the same 0.7 gave 9%, and 0.8 gave 32%.
# The two constants are not independent: a lower threshold admits more
# reversals, and an interrupted forward run reads as slowing. Anyone moving
# either one has to re-sweep the other, which is why both derivations are
# written down rather than just their answers.
MOD_GAIN_PA = 0.7


class MonoamineLayer:
    """Per-ligand release pools and per-cell receptor weights."""

    def __init__(self, conn, genome, peptides: bool = False):
        self.conn = conn
        self.genome = genome
        raw = json.loads((data_dir() / "monoamines.json").read_text())
        self.edges = list(raw["edges"])
        self.ligand_gene = dict(raw["ligand_gene"])
        self.global_genes: list[str] = []
        self._peptide_ligands: set[str] = set()
        if peptides:
            pep = json.loads((data_dir() / "peptides.json").read_text())
            self.edges += pep["edges"]
            self.global_genes = list(pep["global_genes"])
            self._peptide_ligands = {e["ligand"] for e in pep["edges"]}
        self.ligands = sorted({e["ligand"] for e in self.edges})
        idx = conn.index
        # Source cells per ligand, as indices into the simulated network.
        self._src = {
            lig: np.array(sorted({idx[e["pre"]] for e in self.edges
                                  if e["ligand"] == lig and e["pre"] in idx}),
                          dtype=int)
            for lig in self.ligands
        }
        self.refresh()
        self.reset()

    # -- genetics -------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild receptor weights, honouring receptor-gene knockouts.

        Called whenever the genome changes. A receptor knockout drops the
        edges through that receptor and leaves the rest of the layer, so
        dop-3 can be removed without touching dop-1 on the same cell.
        """
        n = self.conn.n
        idx = self.conn.index
        dead = set()
        for rec in {e["receptor"] for e in self.edges}:
            resolved = self.genome.resolve(rec)
            if resolved is not None and resolved in self.genome.knockouts:
                dead.add(rec)
        self._dead_receptors = sorted(dead)
        self.W = {lig: np.zeros(n) for lig in self.ligands}
        for e in self.edges:
            if e["receptor"] in dead or e["post"] not in idx:
                continue
            self.W[e["ligand"]][idx[e["post"]]] += e["sign"]

    def _global_scale(self, ligand: str) -> float:
        """Dense-core vesicle machinery, peptides only.

        unc-31 removes regulated peptide release outright and egl-3 removes
        the peptides needing proprotein processing, which is most of them,
        so either silences the peptidergic half and spares the monoamines.
        """
        if ligand not in self._peptide_ligands:
            return 1.0
        for gene in self.global_genes:
            resolved = self.genome.resolve(gene)
            if resolved is not None and resolved in self.genome.knockouts:
                return 0.0
        return 1.0

    def _ligand_scale(self, ligand: str) -> float:
        """1.0 if the animal can still make this monoamine, else 0."""
        gene = self.ligand_gene.get(ligand)
        if not gene:
            return 1.0
        resolved = self.genome.resolve(gene)
        if resolved is not None and resolved in self.genome.knockouts:
            return 0.0
        return float(np.clip(self.genome.nt_scale(ligand.capitalize()),
                             0.0, 1.0))

    # -- dynamics -------------------------------------------------------
    def reset(self) -> None:
        self._pool: dict[str, float | None] = {l: None for l in self.ligands}
        self._baseline: dict[str, float | None] = {l: None for l in self.ligands}
        self.last_release: dict[str, float] = {l: 0.0 for l in self.ligands}

    def current(self, act: np.ndarray, dt: float) -> np.ndarray:
        """Modulatory current, in pA, for this step."""
        I = np.zeros(self.conn.n)
        for lig in self.ligands:
            src = self._src[lig]
            if not len(src):
                continue
            raw = (float(np.mean(act[src])) * self._ligand_scale(lig)
                   * self._global_scale(lig))
            pool = self._pool[lig]
            if pool is None:
                pool = self._pool[lig] = raw
            self._pool[lig] = pool + (dt / MOD_RELEASE_TAU_S) * (raw - pool)
            r = self._pool[lig]
            base = self._baseline[lig]
            if base is None:
                base = self._baseline[lig] = r
            self._baseline[lig] = base + (dt / MOD_TAU_S) * (r - base)
            dev = r - base
            self.last_release[lig] = dev
            if dev:
                I += MOD_GAIN_PA * dev * self.W[lig]
        return I
