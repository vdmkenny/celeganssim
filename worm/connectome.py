"""Connectome loading: Cook et al. 2019 wiring turned into dense matrices.

Two graphs are built over the same cell index:
  Gs  chemical synapses   directed, weight = number of synaptic contacts
  Gg  gap junctions       symmetric, weight = number of contacts

Chemical synapses carry a sign derived per edge from the POSTSYNAPTIC cell's
measured receptor expression (CeNGEN): a synapse's sign is a property of the
receptor, not the transmitter -- glutamate excites through GLR-1-class cation
channels and inhibits through the glutamate-gated chloride channels, so the
same transmitter gets opposite signs on different cells. Edges that
expression cannot resolve (metabotropic transmitters, cells outside CeNGEN,
exact ties) fall back to a transmitter-level heuristic, tagged as such.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .paths import data_dir as _data_dir

# Transmitter-level sign heuristic: the FALLBACK for edges where measured
# receptor expression cannot decide. Monoamines here are mostly metabotropic
# (GPCRs), whose postsynaptic effect depends on downstream signalling rather
# than ion selectivity, so their entries stay permanently heuristic.
NT_POLARITY: dict[str, float] = {
    "Acetylcholine": +1.0,
    "Glutamate": +0.4,
    "GABA": -1.0,
    "Serotonin": -0.2,
    "Dopamine": -0.2,
    "Octopamine": -0.2,
    "Tyramine": -0.4,
}
DEFAULT_POLARITY = +0.5  # unknown transmitter: weak excitation

E_EXC = 0.0     # excitatory synaptic reversal potential, mV
E_INH = -48.0   # inhibitory synaptic reversal potential, mV


class Connectome:
    def __init__(self, cells: dict, edges: list[dict]) -> None:
        self.cell_info = cells
        self.names: list[str] = sorted(cells)
        self.index: dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self.n = len(self.names)
        self.edges = edges

        self.kind = np.array([cells[n]["kind"] for n in self.names])
        self.is_neuron = self.kind == "neuron"
        self.is_muscle = self.kind == "muscle"

        self._build_matrices()

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "Connectome":
        d = Path(data_dir) if data_dir is not None else _data_dir()
        cells = json.loads((d / "cells.json").read_text())["cells"]
        conn = json.loads((d / "connectome.json").read_text())
        return cls(cells, conn["edges"])

    def _build_matrices(self) -> None:
        n = self.n
        self.Gs = np.zeros((n, n), dtype=np.float64)  # [post, pre] chemical
        self.Gg = np.zeros((n, n), dtype=np.float64)  # symmetric electrical
        self.polarity = np.zeros(n, dtype=np.float64)  # per presynaptic cell

        for name in self.names:
            nts = self.cell_info[name].get("neurotransmitters") or []
            if nts:
                self.polarity[self.index[name]] = float(
                    np.mean([NT_POLARITY.get(nt, DEFAULT_POLARITY) for nt in nts])
                )
            else:
                self.polarity[self.index[name]] = DEFAULT_POLARITY

        pol = np.clip(self.polarity, -1.0, 1.0)
        self.E_pre_default = E_INH + (pol + 1.0) * 0.5 * (E_EXC - E_INH)

        # Per-edge reversal potential, defaulting to the presynaptic cell's
        # transmitter but overridable for documented exceptions.
        self.E_syn = np.tile(self.E_pre_default[np.newaxis, :], (n, 1))

        skipped = 0
        for e in self.edges:
            pre, post = e["pre"], e["post"]
            i, j = self.index.get(pre), self.index.get(post)
            if i is None or j is None:
                skipped += 1
                continue
            w = e["w"]
            if e["type"] == "chemical":
                self.Gs[j, i] += w
            else:
                # The edgelist lists each gap junction in both directions
                # (1339 of 1359 unordered pairs), so the matrix is filled
                # directly rather than symmetrised. Self-referential rows are
                # reconstruction artifacts and are dropped.
                if i != j:
                    self.Gg[j, i] += w
        self.skipped_edges = skipped

        # Per-cell transmitter label, used for gene-knockout scaling and for
        # the receptor-sign derivation below.
        self.pre_nt: list[tuple[str, ...]] = [
            tuple(self.cell_info[n].get("neurotransmitters") or ()) for n in self.names
        ]
        self._derive_signs()

    def _derive_signs(self) -> None:
        """Per-edge reversal potentials from postsynaptic receptor expression.

        For each chemical edge, sum the CeNGEN expression levels of the
        excitatory and of the inhibitory receptor genes for the presynaptic
        transmitter in the postsynaptic cell; the larger side sets the sign
        and the margin is reported. Body-wall muscles are not in CeNGEN, but
        their junction is measured directly: acetylcholine excites through
        the nicotinic UNC-29/UNC-38/UNC-63/LEV-1/LEV-8 receptors, GABA
        inhibits through UNC-49. Edges with no receptor evidence keep the
        transmitter-level fallback. Every edge's provenance is counted in
        self.sign_provenance so the mix of derived and heuristic signs is
        inspectable.
        """
        d = _data_dir()
        rec_path, exp_path = d / "receptors.json", d / "expression.json"
        self.sign_provenance: Counter = Counter()
        self.sign_flips: list[tuple[str, str, float, float]] = []
        if not (rec_path.exists() and exp_path.exists()):
            self.sign_provenance["transmitter_fallback"] = \
                int((self.Gs > 0).sum())
            return

        receptors = json.loads(rec_path.read_text())["receptors"]
        expression = json.loads(exp_path.read_text())["genes"]
        by_nt: dict[tuple[str, str], list[str]] = {}
        for gene, rec in receptors.items():
            if rec["sign"] in ("excitatory", "inhibitory"):
                by_nt.setdefault((rec["transmitter"], rec["sign"]), []).append(gene)

        def level(gene: str, cell: str) -> float:
            entry = expression.get(gene)
            return entry["levels"].get(cell, 0.0) if entry else 0.0

        # Gs is [post, pre]: np.nonzero returns (postsynaptic, presynaptic).
        posts, pres = np.nonzero(self.Gs)
        for j, i in zip(posts, pres):
            post = self.names[j]
            nts = self.pre_nt[i]
            if not nts:
                self.sign_provenance["transmitter_fallback"] += 1
                continue
            exc = inh = 0.0
            tag = "receptor_expression"
            if self.is_muscle[j]:
                # Measured NMJ, no scRNA-seq needed: ACh excites, GABA inhibits.
                exc = float("Acetylcholine" in nts)
                inh = float("GABA" in nts)
                tag = "nmj_literature"
            else:
                for nt in nts:
                    for g in by_nt.get((nt, "excitatory"), ()):
                        exc += level(g, post)
                    for g in by_nt.get((nt, "inhibitory"), ()):
                        inh += level(g, post)
            if exc == inh:
                self.sign_provenance["transmitter_fallback"] += 1
                continue
            before = float(self.E_syn[j, i])
            self.E_syn[j, i] = E_EXC if exc > inh else E_INH
            self.sign_provenance[tag] += 1
            if (before == E_INH) != (exc < inh):
                self.sign_flips.append((self.names[i], post, before,
                                        float(self.E_syn[j, i])))

    # -- queries --------------------------------------------------------
    def idx(self, name: str) -> int:
        return self.index[name]

    def group(self, *, kind: str | None = None, role: str | None = None,
              vnc_class: str | None = None) -> list[str]:
        out = []
        for n in self.names:
            c = self.cell_info[n]
            if kind and c["kind"] != kind:
                continue
            if role and role not in (c.get("roles") or []):
                continue
            if vnc_class and c.get("vnc_class") != vnc_class:
                continue
            out.append(n)
        return out

    def indices(self, names) -> np.ndarray:
        return np.array([self.index[n] for n in names if n in self.index], dtype=int)

    def partners(self, name: str, kind: str = "chemical", top: int = 10):
        """Strongest outgoing partners of a cell, for inspection."""
        i = self.index[name]
        M = self.Gs if kind == "chemical" else self.Gg
        col = M[:, i]
        order = np.argsort(col)[::-1]
        return [(self.names[j], float(col[j])) for j in order[:top] if col[j] > 0]

    # -- layout ---------------------------------------------------------
    def layer_of(self, name: str) -> int:
        """Place a cell in the sensory -> interneuron -> motor -> muscle chain.

        Many cells carry more than one role (plenty of head neurons are both
        sensory and interneuron), so roles are checked in signal-flow order and
        the first match wins.
        """
        info = self.cell_info[name]
        if info["kind"] == "muscle":
            return 3
        # Ventral cord motor neurons are motor output, whatever else they do.
        # The B-class in particular is annotated sensory as well, correctly:
        # they are proprioceptive, and that stretch feedback is what propagates
        # the undulatory wave (Wen et al. 2012). That is an internal sense, not
        # an external modality, so it should not pull them into the input layer.
        if info.get("vnc_class"):
            return 2
        roles = info.get("roles") or []
        if "sensory" in roles:
            return 0
        if "interneuron" in roles:
            return 1
        if "motor" in roles:
            return 2
        return 1

    def layout(self) -> dict:
        """A layered drawing of the network, ordered to keep it readable.

        Columns are the signal-flow layers. Within a column, cells are ordered
        by the average position of what they connect to in the next column
        (the barycentre heuristic from layered graph drawing), which pulls
        connected cells level with each other and cuts down edge crossings.
        Muscles are the fixed anchor: dorsal rows above, ventral below, head to
        tail left to right, matching the body.
        """
        cols: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
        for n in self.names:
            cols[self.layer_of(n)].append(n)

        # Anchor: muscles in anatomical order.
        def muscle_key(n):
            c = self.cell_info[n]
            return (0 if c.get("side") == "dorsal" else 1, c.get("row", 0),
                    c.get("lr", ""))
        cols[3].sort(key=muscle_key)

        order: dict[str, float] = {}
        for i, n in enumerate(cols[3]):
            order[n] = i / max(len(cols[3]) - 1, 1)

        # Work right to left, placing each layer by where its targets sit.
        undirected = self.Gs + self.Gs.T + self.Gg
        for layer in (2, 1, 0):
            nxt = set(cols[layer + 1])
            bary = {}
            for n in cols[layer]:
                i = self.index[n]
                num = den = 0.0
                for m in nxt:
                    w = undirected[self.index[m], i]
                    if w > 0:
                        num += w * order[m]
                        den += w
                bary[n] = num / den if den else 0.5
            cols[layer].sort(key=lambda n: (bary[n], n))
            for i, n in enumerate(cols[layer]):
                order[n] = i / max(len(cols[layer]) - 1, 1)

        xs = {0: 0.06, 1: 0.34, 2: 0.63, 3: 0.92}
        nodes = []
        for layer, names in cols.items():
            for i, n in enumerate(names):
                info = self.cell_info[n]
                nodes.append({
                    "n": n,
                    "x": xs[layer],
                    "y": round(0.03 + 0.94 * (i / max(len(names) - 1, 1)), 5),
                    "layer": layer,
                    "kind": info["kind"],
                    "nt": (info.get("neurotransmitters") or [None])[0],
                    "side": info.get("side"),
                })
        return {"nodes": nodes,
                "layers": ["sensory", "interneuron", "motor", "muscle"]}

    def edge_list(self, min_weight: float = 3.0) -> list:
        """Edges worth drawing, as [pre_index, post_index, weight, is_gap].

        Thresholded because the full 7,379-edge graph redrawn every frame is
        both slow and unreadable; the weak tail carries little signal.
        """
        out = []
        n = self.n
        for i in range(n):
            for j in range(n):
                w = self.Gs[j, i]
                if w >= min_weight:
                    out.append([i, j, round(float(w), 1), 0])
        seen = set()
        for i in range(n):
            for j in range(n):
                w = self.Gg[j, i]
                if w >= min_weight and (j, i) not in seen:
                    seen.add((i, j))
                    out.append([i, j, round(float(w), 1), 1])
        return out

    def stats(self) -> dict:
        # Unordered gap-junction pairs: the edgelist lists most pairs in both
        # directions but ~20 only once, so (Gg>0).sum()//2 would miscount.
        either_direction = (self.Gg > 0) | (self.Gg.T > 0)
        gap_pairs = int(np.count_nonzero(np.triu(either_direction)))
        return {
            "cells": self.n,
            "neurons": int(self.is_neuron.sum()),
            "muscles": int(self.is_muscle.sum()),
            "chemical_edges": int((self.Gs > 0).sum()),
            "electrical_edges": gap_pairs,
            "chemical_contacts": float(self.Gs.sum()),
            "electrical_contacts": float(self.Gg.sum() / 2),
            "skipped_edges": self.skipped_edges,
        }
