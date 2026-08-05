"""The genome layer: real WBcel235 annotation, wired to simulator parameters.

The point of loading 100 Mb of real sequence annotation into a behaviour
simulator is that gene names become a control surface. `GENE_EFFECTS` below maps
a curated set of genes onto the subsystems their products actually implement, so
knocking out `unc-25` removes GABA synthesis, which removes cross-inhibition at
the neuromuscular junction, which makes the model worm coil -- the same causal
chain as in the animal.

Every entry cites the phenotype it is modelling. Effects are deliberately coarse:
this is a behavioural model, not a molecular one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"


@dataclass(frozen=True)
class GeneEffect:
    """What the simulator does when this gene is knocked out.

    Fields name a subsystem and a multiplier applied to its strength. A value of
    0.0 means fully abolished; 0.2 means strongly reduced but not eliminated.
    """

    gene: str
    product: str
    phenotype: str  # observable behaviour of the loss-of-function mutant
    # Subsystem multipliers applied on knockout.
    neurotransmitter_scale: dict[str, float] = field(default_factory=dict)
    sensory_scale: dict[str, float] = field(default_factory=dict)
    global_scale: dict[str, float] = field(default_factory=dict)


# Curated loss-of-function effects. Phenotypes from WormBase/WormAtlas/WormBook.
GENE_EFFECTS: dict[str, GeneEffect] = {
    # --- Neurotransmitter synthesis and packaging ---
    "unc-25": GeneEffect(
        "unc-25",
        "glutamic acid decarboxylase (GAD) - synthesises GABA",
        "Shrinker. With no GABA the dorsal and ventral muscles hypercontract "
        "together on stimulation, so the animal shortens and pulls its head in "
        "rather than bending. Note it still undulates - GABA modulates "
        "amplitude and speed rather than being strictly required for "
        "dorsoventral alternation.",
        neurotransmitter_scale={"GABA": 0.0},
    ),
    "unc-47": GeneEffect(
        "unc-47",
        "vesicular GABA transporter (VGAT)",
        "Shrinker, essentially identical to unc-25: GABA is synthesised but "
        "never loaded into vesicles, so it is never released.",
        neurotransmitter_scale={"GABA": 0.0},
    ),
    "unc-49": GeneEffect(
        "unc-49",
        "GABA-A receptor subunit at the neuromuscular junction",
        "Shrinker, and the cleanest allele of the three: GABA is released "
        "normally but body-wall muscle cannot hear it, so foraging and "
        "expulsion stay normal while body locomotion shrinks.",
        neurotransmitter_scale={"GABA": 0.05},
    ),
    "unc-17": GeneEffect(
        "unc-17",
        "vesicular acetylcholine transporter (VAChT)",
        "COILER: small, slow-growing, coils and is jerky in reverse. "
        "Aldicarb-resistant. Null alleles are lethal, so only hypomorphs are "
        "viable - modelled here as a strong reduction, not a full loss.",
        neurotransmitter_scale={"Acetylcholine": 0.15},
    ),
    "cha-1": GeneEffect(
        "cha-1",
        "choline acetyltransferase - synthesises acetylcholine",
        "COILER, like unc-17 - the two share a locus (the unc-17 ORF sits "
        "inside cha-1 intron 1). Coily, uncoordinated, jerky in reverse. "
        "Nulls hatch but never feed or grow.",
        neurotransmitter_scale={"Acetylcholine": 0.15},
    ),
    "eat-4": GeneEffect(
        "eat-4",
        "vesicular glutamate transporter (VGLUT)",
        "Glutamatergic signalling lost: defective chemotaxis, weakened "
        "mechanosensory responses, and poor pharyngeal pumping.",
        neurotransmitter_scale={"Glutamate": 0.05},
    ),
    "tph-1": GeneEffect(
        "tph-1",
        "tryptophan hydroxylase - rate-limiting step for serotonin",
        "No serotonin. Loses the enhanced slowing response on encountering "
        "food; egg laying and pharyngeal pumping are reduced.",
        neurotransmitter_scale={"Serotonin": 0.0},
    ),
    "cat-2": GeneEffect(
        "cat-2",
        "tyrosine hydroxylase - rate-limiting step for dopamine",
        "No dopamine. Loses the basal slowing response on food (mechanosensory "
        "detection of a bacterial lawn).",
        neurotransmitter_scale={"Dopamine": 0.0},
    ),
    "tdc-1": GeneEffect(
        "tdc-1",
        "tyrosine decarboxylase - makes tyramine (and via tbh-1, octopamine)",
        "No tyramine or octopamine. Anterior touch still starts a reversal, but "
        "head oscillation is no longer suppressed during it, reversals are "
        "shorter, reversal rate goes up, and the ventral omega turn is "
        "defective.",
        neurotransmitter_scale={"Tyramine": 0.0, "Octopamine": 0.0},
        global_scale={"head_suppression": 0.0, "omega_turn": 0.35},
    ),
    # --- Core synaptic release machinery ---
    "unc-13": GeneEffect(
        "unc-13",
        "synaptic vesicle priming factor",
        "Near-complete paralysis with a coily posture; only slow head movement "
        "survives. Vesicles dock but are fusion-incompetent, so evoked release "
        "is abolished. Strongly aldicarb-resistant.",
        global_scale={"chemical_synapse": 0.05},
    ),
    "unc-18": GeneEffect(
        "unc-18",
        "Sec1/Munc18 syntaxin chaperone required for vesicle fusion",
        "Severe paralysis, comparable to unc-13.",
        global_scale={"chemical_synapse": 0.08},
    ),
    "unc-31": GeneEffect(
        "unc-31",
        "CAPS - dense-core vesicle exocytosis (neuropeptide release)",
        "Straight, relaxed, paralysed-looking posture. Crawls at a few percent "
        "of wild-type rate ON food but moves nearly normally OFF food, which is "
        "the tell that this is lost neuromodulation rather than lost fast "
        "transmission. Only mildly aldicarb-resistant.",
        global_scale={"neuromodulation": 0.05, "arousal": 0.5},
    ),
    "egl-3": GeneEffect(
        "egl-3",
        "proprotein convertase 2 - processes neuropeptide precursors",
        "Neuropeptide precursors are never processed. Egg laying is strongly "
        "defective, and gentle body touch is severely diminished (Mec-like) "
        "because the touch circuit loses its peptidergic modulation - though "
        "nose touch stays normal.",
        sensory_scale={"touch": 0.3},
        global_scale={"neuromodulation": 0.15},
    ),
    # --- Sensory transduction ---
    "mec-4": GeneEffect(
        "mec-4",
        "DEG/ENaC channel subunit of the gentle-touch mechanotransducer",
        "Mec: touch-insensitive. The animal ignores a gentle eyelash stroke to "
        "the body but still responds normally to harsh prodding, because that "
        "runs through PVD rather than the MEC-4 channel.",
        sensory_scale={"touch": 0.0},
    ),
    "mec-10": GeneEffect(
        "mec-10",
        "DEG/ENaC subunit partnering MEC-4",
        "Only PARTIAL touch loss for a true deletion null. The classic alleles "
        "that look fully touch-insensitive are recessive gain-of-function, not "
        "nulls - the null is modelled here. Unlike mec-4, MEC-10 is also "
        "required (with DEGT-1) for PVD harsh touch, so hard prodding is "
        "degraded too.",
        sensory_scale={"touch": 0.35, "harsh_touch": 0.3},
    ),
    "mec-2": GeneEffect(
        "mec-2",
        "stomatin-like protein required for the touch channel complex",
        "Mec: touch-insensitive.",
        sensory_scale={"touch": 0.05},
    ),
    "osm-9": GeneEffect(
        "osm-9",
        "TRPV channel in ASH and other polymodal sensory neurons",
        "Osm: fails to avoid high osmolarity, nose touch and repellent odours.",
        sensory_scale={"nociception": 0.0},
    ),
    "tax-4": GeneEffect(
        "tax-4",
        "cyclic-nucleotide-gated channel alpha subunit (AFD, AWC, ASE)",
        "Tax: no thermotaxis and badly defective chemotaxis to water-soluble "
        "and volatile attractants.",
        sensory_scale={"chemotaxis": 0.05, "thermotaxis": 0.0},
    ),
    "tax-2": GeneEffect(
        "tax-2",
        "cyclic-nucleotide-gated channel beta subunit, partners TAX-4",
        "Tax: thermotaxis and chemotaxis defective, like tax-4.",
        sensory_scale={"chemotaxis": 0.1, "thermotaxis": 0.05},
    ),
    "che-1": GeneEffect(
        "che-1",
        "zinc-finger transcription factor specifying ASE gustatory fate",
        "ASE never differentiates, so salt chemotaxis is abolished while "
        "odour responses stay intact.",
        sensory_scale={"salt": 0.0},
    ),
    "odr-1": GeneEffect(
        "odr-1",
        "guanylyl cyclase required for odorant responses in AWC/AWB",
        "Odr: fails to chemotax to AWC- and AWB-sensed volatile odorants.",
        sensory_scale={"odor": 0.05},
    ),
    "gcy-8": GeneEffect(
        "gcy-8",
        "AFD-specific receptor guanylyl cyclase for thermosensation",
        "Thermotaxis is degraded; AFD's temperature response is blunted.",
        sensory_scale={"thermotaxis": 0.3},
    ),
    # --- Neuromodulatory G-protein signalling ---
    "goa-1": GeneEffect(
        "goa-1",
        "heterotrimeric G protein Go alpha - inhibits neurotransmitter release",
        "Loopy: exaggerated deep body bends and hyperactive locomotion and egg "
        "laying. GOA-1 inhibits EGL-30, so removing it raises synaptic output "
        "(the same phenotype as egl-30 gain of function).",
        global_scale={"chemical_synapse": 1.6, "arousal": 1.5,
                      "bend_amplitude": 1.7},
    ),
    "egl-30": GeneEffect(
        "egl-30",
        "heterotrimeric G protein Gq alpha - promotes neurotransmitter release",
        "Loss of function is lethargic and near-paralysed with reduced "
        "locomotion and egg laying. (Gain of function goes the other way and "
        "is loopy, like goa-1 loss of function; only lf is modelled here.)",
        global_scale={"chemical_synapse": 0.25, "arousal": 0.4,
                      "bend_amplitude": 0.55},
    ),
    "npr-1": GeneEffect(
        "npr-1",
        "neuropeptide receptor (FMRFamide-like), NPY-receptor family",
        "Social/bordering feeding behaviour and elevated roaming - the classic "
        "solitary-vs-social polymorphism.",
        global_scale={"arousal": 1.3},
    ),
}

# Common informal names people reach for, mapped to real loci.
ALIASES = {
    "gad": "unc-25",
    "vgat": "unc-47",
    "vacht": "unc-17",
    "vglut": "eat-4",
    "gaba": "unc-25",
    "touch": "mec-4",
    "trpv": "osm-9",
}


class Genome:
    """The real annotated genome plus a set of knocked-out loci."""

    def __init__(self, genes: dict, stats: dict) -> None:
        self.genes = genes
        self.stats = stats
        self.knockouts: set[str] = set()
        self._lower = {k.lower(): k for k in genes}

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "Genome":
        d = Path(data_dir) if data_dir else DATA
        genes = json.loads((d / "genes.json").read_text())
        stats = json.loads((d / "genome_stats.json").read_text())
        return cls(genes, stats)

    # -- lookup ---------------------------------------------------------
    def resolve(self, name: str) -> str | None:
        """Map a user-supplied name onto an annotated locus, or None."""
        key = name.strip()
        if key in self.genes:
            return key
        low = key.lower()
        if low in ALIASES and ALIASES[low] in self.genes:
            return ALIASES[low]
        return self._lower.get(low)

    def gene(self, name: str) -> dict | None:
        resolved = self.resolve(name)
        return self.genes.get(resolved) if resolved else None

    def search(self, prefix: str, limit: int = 25) -> list[dict]:
        p = prefix.lower()
        hits = [g for k, g in self.genes.items() if k.lower().startswith(p)]
        hits.sort(key=lambda g: g["symbol"])
        return hits[:limit]

    # -- perturbation ---------------------------------------------------
    def knock_out(self, name: str) -> dict:
        """Knock out a gene. Returns a record describing what changed."""
        resolved = self.resolve(name)
        if resolved is None:
            raise KeyError(f"{name!r} is not in the WBcel235 annotation")
        self.knockouts.add(resolved)
        self._effects.cache_clear()
        eff = GENE_EFFECTS.get(resolved)
        return {
            "gene": resolved,
            "locus": self.genes[resolved],
            "modelled": eff is not None,
            "product": eff.product if eff else None,
            "phenotype": eff.phenotype if eff else
            "Annotated locus with no behavioural model in this simulator; "
            "knocking it out changes nothing observable here.",
        }

    def restore(self, name: str) -> None:
        resolved = self.resolve(name)
        if resolved:
            self.knockouts.discard(resolved)
            self._effects.cache_clear()

    def reset(self) -> None:
        self.knockouts.clear()
        self._effects.cache_clear()

    @lru_cache(maxsize=1)
    def _effects(self) -> tuple[dict, dict, dict]:
        nt = {}
        sens = {}
        glob = {}
        for g in sorted(self.knockouts):
            eff = GENE_EFFECTS.get(g)
            if not eff:
                continue
            for k, v in eff.neurotransmitter_scale.items():
                nt[k] = min(nt.get(k, 1.0), v) if v <= 1 else nt.get(k, 1.0) * v
            for k, v in eff.sensory_scale.items():
                sens[k] = min(sens.get(k, 1.0), v) if v <= 1 else sens.get(k, 1.0) * v
            for k, v in eff.global_scale.items():
                glob[k] = glob.get(k, 1.0) * v
        return nt, sens, glob

    def nt_scale(self, neurotransmitter: str) -> float:
        return self._effects()[0].get(neurotransmitter, 1.0)

    def sensory_scale(self, modality: str) -> float:
        return self._effects()[1].get(modality, 1.0)

    def global_scale(self, key: str) -> float:
        return self._effects()[2].get(key, 1.0)

    # -- reporting ------------------------------------------------------
    def summary(self) -> dict:
        s = dict(self.stats)
        s["knockouts"] = sorted(self.knockouts)
        return s

    def describe(self) -> str:
        st = self.stats
        lines = [
            f"Genome: {st['assembly']}",
            f"  {st['total_bp']:,} bp across {st['n_sequences']} sequences",
        ]
        for name, c in st["chromosomes"].items():
            lines.append(
                f"    {name:>6}  {c['length_bp']:>11,} bp   GC {c['gc_fraction']*100:.2f}%"
            )
        g = st.get("genes", {})
        if g:
            lines.append(f"  {g['n_genes']:,} annotated genes")
            for bt, n in list(g["by_biotype"].items())[:6]:
                lines.append(f"    {n:>6,}  {bt}")
        if self.knockouts:
            lines.append(f"  knockouts: {', '.join(sorted(self.knockouts))}")
        return "\n".join(lines)
