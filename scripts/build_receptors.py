"""Derive ligand-gated channel pharmacology from the genome annotation.

A synapse's sign is a property of the POSTSYNAPTIC receptor, not the
presynaptic transmitter: glutamate excites through GLR-1-class cation channels
and inhibits through glutamate-gated chloride channels; acetylcholine excites
through nicotinic receptors and inhibits through ACC-1..4; GABA inhibits
through UNC-49 but excites through EXP-1. This script builds the receptor
table the connectome uses to give each chemical edge its sign.

The table is DERIVED, not hand-listed. Three tiers, in order:

  1. annotation_text -- the RefSeq product description says what the channel
     is ("Glutamate-gated chloride channel subunit beta"). This covers most of
     the family and is regenerated whenever the annotation changes.
  2. family_fill -- a handful of well-studied genes carry thin or wrong
     product descriptions ("Ig-like domain-containing protein" for glc-3 and
     avr-14/15; "LITAF domain-containing protein" for nmr-1). The gene NAME is
     itself annotation here: C. elegans names are assigned on genetic grounds
     (glc = glutamate-gated chloride). Each fill cites the paper that
     established the pharmacology.
  3. exception -- the annotation actively misleads. exp-1 is described as a
     generic GABA receptor but is a GABA-gated CATION channel (Beg & Jorgensen
     2003); acr-23 is annotated as an acetylcholine receptor but is the
     betaine-gated channel (Peden et al. 2013).

Metabotropic receptors (GPCRs: dop-*, ser-*, octr-*, gar-*, most monoamine
receptors) are recorded as metabotropic and carry no sign: their effect
depends on downstream signalling, not ion selectivity. The connectome falls
back to its transmitter-level heuristic for those.

Output: data/processed/receptors.json
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

EXC, INH, META = "excitatory", "inhibitory", "metabotropic"

# Tier 1: rules over the RefSeq product description, most specific first.
# Each rule: (substring to match, transmitter, sign). Text is lowercased.
TEXT_RULES: list[tuple[str, str, str]] = [
    ("glutamate-gated chloride", "Glutamate", INH),
    ("glutamate-gated ion channel", "Glutamate", INH),   # GluCl family wording
    ("acetylcholine-gated chloride", "Acetylcholine", INH),
    ("acetylcholine-gated ion channel", "Acetylcholine", INH),  # acc-4 wording
    ("serotonin-gated chloride", "Serotonin", INH),
    ("tyramine-gated chloride", "Tyramine", INH),
    ("glutamate receptor", "Glutamate", EXC),            # AMPA/kainate/NMDA = cation
    ("nicotinic acetylcholine receptor", "Acetylcholine", EXC),
    ("acetylcholine receptor", "Acetylcholine", EXC),    # nAChRs are cation
    ("gamma-aminobutyric acid receptor", "GABA", INH),   # GABA-A chloride
]

# Monoamine GPCRs are named for their ligand; recorded as metabotropic so the
# table documents them, but they carry no sign.
GPCR_RULES: list[tuple[str, str]] = [
    ("dopamine receptor", "Dopamine"),
    ("serotonin receptor", "Serotonin"),
    ("tyramine receptor", "Tyramine"),
    ("octopamine", "Octopamine"),
    ("muscarinic", "Acetylcholine"),
]

# Tier 2: gene-name gap-fill for thin annotations, each with its reference.
# Applied only when tier 1 found nothing.
FAMILY_FILL: dict[str, tuple[str, str, str]] = {
    "glc": ("Glutamate", INH,
            "Cully et al. 1994; Chalasani et al. 2007 (glc-3 in AWC->AIY; "
             "that paper carries a 2016 corrigendum over invalid imaging "
             "movies, but AIY properties were fully supported by the "
             "reanalysis and this is a receptor-identity claim: "
             "see docs/citations.md)"),
    "acc": ("Acetylcholine", INH,
            "Putrenko, Zakikhani & Dent 2005 JBC 280:6392"),
    "nmr": ("Glutamate", EXC,
            "Brockie et al. 2001 (NMDA subunits)"),
}
GENE_FILL: dict[str, tuple[str, str, str]] = {
    "avr-14": ("Glutamate", INH, "Dent et al. 1997 Science 277:126 (GluCl alpha)"),
    "avr-15": ("Glutamate", INH, "Dent et al. 2000 JBC 275:34471 (GluCl alpha2)"),
    "lgc-55": ("Tyramine", INH, "Pirri et al. 2009 Neuron 62:526 (tyramine-gated Cl-)"),
}

# Tier 3: the annotation is wrong or misleading for these; cite the fix.
EXCEPTIONS: dict[str, tuple[str, str, str]] = {
    "exp-1": ("GABA", EXC,
              "Beg & Jorgensen 2003 Nat Neurosci 6:1145 (GABA-gated cation)"),
    "acr-23": ("Betaine", EXC,
               "Peden et al. 2013 Science 340:96 (betaine-gated channel)"),
}


def load_products() -> dict[str, set[str]]:
    """symbol -> set of RefSeq product descriptions across its transcripts."""
    products: dict[str, set[str]] = {}
    with gzip.open(RAW / "celegans_features.txt.gz", "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(r for r in reader if r and r[0] == "# feature")
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            if not row or row[0] != "mRNA":
                continue
            symbol, name = row[col["symbol"]], row[col["name"]]
            if symbol and name:
                products.setdefault(symbol, set()).add(name)
    return products


def classify(symbol: str, descriptions: set[str]) -> dict | None:
    """One gene's pharmacology, or None if it is not a ligand-gated channel."""
    if symbol in EXCEPTIONS:
        nt, sign, ref = EXCEPTIONS[symbol]
        return {"gene": symbol, "transmitter": nt, "sign": sign,
                "rule": "exception", "evidence": ref}
    texts = [d.lower() for d in descriptions]
    joined = " | ".join(texts)
    # Muscarinic/GPCR annotations disqualify ionotropic reading even if the
    # word "receptor" appears.
    metabotropic = ("g-protein coupled" in joined or "g protein-coupled" in joined
                    or "muscarinic" in joined)
    for needle, nt, sign in TEXT_RULES:
        if any(needle in t for t in texts):
            if metabotropic and sign != META:
                break
            return {"gene": symbol, "transmitter": nt, "sign": sign,
                    "rule": "annotation_text",
                    "evidence": "; ".join(sorted(descriptions))}
    if metabotropic:
        # Which amine is only recorded for reporting; GPCR sign is not
        # resolvable from ion selectivity.
        for needle, candidate in GPCR_RULES:
            if needle in joined:
                return {"gene": symbol, "transmitter": candidate, "sign": META,
                        "rule": "annotation_text",
                        "evidence": "; ".join(sorted(descriptions))}
        # A "GPCR family profile domain" with no named ligand is one of the
        # ~450 chemosensory receptors, not a neurotransmitter receptor: drop.
        return None
    for needle, candidate in GPCR_RULES:
        if any(needle in t for t in texts):
            return {"gene": symbol, "transmitter": candidate, "sign": META,
                    "rule": "annotation_text",
                    "evidence": "; ".join(sorted(descriptions))}
    family = symbol.split("-")[0]
    if family in FAMILY_FILL and symbol[len(family):].lstrip("-").isdigit():
        nt, sign, ref = FAMILY_FILL[family]
        return {"gene": symbol, "transmitter": nt, "sign": sign,
                "rule": "family_fill", "evidence": ref}
    if symbol in GENE_FILL:
        nt, sign, ref = GENE_FILL[symbol]
        return {"gene": symbol, "transmitter": nt, "sign": sign,
                "rule": "family_fill", "evidence": ref}
    return None


def main() -> None:
    products = load_products()
    out: dict[str, dict] = {}
    for symbol in sorted(products):
        rec = classify(symbol, products[symbol])
        if rec:
            out[symbol] = rec

    counts = Counter((r["transmitter"], r["sign"]) for r in out.values())
    rules = Counter(r["rule"] for r in out.values())
    payload = {
        "source": "derived from RefSeq GCF_000002985.6 product descriptions; "
                  "tiered rules in scripts/build_receptors.py",
        "receptors": out,
    }
    (OUT / "receptors.json").write_text(json.dumps(payload, indent=1))
    print(f"{len(out)} ligand-gated / metabotropic receptor genes classified")
    for (nt, sign), n in sorted(counts.items()):
        print(f"  {nt:<15} {sign:<12} {n}")
    print("by evidence tier: " + ", ".join(f"{k} {v}" for k, v in rules.most_common()))
    print(f"wrote -> {OUT / 'receptors.json'}")


if __name__ == "__main__":
    main()
