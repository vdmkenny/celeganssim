"""Cache CeNGEN single-cell expression for the genes the simulator models.

CeNGEN is measured single-cell RNA-seq across 128 neuron classes (Taylor et al.
2021 Cell 184:4329), reached through the wormneuroatlas API (Randi et al. 2023
Nature 623:406). It determines which cells a gene knockout reaches, so an effect
lands only on the cells that transcribe the gene rather than scaling a
transmitter everywhere.

Consequences follow from the data rather than from rules: mec-4 is not expressed
in PVD or FLP, so harsh touch survives a mec-4 knockout.

Run once; the result is cached to data/processed/expression.json so the
simulator never needs wormneuroatlas at runtime.

    python scripts/build_expression.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))

# CeNGEN reports by neuron CLASS, sometimes merging classes it cannot separate
# (VD_DD) or splitting by subtype (RME_DV vs RME_LR). Expand those onto the
# individual cells the connectome names.
CLASS_EXPANSIONS: dict[str, list[str]] = {
    "VD_DD": ["VD", "DD"],
    "RME_DV": ["RMED", "RMEV"],
    "RME_LR": ["RMEL", "RMER"],
    "IL1_DV": ["IL1D", "IL1V"],
    "IL2_DV": ["IL2D", "IL2V"],
    "IL2_LR": ["IL2L", "IL2R"],
    "CEP_DV": ["CEPD", "CEPV"],
    "OLQ_DV": ["OLQD", "OLQV"],
    "SMD_DV": ["SMDD", "SMDV"],
    "RMD_DV": ["RMDD", "RMDV"],
    "SIA_DV": ["SIAD", "SIAV"],
    "SIB_DV": ["SIBD", "SIBV"],
    "URY_DV": ["URYD", "URYV"],
    "AWC_ON": ["AWC"],
    "AWC_OFF": ["AWC"],
    "DA9": ["DA09"],
    "VB01": ["VB01"],
    "VB02": ["VB02"],
    "DB01": ["DB01"],
    "VA12": ["VA12"],
}

# Positional suffixes on a class stem: dorsal/ventral, left/right, or both.
SUFFIX_RE = re.compile(r"^[DV]?[LR]?$|^[DV][LR]$")


def expand(cengen_class: str, cell_names: list[str]) -> list[str]:
    """Map one CeNGEN class label onto the connectome cells it covers."""
    stems = CLASS_EXPANSIONS.get(cengen_class, [cengen_class])
    hits: list[str] = []
    for stem in stems:
        if stem in cell_names:                      # exact cell, e.g. AVM
            hits.append(stem)
            continue
        for n in cell_names:
            if not n.startswith(stem):
                continue
            suffix = n[len(stem):]
            # A class stem plus a positional suffix names the same cells:
            #   AVA  -> AVAL, AVAR          (left/right)
            #   DD   -> DD01..DD06          (cord index)
            #   CEP  -> CEPDL, CEPDR, ...   (dorsal/ventral AND left/right)
            #   RME  -> RMED, RMEV, RMEL, RMER
            if (suffix == ""
                    or suffix.isdigit()
                    or SUFFIX_RE.fullmatch(suffix)):
                hits.append(n)
    return sorted(set(hits))


# Voltage-gated and leak channels, cached per neuron class so cell
# biophysics can be derived from measured expression rather than every
# neuron sharing one leak conductance and one capacitance. This is issue
# #28, and it is issue #17 approached from the data side: a network of
# identical passive cells has no dynamic range to give, and the missing
# range has to come from cells actually differing.
#
# The set is the well-characterised families, named on genetic grounds the
# way the receptor table is (Salkoff et al. 2005 WormBook, potassium
# channels; Bargmann 1998 Science 282:2028 for the family census; Liu,
# Chen & Wang 2018 Cell 175:57 for whole-cell recordings showing neurons
# differ in exactly these currents).
ION_CHANNELS = {
    # Voltage-gated potassium, the main repolarising currents
    "kv": ["shk-1", "shl-1", "shw-1", "shw-3", "egl-2", "egl-36", "exp-2",
           "unc-103", "kqt-1", "kqt-3"],
    # Calcium-activated potassium: BK (slo-1, slo-2) and SK (kcnl-*)
    "kca": ["slo-1", "slo-2", "kcnl-1", "kcnl-2", "kcnl-3", "kcnl-4"],
    # Two-pore leak potassium, the TWK family, which sets resting potential
    # CeNGEN lists these under their canonical names, so the aliases are
    # used: nca-1 appears as unc-77, and twk-1/unc-110 are absent from the
    # table entirely.
    "k_leak": ["twk-3", "twk-14", "twk-20", "twk-40", "twk-44", "unc-58",
               "sup-9"],
    # Inward rectifier
    "kir": ["irk-1", "irk-2", "irk-3"],
    # Voltage-gated calcium: L-type, P/Q-type, T-type, and the subunits
    "cav": ["egl-19", "unc-2", "cca-1", "unc-36", "ccb-1"],
    # NALCN-family sodium leak, which depolarises toward threshold
    "na_leak": ["unc-77", "nca-2", "unc-80", "nlf-1"],
}


def main() -> int:
    try:
        import wormneuroatlas as wa
    except ImportError:
        print("wormneuroatlas is not installed.\n"
              "  pip install wormneuroatlas\n"
              "This step is optional: without expression.json the simulator "
              "falls back to applying gene effects globally.")
        return 1

    import numpy as np

    from worm.genome import GENE_EFFECTS

    cells = json.loads((OUT / "cells.json").read_text())["cells"]
    cell_names = sorted(cells)

    # Genes to cache: the modelled knockout loci plus every classified
    # ligand-gated receptor, so postsynaptic signs can be derived from
    # measured expression rather than assumed per transmitter.
    genes_wanted = set(GENE_EFFECTS)
    channels = {g for fam in ION_CHANNELS.values() for g in fam}
    genes_wanted |= channels
    receptors_path = OUT / "receptors.json"
    if receptors_path.exists():
        receptors = json.loads(receptors_path.read_text())["receptors"]
        genes_wanted |= set(receptors)
        print(f"caching {len(genes_wanted)} genes "
                  f"({len(GENE_EFFECTS)} effect loci + {len(receptors)} "
                  f"receptors + {len(channels)} ion channels)")
    else:
        print(f"caching {len(genes_wanted)} effect loci "
              "(no receptors.json; run scripts/build_receptors.py)")

    cg = wa.Cengen()
    neuron_ids = [str(x) for x in cg.get_neuron_ids()]
    gene_names = [str(g) for g in cg.get_gene_names()]

    out: dict[str, dict] = {}
    channel_families = ION_CHANNELS
    missing: list[str] = []
    for gene in sorted(genes_wanted):
        if gene not in gene_names:
            missing.append(gene)
            continue
        # th=4 is CeNGEN's own moderately stringent expression threshold.
        ex = np.asarray(cg.get_expression(gene_names=[gene], th=4)).ravel()
        # Keep the LEVELS, not just above-threshold presence: sign derivation
        # weighs excitatory against inhibitory receptor expression, and a
        # binary cache cannot say by how much.
        levels = {neuron_ids[i]: round(float(ex[i]), 3)
                  for i in np.where(ex > 0)[0]}
        cells_out: dict[str, float] = {}
        for c, lvl in levels.items():
            for cell in expand(c, cell_names):
                # Expanded classes (VD_DD etc.) share one measurement; each
                # covered cell inherits the class level, documented as such.
                cells_out[cell] = lvl
        out[gene] = {"cengen_classes": sorted(levels),
                     "cells": sorted(cells_out),
                     "levels": cells_out,
                     "n_cells": len(cells_out)}
        print(f"  {gene:8} {len(levels):3} classes -> {len(cells_out):3} cells")

    payload = {
        "source": "CeNGEN scRNA-seq (Taylor et al. 2021 Cell 184:4329) via "
                  "wormneuroatlas (Randi et al. 2023 Nature 623:406)",
        "threshold": 4,
        "genes": out,
        "channel_families": channel_families,
        "not_in_cengen": missing,
    }
    (OUT / "expression.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {len(out)} genes to {OUT/'expression.json'}")
    if missing:
        print(f"not in the CeNGEN table (non-neuronal or absent): "
              f"{', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
