"""Parse raw downloads into compact JSON the simulator loads at runtime.

Inputs (data/raw/):
  celegans_genome.fna.gz      NCBI WBcel235 assembly (RefSeq GCF_000002985.6)
  celegans_annotation.gff.gz  matching RefSeq annotation
  cook_2020_adjacency.xlsx    Cook et al. connectome adjacency matrices,
                              corrected July 2020, hermaphrodite chemical and
                              symmetric gap-junction sheets. Read through
                              scripts/xlsx.py, a standard-library reader, so
                              the project keeps its numpy-only dependency.
                              This is the connectome; the 2019 edgelist and
                              the White 1986 file that fetch_data.py also
                              downloads are comparison material and are not
                              read here.
  owmeta_cache.json           OpenWorm curated neuron/muscle metadata
  all_cell_info.csv           WormAtlas cell classifications

Outputs (data/processed/): genes.json, genome_stats.json, connectome.json, cells.json
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import xlsx

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# RefSeq accession -> the chromosome name every C. elegans paper actually uses.
ACCESSION_TO_CHROM = {
    "NC_003279.8": "I",
    "NC_003280.10": "II",
    "NC_003281.10": "III",
    "NC_003282.8": "IV",
    "NC_003283.11": "V",
    "NC_003284.9": "X",
    "NC_001328.1": "MtDNA",
}


def parse_attributes(field: str) -> dict[str, str]:
    out = {}
    for part in field.rstrip(";").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def build_genome_stats() -> dict:
    """Walk the assembly once, recording length and base composition per chromosome."""
    path = RAW / "celegans_genome.fna.gz"
    chroms: dict[str, Counter] = {}
    current = None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                acc = line[1:].split()[0]
                current = ACCESSION_TO_CHROM.get(acc, acc)
                chroms[current] = Counter()
            else:
                chroms[current].update(line.strip().upper())

    stats, total = {}, 0
    for name, counts in chroms.items():
        length = sum(counts.values())
        gc = counts["G"] + counts["C"]
        at = counts["A"] + counts["T"]
        stats[name] = {
            "length_bp": length,
            "gc_fraction": round(gc / (gc + at), 5) if gc + at else 0.0,
            "n_bases": counts.get("N", 0),
        }
        total += length

    return {
        "assembly": "WBcel235 (RefSeq GCF_000002985.6)",
        "source": "NCBI RefSeq",
        "total_bp": total,
        "n_sequences": len(stats),
        "chromosomes": stats,
    }


def build_genes() -> tuple[dict, dict]:
    """Extract every annotated gene, plus per-gene exon/transcript counts."""
    path = RAW / "celegans_annotation.gff.gz"
    genes: dict[str, dict] = {}
    # locus_tag -> gene symbol, so we can attribute mRNA/exon lines back to a gene
    tag_to_symbol: dict[str, str] = {}
    exon_counts: Counter = Counter()
    transcript_counts: Counter = Counter()

    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            seqid, _, feature, start, end, _, strand, _, attrs_raw = f
            chrom = ACCESSION_TO_CHROM.get(seqid)
            if chrom is None:
                continue
            attrs = parse_attributes(attrs_raw)

            if feature == "gene" or feature == "pseudogene":
                symbol = attrs.get("gene") or attrs.get("Name")
                if not symbol:
                    continue
                locus = attrs.get("locus_tag", "")
                wb = ""
                for ref in attrs.get("Dbxref", "").split(","):
                    if ref.startswith("WormBase:"):
                        wb = ref.split(":", 1)[1]
                genes[symbol] = {
                    "symbol": symbol,
                    "wormbase_id": wb,
                    "locus_tag": locus,
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "biotype": attrs.get("gene_biotype", "unknown"),
                }
                if locus:
                    tag_to_symbol[locus] = symbol
            elif feature == "exon":
                tag = attrs.get("locus_tag") or ""
                if tag:
                    exon_counts[tag] += 1
            elif feature in ("mRNA", "transcript", "ncRNA"):
                tag = attrs.get("locus_tag") or ""
                if tag:
                    transcript_counts[tag] += 1

    for tag, symbol in tag_to_symbol.items():
        genes[symbol]["exons"] = exon_counts.get(tag, 0)
        genes[symbol]["transcripts"] = transcript_counts.get(tag, 0)

    by_type = Counter(g["biotype"] for g in genes.values())
    by_chrom = Counter(g["chrom"] for g in genes.values())
    summary = {
        "n_genes": len(genes),
        "by_biotype": dict(by_type.most_common()),
        "by_chromosome": dict(by_chrom.most_common()),
    }
    return genes, summary


# Connectome muscle labels (dBWML7) vs owmeta labels (MDL07) are the same cells
# under two naming conventions. Normalise everything to the connectome's form.
MUSCLE_RE = re.compile(r"^([dv])BWM([LR])(\d+)$")


def canonical_muscle(name: str) -> str | None:
    m = MUSCLE_RE.match(name)
    if m:
        return name
    m = re.match(r"^M([DV])([LR])(\d+)$", name)
    if m:
        side, lr, num = m.groups()
        return f"{side.lower()}BWM{lr}{int(num)}"
    return None


# The connectome edgelist zero-pads ventral-cord motor neurons (DA01); owmeta does
# not (DA1). Same cells. Strip padding when cross-referencing.
VNC_RE = re.compile(r"^(AS|DA|DB|DD|VA|VB|VC|VD)(\d+)$")


def owmeta_key(name: str, neuron_info: dict) -> str | None:
    if name in neuron_info:
        return name
    m = VNC_RE.match(name)
    if m:
        alt = f"{m.group(1)}{int(m.group(2))}"
        if alt in neuron_info:
            return alt
    return None


# owmeta's table stops at index 9, but the real animal (and the Cook edgelist) has
# VA01-12, VB01-11, AS01-11, VD01-13. Fill the remainder from the textbook
# assignment for each class -- these are uniform within a class.
# Refs: WormAtlas Ventral Cord Motor Neurons; Pereira et al. 2015 eLife
# (cholinergic atlas); Gendrel et al. 2016 eLife (GABAergic atlas).
VNC_CLASS_DEFAULTS = {
    "DA": (["motor"], ["Acetylcholine"]),
    "DB": (["motor"], ["Acetylcholine"]),
    "VA": (["motor"], ["Acetylcholine"]),
    "VB": (["motor"], ["Acetylcholine"]),
    "AS": (["motor"], ["Acetylcholine"]),
    "DD": (["motor"], ["GABA"]),
    "VD": (["motor"], ["GABA"]),
    "VC": (["motor"], ["Acetylcholine"]),
}


# owmeta leaves 57 neurons without a transmitter, including every command
# interneuron -- which matters a lot here, because the behavioural readout is
# taken straight off AVA/AVB/AVD/AVE. These are filled from the systematic
# transmitter atlases. Only well-attested assignments are listed; genuinely
# ambiguous cells are left blank and fall back to the neutral default.
# Refs: Pereira et al. 2015 eLife 12432 (cholinergic map);
#       Serrano-Saiz et al. 2013 Cell 155:659 (glutamatergic map);
#       Gendrel et al. 2016 eLife 17686 (GABAergic map).
# Note: ALA and AVF stain for GABA but lack unc-25 -- they take GABA up rather
# than synthesising it, so they are deliberately NOT marked GABAergic.
SUPPLEMENTARY_NT: dict[str, list[str]] = {
    # Command and premotor interneurons -- cholinergic (Pereira 2015).
    "AVAL": ["Acetylcholine"], "AVAR": ["Acetylcholine"],
    "AVBL": ["Acetylcholine"], "AVBR": ["Acetylcholine"],
    "AVDL": ["Acetylcholine"], "AVDR": ["Acetylcholine"],
    "AVEL": ["Acetylcholine"], "AVER": ["Acetylcholine"],
    # Head/turn motor and interneurons -- cholinergic.
    "RIVL": ["Acetylcholine"], "RIVR": ["Acetylcholine"],
    "RMFL": ["Acetylcholine"], "RMFR": ["Acetylcholine"],
    "RMHL": ["Acetylcholine"], "RMHR": ["Acetylcholine"],
    "AVG": ["Acetylcholine"], "AVJL": ["Acetylcholine"],
    "AVJR": ["Acetylcholine"], "PVNL": ["Acetylcholine"],
    "PVNR": ["Acetylcholine"], "PDB": ["Acetylcholine"],
    # Sensory and interneurons -- glutamatergic (eat-4 positive).
    "ADLL": ["Glutamate"], "ADLR": ["Glutamate"],
    "AWAL": ["Glutamate"], "AWAR": ["Glutamate"],
    "AWBL": ["Glutamate"], "AWBR": ["Glutamate"],
    "URXL": ["Glutamate"], "URXR": ["Glutamate"],
    "AIBR": ["Glutamate"], "PVT": ["Glutamate"],
    "ASJL": ["Glutamate"], "ASJR": ["Glutamate"],
    "BDUL": ["Glutamate"], "BDUR": ["Glutamate"],
    "RIR": ["Glutamate"], "PVM": ["Glutamate"],
}


def _matrix_edges(path: Path, sheet: str, etype: str) -> list[dict]:
    """Edges from one adjacency sheet.

    Layout: row 2 holds the postsynaptic column labels, column 2 the
    presynaptic row labels, and the body is weights. Rows 0 and 1 are the
    category banner ("PHARYNX", "SENSORY NEURONS", ...), not data.

    The symmetric gap-junction sheet carries each pair in both directions,
    matching how the previous edgelist listed them, so both are emitted and
    the consumer fills its matrix directly rather than symmetrising.
    """
    grid = xlsx.read_sheet(path, sheet)
    header = {str(c).strip(): j for j, c in enumerate(grid[2]) if c}
    edges = []
    for row in grid[3:]:
        pre = row[2] if len(row) > 2 else None
        if not pre:
            continue
        pre = str(pre).strip()
        for post, j in header.items():
            if j >= len(row):
                continue
            try:
                w = float(row[j])
            except (TypeError, ValueError):
                continue
            if w > 0:
                edges.append({"pre": pre, "post": post, "w": w, "type": etype})
    return edges


def build_connectome(neuron_info: dict, muscle_info: dict) -> tuple[dict, dict]:
    # Cook et al.'s corrected July 2020 release. Weights are the same numbers
    # as the 2019 edgelist (AVAL-AVAR 18, PVCL->AVBL 7), so nothing calibrated
    # against contact counts changes, but the posterior body is better covered.
    path = RAW / "cook_2020_adjacency.xlsx"
    edges = _matrix_edges(path, "hermaphrodite chemical", "chemical")
    edges += _matrix_edges(path, "hermaphrodite gap jn symmetric", "electrical")
    cells_seen: set[str] = {e["pre"] for e in edges} | {e["post"] for e in edges}

    # Classify every cell that appears in the wiring diagram.
    wormatlas: dict[str, dict] = {}
    ci_path = RAW / "all_cell_info.csv"
    if ci_path.exists():
        with open(ci_path, newline="") as fh:
            for row in csv.DictReader(fh):
                wormatlas[row["Cell name"].strip()] = {
                    "wormatlas_type": row.get("Type", "").strip(),
                    "classification": row.get("Classification", "").strip(),
                    "lineage": row.get("Lineage", "").strip(),
                }

    muscle_alias = {}
    for owmeta_name in muscle_info:
        canon = canonical_muscle(owmeta_name)
        if canon:
            muscle_alias[canon] = owmeta_name

    cells: dict[str, dict] = {}
    for name in sorted(cells_seen):
        entry: dict = {"name": name}
        key = owmeta_key(name, neuron_info)
        vnc = VNC_RE.match(name)
        if key is not None:
            info = neuron_info[key]
            types = [t.lower() for t in (info[1] or [])]
            nts = [n for n in (info[3] or []) if n]
            entry["kind"] = "neuron"
            entry["roles"] = types
            entry["neurotransmitters"] = nts
            entry["nt_source"] = "owmeta"
        elif vnc and vnc.group(1) in VNC_CLASS_DEFAULTS:
            roles, nts = VNC_CLASS_DEFAULTS[vnc.group(1)]
            entry["kind"] = "neuron"
            entry["roles"] = list(roles)
            entry["neurotransmitters"] = list(nts)
            entry["nt_source"] = "class_default"
        elif canonical_muscle(name):
            entry["kind"] = "muscle"
            entry["roles"] = ["bodywall_muscle"]
            entry["neurotransmitters"] = []
            m = MUSCLE_RE.match(name)
            entry["side"] = "dorsal" if m.group(1) == "d" else "ventral"
            entry["lr"] = "left" if m.group(2) == "L" else "right"
            entry["row"] = int(m.group(3))
        else:
            entry["kind"] = "other"
            entry["roles"] = []
            entry["neurotransmitters"] = []
        if entry.get("kind") == "neuron" and not entry.get("neurotransmitters"):
            supp = SUPPLEMENTARY_NT.get(name)
            if supp:
                entry["neurotransmitters"] = list(supp)
                entry["nt_source"] = "atlas"

        if vnc:
            # Class and index along the cord: DB07 -> ("DB", 7). Locomotion code
            # uses the index to map motor neurons onto body segments.
            entry["vnc_class"] = vnc.group(1)
            entry["vnc_index"] = int(vnc.group(2))
        entry.update(wormatlas.get(name, {}))
        cells[name] = entry

    kinds = Counter(c["kind"] for c in cells.values())
    nt_counts: Counter = Counter()
    for c in cells.values():
        for nt in c["neurotransmitters"]:
            nt_counts[nt] += 1
    role_counts: Counter = Counter()
    for c in cells.values():
        for r in c["roles"]:
            role_counts[r] += 1

    by_type = Counter(e["type"] for e in edges)
    summary = {
        "nt_provenance": dict(
            Counter(c.get("nt_source", "none") for c in cells.values())
        ),
        "n_cells": len(cells),
        "n_edges": len(edges),
        "edges_by_type": dict(by_type),
        "cells_by_kind": dict(kinds),
        "neurons_by_role": dict(role_counts.most_common()),
        "neurotransmitters": dict(nt_counts.most_common()),
        "unmatched_muscles": sorted(
            set(muscle_alias) - {c for c, v in cells.items() if v["kind"] == "muscle"}
        ),
    }
    connectome = {
        "source": "Cook et al. 2019 hermaphrodite full edgelist (via OpenWorm ConnectomeToolbox)",
        "edges": edges,
    }
    return connectome, {"cells": cells, "summary": summary}


# Receptor pharmacology for the monoamine layer. Each entry is the sign of
# the receptor's action on the cell expressing it, from its G-protein
# coupling or channel selectivity. These are the measured classes, not a
# guess: Gi/Go coupling inhibits (lowers cAMP, opens GIRK-like conductance),
# Gq and Gs excite, and the ligand-gated chloride channels inhibit directly.
#
# dop-1 Gq (Chase, Pepper & Koelle 2004 Nat Neurosci 7:1096)
# dop-2 Gi/Go autoreceptor (Suo, Kimura & Van Tol 2006 J Neurosci 26:10082)
# dop-3 Gi/Go, the receptor carrying basal slowing on cholinergic motor
#       neurons (Chase 2004; Sanyal et al. 2004 EMBO J 23:473)
# dop-4 Gs (Sugiura et al. 2005 J Neurochem 94:1146)
# lgc-53 dopamine-gated chloride channel (Ringstad, Abe & Horvitz 2009
#       Science 325:96)
# ser-1 Gq, ser-4 Gi/Go, ser-7 Gs (Hobson et al. 2006; Olde & McCombie 1997;
#       Hobson et al. 2003), mod-1 serotonin-gated chloride channel
#       (Ranganathan, Cannon & Horvitz 2000 Nature 408:470)
# ser-2 tyramine Gi/Go, tyra-2 Gq, tyra-3 Gq, lgc-55 tyramine-gated chloride
#       (Rex & Komuniecki 2002; Pirri et al. 2009 Neuron 62:526)
# octr-1 Gi/Go, ser-3 Gq, ser-6 Gs (Wragg et al. 2007; Suo et al. 2006)
RECEPTOR_SIGN = {
    "dop-1": +1.0, "dop-2": -1.0, "dop-3": -1.0, "dop-4": +1.0,
    "lgc-53": -1.0,
    "ser-1": +1.0, "ser-4": -1.0, "ser-5": +1.0, "ser-7": +1.0,
    "mod-1": -1.0,
    "ser-2": -1.0, "tyra-2": +1.0, "tyra-3": +1.0, "lgc-55": -1.0,
    "octr-1": -1.0, "ser-3": +1.0, "ser-6": +1.0,
}

# Which gene has to be intact for a cell to RELEASE each monoamine, so a
# knockout removes the ligand while leaving every receptor in place, which
# is what the mutant actually is.
LIGAND_GENE = {
    "dopamine": "cat-2",       # tyrosine hydroxylase
    "serotonin": "tph-1",      # tryptophan hydroxylase
    "tyramine": "tdc-1",       # tyrosine decarboxylase
    "octopamine": "tbh-1",     # tyramine beta-hydroxylase, downstream of tdc-1
}


def build_monoamines() -> dict:
    """Bentley et al. 2016 monoaminergic edges, per ligand and receptor.

    Extrasynaptic: an edge means the source expresses the biosynthetic
    enzyme for that monoamine and the target expresses a cognate receptor,
    so this layer does NOT follow the wired connectome and cannot be
    derived from it. Kept as an edge list rather than a matrix because the
    receptor identity is the whole point: dop-3 inhibits where dop-1
    excites, and a receptor knockout has to be able to remove one without
    the other.
    """
    rows = []
    with open(RAW / "edgelist_MA.csv", newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 4:
                continue
            src, tgt, lig, rec = (x.strip() for x in r[:4])
            if not src or not tgt:
                continue
            rows.append({"pre": src, "post": tgt, "ligand": lig,
                         "receptor": rec,
                         "sign": RECEPTOR_SIGN.get(rec, 0.0)})
    unknown = sorted({r["receptor"] for r in rows if r["sign"] == 0.0})
    return {"edges": rows, "ligand_gene": LIGAND_GENE,
            "receptor_sign": RECEPTOR_SIGN,
            "receptors_without_sign": unknown,
            "source": "Bentley et al. 2016 PLoS Comput Biol 12:e1005283"}


# Neuropeptide receptor pharmacology. Twelve receptors carry the whole
# Bentley 2016 peptidergic layer, and each is signed from its published
# G-protein coupling rather than a family guess.
#
# npr-1  Gi/Go, the NPY-like receptor for FLP-18 and FLP-21 whose loss makes
#        animals social and O2-avoidant (de Bono & Bargmann 1998 Cell 94:679;
#        Rogers et al. 2003 Nat Neurosci 6:1178)
# npr-2, npr-3, npr-4, npr-5, npr-11  NPY/FMRFamide-like, Gi/Go coupled
#        (Cohen et al. 2009 Cell Metab 9:375 npr-1 family; Chalasani et al.
#        2010 Nat Neurosci 13:615 npr-11 on AIA; Nathoo et al. 2001 PNAS
#        98:14000 for the family's deorphanisation)
# pdfr-1 Gs, the PDF receptor driving roaming (Barrios et al. 2012 Nat
#        Neurosci 15:1675; Janssen et al. 2008 J Biol Chem 283:15241)
# ntr-1  Gq, nematocin receptor, the oxytocin/vasopressin homologue
#        (Beets et al. 2012 Science 338:543)
# frpr-4 Gi/Go (Mertens et al. 2005 Biochem Biophys Res Commun 330:967)
# egl-6  Gi/Go, inhibits HSN egg laying (Ringstad & Horvitz 2008 Nat
#        Neurosci 11:1168)
# ckr-2  Gq, cholecystokinin-like, raises feeding and locomotion (Janssen et
#        al. 2008 Endocrinology 149:2826)
# npr-17 Gi/Go, the opioid-like receptor (Cheong et al. 2015 Nat Commun
#        6:9442)
PEPTIDE_RECEPTOR_SIGN = {
    "npr-1": -1.0, "npr-2": -1.0, "npr-3": -1.0, "npr-4": -1.0,
    "npr-5": -1.0, "npr-11": -1.0, "npr-17": -1.0,
    "frpr-4": -1.0, "egl-6": -1.0,
    "pdfr-1": +1.0, "ntr-1": +1.0, "ckr-2": +1.0,
}

# Peptide release needs the dense-core vesicle machinery, so unc-31 (CAPS)
# removes ALL peptidergic signalling at once and egl-3 (proprotein
# convertase PC2) removes the peptides that need processing, which is most
# of them. Both are network-wide lesions rather than per-ligand ones
# (Speese et al. 2007 J Neurosci 27:6150; Kass et al. 2001 Dev Biol 237:173).
PEPTIDE_GLOBAL_GENES = ["unc-31", "egl-3"]


def build_peptides() -> dict:
    """Bentley et al. 2016 peptidergic edges, per ligand and receptor.

    Same extrasynaptic logic as the monoamines and the same shape, so the
    modulation layer can carry both: source expresses the peptide, target
    expresses a cognate receptor, no wiring implied.
    """
    rows = []
    with open(RAW / "edgelist_NP.csv", newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 4:
                continue
            src, tgt, lig, rec = (x.strip() for x in r[:4])
            if not src or not tgt:
                continue
            rows.append({"pre": src, "post": tgt, "ligand": lig,
                         "receptor": rec,
                         "sign": PEPTIDE_RECEPTOR_SIGN.get(rec, 0.0)})
    unknown = sorted({r["receptor"] for r in rows if r["sign"] == 0.0})
    return {"edges": rows, "receptor_sign": PEPTIDE_RECEPTOR_SIGN,
            "global_genes": PEPTIDE_GLOBAL_GENES,
            "receptors_without_sign": unknown,
            "source": "Bentley et al. 2016 PLoS Comput Biol 12:e1005283"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    owmeta = json.loads((RAW / "owmeta_cache.json").read_text())
    neuron_info = owmeta["neuron_info"]
    muscle_info = owmeta["muscle_info"]

    print("parsing genome assembly ...")
    genome_stats = build_genome_stats()
    (OUT / "genome_stats.json").write_text(json.dumps(genome_stats, indent=1))
    print(f"  {genome_stats['total_bp']:,} bp over {genome_stats['n_sequences']} sequences")

    print("parsing gene annotation ...")
    genes, gene_summary = build_genes()
    genome_stats["genes"] = gene_summary
    (OUT / "genome_stats.json").write_text(json.dumps(genome_stats, indent=1))
    (OUT / "genes.json").write_text(json.dumps(genes))
    print(f"  {gene_summary['n_genes']:,} genes")
    for bt, n in list(gene_summary["by_biotype"].items())[:5]:
        print(f"    {n:>6,}  {bt}")

    print("parsing connectome ...")
    connectome, cellinfo = build_connectome(neuron_info, muscle_info)
    (OUT / "connectome.json").write_text(json.dumps(connectome))
    (OUT / "cells.json").write_text(json.dumps(cellinfo, indent=1))
    pep = build_peptides()
    (OUT / "peptides.json").write_text(json.dumps(pep))
    print(f"  peptides.json: {len(pep['edges'])} edges, "
          f"{len(set(e['ligand'] for e in pep['edges']))} ligands, "
          f"unsigned receptors {pep['receptors_without_sign']}")
    mono = build_monoamines()
    (OUT / "monoamines.json").write_text(json.dumps(mono))
    print(f"  monoamines.json: {len(mono['edges'])} edges, "
          f"{len(set(e['ligand'] for e in mono['edges']))} ligands, "
          f"unsigned receptors {mono['receptors_without_sign']}")
    s = cellinfo["summary"]
    print(f"  {s['n_cells']} cells, {s['n_edges']} edges {s['edges_by_type']}")
    print(f"  kinds: {s['cells_by_kind']}")
    print(f"  neurotransmitters: {s['neurotransmitters']}")
    print(f"  roles: {s['neurons_by_role']}")
    if s["unmatched_muscles"]:
        print(f"  WARNING unmatched muscles: {s['unmatched_muscles']}")

    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
