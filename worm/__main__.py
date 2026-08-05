"""Command line: python -m worm <command>"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_serve(a) -> int:
    from .server import serve
    serve(host=a.host, port=a.port, fps=a.fps, seed=a.seed)
    return 0


def cmd_info(a) -> int:
    from .connectome import Connectome
    from .genome import Genome
    g = Genome.load()
    c = Connectome.load()
    print(g.describe())
    print("\nConnectome: Cook et al. 2019 hermaphrodite")
    s = c.stats()
    print(f"  {s['cells']} cells  ({s['neurons']} neurons, {s['muscles']} body-wall muscles)")
    print(f"  {s['chemical_edges']} chemical synapse pairs "
          f"({s['chemical_contacts']:,.0f} contacts)")
    print(f"  {s['electrical_edges']} gap junction pairs "
          f"({s['electrical_contacts']:,.0f} contacts)")
    nt = {}
    for name in c.names:
        for x in c.cell_info[name].get("neurotransmitters") or []:
            nt[x] = nt.get(x, 0) + 1
    print("  neurotransmitters: " +
          ", ".join(f"{k} {v}" for k, v in sorted(nt.items(), key=lambda kv: -kv[1])))
    sp = c.sign_provenance
    total = sum(sp.values())
    print("  synapse signs: " + ", ".join(f"{k} {v}" for k, v in sp.most_common()))
    if c.sign_flips:
        print(f"  {len(c.sign_flips)} edges changed sign vs the transmitter "
              f"heuristic (of {total})")
    return 0


def cmd_gene(a) -> int:
    from .genome import GENE_EFFECTS, Genome
    g = Genome.load()
    rec = g.gene(a.name)
    if rec is None:
        hits = g.search(a.name, 12)
        print(f"no gene named {a.name!r}.")
        if hits:
            print("did you mean: " + ", ".join(h["symbol"] for h in hits))
        return 1
    print(f"{rec['symbol']}   {rec['wormbase_id']}   {rec['locus_tag']}")
    print(f"  chromosome {rec['chrom']}:{rec['start']:,}-{rec['end']:,} ({rec['strand']})")
    print(f"  {rec['end']-rec['start']+1:,} bp, biotype {rec['biotype']}, "
          f"{rec.get('exons', 0)} exons, {rec.get('transcripts', 0)} transcripts")
    eff = GENE_EFFECTS.get(rec["symbol"])
    if eff:
        print(f"  product:   {eff.product}")
        print(f"  phenotype: {eff.phenotype}")
    else:
        print("  (no behavioural model for this locus in the simulator)")
    return 0


def cmd_run(a) -> int:
    from .environment import Environment
    from .simulation import SimConfig, WormSimulation
    env = Environment(width=44.0, height=32.0)
    if a.salt:
        env.add_source(12.0, 6.0, kind="salt", strength=1.0, sigma=8.0)
    sim = WormSimulation(env=env, config=SimConfig(seed=a.seed))
    for gene in a.knockout or []:
        rec = sim.knock_out(gene)
        print(f"# knocked out {rec['gene']}: {rec['phenotype']}", file=sys.stderr)
    n = int(a.seconds / sim.cfg.dt)
    every = max(1, int(a.interval / sim.cfg.dt))
    for i in range(n):
        tel = sim.step()
        if i % every == 0:
            print(json.dumps(tel) if a.json else
                  f"t={tel['t']:7.2f}  {tel['behavior']:<10} "
                  f"speed={tel['speed_mm_s']:.3f}  x={tel['x']:+7.2f} y={tel['y']:+7.2f} "
                  f"rev={tel['reversals']} omega={tel['omegas']} len={tel['length_scale']:.2f}")
    return 0


def cmd_assay(a) -> int:
    from .assays import ASSAYS, run_assay
    names = sorted(ASSAYS) if a.name == "all" else [a.name]
    if a.name != "all" and a.name not in ASSAYS:
        print(f"unknown assay {a.name!r}. available: {', '.join(sorted(ASSAYS))}")
        return 1
    out = []
    for n in names:
        kw = {"knockouts": tuple(a.knockout or ()),
              "ablations": tuple(a.ablate or ()), "seed": a.seed,
              "workers": a.jobs if a.jobs > 0 else None}
        rec = run_assay(n, **kw)
        out.append(rec)
        if not a.json:
            print(f"\n=== {rec['assay']} ===")
            if a.knockout:
                print(f"  genotype: {', '.join(a.knockout)}")
            if a.ablate:
                print(f"  ablated:  {', '.join(a.ablate)}")
            print(f"  metric:   {rec['metric']}")
            print(f"  expected: {rec['expected']}")
            for k, v in rec["result"].items():
                if isinstance(v, list) and len(v) > 12:
                    v = f"{v[:12]} ... ({len(v)} values)"
                print(f"    {k:<26} {v}")
            print(f"  source:   {rec['source']}")
    if a.json:
        print(json.dumps(out, indent=1))
    return 0


def cmd_export(a) -> int:
    """Dump a run to CSV/JSON for offline analysis."""
    import csv

    from .environment import Environment
    from .simulation import SimConfig, WormSimulation
    env = Environment(width=44.0, height=32.0)
    if a.salt:
        env.add_source(12.0, 6.0, kind="salt", strength=1.0, sigma=10.0)
    if a.food:
        env.add_source(12.0, 6.0, kind="food", strength=1.0, sigma=6.0)
    sim = WormSimulation(env=env, config=SimConfig(seed=a.seed))
    for g in a.knockout or []:
        sim.knock_out(g)
    for c in a.ablate or []:
        sim.ablate(c)

    watch = a.neuron or []
    idx = [sim.conn.idx(n) for n in watch if n in sim.conn.index]
    missing = [n for n in watch if n not in sim.conn.index]
    if missing:
        print(f"# unknown cells ignored: {', '.join(missing)}", file=sys.stderr)

    rows = []
    n = int(a.seconds / sim.cfg.dt)
    every = max(1, int(a.interval / sim.cfg.dt))
    import numpy as np
    for i in range(n):
        tel = sim.step()
        if i % every:
            continue
        row = {"t": tel["t"], "x": tel["x"], "y": tel["y"],
               "heading": tel["heading"], "speed_mm_s": tel["speed_mm_s"],
               "behavior": tel["behavior"], "reversals": tel["reversals"],
               "omegas": tel["omegas"], "stage": tel["life"]["stage"],
               "body_length_mm": tel["life"]["body_length_mm"],
               "reserves": tel["life"]["reserves"]}
        if idx:
            act = sim.ns.activation()
            for name, j in zip([w for w in watch if w in sim.conn.index], idx):
                row[f"act_{name}"] = round(float(act[j]), 4)
        rows.append(row)

    path = a.out
    if path.endswith(".json"):
        with open(path, "w") as fh:
            json.dump({"config": {"seed": a.seed,
                                  "knockouts": sorted(sim.genome.knockouts),
                                  "ablated": sorted(sim.ns.ablated),
                                  "dt": sim.cfg.dt, "seconds": a.seconds},
                       "rows": rows}, fh, indent=1)
    else:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")
    return 0


def cmd_batch(a) -> int:
    """Run several genotypes side by side and print a comparison table."""
    import numpy as np

    from .validate import gait
    genotypes = [g.split("+") if "+" in g else [g] for g in (a.genotype or [])]
    genotypes = [[] if g == ["wild-type"] else g for g in genotypes] or [[]]

    print(f"{'genotype':<24}{'n':>3}{'speed mm/s':>16}{'amplitude %BL':>16}"
          f"{'length':>9}")
    out = []
    for genes in genotypes:
        speeds, amps, lens = [], [], []
        for s in range(a.replicates):
            g = gait(knockouts=tuple(genes), seed=s, seconds=a.seconds)
            speeds.append(g["speed"])
            amps.append(g["amplitude"] * 100)
            lens.append(g["length_scale"])
        label = "+".join(genes) if genes else "wild type"
        rec = {"genotype": label, "n": a.replicates,
               "speed_mean": round(float(np.mean(speeds)), 4),
               "speed_sd": round(float(np.std(speeds)), 4),
               "amplitude_mean_pct": round(float(np.mean(amps)), 2),
               "amplitude_sd": round(float(np.std(amps)), 2),
               "length_scale_mean": round(float(np.mean(lens)), 3)}
        out.append(rec)
        print(f"{label:<24}{a.replicates:>3}"
              f"{rec['speed_mean']:>10.3f} +-{rec['speed_sd']:<5.3f}"
              f"{rec['amplitude_mean_pct']:>10.1f} +-{rec['amplitude_sd']:<5.1f}"
              f"{rec['length_scale_mean']:>9.2f}")
    if a.json:
        print(json.dumps(out, indent=1))
    return 0


def cmd_validate(a) -> int:
    from .validate import CHECKS, main as run_validation
    import os
    jobs = a.jobs if a.jobs > 0 else min(os.cpu_count() or 1, len(CHECKS))
    return run_validation(verbose=not a.quiet, jobs=jobs, match=a.match)


def cmd_params(a) -> int:
    from .parameters import audit
    print(audit())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="worm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the live browser viewer")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--fps", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("info", help="genome and connectome summary")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("gene", help="look up a gene in the WBcel235 annotation")
    s.add_argument("name")
    s.set_defaults(fn=cmd_gene)

    s = sub.add_parser("run", help="headless run, printing telemetry")
    s.add_argument("--seconds", type=float, default=30.0)
    s.add_argument("--interval", type=float, default=1.0)
    s.add_argument("--knockout", action="append", metavar="GENE")
    s.add_argument("--salt", action="store_true", help="place a salt gradient")
    s.add_argument("--json", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("assay", help="run a standard behavioural assay")
    s.add_argument("name", help="assay name, or 'all'")
    s.add_argument("--knockout", action="append", metavar="GENE")
    s.add_argument("--ablate", action="append", metavar="CELL")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--jobs", type=int, default=0,
                   help="parallel workers for replicated assays; 0 = all cores")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_assay)

    s = sub.add_parser("export", help="dump a run to CSV or JSON")
    s.add_argument("out", help="output path (.csv or .json)")
    s.add_argument("--seconds", type=float, default=120.0)
    s.add_argument("--interval", type=float, default=0.5)
    s.add_argument("--neuron", action="append", metavar="CELL",
                   help="also record this cell's activation (repeatable)")
    s.add_argument("--knockout", action="append", metavar="GENE")
    s.add_argument("--ablate", action="append", metavar="CELL")
    s.add_argument("--salt", action="store_true")
    s.add_argument("--food", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("batch", help="compare genotypes with replicates")
    s.add_argument("--genotype", action="append", metavar="GENE[+GENE]",
                   help="repeatable; use 'wild-type' for the control")
    s.add_argument("--replicates", type=int, default=3)
    s.add_argument("--seconds", type=float, default=30.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_batch)

    s = sub.add_parser("validate", help="check modelled phenotypes against the literature")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--jobs", type=int, default=0,
                   help="parallel workers; 0 = one per check, up to core count")
    s.add_argument("--match", help="run only checks whose name contains this")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("params", help="audit every model parameter and its provenance")
    s.set_defaults(fn=cmd_params)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
