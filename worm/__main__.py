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
    print(f"  inhibitory-glutamate overrides active: {len(set(c.overrides))}")
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


def cmd_validate(a) -> int:
    from .validate import main as run_validation
    return run_validation(verbose=not a.quiet)


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

    s = sub.add_parser("validate", help="check modelled phenotypes against the literature")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_validate)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
