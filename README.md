# celeganssim

A whole-organism simulator for *Caenorhabditis elegans*.

It simulates all 300 neurons and 95 body-wall muscles over the published
electron-microscopy connectome, drives a biomechanical body through a viscous
medium, and runs the animal from a fertilised egg through four larval stages to
senescence and death. You can poke it anywhere on its body, knock out genes by
name, laser-ablate individual neurons, and run the standard behavioural assays.

![Arena view: an adult worm crawling, with dorsal and ventral body-wall muscle activation drawn along the body and live neuron traces in the sidebar](docs/screenshot.png)

*Teal is dorsal muscle, purple is ventral. The alternation running down the body
is the undulatory wave that moves the animal.*

---

## Install and run

```bash
git clone https://github.com/vdmkenny/celeganssim && cd celeganssim
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/fetch_data.py     # downloads ~43 MB, then builds
.venv/bin/worm serve                       # http://127.0.0.1:8080
```

Requires Python 3.10+ and **numpy**. The viewer is standard library and vanilla
JavaScript, with no build step. Datasets are not redistributed here:
`fetch_data.py` pulls each from its publisher, verifies it, and builds the
processed files.

```bash
worm serve            # live browser viewer
worm validate         # 24 checks against published measurements, in parallel
worm params           # audit every model parameter and its provenance
worm assay all        # run the standard behavioural assays
worm gene unc-25      # look up a locus
worm info             # genome and connectome summary
worm export run.csv --seconds 120 --neuron AVAL
worm batch --genotype wild-type --genotype unc-25 --replicates 5
```

---

## What it does

### Mechanosensation

Touch is a position along the body. The six touch receptor neurons tile the
animal with overlapping graded receptive fields, so where you prod it determines
which cells respond. Anterior touch drives reversal, posterior touch drives
forward acceleration, and mid-body touch falls where the ALM and PLM fields
cross over and is unreliable. These follow from the connectome rather than from
rules: PLM has no chemical output at all, reaching the forward command
interneuron PVC purely through gap junctions.

Harsh touch (100-200 uN) is a separate channel. FLP covers the head and is
reversal-biased in the wiring; PVD tiles the rest of the body and favours
forward. Both are MEC-4 independent, so harsh responses survive in `mec-4`
mutants.

### Genetics

Thirty-one loci are mapped onto the subsystems their products implement.
Knockouts reach only the cells that CeNGEN single-cell RNA-seq measures the gene
in:

| Gene | Expressed in | Effect |
|---|---|---|
| `mec-4` | the six touch receptor neurons | gentle touch abolished, harsh touch intact |
| `unc-25` | 28 cells: DD, VD, RME, AVL, DVB, RIS, RIB | GABA lost, animal hypercontracts |
| `che-1` | ASEL, ASER | salt chemotaxis lost, odour intact |
| `cat-2` | the eight dopaminergic neurons | basal slowing on food lost |
| `daf-2` | 158 cells | lifespan roughly doubles, requires `daf-16` |

`worm gene <name>` resolves any of the 46,926 annotated loci to its WormBase ID,
chromosome and coordinates.

### Neuron ablation

Alt-click any cell in the network view to kill it: it stops sending and
receiving, and the surviving network's resting state is re-solved around the
loss.

| Ablation | Result |
|---|---|
| AVB + PVC | forward drive falls to 0.00, reversals intact |
| AVA + AVD + AVE | touch-evoked reversal abolished |
| ALM + AVM | anterior touch sensitivity lost |

### Network view

![Network view with AVAL pinned, showing all 106 of its connections highlighted while the rest of the graph dims](docs/network.png)

All 448 cells laid out sensory to interneuron to motor to muscle, ordered to
minimise edge crossings, with node brightness tracking live drive. Hover any
cell for its transmitter, current drive and strongest partners with weights.
Pin it, or watch it in the trace panel to see its activation over the last 30
seconds.

### Life cycle

Runs from fertilised egg to death on measured timings: 14.2 h of embryogenesis
with milestones, 50.7 h from hatch to adult across four larval stages, dauer
entry under crowding and scarcity, a sperm-limited brood of ~300 over ~5 days,
then senescence and death at a lifespan that scales with temperature and
genotype.

### Assays

Each reports the metric its original paper reports and carries that paper's
reference value.

| Assay | Measures |
|---|---|
| `chemotaxis` | chemotaxis index on a salt gradient |
| `thermotaxis` | drift relative to remembered cultivation temperature |
| `touch-habituation` | response decrement over repeated taps |
| `basal-slowing` | speed drop on encountering food |
| `touch-response` | reversal probability by body position |
| `lifespan` | egg to death, with brood size and reproductive period |

### Data export

CSV or JSON of trajectory, behavioural state, developmental stage and named
neuron activations, with fixed seeds. `worm batch` compares genotypes across
replicates.

---

## How it works

```
Environment      gradients, bacterial lawn, temperature, oxygen, touch
      |
SensorySystem    modality -> named neurons, via receptive fields, gated by genes
      |
NervousSystem    448 cells, graded-potential ODEs over the real connectome
      |
Simulation       reads command interneurons -> behavioural state machine
      |
Body             ventral-cord oscillator -> 95 muscles -> viscoelastic body
      |
                 resistive force theory -> movement -> back to Environment
```

*C. elegans* neurons do not spike; they are isopotential, graded-release cells.
The model is the standard leaky-integrator formulation of Wicks et al. (1996)
and Kunert et al. (2014):

```
C dV_i/dt = -G_leak(V_i - E_leak,i) - I_gap_i - I_syn_i + I_ext_i
I_gap_i   = sum_j g_gap * Gg[i,j] * (V_i - V_j)
I_syn_i   = sum_j g_syn * Gs[i,j] * s_j * (V_i - E_ij)
ds_j/dt   = a_r * phi(V_j) * (1 - s_j) - a_d * s_j
```

Integration is exponential Euler, unconditionally stable, at roughly 4x real
time on a laptop. The per-cell activation threshold is solved as a linear system
so the resting network sits at its own equilibrium. Passive properties follow
patch clamp: a 0.25 nS leak for a 4 GOhm input resistance, and 5 pS per
gap-junction contact, since measured coupling is reported whole-cell across all
contacts a pair shares. Leak reversals are solved so the network equilibrium
lands on measured resting potentials.

| Module | Role |
|---|---|
| `worm/genome.py` | annotation, gene lookup, knockouts, expression |
| `worm/connectome.py` | wiring to conductance matrices, network layout |
| `worm/nervous_system.py` | graded-potential dynamics, ablation |
| `worm/sensory.py` | environment to current, mechanosensory receptive fields |
| `worm/body.py` | oscillator, muscle to curvature, low-Reynolds locomotion |
| `worm/lifecycle.py` | embryo, larval stages, dauer, feeding, ageing, death |
| `worm/simulation.py` | closed loop, escape-response state machine |
| `worm/assays.py` | standard assays with reference values |
| `worm/validate.py` | 24 checks: 19 behavioural, 5 consistency, gaps as expected failures |
| `worm/parameters.py` | audited parameter registry with provenance tags |
| `worm/server.py`, `viewer/` | live browser viewer |

---

## Data

| Dataset | Source | Contents |
|---|---|---|
| Genome annotation | NCBI RefSeq GCF_000002985.6 (WBcel235) | 46,926 genes, 19,983 protein-coding |
| Connectome | Cook et al. 2019, via OpenWorm ConnectomeToolbox | 448 cells, 7,379 edges |
| Neuron metadata | OpenWorm owmeta | type and transmitter for 302 neurons |
| Expression | CeNGEN via wormneuroatlas | 128 neuron classes x 13,669 genes |
| Cell classification | WormAtlas | lineage and anatomical class |

The datasets disagree in specific ways, each handled in code where it arises:
gap junctions are listed in both directions, so the matrix is filled directly
rather than symmetrised; cell naming differs between sources and is normalised;
owmeta leaves 57 neurons unannotated, which come from the systematic transmitter
atlases; and cells that stain GABA-positive but lack `unc-25` take GABA up
rather than making it, so are not modelled as GABAergic.

---

## Validation

`worm validate` runs 24 checks against published measurements: 19 behavioural
(the animal is run and measured) and 5 consistency checks (parameter and data
invariants). 22 pass. The 2 failures are known gaps, reported as expected
failures and tracked, not hidden: without spontaneous reversals there are no
pirouettes, so chemotaxis cannot yet discriminate salt-blind mutants
([issue #6](https://github.com/vdmkenny/celeganssim/issues/6)), and the
fixed-frequency oscillator cannot adapt gait to the medium
([issue #10](https://github.com/vdmkenny/celeganssim/issues/10)). When those
issues are fixed, the checks flip from XFAIL to PASS and the suite will say so.

| Check | Model | Published |
|---|---|---|
| Input resistance | 4.0 GOhm | 1.6 to 8 GOhm |
| Membrane time constant | 6.0 ms | 3 to 10 ms |
| AVAL-AVAR gap coupling | 90 pS (1.6x high) | 56 pS |
| VA5 / VB6 resting potential | -71.7 / -53.2 mV | -71.7 / -53.2 mV |
| Crawling speed | 0.239 mm/s | 0.20 +/- 0.04 mm/s |
| Undulation amplitude | 21.2% body length | 19.3% |
| Embryogenesis | 14.2 h | 14.2 h |
| Hatch to adult | 50.8 h | 50.67 +/- 1.95 h |
| Adult lifespan | mean 15.9 d, sd 3.3 d over a cohort | 15.2 +/- 3.6 d |
| Self-fertile brood | 300 | ~327, sperm-limited |
| `daf-2` lifespan | 31.3 d (2x the animal's own draw) | ~29.5 d |

The other behavioural checks cover touch responses by body position, the
`mec-4`/`mec-10` dissociation, the `unc-25`/`unc-47`/`unc-49` shrinker class,
`unc-13` paralysis, the opposing `goa-1` and `egl-30` phenotypes, `tdc-1`
omega-turn loss, and command-interneuron ablation. The consistency checks pin
the `daf-16` epistasis, `che-1` modality gating, developmental timings, and
the sperm-limited brood.

Every parameter the model is told is collected in one audited registry
(`worm params`), tagged measured, published, tuned or scripted; the scripted
tags are the ones the open issues exist to delete.

Independently, the 26 GABAergic neurons the pipeline derives from transmitter
data match the known count exactly (6 DD + 13 VD + 4 RME + AVL + DVB + RIS).

---

## Scope and limitations

**Measured data:** connectivity and synapse counts, transmitter identity, gene
expression, receptive field extents, passive membrane properties, resting
potentials, developmental timings, body sizes, brood size, lifespan.

**Standard published models:** graded-potential dynamics (Wicks 1996 / Kunert
2014), resistive force theory for low-Reynolds locomotion.

**Approximations:**

- **The locomotor rhythm is imposed, not emergent.** The network sets its
  amplitude, direction and frequency, but the oscillation itself is modelled.
  [docs/emergent-cpg.md](docs/emergent-cpg.md) sets out what an emergent version
  requires. No published model produces C. elegans locomotion emergently from
  the connectome.
- **Synapse signs are inferred** from transmitter identity, with documented
  exceptions overridden per edge. This is genuinely uncertain in the literature.
- **The genome sequence is not load-bearing.** The annotation drives gene lookup
  and knockouts, but the 100 Mb of sequence yields chromosome lengths and GC
  content. Mapping a gene to what its loss *does* is a curated table.
- **Escape is a state machine.** Reversal and omega turn have separable motor
  pathways; the network decides when each fires.
- **Not modelled:** pharynx, gonad, intestine, embryonic lineage, hydrodynamics
  beyond RFT. One animal, so crowding is a scalar.

---

## References

**Connectome**
- White, Southgate, Thomson & Brenner (1986), Phil. Trans. R. Soc. B 314:1
- [Cook et al. (2019), Nature 571:63](https://doi.org/10.1038/s41586-019-1352-7)
- [Witvliet et al. (2021), Nature 596:257](https://doi.org/10.1038/s41586-021-03778-8)
- [OpenWorm ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox) · [c302](https://github.com/openworm/c302) · [owmeta](https://github.com/openworm/owmeta)

**Neural dynamics and electrophysiology**
- [Wicks, Roehrig & Rankin (1996), J. Neurosci. 16:4017](https://doi.org/10.1523/JNEUROSCI.16-12-04017.1996)
- [Kunert, Shlizerman & Kutz (2014), Phys. Rev. E 89:052805](https://doi.org/10.1103/PhysRevE.89.052805)
- Goodman, Hall, Avery & Lockery (1998), Neuron 20:763 (input resistance, capacitance)
- [Liu, Chen & Wang (2014), Nat. Commun. 5:5155](https://doi.org/10.1038/ncomms6155) (motor neuron resting potentials)
- [Liu, Chen & Wang (2020), Nat. Commun. 11:5076](https://doi.org/10.1038/s41467-020-18893-9) (gap junction conductance)
- [Shindou et al. (2019), Sci. Rep. 9:3430](https://doi.org/10.1038/s41598-019-40158-9)
- Jospin et al. (2002), J. Cell Biol. 159:337 (calcium reversal)
- [C. elegans Neural Interactome](https://github.com/shlizee/C-elegans-Neural-Interactome)

**Transmitters and expression**
- [Pereira et al. (2015), eLife 4:e12432](https://doi.org/10.7554/eLife.12432) (cholinergic)
- [Serrano-Saiz et al. (2013), Cell 155:659](https://doi.org/10.1016/j.cell.2013.09.052) (glutamatergic)
- [Gendrel, Atlas & Hobert (2016), eLife 5:e17686](https://doi.org/10.7554/eLife.17686) (GABAergic)
- [Taylor et al. (2021), Cell 184:4329](https://doi.org/10.1016/j.cell.2021.06.023) (CeNGEN)
- [Randi et al. (2023), Nature 623:406](https://doi.org/10.1038/s41586-023-06683-4) (wormneuroatlas)

**Locomotion**
- [Boyle, Berri & Cohen (2012), Front. Comput. Neurosci. 6:10](https://doi.org/10.3389/fncom.2012.00010)
- [Wen et al. (2012), Neuron 76:750](https://doi.org/10.1016/j.neuron.2012.08.039) (proprioceptive coupling)
- [Gao et al. (2018), eLife 7:e29915](https://doi.org/10.7554/eLife.29915) (A-class oscillators)
- [Kawano et al. (2011), Neuron 72:572](https://doi.org/10.1016/j.neuron.2011.09.005)
- [Deng et al. (2021), eNeuro 8:ENEURO.0241-20.2020](https://doi.org/10.1523/ENEURO.0241-20.2020)
- [Cronin et al. (2005), BMC Genet. 6:5](https://doi.org/10.1186/1471-2156-6-5) (gait metrics)
- [Fang-Yen et al. (2010), PNAS 107:20323](https://doi.org/10.1073/pnas.1003016107)

**Mechanosensation and escape**
- Chalfie & Sulston (1981), Dev. Biol. 82:358 · Chalfie et al. (1985), J. Neurosci. 5:956
- [Arnadottir et al. (2011), J. Neurosci. 31:12695](https://doi.org/10.1523/JNEUROSCI.4580-10.2011)
- [Li, Kang, Piggott, Feng & Xu (2011), Nat. Commun. 2:315](https://doi.org/10.1038/ncomms1308) (harsh touch)
- [Husson, Steuer Costa et al. (2012), Curr. Biol. 22:743](https://doi.org/10.1016/j.cub.2012.02.066)
- [Donnelly et al. (2013), PLoS Biol. 11:e1001529](https://doi.org/10.1371/journal.pbio.1001529)
- [Pirri et al. (2009), Neuron 62:526](https://doi.org/10.1016/j.neuron.2009.04.013)

**Chemosensation and navigation**
- [Chalasani et al. (2007), Nature 450:63](https://doi.org/10.1038/nature06292)
- [Pierce-Shimomura, Morse & Lockery (1999), J. Neurosci. 19:9557](https://doi.org/10.1523/JNEUROSCI.19-21-09557.1999)
- [Iino & Yoshida (2009), J. Neurosci. 29:5370](https://doi.org/10.1523/JNEUROSCI.3633-08.2009)
- [Gray, Hill & Bargmann (2005), PNAS 102:3184](https://doi.org/10.1073/pnas.0409009101)
- [Sawin, Ranganathan & Horvitz (2000), Neuron 26:619](https://doi.org/10.1016/S0896-6273(00)81199-X)

**Genetics and phenotypes**
- Brenner (1974), Genetics 77:71
- [Jin, Jorgensen, Hartwieg & Horvitz (1999), J. Neurosci. 19:539](https://doi.org/10.1523/JNEUROSCI.19-02-00539.1999)
- [Bamber et al. (1999), J. Neurosci. 19:5348](https://doi.org/10.1523/JNEUROSCI.19-13-05348.1999)
- Richmond, Davis & Jorgensen (1999), Nat. Neurosci. 2:959
- [WormBook: GABA](https://www.ncbi.nlm.nih.gov/books/NBK19793/) · [Acetylcholine](https://www.ncbi.nlm.nih.gov/books/NBK19736/) · [Mechanosensation](https://www.ncbi.nlm.nih.gov/books/NBK19654/) · [Chemosensation](https://www.ncbi.nlm.nih.gov/books/NBK19746/)

**Life cycle and ageing**
- Sulston, Schierenberg, White & Thomson (1983), Dev. Biol. 100:64
- [Faerberg, Gurarie & Ruvinsky (2022), BMC Biol. 20:87](https://doi.org/10.1186/s12915-022-01282-7)
- Cassada & Russell (1975), Dev. Biol. 46:326 · Golden & Riddle (1984), Dev. Biol. 102:368
- Hodgkin & Barnes (1991), Proc. R. Soc. B 246:19 · Ward & Carrel (1979)
- [Huang, Xiong & Kornfeld (2004), PNAS 101:8084](https://doi.org/10.1073/pnas.0400848101)
- Herndon et al. (2002), Nature 419:808
- Kenyon et al. (1993), Nature 366:461 · [Lakowski & Hekimi (1998), PNAS 95:13091](https://doi.org/10.1073/pnas.95.22.13091)
- [Baugh (2013), Genetics 194:539](https://doi.org/10.1534/genetics.113.150847)

**Reference resources**
- [WormBase](https://wormbase.org) · [WormAtlas](https://www.wormatlas.org) · [WormBook](https://www.ncbi.nlm.nih.gov/books/NBK19559/)

---

## Licence

MIT, code only. No datasets are redistributed; `fetch_data.py` pulls each from
its publisher and they remain under their own terms. If you use this for
research, cite the primary sources above rather than this repository.
