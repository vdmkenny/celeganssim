# celeganssim

A whole-organism simulator for *Caenorhabditis elegans*, built on the real genome
and the real wiring diagram.

![The live viewer: an adult worm crawling, with dorsal and ventral body-wall muscle activation drawn along the body, and live neuron traces](docs/screenshot.png)

*Arena view. Teal is dorsal muscle, purple is ventral; the alternation running down
the body is the undulatory wave that moves the animal. Pick a tool, then click
anywhere on the worm to poke it.*

Its nervous system is the actual electron-microscopy connectome: 300 neurons and
95 body-wall muscles, 4,681 chemical synapse pairs and 1,342 gap junctions. Those
cells are integrated as coupled differential equations, their output drives
muscle, muscle drives a body, and the body moves through a fluid. It develops from
a fertilised egg, feeds, ages and dies.

**On the genome, precisely.** The WBcel235 *annotation* is load-bearing: gene
lookup, real WormBase IDs and coordinates, and the knockout system are all driven
by it. The 100 Mb of *sequence* is not. It yields chromosome lengths and GC
content and nothing else, and going from base pairs to behaviour is not
computable for any animal. This is a behavioural model with a real annotation
layer, not a genome-to-phenotype engine.

What the genetics layer *does* rest on is measured **single-cell expression**.
CeNGEN scRNA-seq (Taylor et al. 2021, via the wormneuroatlas API) says which of
the 128 neuron classes transcribe each gene, so a knockout reaches exactly the
cells that express it rather than scaling a transmitter everywhere:

| Gene | Cells it is measured in |
|---|---|
| `mec-4` | ALML, ALMR, AVM, PLML, PLMR, PVM (the six touch receptor neurons) |
| `che-1` | ASEL, ASER |
| `cat-2` | ADE, CEP and PDE (the eight dopaminergic neurons) |
| `unc-25` | 28 cells: DD1-6, VD1-13, RME x4, AVL, DVB, RIS, RIB |

Two things that were previously hand-written now fall out of the data. `mec-4` is
simply not expressed in PVD or FLP, so harsh touch survives a `mec-4` knockout
without a rule saying it should. And AVA and AVB, which stain GABA-positive but
lack `unc-25`, are correctly left untouched by an `unc-25` knockout.

The mapping from a gene to *what its loss does* is still a curated table of 31
loci, and that part is modelled, not derived.

```bash
python -m worm serve                    # live viewer at http://127.0.0.1:8080
python -m worm validate                 # 21 checks against published results
python -m worm assay chemotaxis         # run a standard behavioural assay
python -m worm gene unc-25              # look up a locus
```

---

## What you can do with it

### Poke it anywhere and watch the signal propagate

Touch is a position along the body, not three buttons. The six touch receptor
neurons tile the animal with overlapping graded receptive fields, so where you
prod it changes which cells hear it. Click the head and it reverses; click the
tail and it speeds up. Neither response is written down anywhere: current goes
into the real touch neurons and the behaviour comes out of the connectome.

Inspecting the wiring shows why. PLM has **no chemical output at all** in the Cook
dataset; it reaches the forward command interneuron PVC purely through gap
junctions. That is the whole reason tail touch accelerates rather than reverses.

Harsh prodding is a separate channel through separate cells. FLP covers the head
and is reversal-biased in the connectome (145 contacts onto the backward command
pool against 50 onto the forward one); PVD tiles the rest of the body and is
near-balanced, favouring forward. So harsh touch keeps the same directional logic
while being MEC-4 independent, which is why it survives in `mec-4` mutants.

### Knock out a gene and get the documented mutant

Thirty-one real loci are wired to the subsystems their products implement:

```bash
python -m worm run --knockout unc-25 --seconds 30
python -m worm batch --genotype wild-type --genotype unc-25 --genotype goa-1
```

`unc-25` encodes glutamic acid decarboxylase, so knocking it out removes GABA
synthesis, which removes inhibition at the neuromuscular junction, which makes
dorsal and ventral muscle co-contract. The animal shortens and its bends go
shallow. That is the shrinker phenotype, reached through the same causal chain as
in the animal rather than by special-casing the gene.

`daf-2` roughly doubles lifespan, and `daf-2; daf-16` does not, because the
insulin arm acts only through DAF-16. `eat-2` extends life and keeps doing so
without `daf-16`, because dietary restriction is a different route.

### Laser-ablate any cell

Alt-click a neuron in the network view and it dies: it stops sending, stops
receiving, and the surviving network's resting thresholds are re-solved around the
loss. This is how the touch circuit was mapped in the first place.

| Ablation | Result |
|---|---|
| AVB + PVC | forward drive falls to exactly 0.00, reversals intact |
| AVA + AVD + AVE | touch-evoked reversal abolished |
| ALM + AVM | anterior touch sensitivity gone |

### Watch the whole network live

![Network view with AVAL pinned, showing all 106 of its connections highlighted while the rest of the graph dims](docs/network.png)

*Network view. All 448 cells laid out sensory to interneuron to motor to muscle,
ordered by barycentre to cut edge crossings. Node brightness tracks live drive.
Here AVAL is pinned, showing all 106 of its strong connections and its top
partners with weights straight from the connectome.*

Hover any cell for its transmitter, live drive and strongest partners. Pin it to
keep it. Watch it in the trace panel to see its activation over the last 30
seconds, the way a calcium recording reads.

### Run the standard assays

Each reports the metric its original paper reports, and carries its own reference
value so results sit next to published numbers.

```bash
python -m worm assay all
python -m worm assay basal-slowing --knockout cat-2
python -m worm assay touch-response --ablate ALML --ablate ALMR
```

| Assay | Measures |
|---|---|
| `chemotaxis` | chemotaxis index on a salt gradient |
| `thermotaxis` | drift along a thermal gradient relative to remembered temperature |
| `touch-habituation` | response decrement over repeated taps (Rankin paradigm) |
| `basal-slowing` | speed drop on encountering food |
| `touch-response` | reversal probability by body position |
| `lifespan` | egg to death, with brood size and reproductive period |

Wild type slows 15.6% on food; `cat-2`, which cannot make dopamine, shows none.
That is the Sawin et al. dissociation.

### Get the data out

```bash
python -m worm export run.csv --seconds 120 --neuron AVAL --neuron AVBL --salt
python -m worm batch --genotype wild-type --genotype unc-13 --replicates 5 --json
```

CSV or JSON of trajectory, behavioural state, developmental stage and named
neuron activations, with fixed seeds for reproducibility.

---

## Why this is more than an animation

**The wiring is measured, not invented.** Every synapse comes from Cook et al.
(2019). Nothing in the connectivity was authored.

**The neurons are solved, not scripted.** *C. elegans* neurons do not spike; they
are isopotential, graded-release cells. So the model is the standard
leaky-integrator formulation of Wicks et al. (1996) and Kunert et al. (2014):

```
C dV_i/dt = -G_leak(V_i - E_cell) - I_gap_i - I_syn_i + I_ext_i
I_gap_i = Σ_j g_gap·Gg[i,j]·(V_i - V_j)
I_syn_i = Σ_j g_syn·Gs[i,j]·s_j·(V_i - E_ij)
ds_j/dt = a_r·φ(V_j)·(1 - s_j) - a_d·s_j
```

448 cells integrated every step. The per-cell threshold `V_th` is not a tuned
parameter: it is solved as a linear system so the resting network sits at its own
equilibrium with every synapse half-activated. Without that, a network with this
much heterogeneous recurrence saturates immediately.

**It makes checkable predictions.** `python -m worm validate` runs 21 checks
against published results, and all 21 pass.

| Check | Model | Published |
|---|---|---|
| Crawling speed | 0.239 mm/s | 0.20 ± 0.04 mm/s |
| Undulation amplitude | 21.2% body length | 19.3% |
| Embryogenesis | 14.2 h | 14.2 h |
| Hatch to adult | 50.8 h | 50.67 ± 1.95 h |
| Adult lifespan | 15.2 d | 15.2 ± 3.6 d |
| Self-fertile brood | 300 | ~327, sperm-limited |
| `daf-2` lifespan | 30.4 d | ~29.5 d |

Independently, the 26 GABAergic neurons the pipeline derives from the transmitter
data match the known count exactly (6 DD + 13 VD + 4 RME + AVL + DVB + RIS).

---

## Architecture

```
Environment          gradients, bacterial lawn, temperature, oxygen, pokes
      │
      ▼
SensorySystem        modality → named neurons, via receptive fields, gated by genes
      │
      ▼
NervousSystem        448 cells, graded-potential ODEs over the real connectome
      │
      ▼
Simulation           reads command interneurons → behavioural state machine
      │
      ▼
Body                 ventral-cord oscillator → 95 muscles → viscoelastic body
      │
      ▼
                     resistive force theory → movement → back to Environment
```

| Module | What it does |
|---|---|
| `worm/genome.py` | WBcel235 annotation; gene lookup; knockouts and their effects |
| `worm/connectome.py` | Cook 2019 wiring → conductance matrices; layered layout |
| `worm/nervous_system.py` | Graded-potential dynamics; laser ablation |
| `worm/sensory.py` | Environment → current, with mechanosensory receptive fields |
| `worm/body.py` | Oscillator chain, muscle → curvature, low-Reynolds locomotion |
| `worm/lifecycle.py` | Embryo, larval stages, dauer, feeding, senescence, death |
| `worm/simulation.py` | Closed loop and escape-response state machine |
| `worm/assays.py` | Standard behavioural assays with published reference values |
| `worm/validate.py` | 21 checks against the literature |
| `worm/server.py` + `viewer/` | Live browser viewer (standard library only) |

---

## The data

Everything under `data/raw/` is downloaded from primary sources by
`scripts/fetch_data.py`; `scripts/build_data.py` parses it into `data/processed/`.
Nothing is hand-authored.

| Dataset | Source | Contents |
|---|---|---|
| Genome | NCBI RefSeq **GCF_000002985.6 (WBcel235)** | 100,286,401 bp; I–V, X + mtDNA |
| Annotation | matching RefSeq GFF | 46,926 genes, **19,983 protein-coding** |
| Connectome | **Cook et al. 2019**, via OpenWorm ConnectomeToolbox | 448 cells, 7,379 edges |
| Neuron metadata | **OpenWorm owmeta** | type and transmitter for 302 neurons |
| Cell classification | **WormAtlas** | lineage and anatomical class |

### Data problems found and fixed

Real data means real inconsistencies. Each is documented in code where it is
handled:

- **Gap junctions were double-counted.** The edgelist lists each electrical
  connection in *both* directions (1,339 of 1,359 unordered pairs), so
  symmetrising on load doubled every weight.
- **Zero-padding mismatch.** The connectome writes `DA01`, owmeta writes `DA1`.
  This silently dropped all 75 ventral-cord motor neurons, the cells that drive
  locomotion.
- **57 neurons had no transmitter annotation**, including every command
  interneuron. Filled from the systematic atlases (Pereira 2015; Serrano-Saiz
  2013; Gendrel 2016).
- **Glutamate polarity is target-dependent.** A single global sign gets the touch
  circuit backwards, so documented inhibitory glutamatergic connections are
  overridden per edge.
- **ALA, AVF, AVA, AVB and SMD stain GABA-positive but lack `unc-25`.** They take
  GABA up rather than making it, and are deliberately not modelled as GABAergic.

---

## What is modelled vs. approximated

**Real, measured data.** Genome sequence and annotation. Connectivity and synapse
counts. Transmitter identity. Motor neuron class membership. Receptive field
extents. Developmental timings, body sizes, brood size, lifespan.

**Standard published models.** Graded-potential dynamics (Wicks 1996 / Kunert
2014). Resistive force theory for low-Reynolds locomotion.

**Deliberate approximations, stated plainly:**

- **The locomotor rhythm is generated explicitly**, not by the connectome. Its
  amplitude, direction, frequency and dorsoventral balance are all set by the
  network, but the oscillator itself is modelled. This is the standard choice in
  whole-animal models (Boyle & Cohen 2012; Olivares et al. 2021): the wiring
  diagram alone does not determine the ionic conductances needed to make a
  302-cell network oscillate. **If you want emergent locomotion from connectivity
  alone, this is not that.** Note the field's current view is that GABA is not
  required for dorsoventral alternation, since `unc-25` nulls still undulate, so
  GABA is modelled as a modulator of amplitude and speed (Wen et al. 2012; Deng
  et al. 2021).
- **Synapse signs are inferred from transmitter identity**, with documented
  exceptions. This is genuinely uncertain: Wicks et al. could not reliably predict
  the polarity of AVA or DVA at any significance level.
- **Escape is a state machine.** Reversal and omega turn have separable final
  motor pathways, so they are separate states; the network decides when.
- **No pharynx, gonad, intestine or embryonic lineage.** No hydrodynamics beyond
  RFT. One animal, so crowding is a scalar rather than other worms.

---

## Sources

**Genome and data**
- [*C. elegans* Sequencing Consortium (1998), Science 282:2012](https://doi.org/10.1126/science.282.5396.2012)
- [NCBI RefSeq GCF_000002985.6, WBcel235](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000002985.6/)
- [WormBase](https://wormbase.org) · [WormAtlas](https://www.wormatlas.org) · [WormBook](https://www.ncbi.nlm.nih.gov/books/NBK19559/)

**Connectome**
- White, Southgate, Thomson & Brenner (1986), Phil. Trans. R. Soc. B 314:1
- [Cook et al. (2019), *Whole-animal connectomes of both C. elegans sexes*, Nature 571:63](https://doi.org/10.1038/s41586-019-1352-7)
- [Witvliet et al. (2021), *Connectomes across development*, Nature 596:257](https://doi.org/10.1038/s41586-021-03778-8)
- [OpenWorm ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox) · [c302](https://github.com/openworm/c302) · [owmeta](https://github.com/openworm/owmeta)

**Neural dynamics**
- [Wicks, Roehrig & Rankin (1996), J. Neurosci. 16:4017](https://doi.org/10.1523/JNEUROSCI.16-12-04017.1996)
- [Kunert, Shlizerman & Kutz (2014), Phys. Rev. E 89:052805](https://doi.org/10.1103/PhysRevE.89.052805)
- [C. elegans Neural Interactome](https://github.com/shlizee/C-elegans-Neural-Interactome)

**Neurotransmitters**
- [Pereira et al. (2015), eLife 4:e12432](https://doi.org/10.7554/eLife.12432) (cholinergic map)
- [Serrano-Saiz et al. (2013), Cell 155:659](https://doi.org/10.1016/j.cell.2013.09.052) (glutamatergic map)
- [Gendrel, Atlas & Hobert (2016), eLife 5:e17686](https://doi.org/10.7554/eLife.17686) (GABAergic map)
- [Beg & Jorgensen (2003), Nat. Neurosci. 6:1145](https://doi.org/10.1038/nn1136) (excitatory GABA via EXP-1)

**Locomotion and biomechanics**
- Gray & Lissmann (1964), J. Exp. Biol. 41:135
- [Boyle, Berri & Cohen (2012), Front. Comput. Neurosci. 6:10](https://doi.org/10.3389/fncom.2012.00010)
- [Wen et al. (2012), Neuron 76:750](https://doi.org/10.1016/j.neuron.2012.08.039) (proprioceptive coupling)
- [Deng et al. (2021), eNeuro 8:ENEURO.0241-20.2020](https://doi.org/10.1523/ENEURO.0241-20.2020) (GABA and fast undulation)
- [Cronin et al. (2005), BMC Genet. 6:5](https://doi.org/10.1186/1471-2156-6-5) (gait metrics)
- [Fang-Yen et al. (2010), PNAS 107:20323](https://doi.org/10.1073/pnas.1003016107) (gait adaptation)
- [Gao et al. (2018), eLife 7:e29915](https://doi.org/10.7554/eLife.29915) (A-class oscillators)
- [Kawano et al. (2011), Neuron 72:572](https://doi.org/10.1016/j.neuron.2011.09.005) (gap-junction bias)

**Mechanosensation and escape**
- Chalfie & Sulston (1981), Dev. Biol. 82:358 · Chalfie et al. (1985), J. Neurosci. 5:956
- [Árnadóttir et al. (2011), J. Neurosci. 31:12695](https://doi.org/10.1523/JNEUROSCI.4580-10.2011) (mec-10 is partial)
- [Li, Kang, Piggott, Feng & Xu (2011), Nat. Commun. 2:315](https://doi.org/10.1038/ncomms1308) (harsh touch)
- [Husson, Steuer Costa et al. (2012), Curr. Biol. 22:743](https://doi.org/10.1016/j.cub.2012.02.066) (PVD drives forward)
- [Donnelly et al. (2013), PLoS Biol. 11:e1001529](https://doi.org/10.1371/journal.pbio.1001529) (tyramine and the omega turn)
- [Pirri et al. (2009), Neuron 62:526](https://doi.org/10.1016/j.neuron.2009.04.013) (LGC-55)

**Chemosensation and navigation**
- [Chalasani et al. (2007), Nature 450:63](https://doi.org/10.1038/nature06292) (AWC circuit)
- [Pierce-Shimomura, Morse & Lockery (1999), J. Neurosci. 19:9557](https://doi.org/10.1523/JNEUROSCI.19-21-09557.1999) (pirouettes)
- [Iino & Yoshida (2009), J. Neurosci. 29:5370](https://doi.org/10.1523/JNEUROSCI.3633-08.2009) (two strategies)
- [Gray, Hill & Bargmann (2005), PNAS 102:3184](https://doi.org/10.1073/pnas.0409009101) (navigation circuit)
- [Sawin, Ranganathan & Horvitz (2000), Neuron 26:619](https://doi.org/10.1016/S0896-6273(00)81199-X) (slowing responses)

**Genetics and phenotypes**
- Brenner (1974), Genetics 77:71
- [Jin, Jorgensen, Hartwieg & Horvitz (1999), J. Neurosci. 19:539](https://doi.org/10.1523/JNEUROSCI.19-02-00539.1999) (unc-25)
- [Bamber et al. (1999), J. Neurosci. 19:5348](https://doi.org/10.1523/JNEUROSCI.19-13-05348.1999) (unc-49)
- Richmond, Davis & Jorgensen (1999), Nat. Neurosci. 2:959 (unc-13)
- [WormBook: GABA](https://www.ncbi.nlm.nih.gov/books/NBK19793/) · [Acetylcholine](https://www.ncbi.nlm.nih.gov/books/NBK19736/) · [Mechanosensation](https://www.ncbi.nlm.nih.gov/books/NBK19654/) · [Chemosensation](https://www.ncbi.nlm.nih.gov/books/NBK19746/)

**Life cycle and ageing**
- Sulston, Schierenberg, White & Thomson (1983), Dev. Biol. 100:64 (embryonic lineage)
- Byerly, Cassada & Russell (1976), Dev. Biol. 51:23
- [Faerberg, Gurarie & Ruvinsky (2022), BMC Biol. 20:87](https://doi.org/10.1186/s12915-022-01282-7) (larval timings)
- Cassada & Russell (1975), Dev. Biol. 46:326 · Golden & Riddle (1984), Dev. Biol. 102:368 (dauer)
- Hodgkin & Barnes (1991), Proc. R. Soc. B 246:19 (brood size) · Ward & Carrel (1979) (sperm limitation)
- [Huang, Xiong & Kornfeld (2004), PNAS 101:8084](https://doi.org/10.1073/pnas.0400848101) (lifespan, decline)
- Herndon et al. (2002), Nature 419:808 (movement classes)
- Kenyon et al. (1993), Nature 366:461 (daf-2) · [Lakowski & Hekimi (1998), PNAS 95:13091](https://doi.org/10.1073/pnas.95.22.13091) (eat-2)
- [Baugh (2013), Genetics 194:539](https://doi.org/10.1534/genetics.113.150847) (L1 arrest)

---

## Setup

```bash
python -m venv .venv && .venv/bin/pip install numpy
.venv/bin/python scripts/fetch_data.py     # downloads ~43 MB, then builds
.venv/bin/python -m worm serve
```

The simulator core needs only **numpy**. The viewer is Python standard library and
vanilla JavaScript, with no build step. Datasets are not vendored:
`fetch_data.py` pulls each one from its publisher, verifies it, and runs the
build. Re-running it only fetches what is missing.

Performance: the 448-cell network runs at roughly 4x real time on a laptop. The
network view bakes its static edge layer into an offscreen canvas and batches the
rest, so a repaint costs about 0.6 ms.

If you want to extend the neural model, [**Brian2**](https://brian2.readthedocs.io)
is a good fit and installs cleanly here. It is not used by default because it is a
spiking-network simulator by design and this graded system solves fine in numpy.
[NEURON](https://neuron.yale.edu) and [NetPyNE](http://netpyne.org) are the
options for multi-compartment morphology; OpenWorm's
[c302](https://github.com/openworm/c302) generates NeuroML models of this same
network.

---

## License and provenance

Code here is original. No datasets are redistributed; `fetch_data.py` pulls each
one from its publisher, so they stay under their own terms. WormBase and WormAtlas
data is freely available for academic use, and the OpenWorm projects are
MIT-licensed. If you use this for anything real, cite the primary sources above
rather than this repository.
