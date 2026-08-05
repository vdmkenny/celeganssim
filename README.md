# celeganssim

A whole-organism simulator for *Caenorhabditis elegans*, built on the real genome
and the real wiring diagram.

![The live viewer: an adult worm crawling, with dorsal and ventral body-wall
muscle activation drawn along the body, the travelling wave of muscle drive from
head to tail, and live command-neuron
activity](docs/screenshot.png)

*The live viewer. Teal is dorsal muscle, purple is ventral; the alternation you
can see running down the body is the undulatory wave that moves the animal. The
muscle drive panel shows the same wave as a strip from head to tail, and the
command neurons below it are the AVB/PVC and AVA/AVD/AVE pools whose balance
decides whether the worm goes forwards or backwards.*

The animal's 100,286,401 bp genome is loaded from the actual reference assembly.
Its nervous system is the actual electron-microscopy connectome: 300 neurons and
95 body-wall muscles, 4,681 chemical synapse pairs and 1,342 gap junctions. Those
neurons are integrated as coupled differential equations, their output drives
muscle, muscle drives a body, and the body moves through a fluid. You can watch it
in a browser, poke it, feed it, and knock out genes by name.

```bash
python -m worm serve      # live viewer at http://127.0.0.1:8080
python -m worm validate   # check modelled phenotypes against the literature
python -m worm info       # genome + connectome summary
python -m worm gene unc-25
```

---

## Why this is more than an animation

An animation would be a sine wave drawn on a canvas. The distinction here is that
**the causal chain runs in the right direction, through real data, and it is
falsifiable.**

**1. The wiring is measured, not invented.** Every synapse comes from Cook et al.
(2019), the whole-animal EM reconstruction. Nothing in the connectivity was
authored. When you poke the worm's head, the signal propagates through the actual
anatomy. Inspecting the touch circuit in the loaded data shows PLM has *no chemical
output at all*. It reaches the forward command interneuron PVC purely through gap
junctions, which is precisely why posterior touch accelerates the animal forward
instead of reversing it. That behaviour is a consequence of the dataset, not
something coded in.

**2. The neurons are solved, not scripted.** *C. elegans* neurons do not spike;
they are isopotential, graded-release cells. So the model is the standard
leaky-integrator formulation of Wicks et al. (1996) and Kunert et al. (2014):

```
C dV_i/dt = -G_leak(V_i - E_cell) - I_gap_i - I_syn_i + I_ext_i
I_gap_i = Σ_j g_gap·Gg[i,j]·(V_i - V_j)
I_syn_i = Σ_j g_syn·Gs[i,j]·s_j·(V_i - E_ij)
ds_j/dt = a_r·φ(V_j)·(1 - s_j) - a_d·s_j
```

448 cells integrated every step. The per-cell activation threshold `V_th` is not a
tuned parameter. It is *solved* as a linear system so the resting network sits at
its own equilibrium with every synapse half-activated. Without that, a network with
this much heterogeneous recurrence saturates immediately.

**3. Genes are wired to the subsystems their products actually implement.** The
genome is not decoration. `unc-25` encodes glutamic acid decarboxylase, so knocking
it out removes GABA synthesis, which removes inhibition at the neuromuscular
junction, which makes dorsal and ventral muscle co-contract, and the simulated
animal shortens and flattens its bends. That is the shrinker phenotype, arrived at
through the same causal chain as in the animal. Each of the 27 modelled genes is a
real locus with real coordinates in the assembly, and `python -m worm gene unc-25`
will print them.

**4. It makes checkable predictions.** `python -m worm validate` runs 13 checks
against published results. Wild-type gait lands at **0.24 mm/s with 21.5% body-length
undulation amplitude** against a literature reference of 0.20 ± 0.04 mm/s and 19.3%
(Cronin et al. 2005). Development runs **L1 → L2 at 11.3 h, L3 at 19.1 h, L4 at
27.1 h, adult at ~38 h** against published 12/20/28/38 h. Independently, the 26
GABAergic neurons the pipeline derives from the transmitter data match the known
count exactly (6 DD + 13 VD + 4 RME + AVL + DVB + RIS).

The honest counterpart: **this is a behavioural model, not a molecular one**, and
Section "What is modelled vs. approximated" below states exactly where the seams are.

---

## Architecture

```
Environment          gradients, bacterial lawn, temperature, oxygen, pokes
      │
      ▼
SensorySystem        modality → named sensory neurons, gated by transduction genes
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
| `worm/genome.py` | WBcel235 annotation; gene lookup; knockouts and their subsystem effects |
| `worm/connectome.py` | Cook 2019 wiring → conductance matrices; per-edge synaptic polarity |
| `worm/nervous_system.py` | Graded-potential network dynamics |
| `worm/sensory.py` | Environment → injected current in specific neurons |
| `worm/body.py` | Oscillator chain, muscle → curvature, low-Reynolds locomotion |
| `worm/lifecycle.py` | Larval stages, feeding, dauer, egg laying |
| `worm/simulation.py` | Closed loop and escape-response state machine |
| `worm/validate.py` | Phenotype checks against the literature |
| `worm/server.py` + `viewer/` | Live browser viewer (stdlib only, no dependencies) |

---

## The data

Everything under `data/raw/` is downloaded from primary sources; `scripts/build_data.py`
parses it into `data/processed/`. Nothing is hand-authored.

| Dataset | Source | Contents |
|---|---|---|
| Genome | NCBI RefSeq **GCF_000002985.6 (WBcel235)** | 100,286,401 bp; chromosomes I–V, X + mtDNA |
| Annotation | matching RefSeq GFF | 46,926 genes, of which **19,983 protein-coding** |
| Connectome | **Cook et al. 2019** hermaphrodite edgelist, via OpenWorm ConnectomeToolbox | 448 cells, 7,379 edges |
| Neuron metadata | **OpenWorm owmeta** curated database | type + neurotransmitter for 302 neurons |
| Cell classification | **WormAtlas** | lineage and anatomical class |

Verified chromosome lengths, straight from the assembly:

| | I | II | III | IV | V | X | MtDNA |
|---|---|---|---|---|---|---|---|
| bp | 15,072,434 | 15,279,421 | 13,783,801 | 17,493,829 | 20,924,180 | 17,718,942 | 13,794 |

### Data problems found and fixed

Working with real data means hitting real inconsistencies. These are documented in
code at the point they are handled:

- **Gap junctions were double-counted.** The Cook edgelist lists each electrical
  connection in *both* directions (1,339 of 1,359 unordered pairs). Symmetrising it
  on load doubled every gap-junction weight.
- **Zero-padding mismatch.** The connectome writes `DA01`, owmeta writes `DA1`. This
  silently dropped all 75 ventral-cord motor neurons (the cells that drive
  locomotion) until they were normalised.
- **57 neurons had no transmitter annotation**, including every command
  interneuron (AVA, AVB, AVD, AVE). Filled from the systematic transmitter atlases
  (Pereira et al. 2015; Serrano-Saiz et al. 2013; Gendrel et al. 2016).
- **owmeta truncates the ventral cord at index 9**, but the real animal has VA01–12,
  VB01–11, AS01–11, VD01–13. Filled from the textbook class assignment.
- **Glutamate polarity is target-dependent.** A single global sign gets the touch
  circuit backwards. Documented inhibitory glutamatergic connections are overridden
  per-edge.
- **ALA, AVF, AVA, AVB, SMD stain GABA-positive but lack `unc-25`**. They take GABA
  up rather than synthesising it, and are deliberately *not* modelled as GABAergic.

---

## Inputs you can drive

| Input | Sensory neurons | Notes |
|---|---|---|
| Salt gradient | ASEL / ASER | ASEL is the ON cell (up-steps), ASER the OFF cell (down-steps) |
| Volatile odour | AWA, AWC | AWC is a tonically active **OFF** cell, so odour *suppresses* it |
| Repellents | ASH, AWB | polymodal nociception |
| Temperature | AFD | responds relative to remembered cultivation temperature |
| Gentle touch, anterior | ALM, AVM | → reversal |
| Gentle touch, posterior | PLM | → forward acceleration |
| Nose touch | ASH, FLP, OLQ | → escape |
| Oxygen | URX/AQR/PQR (high), BAG (low) | URX is a tonic high-O₂ sensor |
| Bacteria | CEP, ADE (dopaminergic) | drives the basal slowing response |
| Crowding | n/a | ascaroside pheromone; drives the dauer decision |

Outputs: position and trajectory, behavioural state (forward / reversal / omega /
refractory), per-muscle activation for all 95 body-wall muscles, command-neuron
activity, body length, developmental stage, pumping rate, reserves, eggs laid.

---

## Life cycle and feeding

The post-embryonic programme runs L1 → L2 → L3 → L4 → adult on measured timings
(Byerly et al. 1976), with body length scaling from 0.25 mm to ~1.1 mm and feeding
the biomechanics directly. Two branch points are modelled:

- **L1 arrest**: hatching without food halts development outright.
- **Dauer**: late L1 assesses crowding, food and temperature; a bad verdict produces
  a non-feeding, stress-resistant alternative L3 that recovers when conditions improve.

Feeding is pharyngeal pumping (≈4.5 Hz well-fed, collapsing off food, potentiated by
serotonin so `tph-1` mutants eat less). Grazing depletes the lawn. Reserves are
calibrated so a starved adult survives days, not minutes, matching measured
starvation survival.

---

## Validation

```
$ python -m worm validate
```

13 checks, each stating its literature expectation and source. Current status:
**all 13 pass.** Representative results:

| Check | Result |
|---|---|
| Wild-type gait | 0.241 mm/s, 21.5% BL amplitude (ref: 0.20 ± 0.04 mm/s, 19.3%) |
| Anterior touch → reversal | ✅ |
| Posterior touch → no reversal, speeds up | ✅ |
| `mec-4` touch-insensitive | ✅ abolished |
| `mec-10` only *partially* insensitive | ✅ (classic full-Mec alleles are gain-of-function, not nulls) |
| `unc-25` shrinker | length 0.90, amplitude 18.0% vs 21.5%, shorter and shallower, still moving |
| `unc-47`, `unc-49` shrink likewise | ✅ |
| `egl-30(lf)` lethargic | 0.062 vs 0.241 mm/s, amplitude 8.1% vs 21.5% |
| `goa-1(lf)` loopy | amplitude 26.8% vs 21.5%, deeper bends |
| `che-1` loses salt but keeps odour | ✅ |
| `unc-13` near-paralysed | 0.026 vs 0.241 mm/s |
| `tdc-1` degrades the omega turn | ✅ |

---

## What is modelled vs. approximated

Stated plainly, because the difference matters:

**Real, measured data.** Genome sequence and annotation. Connectivity and synapse
counts. Neurotransmitter identity. Motor neuron class membership. Developmental
timings and body sizes.

**Standard published models.** Graded-potential neural dynamics (Wicks 1996 /
Kunert 2014). Resistive force theory for low-Reynolds-number locomotion.

**Deliberate approximations: the seams.**

- **The locomotor rhythm is generated explicitly**, not by the connectome. Its
  amplitude, direction, frequency and dorsoventral balance are all *set by* the
  network, but the oscillator itself is modelled. This is the standard choice in
  whole-animal models (Boyle & Cohen 2012; Olivares et al. 2021): the wiring diagram
  alone does not determine the ionic conductances needed to make a 302-cell network
  oscillate. Notably, the field's current view is that **GABA is not required for
  dorsoventral alternation** (`unc-25` nulls still undulate), so GABA is modelled as
  a modulator of amplitude and speed, not as the alternation mechanism (Wen et al.
  2012; Deng et al. 2021).
- **Synapse signs are inferred from transmitter identity**, with documented
  exceptions. This is genuinely uncertain in the literature; Wicks et al. could not
  reliably predict the polarity of AVA or DVA at any significance level.
- **Escape is a state machine.** Reversal and omega turn have separable final motor
  pathways in the animal, so they are separate states; the network decides *when*.
- **No pharynx, gonad, intestine or embryogenesis.** No hydrodynamics beyond RFT.
  Single animal, so crowding is a scalar rather than other worms.

---

## Sources

**Genome and data**
- Genome sequence: [*C. elegans* Sequencing Consortium (1998), Science 282:2012](https://doi.org/10.1126/science.282.5396.2012)
- Assembly: [NCBI RefSeq GCF_000002985.6, WBcel235](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000002985.6/)
- [WormBase](https://wormbase.org) · [WormAtlas](https://www.wormatlas.org) · [WormBook](https://www.ncbi.nlm.nih.gov/books/NBK19559/)

**Connectome**
- White, Southgate, Thomson & Brenner (1986), *The structure of the nervous system of the nematode C. elegans*, Phil. Trans. R. Soc. B 314:1. The original reconstruction
- [Cook et al. (2019), *Whole-animal connectomes of both C. elegans sexes*, Nature 571:63](https://doi.org/10.1038/s41586-019-1352-7)
- [Witvliet et al. (2021), *Connectomes across development*, Nature 596:257](https://doi.org/10.1038/s41586-021-03778-8)
- [OpenWorm ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox) · [c302](https://github.com/openworm/c302) · [owmeta](https://github.com/openworm/owmeta)

**Neural dynamics**
- [Wicks, Roehrig & Rankin (1996), *A dynamic network simulation of the nematode tap withdrawal circuit*, J. Neurosci. 16:4017](https://doi.org/10.1523/JNEUROSCI.16-12-04017.1996)
- [Kunert, Shlizerman & Kutz (2014), *Low-dimensional functionality of complex network dynamics*, Phys. Rev. E 89:052805](https://doi.org/10.1103/PhysRevE.89.052805)
- [C. elegans Neural Interactome](https://github.com/shlizee/C-elegans-Neural-Interactome), the reference implementation
- Goodman et al. (1998), *Active currents regulate sensitivity and dynamic range in C. elegans neurons*, Neuron 20:763

**Neurotransmitters**
- [Pereira et al. (2015), *A cellular and regulatory map of the cholinergic nervous system*, eLife 4:e12432](https://doi.org/10.7554/eLife.12432)
- [Serrano-Saiz et al. (2013), *Modular control of glutamatergic neuronal identity*, Cell 155:659](https://doi.org/10.1016/j.cell.2013.09.052)
- [Gendrel, Atlas & Hobert (2016), *A cellular and regulatory map of the GABAergic nervous system*, eLife 5:e17686](https://doi.org/10.7554/eLife.17686)
- [Beg & Jorgensen (2003), *EXP-1 is an excitatory GABA-gated cation channel*, Nat. Neurosci. 6:1145](https://doi.org/10.1038/nn1136)

**Locomotion and biomechanics**
- Gray & Lissmann (1964), *The locomotion of nematodes*, J. Exp. Biol. 41:135
- [Boyle, Berri & Cohen (2012), *Gait modulation in C. elegans*, Front. Comput. Neurosci. 6:10](https://doi.org/10.3389/fncom.2012.00010)
- [Wen et al. (2012), *Proprioceptive coupling within motor neurons drives C. elegans forward locomotion*, Neuron 76:750](https://doi.org/10.1016/j.neuron.2012.08.039)
- [Deng et al. (2021), *Inhibition underlies fast undulatory locomotion*, eNeuro 8:ENEURO.0241-20.2020](https://doi.org/10.1523/ENEURO.0241-20.2020)
- [Cronin et al. (2005), *An automated system for measuring parameters of nematode sinusoidal movement*, BMC Genet. 6:5](https://doi.org/10.1186/1471-2156-6-5)
- [Fang-Yen et al. (2010), *Biomechanical analysis of gait adaptation*, PNAS 107:20323](https://doi.org/10.1073/pnas.1003016107)
- [Gao et al. (2018), *Excitatory motor neurons are local oscillators*, eLife 7:e29915](https://doi.org/10.7554/eLife.29915)
- [Kawano et al. (2011), *An imbalancing act: gap junctions reduce the backward motor circuit*, Neuron 72:572](https://doi.org/10.1016/j.neuron.2011.09.005)

**Sensory biology and behaviour**
- Chalfie et al. (1985), *The neural circuit for touch sensitivity in C. elegans*, J. Neurosci. 5:956
- [Árnadóttir et al. (2011), *The DEG/ENaC protein MEC-10 regulates the transduction channel complex*, J. Neurosci. 31:12695](https://doi.org/10.1523/JNEUROSCI.4580-10.2011)
- [Chalasani et al. (2007), *Dissecting a circuit for olfactory behaviour*, Nature 450:63](https://doi.org/10.1038/nature06292)
- [Pierce-Shimomura, Morse & Lockery (1999), *The fundamental role of pirouettes in C. elegans chemotaxis*, J. Neurosci. 19:9557](https://doi.org/10.1523/JNEUROSCI.19-21-09557.1999)
- [Iino & Yoshida (2009), *Parallel use of two behavioral mechanisms for chemotaxis*, J. Neurosci. 29:5370](https://doi.org/10.1523/JNEUROSCI.3633-08.2009)
- [Gray, Hill & Bargmann (2005), *A circuit for navigation in C. elegans*, PNAS 102:3184](https://doi.org/10.1073/pnas.0409009101)
- [Sawin, Ranganathan & Horvitz (2000), *C. elegans locomotory rate is modulated by the environment through a dopaminergic pathway*, Neuron 26:619](https://doi.org/10.1016/S0896-6273(00)81199-X)
- [Donnelly et al. (2013), *Monoaminergic orchestration of motor programs in a complex C. elegans behavior*, PLoS Biol. 11:e1001529](https://doi.org/10.1371/journal.pbio.1001529)
- [Pirri et al. (2009), *A tyramine-gated chloride channel coordinates distinct motor programs*, Neuron 62:526](https://doi.org/10.1016/j.neuron.2009.04.013)

**Genetics and phenotypes**
- Brenner (1974), *The genetics of Caenorhabditis elegans*, Genetics 77:71
- [Jin, Jorgensen, Hartwieg & Horvitz (1999), *The C. elegans gene unc-25 encodes glutamic acid decarboxylase*, J. Neurosci. 19:539](https://doi.org/10.1523/JNEUROSCI.19-02-00539.1999)
- [Bamber et al. (1999), *The C. elegans unc-49 locus encodes multiple subunits of a heteromultimeric GABA receptor*, J. Neurosci. 19:5348](https://doi.org/10.1523/JNEUROSCI.19-13-05348.1999)
- Richmond, Davis & Jorgensen (1999), *UNC-13 is required for synaptic vesicle fusion*, Nat. Neurosci. 2:959
- [WormBook: GABA](https://www.ncbi.nlm.nih.gov/books/NBK19793/) · [Acetylcholine](https://www.ncbi.nlm.nih.gov/books/NBK19736/) · [Mechanosensation](https://www.ncbi.nlm.nih.gov/books/NBK19654/) · [Chemosensation](https://www.ncbi.nlm.nih.gov/books/NBK19746/)

**Life cycle**
- Byerly, Cassada & Russell (1976), *The life cycle of the nematode C. elegans*, Dev. Biol. 51:23
- Cassada & Russell (1975), *The dauerlarva, a post-embryonic developmental variant*, Dev. Biol. 46:326
- Golden & Riddle (1984), *The C. elegans dauer larva: developmental effects of pheromone, food, and temperature*, Dev. Biol. 102:368
- [Baugh (2013), *To grow or not to grow: nutritional control of development during L1 arrest*, Genetics 194:539](https://doi.org/10.1534/genetics.113.150847)
- Avery & Horvitz (1990), *Effects of starvation and neuroactive drugs on feeding in C. elegans*, J. Exp. Zool. 253:263

---

## Setup

```bash
python -m venv .venv && .venv/bin/pip install numpy scipy
.venv/bin/python scripts/fetch_data.py     # downloads ~43 MB, then builds
.venv/bin/python -m worm serve
```

The datasets are **not** vendored in this repository -- `fetch_data.py` pulls each
one from whoever published it, verifies it, and runs the build. Re-running it only
fetches what is missing.

The simulator core needs only **numpy**; the viewer is Python standard library and
vanilla JavaScript, with no build step. `scipy` is optional.

If you want to extend the neural model, [**Brian2**](https://brian2.readthedocs.io)
is a good fit and installs cleanly here: you can express the graded-potential
equations directly as strings and get C++ code generation. It is not used by
default because it is a spiking-network simulator by design and the 448-cell graded
system solves fine in numpy at roughly 4× real time. [NEURON](https://neuron.yale.edu)
and [NetPyNE](http://netpyne.org) are the alternatives if you want multi-compartment
morphology; OpenWorm's [c302](https://github.com/openworm/c302) generates NeuroML
models of this same network and is the reference point for anyone wanting to compare.

---

## License and provenance

Code here is original. No datasets are redistributed; `fetch_data.py` pulls each
one from its publisher, so they stay under their own terms. WormBase and WormAtlas
data is freely available for academic use, and the OpenWorm projects are
MIT-licensed. If you use this for anything real, cite the primary
sources above rather than this repository.
