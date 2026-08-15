# The compressed network: eight refutations and what the network actually is

Cell activations in this model sit compressed around 0.5. AVA moves 1.57%
between forward locomotion and reversal and RIM moves 0.08%, against a
sigmoid that needs order 10 mV to be modulated. That caps touch
discrimination, blocks network-recruited neuromodulation
(`worm/modulation.py` documents the tyramine pathway that is correctly
wired and carries no signal), forces a reversal threshold of 0.0034, and
makes an emergent CPG implausible. It is issue #17.

Eight routes have now been measured and all eight failed. This note records
the killing measurement for each, because the pattern in the failures turned
out to be worth more than any of the attempts, and then records what the
network was measured to actually be, which is not what any of the eight
assumed.

## The eight routes

| route | killing measurement |
|---|---|
| **1. Raise `V_th` above rest** (`v_th_offset_mv`) | crawling **0.000 mm/s** at offset 40, having fixed every static quantity: tonic release 54.5% to 0.8%, muscle input resistance 0.15 to 0.85 GOhm, unphysical leak reversals 111 to 59 cells |
| **2. Compensate that with junction strength** | **0.000 mm/s** at `g_syn_nmj` of 1.02, 4.08, 12.24 and 30.60. Large static bends, no travelling wave |
| **3. Per-cell leak from potassium expression** | median deflection 0.0012 to **0.0008** mV over a 10x leak spread, and a shuffled control indistinguishable |
| **4. Release cooperativity, phi^n** | **260 of 473** cells saturate; synaptic depression restores stability at half the gait speed (docs/emergent-cpg.md) |
| **5. Heavy-tailed synaptic weights** | median deflection 0.0208 to **0.0019** mV as the tail is made heavier (w to w^gamma at preserved total, gamma 0.5 to 3.0), monotone, and it **reverses** when the tail is made lighter than anatomy. The prediction was confirmed backwards |
| **6. Recalibrate `g_syn` against input resistance** | AUC rises monotonically all the way to `g_syn` = 0 in the amplitude-free linear regime, so no value is selected by the data; muscle synaptic driveability falls 2.3x; the forced reversal-threshold rescale loses the mec-10 band |
| **7. Current-based synapses instead of conductance** | smallest real eigenvalue of the DC response matrix is **0.164 nS**, crossing zero at chi = 0.520. At chi = 0 rest spreads from -298.7 to -16.6 mV, AUC 0.570, gait 0.042 mm/s |
| **8. Voltage-gated channels in neurons** | Jacobian's largest real part crosses zero at **alpha = 0.145** (cancel 14.5% of resting conductance); gait -39% at 0.14, and 0.0001 mm/s with 19 spontaneous reversals at 0.30. Decompression 2.1x against the 13x to 40x needed |

Routes 5 to 8 were run as four independent arms in one session, each
measuring its own baseline in the same script, with the two that claimed an
improvement then attacked by a separate agent that reproduced them from the
recipe and tried to break them. Routes 1 to 4 are earlier work.

Two subsidiary negatives are worth keeping:

**CeNGEN heterogeneity carries no information for this metric.** Route 8
measured expression-driven, uniform and *shuffled* channel densities at
alpha 0.10: AUC 0.5958, 0.5968, 0.5967. Route 3 found the same for leak.
Two independent attempts to import expression heterogeneity into the
electrical model have now produced nothing distinguishable from a scrambled
control.

**Amplification cannot move a rank statistic.** Route 8 measured the
per-pair gain across all 236x236 atlas pairs: median 2.26x, and the Spearman
rank correlation between baseline and modified responses is **0.988**. Any
future route whose mechanism is "amplify everything" is refuted in advance
on AUC, and needs a different readout.

## What the network actually is

The natural reading of eight symmetric failures is that excitation and
inhibition nearly cancel, so the response is the small residue of large
opposing terms and no scalar on the gain can widen it. That was the leading
hypothesis at the end of the fan-out. It is wrong, and measuring it is what
found the real answer.

Linearise about rest, `M dV = dI` with
`M = diag(G_leak + sum_j Gg_ij + s_eq sum_j Gs_ij) - Gg + K` and
`K_ij = Gs_ij (V_i - E_ij) ds_j/dV_j`. For each of 48 driven sources,
decompose the synaptic input arriving at every neuron into its depolarising
and hyperpolarising parts and take `|net| / (|depolarising| + |hyperpolarising|)`.
Cancellation would put that near 0.

    cancellation index, 16,403 cell/source pairs
      p10  0.371   p25  0.784   p50  0.993   p75  1.000   p90  1.000
      74% of pairs above 0.8, 5% below 0.2
      median (|dep| + |hyp|) / |net| = 1.01x

There is essentially no cancellation. The input a typical neuron receives is
one-sided. And the reason is measurable:

| | neuron rows only |
|---|---|
| live chemical edges | 3,923 |
| depolarising at rest / hyperpolarising at rest | 3,240 / **683** |
| share of chemical conductance hyperpolarising | **16.5%** |
| median driving force, depolarising / hyperpolarising | 24.4 mV / **11.0 mV** |
| current at full release, depolarising / hyperpolarising | 42,749 pA / 5,053 pA (**8x**) |
| neurons receiving ANY hyperpolarising input | **139 of 378** |

Nearly two thirds of the neurons in this model cannot be inhibited through a
chemical synapse at all, and the network delivers eight times more
depolarising than hyperpolarising current.

This is not the receptor assignment being wrong. By receptor identity the
signs are reasonable and reflect the biology: glutamate splits 648
hyperpolarising against 530 depolarising, which is the glutamate-gated
chloride story `worm/connectome.py` is built to capture, GABA is
hyperpolarising, acetylcholine is depolarising. The defect is downstream of
the sign, in the driving force:

**60% of the model's inhibitory synapses push the wrong way.** 1,029 of the
1,712 edges wired to the inhibitory reversal potential are *depolarising* at
rest, because 148 of 378 neurons rest below `E_INH = -48 mV`. The median
neuron rests at **-44.2 mV** (10th to 90th percentile -64.1 to -22.7), only
4 mV above the chloride reversal, so inhibition is nearly powerless where it
is not actively reversed.

That closes a loop. Tonic release at rest, `s_eq/s_max = 0.545`, is 84%
depolarising, so it drags every unmeasured cell up from `E_cell = -65 mV`
toward -44 mV. At -44 mV inhibition has 11 mV of driving force against
excitation's 24 mV, and for the hyperpolarised tail it inverts. The network
becomes more one-sidedly excitatory, which raises the resting potential
further. The compression is the fixed point of that loop, and it explains
why gain-based routes fail symmetrically: there is no E/I structure to
exploit, only a one-sided flood to scale up or down.

The near-marginal mode is also not what it looked like. The gap-junction
block of `M` is a graph Laplacian and therefore singular by construction,
but the leak lifts it cleanly: `diag(G_leak) + L` has a smallest eigenvalue
of exactly 0.25 nS, the leak itself, and adding the tonic term gives 0.2527.
It is the synaptic release feedback `K` that pulls it down to **0.16389 nS**.
And that soft mode is local, not global: 47 of 473 cells carry half of it,
led by DA09, DD02, VD11, VD10, VD13 and VD12, the posterior ventral cord
motor neurons. The stability margin every gain-raising route ran into
belongs to one circuit, not to the network as a whole.

## What the propagation metric can and cannot resolve

`scripts/propagation.py` scores the model against Randi et al. 2023's
measured functional atlas. Its AUC was recorded as 0.615 and quoted as
though a network change could be judged by whether it moved. It cannot.

| source of variation | magnitude in AUC |
|---|---|
| the old 20,000-pair Monte Carlo estimator vs the exact statistic | 0.003 to 0.005 |
| injection current, an arbitrary knob, on the SAME model at 5 / 15 / 45 pA | 0.6121 / 0.6183 / **0.6279** |
| bootstrap over the 236 stimulated cells, the real unit of resampling | **sd 0.0142**, 95% CI [0.5923, 0.6494] |
| the four model changes measured in the fan-out | +0.001 to +0.013, all inside the above |

The script now computes the AUC exactly and always prints the interval. Use
`--paired` reasoning for a model change: resampling the same stimulated
cells in both arms cancels most of the spread.

Two controls make the number interpretable, and the first is uncomfortable:

    gap junctions only (g_syn = g_syn_nmj = 0)   AUC 0.6439
        paired delta +0.0258 +/- 0.0136, 95% CI [+0.0016, +0.0520]
    chemical only (g_gap = 0)                    AUC 0.5997
        paired delta -0.0184 +/- 0.0058, 95% CI [-0.0303, -0.0075]

**Deleting every chemical synapse improves this model's agreement with the
measured functional atlas.** It is not a tie artefact from the sparser
network: restricted to the 89.7% of pairs that respond measurably in both
conditions, the gap-junction-only network still wins, 0.6316 against 0.6031.

It follows directly from the section above. Chemical transmission here is an
8x depolarising flood, so driving any cell spreads an indiscriminate
excitatory halo onto pairs the animal calls silent, while gap junctions are
local, bidirectional and specific. The atlas rewards specificity.

## What to try next

Ranked by what the measurements above actually support.

1. **Fix the inhibitory driving force, then re-measure everything.** This is
   the first lead in this line of work that is a defect rather than a knob:
   60% of inhibitory synapses depolarising their target is not a modelling
   trade-off, it is wrong. The honest fix is not to move `E_INH`, which is a
   cited constant, but to ask why 148 neurons rest below it. Both the
   resting-potential distribution (median -44.2 mV against measured values of
   -71.7, -53.2 and -45.8 for the only three ventral cord neurons ever
   patched) and the 54.5% tonic release that produces it are in scope.
   Predicts a fall in the depolarising bias, a fall in the cancellation
   index, and a rise in the gap-junction control's paired delta toward zero.
2. **Score anything new on the E/I asymmetry and the cancellation index, not
   on AUC.** Both are cheap, both are sensitive to exactly the thing that is
   wrong, and AUC is demonstrably blind to weight magnitude (shuffling every
   chemical weight among the same edges costs only 0.006).
3. **Get the acceptance criterion from the literature.** The 13x to 40x
   target is derived internally from `beta` and an arbitrary "20% of
   activation range". Nobody here has cited a measured AVA membrane-potential
   swing between forward and reversal in a behaving animal. That number sets
   the bar for all of the above and it exists in the whole-cell recording
   literature.
4. **Functional sparsification driven by the atlas** (the 12% / 79% lead in
   docs/citations.md) remains untested, but score it on dynamic range against
   a count-matched random prune, never on AUC, which rises as chemical
   transmission is removed by any means.

**Do not spend another session on:** network-wide current-based synapses
(mechanism established, margin 0.164 nS, crossing at chi 0.520);
total-preserving weight-distribution reshaping (monotone and backwards);
CeNGEN-derived electrical heterogeneity scored against a shuffled control
(two independent nulls); regenerative amplification above `vdep_alpha` 0.10;
or tuning `g_syn` as a fix, whose only mechanism is de-weighting chemical
transmission along a monotone curve whose optimum is zero chemical synapses.

## Loose ends these measurements turned up

- `worm/validate.py`'s neuromuscular check asserts
  `0.3 <= (chol * s_max) / 8.5 <= 3.0`, an equality-shaped test on a
  quantity that is properly an inequality. Richmond & Jorgensen's 774 pA is
  the response to pressure-ejected acetylcholine, a saturating agonist dose
  opening the whole receptor complement, so it bounds what the receptors can
  carry rather than measuring what a synapse delivers. Meanwhile Jospin et
  al. 2002's 1.0 GOhm resting muscle input resistance is violated 6.6-fold.
  The two cannot both be satisfied at fixed `V_th`, because resting
  conductance is pinned at `0.545 x ceiling` by `s_eq/s_max`. The check
  should be rewritten as Richmond-as-ceiling plus Jospin-as-resting-equality.
- The Wen 2012 bend-propagation slope read 1.074 on the unmodified model in
  one arm's hands against the 0.364 recorded in the xfail text. Either the
  check or the model has drifted; that comparison should not be trusted
  until it is re-derived.
