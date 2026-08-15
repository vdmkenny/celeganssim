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

## Route 9: the inhibitory driving force

Attempted, correct, and not merged. It is on the `chloride-gradient` branch
and it is the ninth entry in the table above in everything but placement.

`E_INH` was one constant, -48 mV, applied to every neuron while neurons in
this model rest anywhere from -64 to -23 mV, so 1,029 of the 1,712 edges
wired to that reversal were DEPOLARISING their target.

**The literature settles that this is wrong without any appeal to the
model.** Every C. elegans neuron whose resting potential has been measured,
except one, rests below -48 mV:

| cell | rest | source |
|---|---|---|
| VA5 | -71.7 +/- 2.4 mV (n = 15) | Liu, Chen & Wang 2014 Nat Commun 5:5155 |
| ASEL | -61.7 +/- 1.1 mV (n = 22) | Shindou et al. 2019 Sci Rep 9:3430 |
| AVB | -57 mV median (n = 8) | Meng et al. 2024 Sci Adv 10:eadk0002 |
| ASER | -56.6 +/- 1.0 mV (n = 7) | Shindou et al. 2019 |
| VB6 | -53.2 +/- 2.5 mV (n = 13) | Liu et al. 2014 |
| VD5 | -45.8 +/- 1.9 mV (n = 15) | Liu et al. 2014 |
| AVA | -23 mV median (n = 7) | Meng et al. 2024 |

Bellemer et al. 2011 EMBO J 30:1852 names the condition that produces:
C. elegans extrudes chloride through KCC-2 and ABTS-1, and an animal
lacking both is paralysed at 510 um against a wild-type 1212 um, with body
bends almost absent, precisely because chloride flow reverses and
inhibitory transmitters begin to excite. The model was running that mutant
network-wide and calling it wild type.

**The fix, and what it achieved.** Hold E_Cl below each cell's OWN rest,
solved jointly with the resting calibration since the two are circular. The
offset is read off the one chloride reversal in this animal with a measured
basis: muscle rests at -25 mV (Gao & Zhen 2011) against a -30 mV chloride
reversal (Richmond & Jorgensen 1999), so the extruders win by 5 mV. That
rule reproduces the muscle value to 0.00 mV, and takes chloride synapses
depolarising their target from 1,029 of 1,712 to **0 of 785**.

**Why it is not merged.** It breaks five behavioural checks and they do not
come back: mec-10 goes fully touch-insensitive where it should be partial,
unc-13 crawls at 0.152 mm/s against a near-paralysed target, **no omega
turns occur at all**, and food slowing falls from 18.3% to 4.4%, which is
the monoamine layer's one earned result. It also drove the resting
distribution past the -85 mV floor, to -86.7 mV; a physiological clamp on
the pole fixes that and fixes nothing else.

The cause is the same one that killed route 1. Those constants were derived
against the network as it was, and the network as it was had inverted
inhibition doing load-bearing work: it was acting as a restoring force on
the resting distribution, which is why removing it lets the spread run to
the floor. Absorbing this needs the reversal threshold, the omega coupling,
the modulation gain and the chemotaxis scoring re-derived together.

**Two values were declined on purpose,** and both would have made this look
better. An offset of 10 mV scores better on the propagation AUC (+0.0089
+/- 0.0036 against +0.0003 for the anchored 5 mV), and 2.5 mV keeps the
resting range without needing any clamp. Choosing either would be fitting
the constant to the metric or to the model, which is the failure mode this
document exists to record.

### The re-derivation, attempted and measured

The obvious rescue is to re-derive the constants those five checks rest on.
That was done, on the same branch, and it does not converge: **38 passing
against main's 42**, with 5 failures and 11 expected failures.

What re-derived cleanly. The resting potentials, pinned to Meng et al. 2024
and Shindou et al. 2019, which is pure literature and adds no constant.
`reversal_threshold` 0.0034 to 0.0018 against re-measured bands.
`MOD_GAIN_PA` 0.6 to 0.7. mec-10, unc-13 and the resting-range check all
came back. The gain had to be swept **twice**, which is itself the result:
at a threshold of 0.0026 a gain of 0.7 gives 9% slowing and 0.8 gives 32%,
while at 0.0018 the same 0.7 gives 27.5%. A lower threshold admits more
reversals and an interrupted forward run reads as slowing, so the two
constants are not independent and neither can be moved alone.

**The Sawin pass is hollow, and the suite caught it rather than the author.**
Food slowing reads 27.5% with cat-2 at about zero, which is inside the
measured band and looks like the pathway working. But spontaneous reversals
run 13.3/min ON FOOD against 2.3 off it, which is backwards: the animal is
not slowing, it is reversing often enough that forward progress drops. That
is the cliff `worm/modulation.py` already documents for the peptide layer,
reached now by the monoamine layer because the corrected network needs a
larger gain to move at all. A check passing for the wrong reason is worse
than one failing honestly.

**The omega coupling has no feasible threshold.** During a reversal the
decision variable does not decay, it oscillates at gait frequency: after a
harsh anterior poke it runs 0.0297, 0.0041, 0.0305, 0.0041, 0.0013 at 0.2 s
intervals. The reversal ends at the first TROUGH below threshold after
`reversal_min_s`, so its duration is quantised to the gait cycle, and
baseline's 1.51 s is exactly one cycle more than the corrected network's
0.90 s. Holding it one cycle longer needs the threshold below the 0.0013
trough; the touch bands need it above the 0.0013 ceiling. Same number, no
margin. This is the single-threshold readout that issue #12 already calls
terminal and issue #7 owns the redesign of. The corrected network does not
create that conflict, it removes the slack that was hiding it.

Three more fell out of the same machinery being off its calibration: tap
habituation increments instead of decrementing, tdc-1 loses every omega
where it should merely impair them, and the sharpness check finds omegas but
no plain reversal exits to compare them against.

### The finding that outlived the attempt

Every cell without a resting-potential target rests **14 to 40 mV too
depolarised**, while the three pinned cord cells are exact to 0.1 mV:

| cell | model | measured | error |
|---|---|---|---|
| AVA | -9.0 | -23.0 | +14 |
| AVB | -30.0 | -57.0 | **+27** |
| ASEL | -30.3 | -61.7 | +31 |
| ASER | -39.1 | -56.6 | +18 |

AVB is the consequential one. A command neuron resting 27 mV high cannot be
hyperpolarised by an inhibitory synapse, and the forward/backward balance is
a difference between exactly AVA and AVB. Meng et al. measured them 34 mV
apart; this model rests them 20 mV apart, both in the wrong place. So the
chloride defect is downstream of a resting-potential defect, and pinning
those four cells is right on its own terms: it is literature, it introduces
no constant, and it fixed unc-13 by itself.

**A correction to how this was first reported.** Issue #35 framed the
finding as "the network delivers 8x more depolarising than hyperpolarising
current". That framing is weaker than it looked: fixing the sign inversion
makes the ratio WORSE, 8.5x to 18.2x, because a correctly placed E_Cl sits
close to rest and inhibition becomes almost purely shunting, which is what
inhibition in this animal largely is. A current-asymmetry measure cannot
see shunting inhibition, so it was the wrong instrument. The sign inversion
was always the defensible finding.

## What to try next

Ranked by what the measurements above actually support.

1. **Decide whether to pay for route 9** (issue #35). The fix is correct and
   already written, on the `chloride-gradient` branch. What it costs is a
   joint re-derivation of the reversal threshold, the omega coupling, the
   modulation gain and the chemotaxis scoring, because all four were fitted
   against a network whose inhibition was inverted. That is a piece of work
   to schedule rather than an experiment to run, and it should be decided on
   explicitly, because the alternative is knowingly keeping a network that
   inverts inhibition on almost every neuron this animal has been measured on.
2. **Ground the resting distribution in the measurements** that route 9
   turned up. Only six ventral cord classes plus muscle are pinned today, and
   everything else lands wherever the network solve puts it. ASEL, ASER, AVA
   and AVB now have published values (table above) and would pin four more
   cells, including both premotor interneurons. Expect this to perturb
   behaviour for the same reason route 9 did, so it belongs with that
   re-derivation rather than before it.
3. **TWK-40 is the per-cell channel handle #28 could not find.** Meng et al.
   attribute AVA's depolarised rest to that potassium leak, and show a
   loss-of-function depolarises AVA further while a gain-of-function
   hyperpolarises it to a median of -35 mV. That is a named channel, a
   measured direction, a knockout and a quantity, which is more than the
   CeNGEN-wide approach produced: two independent attempts to spread leak or
   channel density by expression came back indistinguishable from a shuffled
   control. The lesson may be that the effect is concentrated in particular
   cells rather than smeared across the population.
4. **Score anything new on the cancellation index, not on AUC and not on the
   E/I current ratio.** AUC is demonstrably blind to weight magnitude
   (shuffling every chemical weight among the same edges costs 0.006). The
   current ratio is worse than useless here: route 9 fixed the sign inversion
   and made that ratio go the wrong way, because correctly placed inhibition
   is mostly a shunt and a current measure cannot see a shunt.
5. **Get the acceptance criterion from the literature.** The 13x to 40x
   target is derived internally from `beta` and an arbitrary "20% of
   activation range". Meng et al. 2024 is the obvious place to look, having
   recorded AVA and AVB across motor states; that number would set the bar
   for everything above and replace a target this project invented.
6. **Functional sparsification driven by the atlas** (the 12% / 79% lead in
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
