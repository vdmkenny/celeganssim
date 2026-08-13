# What this simulator claims, and on what authority

Every README claim, mapped to the mechanism that produces it and tagged the
way the parameter registry tags numbers: **measured** in this animal,
**published** from a standard model, **tuned** to a named target, or
**scripted** as a labelled stand-in. `worm validate` enforces most rows; the
check named is where to look. `worm params` audits every constant.

The point of this file is that a student can tell the difference between
what the model knows and what it assumes.

## The network

| Claim | Basis | Evidence |
|---|---|---|
| 473 cells, 7,762 edges of real wiring | measured (Cook et al., corrected July 2020 EM reconstruction) | build pinned by SHA256; check "posterior muscle innervation" records where the data itself is thin |
| Graded-potential dynamics, no spikes in neurons | published (Wicks et al. 1996; Kunert et al. 2014) | module docstring, `worm/nervous_system.py` |
| Neuron passive properties | measured (Goodman et al. 1998; Shindou et al. 2019; Liu et al. 2018) | check "passive properties match patch-clamp measurements" |
| Gap-junction coupling 3 pS/contact | measured against the AVAL-AVAR pair, 54 vs 56 pS (Liu, Chen & Wang 2020) | same check |
| Motor neurons rest at their measured potentials; muscle at -25 mV | measured (Liu, Chen & Wang 2014; Gao & Zhen 2011) | check "cells rest where they were measured to rest" |
| Synapse signs from postsynaptic receptor expression | measured expression (CeNGEN), derivation modelled; behavioural overrides counted separately | check "receptor-derived signs match documented synapses" |

## Muscle and junction

| Claim | Basis | Evidence |
|---|---|---|
| Muscle passive properties distinct from neurons (70 pF, 1 nS, -25 mV) | measured (Jospin et al. 2002; Gao & Zhen 2011; Richmond WormBook) | checks "passive properties", "cells rest" |
| Neuromuscular strength matches the measured junction | tuned to measurement: achievable conductance calibrated to 8.5 nS (Richmond & Jorgensen 1999) | check "neuromuscular conductance matches the measured junction" |
| GABA at the junction is chloride shunting near -30 mV | measured (Richmond & Jorgensen 1999; Gao & Zhen 2011) | same check |
| Muscle fires calcium action potentials | inward side measured twice over (Jospin 2002 density; max dV/dt); outward side tuned to the measured waveform | `worm/nervous_system.py` muscle AP block |
| Muscle calcium kinetics | measured fluorescence (Butler et al. 2015), an upper bound on the true transient | check "muscle calcium reproduces the measured transient" |
| Calcium-to-force relation | tuned: no length-tension, force-velocity or calcium-to-force curve has ever been measured in this animal | `muscle_force_gain`, fit to 19.3% BL amplitude |

## Locomotion

| Claim | Basis | Evidence |
|---|---|---|
| The undulatory wave | **scripted**: one function delivering per-muscle currents at the measured frequency and wavelength. No published model produces this rhythm from the connectome, and no ventral cord motor neuron has ever been recorded during locomotion | `muscle_pacemaker_current`; docs/emergent-cpg.md for what was tried |
| Which pool drives which direction | measured (Haspel et al. 2010; Kawano et al. 2011: B forward, A backward) | pacer docstring |
| Muscles are load-bearing; ablation has consequences | modelled through real calcium and shared mechanics | check "ablating muscle bends the animal" |
| Wild-type gait 0.25 mm/s at 19.2% BL | emerges from the above against Cronin et al. 2005 | check "wild-type gait" (requires net displacement) |
| Shrinker, unc-13 paralysis, egl-30, goa-1 | knockout scaling through junction, drive gate and rhythm rate | their named checks |
| What is NOT claimed | an emergent rhythm; bend propagation at the measured 0.62; gait adaptation to the medium | five expected failures owned by issue #10 |

## Sensation and learning

| Claim | Basis | Evidence |
|---|---|---|
| Touch is positional, via real receptive-field anatomy | measured extents (Chalfie & Sulston 1981; Goodman 2006) | check "touch is positional, not bucketed" |
| Mechanotransduction is phasic (onset and offset, tens-of-ms adaptation) | measured (O'Hagan, Chalfie & Goodman 2005) | `TOUCH_ADAPT_TAU_S` |
| mec-4 null insensitive, mec-10 partial (~50%), harsh touch independent | measured phenotypes (Chalfie 1985; Arnadottir et al. 2011; Li et al. 2011) | their named checks |
| Tap habituation: magnitude decrements ~30%, rest restores it | modelled depression at the measured locus (Wicks & Rankin 1997), through the measured antagonistic-reflex integration (Wicks & Rankin 1995) | check "tap habituation decrements the response" |
| What is NOT claimed | habituation of the suprathreshold binary response: sensory-output depression cannot produce it in this wiring and deeper depression sensitises | expected failure, issue #16 |
| Sensory injection amplitudes | tuned, one global constant, known-wrong scale | issue #12 |

## Modulation

| Claim | Basis | Evidence |
|---|---|---|
| Monoamines act extrasynaptically, by ligand and receptor | measured (Bentley et al. 2016): 2,626 edges, only 6% of them also chemical synapses | check "the monoamine layer is extrasynaptic and pharmacologically signed" |
| Receptor signs from measured coupling, not heuristic | published pharmacology per receptor (Chase et al. 2004 dop-1 Gq, dop-3 Gi; Ringstad et al. 2009 lgc-53 chloride; Ranganathan et al. 2000 mod-1) | `RECEPTOR_SIGN` in scripts/build_data.py, all 2,626 edges signed |
| Basal slowing on food, and its dissociation | mechanism: food contact excites the dopaminergic cells, DOP-3 inhibits cholinergic motor neurons. Gain tuned to Sawin et al. 2000's magnitude; the three knockout results are not tuned | check "food slowing dissociates by transmitter, as Sawin measured" |
| Enhanced slowing | NOT claimed as mechanism: still a scalar, since serotonin release does not yet depend on feeding history | `food_slowing`, issue #11 |
| The peptidergic layer | NOT claimed: 8,931 edges fetched and pinned, unbuilt | issue #13 |

## Life

| Claim | Basis | Evidence |
|---|---|---|
| Development, brood, lifespan timings | measured (Byerly 1976; Sulston 1983; Faerberg 2022; Hodgkin & Barnes 1991; Huang 2004) | their named checks |
| daf-2 longevity requires daf-16; eat-2 does not | measured epistasis (Kenyon 1993; Lakowski & Hekimi 1998) | check "daf-2 longevity requires daf-16" |
| eat-2 is dietary restriction by mechanism | measured pumping deficit, ~0.2x (Raizen, Lee & Avery 1995; McKay et al. 2004), driving real intake | `pumping_rate(pump_scale=...)` |
| Spontaneous reversals at measured rates; food and goa-1 modulate them | measured rates (Gray et al. 2005; Segalat 1995), **scripted** generator: Poisson pulses into the real backward command cells | check "the animal reverses spontaneously" |
| Chemotaxis | NOT claimed: an earlier CI pass was a torus-arena artifact; with real 133-degree turns klinokinesis now beats a blind animal by 7 mm over six minutes, but rate modulation alone cannot overcome the outward drift of a 2D walk. The missing half is klinotaxis (Iino & Yoshida 2009, issue #20) | expected failure, checks "chemotaxis discriminates salt-blind mutants" and "the omega turn reorients as sharply as the real one" |
| Local search decays to dispersal after leaving food; dopamine required | measured endpoints (Gray et al. 2005), decay constant a labelled GUESS through their time course; cat-2 by mechanism (Hills et al. 2004) | check "leaving food starts local search" |
| Omega probability rises with reversal length; stimulus strength holds reversals longer | published curve (Gray et al. 2005 Fig. 2, read off the plot); duration held by the real command balance, floor tuned | check "omega turns follow long reversals, not a coin" |
| The omega turn reorients about 133 degrees, as measured | tuned to the measured reorientation (Gray et al. 2005; Broekmans et al. 2016), through a mechanism that is itself a finding: turning comes from a wave on a gently curved body, not a deep bend | check "the omega turn reorients as sharply as the real one" |

## Where the honest edges are enforced

Nine expected failures, each citing the issue that owns it (#10, #16,
#17, #18), report the model's known gaps on every run. A provenance check
fails the suite if any parameter tagged measured or published lacks a
citation at its definition. Raw datasets are pinned by SHA256, so upstream
drift is an error, not a silent change.
