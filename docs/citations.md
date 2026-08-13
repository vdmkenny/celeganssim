# Source integrity

Every parameter in this model points at a paper. Papers get corrected and
occasionally retracted, and a model that cites a figure which no longer says
what it used to is quietly wrong in a way no test will catch. So the sources
are audited the same way the datasets are: mechanically, repeatably, and with
the result written down.

Run it yourself:

```bash
.venv/bin/python scripts/audit_citations.py
```

It extracts every DOI in the repository (markdown, `worm/`, `scripts/`) and
asks Crossref about each one. Crossref has held the Retraction Watch database
since 2023, so retractions, corrigenda and errata all appear as `updated-by`
entries against the original work.

## What it checks, and why each matters

**Peer review.** `journal-article` went through review; `posted-content` is a
preprint. A preprint is not disqualifying, but a claim resting on one should
say so at the point of use, because the reader cannot tell from a DOI alone.

**Corrections.** A correction can invalidate the exact figure a constant was
read from while leaving the paper's conclusions intact. That distinction only
becomes visible if someone reads the notice, so the audit surfaces it and this
file records what the notice actually said.

**Reachability.** A DOI that does not resolve is a citation nobody can check.

It is a check, not a gate. Nothing here is dropped for carrying a correction.

## Findings, 2026-08-14

59 DOIs cited. 58 peer-reviewed journal articles, 1 preprint, 0 unresolved,
0 retractions, 2 carrying correction notices.

### Chalasani et al. 2007, Nature 450:63, has a substantial corrigendum

[10.1038/nature06292](https://doi.org/10.1038/nature06292) carries a 2008
erratum and a 2016 corrigendum. The corrigendum is the serious one: 163 of
851 calcium-imaging movie files, 19%, were duplicated or mislabelled. The
authors reanalysed using only valid movies and regenerated every figure.
36 of 41 imaging experiments were unaffected and remained statistically
robust; for 5 of 41 the corrected sample size fell to two or three movies and
those conclusions are labelled preliminary by the authors themselves.

**Where we rely on it, and whether it survives.** We cite this paper for
glutamate-gated chloride signalling from AWC onto AIY, specifically `glc-3`
expression in AIY, used when deriving synapse signs from receptor expression.
The corrigendum states that the properties of AWC-ON, AIB and AIY neurons
were fully supported by the reanalysis, and our use is a receptor-identity
claim rather than one of the five reduced-sample results. So the citation
stands, and the note travels with it at the point of use.

### Bentley et al. 2016, PLOS Comput Biol 12:e1005283, version 2

[10.1371/journal.pcbi.1005283](https://doi.org/10.1371/journal.pcbi.1005283)
shows a `new_version` notice. Read directly, this is PLOS replacing the
uncorrected proof with the final version, not a scientific correction. No
action needed, recorded so the next audit does not re-investigate it.

This one matters more than most because both the monoamine layer (2,626
edges) and the peptidergic layer (8,931 edges) are built from its data.

### The preprint

[MetaWorm](https://doi.org/10.1101/2024.02.22.581686) is cited in
docs/comparison.md as a bioRxiv preprint. It is labelled there, and the
peer-reviewed version of that work (Nature Computational Science, 2024) is
cited alongside it and is what the comparison relies on.

## Trust weighting in practice

The audit gives a hierarchy, applied when a claim needs a source:

1. **Peer-reviewed, uncorrected, primary measurement.** The default, and what
   every `measured` provenance tag should point at.
2. **Peer-reviewed but corrected.** Usable, with the notice read and its scope
   stated where the citation is used, as with Chalasani above.
3. **Preprint.** Usable for context and for comparison against other projects,
   not as the anchor for a `measured` parameter.
4. **Unresolvable.** Fix or remove.

`worm/parameters.py` enforces that anything tagged `measured` or `published`
carries a citation. This audit checks that those citations still say what they
said when they were written.
