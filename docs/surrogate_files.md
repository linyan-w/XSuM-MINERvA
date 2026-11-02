# The surrogate output files and their uncertainties

The surrogate writes two trees, like the open dataset: `Truth`, one row per generated event with the truth branches of the
input sample (the efficiency denominator), and a pruned `MasterAnaDev` tree with the open dataset's branch names, one row
per reconstructed event, truth columns copied, plus `surrogate_*` branches. Events are cross-linked by
`(mc_run, mc_subrun, mc_nthEvtInFile)` or `eventID`. Files written before 2026-10-29 have the `MasterAnaDev` tree only.
This note describes the **ensemble files**
`surrogate_Eall_<sample>.root` (the GENIE sample and the NuWro and GiBUU samples; `scripts/infer/surrogate_to_ntuple.py --ensemble`), which
carry the model uncertainty in a form an analysis can propagate. The single-model files `surrogate_Aall_*.root` have the
same reconstruction branches without the universes.

## The model and its ensemble

One deployed model (`weights/full`, trained on every interaction in the training volume) and three further members
trained with different initialisation and shuffling seeds on the same data (`weights/full_s1..s3`). The members are
statistically indistinguishable on every closure metric; their disagreement on an event measures how much the training
data constrain the response there.

The **nominal sample** draws each batch of events from a random member, so the file as a whole is the ensemble mixture;
`surrogate_member` (int, 0 = the deployed model, 1..3 = the seeds) says which member drew the event.

## Universes

Every written event carries, for each member `k = 0..3` (`surrogate_univ_members` = 4), the member's event-level outputs
drawn with the **same random numbers** as the nominal, so the nominal is exactly the universe of its own member:

| branch (fixed-size array `[4]`) | content |
|---|---|
| `surrogate_univ_muon_P`, `_muon_theta` | muon momentum [MeV] and beam-frame angle [rad] |
| `surrogate_univ_muon_Px/Py/Pz` | muon momentum components, detector frame [MeV] |
| `surrogate_univ_vtx_x/y/z` | reconstructed vertex [mm] (z snapped to a plane when the member's class head says so) |
| `surrogate_univ_recoil_E`, `_recoil_nonvtx100`, `_nonvtx_iso_blobs_E` | calorimetric energies [MeV] |
| `surrogate_univ_n_prongs` | prong count (muon track + hadron prongs) |
| `surrogate_univ_weight_reco`, `_minos`, `_charge` | the weight of member `k` for each flag separately: reconstruction, MINOS match, charge sign (see below) |
| `surrogate_univ_weight` | their product: the weight that turns the nominal row set into member `k`'s reconstruction |

**Weights.** The file contains only the events the nominal sample reconstructed, with the nominal's MINOS-match and
charge outcomes. Member `k` would reconstruct and match a slightly different set. Instead of separate row sets, each
universe reweights the nominal rows by the ratio of Bernoulli probabilities,

`w_k = p_k(reco) / p_nom(reco) × [p_k(MINOS) / p_nom(MINOS) if the nominal has a MINOS match, else (1−p_k)/(1−p_nom)] × [likewise for the charge sign, if matched]`,

which reproduces member `k`'s reconstruction, matching and charge probabilities in expectation. The three factors are
written separately (`surrogate_univ_weight_reco`, `_minos`, `_charge`) so that the uncertainty of each flag can be
propagated on its own, and their product as `surrogate_univ_weight`. The weights average to 1.000 ± 0.001 per universe
and lie within 0.7–1.3 for 98% of events.

**Recipe.** For a histogram `H` of any event-level quantity after any selection:

1. fill the nominal `H_nom` from the standard branches (`MasterAnaDev_muon_P`, `MasterAnaDev_vtx`, ... ), unit weight;
2. for each `k`, fill `H_k` from the `surrogate_univ_*[k]` values with weight `surrogate_univ_weight[k]`, applying the
   selection to the universe's own values (vertex cuts on `surrogate_univ_vtx_*[k]`, angle on `_muon_theta[k]`, ...);
3. the model uncertainty band is the spread of the `H_k` around `H_nom` (bin-to-bin correlated within a universe), or
   the covariance of the four `H_k` for a fit. This is the same mechanics as MINERvA's vertical error bands; with four
   universes it is a band, not a well-determined covariance matrix.

Prong-level variables (`MasterAnaDev_pion_*`, `_proton_*`, `_hadron_*`) are nominal only: a prong-flow pass per member
would multiply the inference cost, and the analysis variables (muon, vertex, selection) do not need them.

## What the universes do and do not cover

They cover the part of the model uncertainty that the ensemble *sees*: where the training data are sparse (large muon
angle, migrated events, high `p_T`) the members disagree by up to ten times more than in the bulk (muon momentum 0.4% at
small angle against 8% above 45°, vertex z 0 against 16 mm). They do **not** cover the model's residual misfit, which
all members share: on the open dataset's own events the ensemble mean differs from the reconstruction by 1–3% per bin
where the spread is 0.4–1.2%. That residual is measured as a **model-error floor**, a relative uncertainty per bin to add
in quadrature, uncorrelated between bins:

| observable (marginal) | floor (relative, per bin) |
|---|---|
| muon momentum | 2.6% |
| muon angle | 2.4% |
| recoil energy | 3.4% |
| non-vertex energy within 100 mm | 4.6% |
| isolated-blob energy | 5.2% |
| prong count | none resolved |

The honest per-bin uncertainty is the universe spread added in quadrature to the floor; neither alone.

## Per-event columns

`surrogate_epi_<variable>` (muon P and components, vertex, recoil, the two non-vertex energies, prong count, the three
flag probabilities) is the standard deviation over the four universes for that event, in physical units; `surrogate_epi_draws`
= 4. `surrogate_epi_logit_*`, `surrogate_epi_multiplicity_logits` and `surrogate_epi_flow_flag` are the between-member
variances of the flag logits, the multiplicity log-probabilities and the model-space event vector; they are per-event
flags of unfamiliar events (migrated events: AUC 0.77 for the flow flag), not uncertainties to propagate.

## The `Truth` tree

Every input event, with all branches of the input truth file, and: `surrogate_in_population` (true vertex z inside the
training volume, 4000–8700 mm; events outside are never reconstructed by the surrogate and have probability 0),
`surrogate_p_reco_exists` (the nominal member's reconstruction probability), `surrogate_reco_exists` (the sampled flag:
true for exactly the events in `MasterAnaDev`), and with the ensemble `surrogate_member` (−1 outside the population) and
`surrogate_univ_p_reco_exists[4]`, each member's reconstruction probability. An efficiency in universe `k` is the
`MasterAnaDev` numerator filled with `surrogate_univ_weight_reco[k]` (times the other flag weights the selection uses)
over the `Truth` denominator; the denominator does not change between universes.

## The geometry-placed generator files (2026-11)

`surrogate_Eall_nuwro_geo_2026-11.root` and `surrogate_Eall_gibuu_geo_2026-11.root` are the ensemble files of the NuWro and
GiBUU samples generated by the generator side with vertices drawn from the MINERvA GDML and per-nuclide rates from the
published ME FHC flux times each generator's own cross section (the generator side's hand-off note; 8 campaigns per generator,
numu / numubar / nue / nuebar x CC / NC, POT_gen = 1.998e20). Nothing is re-vertexed or rescaled here: `scripts/curate/merge_truth_shards.py`
concatenates the delivered shards (docs/generator_input.md), `scripts/curate/thin_truth.py` keeps a random 0.4856 of the
events so that the samples sit at the GENIE sample's exposure (9.702e19 POT = 0.2383 of the open dataset; `gen_pot` is set to that
POT, so the prediction formula below is unchanged; 4.31 M NuWro and 4.97 M GiBUU events), and `scripts/infer/infer_sliced.sh` runs the ensemble in
slices of 2 M events (`surrogate_to_ntuple.py --range`, one seed per slice) that `scripts/infer/merge_ntuples.py` concatenates in
input order. Both trees carry the generator columns of the input, in particular:

| branch | meaning |
|---|---|
| `gen_weight` | event weight at POT_gen; a prediction at the data exposure is `sum(gen_weight) x POT_data / gen_pot`. It already contains the per-nuclide normalisation (and, for GiBUU, `weight_rel`): multiply by nothing else |
| `gen_pot` | the sample's exposure: 9.702e19 after the thinning (1.998e20 as delivered) |
| `gen_subregion`, `gen_material`, `gen_module_plane`, `gen_sigma_cm2_per_nucleon` | where the vertex was placed and the job's flux-averaged cross section |
| `nuwro_dyn`, `nuwro_job`, `nuwro_weight` / `gibuu_prod`, `gibuu_job`, `gen_relweight` | the generator's own channel, job and weight bookkeeping (informational) |

Every event of the input appears in `Truth` (the whole generation region, z 4284-8605 mm, lies inside the training volume, so
`surrogate_in_population` is true for all of them); the surrogate uncertainties combine with `gen_weight` by multiplication, e.g.
universe k of a selection is `sum(gen_weight x surrogate_univ_weight[k])` over the events passing with the universe's values.

## Other branches

`surrogate_p_reco_exists`: the nominal member's probability that the event is reconstructed (the written events are the
Bernoulli draws of it). `phys_n_dead_discr_pair_upstream_prim_track_proj`: the dead-time cut variable, drawn from the open
dataset's distribution by reconstructed vertex z, independently of the event (it is overlay pile-up, not physics); the
standard cut `<= 1` passes 99.0%. Generator extras (`gen_*`, `nuwro_*`, `gibuu_*`, `ref_*`) pass through from the
truth sample into both trees.
