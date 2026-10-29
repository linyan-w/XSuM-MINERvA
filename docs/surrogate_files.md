# The surrogate output files and their uncertainties

The surrogate writes a pruned `MasterAnaDev` tree with the open dataset's branch names (one row per reconstructed event,
truth columns copied from the generator sample), plus `surrogate_*` branches. This note describes the **ensemble files**
`surrogate_Eall_<sample>.root` (GENIE and GiBUU inclusive samples; `scripts/infer/surrogate_to_ntuple.py --ensemble`), which
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
| `surrogate_univ_weight` | the weight that turns the nominal row set into member `k`'s reconstruction (see below) |

**Weights.** The file contains only the events the nominal sample reconstructed, with the nominal's MINOS-match and
charge outcomes. Member `k` would reconstruct and match a slightly different set. Instead of separate row sets, each
universe reweights the nominal rows by the ratio of Bernoulli probabilities,

`w_k = p_k(reco) / p_nom(reco) × [p_k(MINOS) / p_nom(MINOS) if the nominal has a MINOS match, else (1−p_k)/(1−p_nom)] × [likewise for the charge sign, if matched]`,

which reproduces member `k`'s reconstruction, matching and charge probabilities in expectation. The weights average to
1.000 ± 0.001 per universe and lie within 0.7–1.3 for 98% of events.

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

## Other branches

`surrogate_p_reco_exists`: the nominal member's probability that the event is reconstructed (the written events are the
Bernoulli draws of it). `phys_n_dead_discr_pair_upstream_prim_track_proj`: the dead-time cut variable, drawn from the open
dataset's distribution by reconstructed vertex z, independently of the event (it is overlay pile-up, not physics); the
standard cut `<= 1` passes 99.0%. Generator extras (`nuwro_*`, `gibuu_*`, `gen_relweight`, `ref_*`) pass through from the
truth sample; `nuwro_wcomp` is the sample weight of the inclusive construction (1 for almost all events).
