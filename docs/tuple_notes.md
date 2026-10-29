# What is actually in the open-data MC AnaTuple

Measured on `MasterAnaDev_mc_AnaTuple_run00113069_Playlist.root` (ME FHC, `mc_beamConfig = 11`)
with uproot, on the first 20k–30k entries of each tree unless stated. Numbers are rough
guides for design, not final results.

## Trees

| Tree | Entries | Branches | Content |
|---|---|---|---|
| `Truth` | 518,653 | 560 | Every generated interaction in the file, truth only. This is the efficiency denominator. |
| `MasterAnaDev` | 167,871 | 4,190 | Events with a reconstructed muon candidate. Carries the full reco record **and** a copy of the truth (`mc_*`, `truth_*`) for the same event, so (truth, reco) pairs need no joining. |
| `Meta` | 1 | 4 | `POT_Used`, `POT_Total`, entry counts. |

Reco tree / Truth tree = 0.324. Cross-link between trees is `(mc_run, mc_subrun, mc_nthEvtInFile)` or `eventID`.
Of the Truth entries, 24.7% are `truth_is_fiducial`, 75.2% CC, 18.3% fiducial AND CC AND numu.

## Truth side: what the generator hands to the detector simulation

| Branch | Type | Meaning |
|---|---|---|
| `mc_nFSPart`, `mc_FSPartPDG/Px/Py/Pz/E` | jagged | GENIE final-state particles **after** intranuclear FSI, i.e. the particles that leave the nucleus. Mean 8–9 per event, median 5, 99th pct ~60–70 (DIS tails). |
| `mc_er_*` (`nPart`, `ID`, `status`, `Px..E`, `mother`, `FD/LD`, `posInNuc*`) | jagged | Full GENIE event record. `status==1` are the stable final-state particles (same set as `mc_FSPart*`); 0 = initial state, 2 = intermediate, 3 = decayed, 11–16 = nucleon/hadron-in-nucleus/remnant bookkeeping. |
| `mc_primFSLepton` | double[4] | Outgoing lepton 4-vector. Also present in `mc_FSPart*`. |
| `mc_vtx` | double[4] | True vertex (x, y, z, t) in mm/ns detector coordinates. z spans 1742–8601 mm in this file (nuclear targets through tracker/ECAL/HCAL). |
| `mc_targetZ`, `mc_targetA`, `mc_targetNucleus`, `mc_targetNucleon` | int | Struck nucleus. Mostly C (tracker), Pb, Fe, H, O; traces of Al, Ti, Si, N, Cl, Mn. |
| `truth_targetID`, `truth_vtx_module`, `truth_vtx_plane`, `truth_target_zDist` | int/double | Detector geometry labels for the true vertex. `targetID` 0 = tracker, 1–5 = nuclear targets 1–5. |
| `mc_incoming`, `mc_incomingE`, `mc_current`, `mc_intType`, `mc_Q2`, `mc_w`, `mc_Bjorkenx/y` | scalars | Neutrino flavour, energy, CC/NC, interaction mode (1 QE, 2 RES, 3 DIS, 4 COH, 8 MEC/2p2h, ...). **Generator-internal**; not visible to the detector. |
| `mc_beamConfig`, `mc_nInteractions`, `mc_pot`, `mc_fr_*` | scalars | Beam configuration (11 = ME FHC here), interactions per event (always 1 here), flux-ancestor record. |
| `mc_*wgt*`, `mc_cvweight*`, `truth_*Reweight*`, `part_response_*` (2,044 branches) | many | Systematics machinery (GENIE knobs, hadron reinteraction, calorimetric particle response). Irrelevant as surrogate inputs; potentially useful later for systematic-aware training. |

Most common FS PDGs (20k Truth events): n 60.7k, p 57.3k, pi+ 15.5k, mu- 14.6k, pi0 13.0k, pi- 7.7k, nu_mu 4.8k,
pseudo-particle 2000000101 (GENIE binding-energy carrier) 2.3k, gamma 1.7k, K+ 0.8k, Lambda 0.6k, K0 0.6k,
nuclear remnants (PDG 100ZZZAAA0) ~0.1k. Neutrons are the single most common FS particle.

## Reco side: what MasterAnaDev reconstructs

The reco record is **hypothesis-based**, not a generic particle list. The same prong can be scored as a proton
and as a pion. Roughly:

| Group | Branches | Shape | Notes |
|---|---|---|---|
| Muon | `MasterAnaDev_muon_P/E/Px/Py/Pz/theta/phi/qp/qpqpe/score`, `MasterAnaDev_minos_trk_*`, `MasterAnaDev_nuHelicity`, `muon_theta/phi`, `muon_corrected_p` | scalars | Always one muon candidate. `minos_trk_is_ok` true for ~70%. Muon P reco/true: median 0.98, 16–84% band [0.52, 1.13]; 36% of events off by more than 20% (non-MINOS-matched tails, exiting muons). Strongly non-Gaussian, multimodal. |
| Reco vertex | `MasterAnaDev_vtx` (x,y,z,t), `MasterAnaDev_vtx_module/plane`, `vtx`, `vtxErr` | fixed | 68% resolution ~10 mm in x, y; ~40 mm in z. |
| Multiplicity | `multiplicity`, `n_prongs`, `MasterAnaDev_hadron_number`, `n_anchored_long/short_trk_prongs`, `n_nonvtx_iso_blobs`, `n_minos_matches` | int | `multiplicity` = tracks at vertex incl. muon: 1: 44%, 2: 42%, 3: 11%, 4: 2%, ≥5: <0.5%, max 9. |
| Primary proton | `MasterAnaDev_proton_P/T/E_fromdEdx`, `_theta`, `_phi`, `_score`, `_score1`, `_score2`, start/end points, `_calib_energy`, `nodes_E[]` | scalars, -9999 if none | Highest-score proton candidate. |
| Secondary protons | `MasterAnaDev_sec_protons_*` | jagged | 0: 93%, 1: 6%, 2: 0.8%, up to 5. |
| Pions / generic hadron prongs | `MasterAnaDev_pion_P/E/T/theta/phi/Px..`, `_score/_score1/_score2`, `MasterAnaDev_hadron_*` incl. `hadron_tm_PDGCode`, `hadron_tm_fraction`, `hadron_isExiting`, `hadron_endMichel_*`, `hadron_piFit_*` | fixed[10], -1 fill | Candidates per event: 0: 54%, 1: 37%, 2: 8%, ≥3: 1.5%. `hadron_tm_*` is truth matching per prong; useful for validation. |
| Calorimetry | `MasterAnaDev_recoil_E`, `_recoil_E_wide_window`, `_recoil_passivecorrected`, `MasterAnaDev_hadron_recoil`, `_hadron_recoil_CCInc`, `_default`, `_two_track`, `recoil_energy_nonmuon_{vtx,nonvtx}{0..300}mm[_nuclTargs]`, `blob_recoil_E_*` (per subdetector / per true particle class), `nonvtx_iso_blobs_energy`, `vtx_blobs_energy` | scalars | `recoil_E` / true visible hadronic KE: median 1.31, 16–84% [0.81, 2.68] (before calorimetric correction, includes neutron contributions). |
| Derived kinematics | `MasterAnaDev_E`, `_Q2`, `_Q2_CCQE`, `_Q2_Inclusive`, `_W`, `_leptonE` | scalars | Functions of the above; do not model separately. |
| Michel / pi0 / EM | `improved_michel_*`, `matched_michel_*`, `FittedMichel_*`, `gamma1_*`, `gamma2_*`, `pi0_*`, `EMLikeTrackMultiplicity`, `blob_nuefuzz_*` | mixed | Later tiers. |
| Detector / beam conditions | `phys_n_dead_discr_pair*` (dead channels; mean 21, 99th pct 214), `numi_pot`, `numi_horn_curr`, `numi_x/y*`, `numi_is_good_*`, `ev_run/subrun/gate/gps_time*`, `batch_structure`, `n_slices`, `slice_n_hits`, `event_*` (other in-time activity) | scalars | Come from the **data overlay** in ME MC, so they are realistic and available identically in data. |

## Reco multiplicity vs truth (30k reco events)

Rows: true number of protons with KE > 100 MeV plus charged pions. Columns: reco `multiplicity` 1..6.

| n_vis(true) | events | mean mult | mult = 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 3,856 | 1.21 | 3193 | 531 | 112 | 15 | 2 | 2 |
| 1 | 10,541 | 1.63 | 4785 | 5030 | 624 | 81 | 16 | 1 |
| 2 | 7,117 | 1.86 | 2353 | 3597 | 1004 | 139 | 17 | 6 |
| 3 | 4,170 | 1.95 | 1360 | 1891 | 720 | 171 | 22 | 4 |
| 4 | 2,300 | 1.98 | 806 | 911 | 440 | 115 | 23 | 4 |
| 5 | 1,158 | 2.01 | 426 | 423 | 215 | 75 | 11 | 6 |

Two things stand out. Reco multiplicity saturates around 2 while true multiplicity keeps growing, so per-particle
efficiency clearly falls with multiplicity (the motivation for a per-event surrogate). And even at n_vis = 0
about 17% of events have an extra reco track, from neutrons, photons, overlay, or split muon tracks.

## Practical

- Only `uproot`/`awkward` are needed to read the files; PyROOT is not installed here and not required.
- One MC file is 19.8 GB for 168k reco events, almost entirely because of the 4,190 branches. Slimming the
  ~80 branches we need to Parquet should give well under 1 GB per file.
- Upstream branch documentation: https://github.com/MinervaExpt/Tuple-Documentation
  (`MAD_tuple_MainDoc.csv`, 257 documented branches, mostly pi0/neutron/systematics groups; a copy is in
  `docs/branches/`). Full branch lists from this file are in `docs/branches/*_branches.txt`.

## Conventions discovered while writing the encoder/decoder (2026-09-23)

All verified by exact round trip on 20k reco events (`tests/`).

- **Beam frame.** Every `*_theta` / `*_phi` branch (`MasterAnaDev_muon_theta`, `muon_phi`, `pion_theta/phi[10]`,
  `proton_theta/phi`, `sec_protons_theta_fromdEdx`) is measured in the beam frame: detector frame rotated about
  x by MINERvA's `numi_beam_angle_rad = -0.05887` (3.373°). All `Px/Py/Pz` branches are in the detector frame.
  With that constant the muon angles round-trip to 1e-16; with the often-quoted 3.34° they are off by up to
  6e-4 rad. `xsum/prep/frames.py` does the conversion.
- **Prong table.** `pion_*[10]` and `hadron_*[10]` are indexed by prong `i < MasterAnaDev_hadron_number`
  (`= n_prongs - 1`). Empty slots: `pion_P/E/Px/Py/Pz = -1`, `pion_T = -9999`, `pion_theta/phi = -9`,
  `hadron_isExiting = -1`. A valid prong whose pion fit failed has `P = E = -1`, `T = -9999`, but valid angles.
- **"Direction only" sentinel.** ~0.3% of prongs (and 12 primary protons, 6 secondary-proton entries in 20k
  events) have `P == 1`, `E == 0`, `T == -mass`, and unit-vector `Px/Py/Pz`. We treat them as "no fit".
- **Primary / secondary protons.** `MasterAnaDev_proton_*` and each `sec_protons_*` entry match exactly one
  prong by `theta` (identical to 1e-6, 100% of cases); secondaries never coincide with the primary. Picking
  the primary as the highest `proton_score1` reproduces MasterAnaDev's choice in 90.5% of events; for the
  other 9.5% MasterAnaDev used some other criterion, so the encoder keeps an explicit `is_primary_proton` flag.
- **Muon.** `muon_P == |(Px, Py, Pz)|` exactly. `muon_E` is on-shell only for MINOS-matched muons; for 82% of
  unmatched muons `muon_E == muon_P`. `muon_qp` is the MINOS-track q/p in 1/GeV (`-9999.9` when unmatched), so
  it is not derivable from the MINERvA momentum; only its sign (= `nuHelicity`) is modelled.
- **Slimming.** Both trees of the 19.8 GB file slim to 105 MB (reco, 168k events) + 194 MB (truth, 519k
  events) of Parquet in 6 s, because uproot reads only the requested baskets.
- **Training population** (CC nu_mu, true vertex z ≥ 4000 mm): 368,396 Truth events, of which 39.6% have a
  reco entry. After the input cuts (no neutrons, KE ≥ 50 MeV) events have 4.1 particles on average, 99% ≤ 11,
  max 31. Among reconstructed events 78.5% are MINOS-matched and 97.3% have negative reco charge. 462 subruns
  in the file, so a subrun-level 80/10/10 split gives 293k / 37k / 38k events.

## Further conventions found during M2 (2026-09-23)

- **Corrupt rows.** 8 of 4.77M reconstructed CC nu_mu events have NaN muon `Px/Py/Pz` or infinite `MasterAnaDev_vtx`;
  one has a vertex coordinate of order 1e14 mm. They pass `isfinite` checks on most branches and poison any mean.
- **`MasterAnaDev_muon_qp` = -9999.9 when not MINOS-matched**, so "qp < 0" is true for every unmatched muon. A charge
  flag must be defined as (matched AND qp < 0).
- **`MasterAnaDev_recoil_passivecorrected` = 0.6669 x `recoil_E` exactly for > 15% of events** (84th and 99th percentile
  of the ratio coincide), 0.62-0.67 otherwise depending on vertex z; **`MasterAnaDev_hadron_recoil` = 1.385 x
  `recoil_E`** to within a MeV for a large fraction. Both are calibrations of one number, not independent responses.
- **Reconstructed vertex z snaps to plane positions** for 69% of events (about 196 distinct positions 22.1 mm apart in
  the 32-file sample); 55% land within 3 planes of the true z, 14% on a plane further away. The z residual conditional
  on truth is a comb; which plane is chosen depends on the true z's phase within the plane cycle.
