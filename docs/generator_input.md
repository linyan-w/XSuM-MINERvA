# Generator input: the truth file

The surrogate reads one Parquet file with one row per generated event, in the layout of the open dataset's `Truth` tree
(the same columns `scripts/curate/genie_truth.py` writes for the open dataset itself). Any generator can be run through
the model once its events are written in this layout. Units are MeV and mm; positions and momenta are in the MINERvA
detector frame (the beam points 3.34 degrees downward: a generator's beam-frame momenta are rotated about x by +0.05887 rad,
`xsum.prep.frames.from_beam`).

| column | type | meaning |
|---|---|---|
| `mc_run`, `mc_subrun`, `mc_nthEvtInFile` | int32 | event identifiers; `mc_subrun` also groups events into batches |
| `eventID` | int64 or double | a unique event number (e.g. `mc_run x 1e7 + mc_nthEvtInFile`) |
| `mc_vtx` | double[4] | true vertex (x, y, z, t) in detector coordinates. The model is trained on vertices with z in 4000-8700 mm (the inner detector from the nuclear-target region to the end of the tracker); events outside are written to the `Truth` tree only |
| `mc_targetZ`, `mc_targetA` | int32 | the struck nucleus |
| `truth_targetID` | int32 | passive-target number (1-5) or 0 |
| `mc_incoming`, `mc_incomingE` | int32, double | neutrino PDG code and energy |
| `mc_current` | int32 | 1 charged current, 2 neutral current |
| `mc_intType` | int32 | interaction type in the open dataset's convention (1 QE, 2 RES, 3 DIS, 4 COH, 8 2p2h, ...); passed through, used in training only |
| `mc_primFSLepton` | double[4] | (px, py, pz, E) of the final-state lepton |
| `mc_nFSPart` | int32 | number of final-state particles |
| `mc_FSPartPDG`, `mc_FSPartPx`, `mc_FSPartPy`, `mc_FSPartPz`, `mc_FSPartE` | lists | the final-state particles after final-state interactions, the lepton included; the model keeps charged hadrons, neutral pions and neutrons above 10 MeV kinetic energy and the leptons (`xsum.prep.particles.select_particles`) |
| `mc_Q2`, `mc_w` | double | passed through |

Where the events sit in the detector and how many there are per nucleus is the generator side's job: the vertices must be
drawn in the detector geometry for the nucleus the event was generated on, and the sample's exposure must be known. Columns
whose names start with `gen_`, `nuwro_` or `gibuu_` are copied into both output trees unchanged, which is where an event
weight and the exposure belong, e.g. `gen_weight` (the event's weight at the generated exposure) and `gen_pot` (that
exposure in protons on target), so that a prediction at the data exposure is `sum(gen_weight) x POT_data / gen_pot`.
Any other column is copied into the `Truth` tree only.

Helpers: `scripts/curate/merge_truth_shards.py` concatenates per-job shards into one file with fixed-size row groups,
`scripts/curate/thin_truth.py` thins a sample to a smaller exposure (keeping `gen_weight` and rewriting `gen_pot`), and
`scripts/infer/infer_sliced.sh` runs the ensemble on a large file in slices (see `docs/surrogate_files.md`).
