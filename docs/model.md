# The model

A condensed description of the surrogate as released. The paper gives the full account; `paper/schematic/` holds the figure.

## Inputs

One event of generator truth, as the open dataset's `Truth` tree stores it:

- **Interaction**: vertex position (x, y, z) in detector coordinates and target nucleus (Z, A).
- **Final-state particles** after final-state interactions: species and three-momentum. Hadrons enter above 10 MeV kinetic
  energy; neutrons are kept; the primary lepton is one token like any other. The particle set is unordered.

The input is a set, so any generator that writes a post-FSI particle list with a vertex can be run through the model
(`scripts/curate/nuwro_to_truth.py`, `gibuu_to_truth.py`).

## Architecture

- **Set encoder**: a transformer over the particle tokens with a global token that carries the vertex and nucleus
  (4 layers, width 128, 4 heads, permutation-invariant). It returns an event-level embedding and one embedding per particle.
- **Event flag heads**, five Bernoulli classifiers on the event embedding: a reconstructed muon candidate exists, the muon is
  matched in MINOS, its charge is negative, and two zero flags for calorimetric energies that are exactly zero.
- **Hadron multiplicity head**: a classifier over the number of hadron prongs (0 to 8), conditioned on the reconstruction flag.
- **Event flow**: a 9-dimensional conditional flow-matching model of the muon momentum ratio and angles, the vertex residual,
  and three calorimetric energies, conditioned on the embedding, the flags and the multiplicity.
- **Vertex-plane head**: a 9-class classifier that decides whether the reconstructed vertex snaps to a scintillator plane,
  given the sampled vertex residual and the true vertex's offset from the nearest plane.
- **Prong set flow**: a masked flow-matching model over the N prongs (a 3-layer transformer that cross-attends to the
  particle embeddings), producing per prong the direction, the momentum under the pion and proton hypotheses, the proton
  score and the flags has-kinematics, has-proton-fit, exiting and primary.
- **Decoder**: inverse transforms and vertex snapping into the open dataset's reconstruction format (`xsum/io/writer.py`).

Sampling follows the heads top to bottom; every head is conditioned on what was sampled above it.

## Training

Teacher forcing on the open dataset's own reconstruction, AdamW with a one-cycle schedule, batches of 1024 events. The
open dataset is split by subrun into training, validation and test samples (80 / 10 / 10 of 15.8 M events in the training
volume). Three stages, chained by `scripts/train/train_all.sh`:

1. **Event stage** (`train_event.py`, 30 epochs): encoder, flag heads, multiplicity head, event flow.
2. **Prong stage** (`train_prong.py`, 16 epochs, warm start from stage 1): adds the vertex-plane head and the prong set flow
   and fine-tunes every parameter.
3. **Vertex head** (`train_prong.py --only-vtx --vtx-cond-z far`, 3 epochs): re-trains the vertex-plane head alone with the
   flow's sampled vertex z as input, so that plane snapping is consistent with the sampled residual.

The released models:

| directory | training data | seed |
|---|---|---|
| `weights/full` | every interaction in the training volume | 0 |
| `weights/full_s1`, `_s2`, `_s3` | same | 1, 2, 3 (initialisation and shuffling) |
| `weights/holdout_2p2h` | same, with GENIE 2p2h events (interaction type 8) excluded | 0 |

## Uncertainties

The surrogate is stochastic: a second draw from the same truth gives a different reconstructed event, and that spread is
the learned detector resolution. The four members of the ensemble are independent trainings; their disagreement is the
model uncertainty. `scripts/infer/surrogate_to_ntuple.py --ensemble` writes the nominal event (drawn from a random member
per batch) and, per event, every member's event-level outputs with the same random numbers, as four systematic
universes, plus per-universe weights for the flags that cannot be re-drawn. The recipe is in `surrogate_files.md`.

## Validation

- Response of reconstructed quantities as a function of the truth (`paper/fig_response.py`).
- Prong multiplicity given the true charged hadrons (`paper/fig_confusion.py`).
- Two-sample classifier closure: a gradient-boosted classifier trained to tell the open dataset from the surrogate,
  with and without the truth features, evaluated by AUC with an event-level split (`scripts/evaluate/closure_classifier.py`).
- Pulls of binned predictions against the open dataset with the statistical, sampling and ensemble terms
  (`scripts/evaluate/ensemble_spread.py`, `paper/fig_ensemble.py`).
- 2p2h holdout: the holdout model, never shown a 2p2h event, compared with the full model on the 2p2h test events
  (`scripts/evaluate/holdout_compare.py`).

## Not modelled

The dead-time cut variable (drawn from the open dataset's distribution by reconstructed vertex z), pion particle
identification scores and Michel tags, and every branch outside the pruned set, which `scripts/infer/conform_ntuple.py`
fills with the open dataset's default values.
