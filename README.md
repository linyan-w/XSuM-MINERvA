# XSuM-MINERvA

A surrogate detector model for the [MINERvA open dataset](https://minerva.fnal.gov/opendata/): a generative model that maps
the final-state particles of a simulated neutrino interaction to the reconstructed quantities of MINERvA's analysis tuples,
trained on the open simulation dataset (Medium Energy, forward horn current, playlist 1A). It produces a pruned
`MasterAnaDev` tree with the open dataset's branch names, so an analysis written for the open dataset runs unchanged on
events from any generator. The model, its validation and an example analysis are described in the paper
*A Surrogate Detector Model for the MINERvA Open Dataset* (in preparation).

![model schematic](paper/schematic/model_schematic.png)

The model is trained in two stages, marked on the schematic: the **event stage** trains the set encoder, the event flag
heads, the hadron multiplicity head and the event flow; the **prong stage** adds the vertex-plane head and the prong set flow
and fine-tunes every parameter. A short third pass re-trains the vertex-plane head alone, conditioned on the sampled vertex.
See [`docs/model.md`](docs/model.md).

## Layout

| path | content |
|---|---|
| `xsum/` | the package: tuple I/O, input preparation, model, training, evaluation |
| `scripts/curate/` | slim the open-data tuples, build truth files from the open dataset and from NuWro / GiBUU output |
| `scripts/train/` | the two training stages and `train_all.sh`, the chain that produced the released weights |
| `scripts/infer/` | run a model on a truth file, write the ntuple, conform it to the full open-dataset schema |
| `scripts/evaluate/` | closure classifier, ensemble spread, 2p2h holdout comparison |
| `paper/` | the paper's figures and tables, regenerated from `paper/inputs/` and the weights |
| `weights/` | the deployed model (`full`), its three ensemble members (`full_s1..3`) and the 2p2h holdout model |
| `docs/` | output-file format and the uncertainty recipe (`surrogate_files.md`), the model, tuple conventions |
| `tests/` | unit tests on a small example of the open dataset (`tests/data/`) |

## Quick start

```bash
bash scripts/setup_env.sh                      # Python >= 3.9 venv with torch (CUDA), uproot, awkward, ...
source .venv/bin/activate
pytest -q                                      # round-trip and model tests on the example file (about a minute)

# 1. curation: stream the open-data tuples from xrootd into slimmed Parquet (about 300 MB per 20 GB file)
python scripts/curate/slim_remote.py configs/playlist1A_files.txt data/slim_1A --budget-gb 9.5
python scripts/curate/genie_truth.py data/genie_test.truth.parquet --population all --volume full   # open-dataset truth with its own reconstruction attached

# 2. inference: truth file -> surrogate ntuple with the four ensemble universes (docs/surrogate_files.md)
python scripts/infer/surrogate_to_ntuple.py weights/full data/genie_test.truth.parquet out.root --ensemble weights/full_s1 weights/full_s2 weights/full_s3
python scripts/infer/conform_ntuple.py out.root REFERENCE_AnaTuple.root out_full.root --strict      # optional: full 3,687-branch schema

# 3. training (about 4 h per model on one GPU) and the paper's evaluations
bash scripts/train/train_all.sh

# 4. the paper's figures and tables from the released weights and paper/inputs/
python paper/fig_response.py paper/figures/full --models A=weights/full --n 300000
python paper/fig_confusion.py --eval weights/full
python paper/fig_ensemble.py --spread paper/inputs/ensemble/spread.json --out paper/figures
python paper/tables.py
```

## Inputs and outputs

The model takes the generator truth of one event: vertex position, target nucleus, and the list of final-state particles
after final-state interactions (species and three-momentum; hadrons above 10 MeV kinetic energy, neutrons included).
It returns whether the event is reconstructed, the muon momentum and direction, the reconstructed vertex, three
calorimetric energies, the number of hadron prongs and, per prong, direction, momentum under the pion and proton
hypotheses, proton score and the fit flags. The dead-time cut variable is not modelled; it is drawn from the open dataset's
distribution. Uncertainties: the surrogate is stochastic (its sampling spread is the learned detector resolution), and the
spread between the four ensemble members is the model uncertainty, carried as systematic universes. The output branches
and the recipe to propagate the universes are in [`docs/surrogate_files.md`](docs/surrogate_files.md).

Generator samples need a vertex in MINERvA detector coordinates. The converters in `scripts/curate/` read NuWro and GiBUU
output that carries one; placing events in the detector geometry is the generator side's job.

## Large files

The slimmed training data (about 12 GB), the generator truth samples and the surrogate output files (about 1.3 GB each)
are not in this repository. The weights are: five checkpoints of 15 MB.

## License

Copyright 2026 Fermi Forward Discovery Group, LLC. The code, the model weights and the example data in this repository
are released under the Apache License, Version 2.0 (see `LICENSE`). Fermilab is operated by Fermi Forward Discovery
Group, LLC under Contract No. 89243024CSC000002 with the U.S. Department of Energy.

## Citation

Paper in preparation; a citation will be added here.
