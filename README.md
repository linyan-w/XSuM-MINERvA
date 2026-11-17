# XSuM-MINERvA

*What would MINERvA have seen?*

XSuM-MINERvA is a surrogate detector model for the [MINERvA open dataset](https://minerva.fnal.gov/opendata/). Hand it the
final-state particles of a simulated neutrino interaction, and it hands back what MINERvA's reconstruction would have made
of them, in the format of MINERvA's analysis tuples. Under the hood it is a generative model trained on the open simulation
dataset (Medium Energy, forward horn current, playlist ME1A). Its output is a pruned `MasterAnaDev` tree with the open
dataset's branch names, so an analysis written for the open dataset runs unchanged on events from any generator, such as
NuWro, GiBUU or GENIE. The model, its validation and an example analysis are described in the paper
*A Surrogate Detector Model for the MINERvA Open Dataset* (in preparation).

**Contents**

- [Quickstart](#quickstart)
- [Choose your path](#choose-your-path)
  - [A. Run the surrogate on your own events](#a-run-the-surrogate-on-your-own-events)
  - [B. Redraw the paper's figures and tables](#b-redraw-the-papers-figures-and-tables)
  - [C. Rebuild everything from the open dataset](#c-rebuild-everything-from-the-open-dataset)
- [Reference](#reference): [the model](#the-model) | [training sample](#training-sample) |
  [inputs, outputs and uncertainties](#inputs-outputs-and-uncertainties) | [scope and limitations](#scope-and-limitations) |
  [repository layout](#repository-layout) | [troubleshooting](#troubleshooting) | [support](#support-and-contributing) |
  [license](#license) | [citation](#citation)

---

## Quickstart

```bash
git clone https://github.com/linyan-w/XSuM-MINERvA.git
cd XSuM-MINERvA
bash scripts/setup_env.sh        # Python >= 3.9 virtualenv with torch, uproot, awkward, ... (~7 GB, mostly CUDA libraries)
source .venv/bin/activate
python -m pytest -q              # 16 tests on the example file in tests/data/, a few seconds
```

If all 16 tests pass, you are set up. Run every command below from the repository root.

<details>
<summary><b>CPU-only machine: a smaller install</b></summary>

`setup_env.sh` installs the CUDA build of PyTorch, which also runs on machines without a GPU. On a CPU-only machine you
can save about 3 GB by installing the CPU build before running `setup_env.sh` (which keeps an existing `.venv`):

```bash
python3 -m venv .venv
.venv/bin/pip install "torch==2.8.*" --index-url https://download.pytorch.org/whl/cpu
bash scripts/setup_env.sh
```
</details>

> **TODO: tested versions.** A few ideas:
> - Python versions tested (3.9 and 3.11 on Linux work so far; 3.13 can need a recent compiler to build `numpy`).
> - GPU and CUDA version used for training and inference.
> - Whether macOS (CPU only) works.
> - Exact package versions used to produce the weights.

---

## Choose your path

Most people want **A**: you have events, you want to know how MINERvA would see them. **B** is for checking our
figures, **C** for checking our work from the ground up. All three start from the [Quickstart](#quickstart); the table
lists what each needs on top of it.

| path | also needs | time |
|---|---|---|
| [**A. Run the surrogate on your own events**](#a-run-the-surrogate-on-your-own-events) | a GPU for large samples (small samples run on CPU automatically) | scales with the number of events and ensemble members: seconds for a thousand events, hours for multi-million-event generator samples |
| [**B. Redraw the paper's figures and tables**](#b-redraw-the-papers-figures-and-tables) | runs on CPU (except the response figure, which needs the dataset from C and a GPU) | seconds |
| [**C. Rebuild everything from the open dataset**](#c-rebuild-everything-from-the-open-dataset) | xrootd access, ~12 GB of disk, a CUDA GPU | scales with the number of training events: ~4 h per model for the full 15.8 M events on an RTX 3090, five models |

---

## A. Run the surrogate on your own events

| also needs | time | you get |
|---|---|---|
| a GPU for large samples; small samples [run on CPU](#running-on-cpu) automatically | seconds for a thousand events; hours for multi-million-event samples | a ROOT file with the open dataset's `Truth` and `MasterAnaDev` trees |

The recipe is three steps: write your events in the open dataset's truth layout, run the model, read the result. In:
one Parquet file of generator truth, one row per event, with the vertex, the target nucleus and the final-state particles
after final-state interactions (columns in [`docs/generator_input.md`](docs/generator_input.md)). Out: a ROOT file with
two trees, `Truth` (every event you put in) and `MasterAnaDev` (the events the surrogate reconstructed).

### A.1 Write a truth file

The simplest case is a lone muon. This example writes 1,000 of them, 3 GeV/c
along the beam, starting at z = 7 m in the tracker on carbon. Units are MeV and mm in the detector frame (the beam points
3.34 degrees below the detector z axis; `from_beam` rotates a beam-frame momentum into the detector frame):

```python
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from xsum.prep.frames import from_beam

N, P, M_MU = 1000, 3000.0, 105.658                       # events, momentum [MeV], muon mass [MeV]
px, py, pz = (float(v[0]) for v in from_beam(np.zeros(1), np.zeros(1), np.full(1, P)))
E = (P**2 + M_MU**2) ** 0.5
f64, i32 = pa.float64(), pa.int32()
fixed  = lambda n: pa.list_(pa.field("item", f64, nullable=False), n)
jagged = lambda t: pa.large_list(pa.field("item", t, nullable=False))
cols = {  # name: (values for every event, type)
    "mc_run": (1, i32), "mc_subrun": (0, i32), "eventID": (0.0, f64),
    "mc_vtx": ([0.0, 0.0, 7000.0, 0.0], fixed(4)), "mc_targetZ": (6, i32), "mc_targetA": (12, i32), "truth_targetID": (0, i32),
    "mc_nFSPart": (1, i32), "mc_FSPartPDG": ([13], jagged(i32)),
    "mc_FSPartPx": ([px], jagged(f64)), "mc_FSPartPy": ([py], jagged(f64)), "mc_FSPartPz": ([pz], jagged(f64)), "mc_FSPartE": ([E], jagged(f64)),
    "mc_primFSLepton": ([px, py, pz, E], fixed(4)), "mc_incoming": (14, i32), "mc_incomingE": (E, f64),
    "mc_current": (1, i32), "mc_intType": (1, i32), "mc_Q2": (0.0, f64), "mc_w": (938.272, f64)}
arrays = {k: pa.array([v] * N, t) for k, (v, t) in cols.items()}
arrays["mc_nthEvtInFile"] = pa.array(np.arange(N, dtype=np.int32))
arrays["mc_subrun"] = pa.array(np.arange(N, dtype=np.int32) // 100)    # events are batched by subrun
arrays["eventID"] = pa.array(1e7 + np.arange(N, dtype=float))
schema = pa.schema([pa.field(k, a.type, nullable=False) for k, a in arrays.items()])
pq.write_table(pa.table(arrays, schema=schema), "mu3gev.truth.parquet")
```

> :warning: **Important:** write every column as **non-nullable**, as above. With pyarrow's default (nullable)
> columns the model runs but the ROOT writer fails at the end with
> `cannot write Awkward Array type to ROOT file: option[...]`.

> **TODO: a real generator example.** The muon gun shows the layout; a full neutrino interaction might help too.
> A few ideas:
> - A small example truth file of real NuWro or GiBUU events (like `tests/data/`), used in A.2-A.3.
> - A short mapping from one generator's native output (e.g. NUISANCE flat tree, GENIE `gst`) to the columns above.
> - How the vertices were placed in the detector geometry for the paper's samples, or a pointer to the tool.
> - Setting `gen_weight` and `gen_pot` to normalise predictions to the data exposure.

### A.2 Run the model

With the deployed model, `weights/full`:

```bash
python scripts/infer/surrogate_to_ntuple.py weights/full mu3gev.truth.parquet mu3gev.root
```

Add `--ensemble weights/full_s1 weights/full_s2 weights/full_s3` to sample from the four-member ensemble and write its
model uncertainty as systematic universes (see [uncertainties](#inputs-outputs-and-uncertainties)). The script uses a GPU
if one is available and the CPU otherwise; `--device cpu` or `--device cuda` chooses explicitly.

### A.3 Check the output

```python
import numpy as np, uproot
f = uproot.open("mu3gev.root")
print(f["Truth"].num_entries, "generated,", f["MasterAnaDev"].num_entries, "reconstructed")
p = f["MasterAnaDev"]["MasterAnaDev_muon_P"].array(library="np")          # MeV
print(f"muon momentum: median {np.median(p):.0f} MeV")
```

With the default seed this prints `1000 generated, 945 reconstructed` and a median of 3,016 MeV. In physics terms: the
muon comes back with ~4% energy resolution and a median angle to the beam of 0.36 degrees, and about one in twenty is not
reconstructed.

### A.4 Optional: the full open-dataset schema

The surrogate writes only the branches it models. To run analysis code that binds every branch of the open-data tuple,
extend it to the full 3,687-branch schema: the branches the surrogate does not model are filled with the open dataset's
default values (they carry no information). Any open-data AnaTuple serves as the reference; only its first 20,000 entries
are read, so it can be streamed:

```bash
python scripts/infer/conform_ntuple.py mu3gev.root \
    root://fndcadoor.fnal.gov:1095/pnfs/fnal.gov/usr/minerva/persistent/OpenData/MediumEnergy_FHC/MC/StandardMC/Playlist1A/MasterAnaDev_mc_AnaTuple_run00110000_Playlist.root \
    mu3gev_full.root --strict
```

### A.5 Use the output in an analysis

> **TODO: a short worked example.** The universe recipe is in `docs/surrogate_files.md`; a quick example here might
> help. A few ideas:
> - A ~10-line muon-momentum histogram with the four universes as a band.
> - The model-error floor added in quadrature (2.6% per bin for muon momentum).
> - An efficiency: `MasterAnaDev` over `Truth`.
> - Weighting a generator sample by `gen_weight` and scaling by `POT_data / gen_pot`.
> - Pointers to the paper's example analysis and the MINERvA 101 tutorial.

### Large generator samples

For samples of millions of events produced in many jobs: concatenate the per-job shards, optionally thin them to a
smaller exposure, and run the ensemble on a GPU in slices of 2 M events, which are merged into one file. Each slice holds
about 10 GB in memory; a slice whose output exists is skipped, so an interrupted run resumes. Expect hours for a sample of
several million events:

```bash
python scripts/curate/merge_truth_shards.py data/nuwro.truth.parquet shards/*.parquet
python scripts/curate/thin_truth.py data/nuwro.truth.parquet data/nuwro_1e20.truth.parquet --pot 1e20   # optional
bash scripts/infer/infer_sliced.sh data/nuwro_1e20.truth.parquet out_nuwro.root
```

An event weight and the sample's exposure travel with the events as `gen_weight` and `gen_pot`
([`docs/generator_input.md`](docs/generator_input.md)). Placing the vertices in the detector geometry for each nucleus is
the generator side's job; the model is trained on true vertices with z between 4000 and 8700 mm.

### Running on CPU

Without a GPU, `surrogate_to_ntuple.py` runs on the CPU automatically; nothing needs to change. Small studies are fine
this way: on two CPU cores the model processes roughly 10 neutrino events per second per ensemble member (single-particle
events are faster), so up to ~10,000-100,000 events is comfortable. Full generator samples need a GPU: 4 M events through
the four-member ensemble would take days on CPU. Training (path C) and the response figure (B.2) also fall back to the CPU, which is fine for a quick
test on a few files but far too slow for the full dataset: use a GPU there.

---

## B. Redraw the paper's figures and tables

| also needs | time | you get |
|---|---|---|
| runs on CPU (the response figure needs the dataset from C and a GPU) | seconds | `paper/figures/*.png`, `paper/tables/*.tex` |

These steps reproduce the figures and tables in the paper.

### B.1 From the shipped inputs

From the released weights and the evaluation outputs in `paper/inputs/`, without the open dataset:

```bash
mkdir -p paper/figures
python paper/fig_confusion.py --eval weights/full                                            # prong multiplicity vs true charged hadrons
python paper/fig_ensemble.py --spread paper/inputs/ensemble/spread.json --out paper/figures   # ensemble spread and pulls
python paper/tables.py                                                                       # closure tables -> paper/tables/
```

### B.2 The response figure

The response figure samples the model on the open dataset's test events, so it needs the curated dataset
([C.1](#c1-curate)) and a GPU:

```bash
python paper/fig_response.py paper/figures/full --models A=weights/full --n 300000
```

---

## C. Rebuild everything from the open dataset

| also needs | time | you get |
|---|---|---|
| xrootd access, ~12 GB of disk, a CUDA GPU | scales with the number of training events: ~4 h per model for the full 15.8 M events on an RTX 3090, five models | retrained models in `reports/`, regenerated `paper/inputs/` |

The full chain, from MINERvA's tuples to the released weights and the paper's numbers. It needs a GPU and takes about a day.

### C.1 Curate

Stream the 41 open-data files of playlist ME1A (about 20 GB each) from Fermilab's xrootd door into slimmed Parquet, about
300 MB per file. Only the needed branches are read (~1.5% of each file); files already slimmed are skipped, so the command
can be re-run after an interruption. The xrootd door is reachable off-site.

```bash
python scripts/curate/slim_remote.py configs/playlist1A_files.txt data/slim_1A --budget-gb 9.5
```

To try the chain on a few files first, use `--max-files 4` (about 1.5 M training-volume events) and pass `--n-files 4`
to the training scripts in [C.3](#c3-train).

:warning: **Slimming takes a while, so run it in the background.** The output for a file appears only once the whole
file is done, so a quiet log for the first few minutes is normal. Expect the slowest runs at sites far from Fermilab,
where every read is a long round trip. Started like this, the job survives logging out, and `-u` keeps the log
up to date:

```bash
nohup python -u scripts/curate/slim_remote.py configs/playlist1A_files.txt data/slim_1A --budget-gb 9.5 > slim.log 2>&1 &
tail -f slim.log    # Ctrl-C stops watching, not the job
```

On a cluster with several login nodes, note which one you started it on: the process is only visible from there.

### C.2 The GENIE reference sample

Open-dataset truth events with the open dataset's own reconstruction attached as `ref_*` columns, for comparing the
surrogate with MINERvA's reconstruction of the same events:

```bash
python scripts/curate/genie_truth.py data/genie_test.truth.parquet --population all --volume full
python scripts/infer/surrogate_to_ntuple.py weights/full data/genie_test.truth.parquet out.root \
    --ensemble weights/full_s1 weights/full_s2 weights/full_s3
```

### C.3 Train

`train_all.sh` trains the five released models (the deployed model, three ensemble members with seeds 1-3, and the 2p2h
holdout model), three stages each. Training time scales with the number of training events: on the full dataset (15.8 M
events in the training volume) it is about 4 h per model on an RTX 3090, so about 20 h in all. It writes to `reports/`,
not `weights/`; every stage is skipped when its output exists.

```bash
bash scripts/train/train_all.sh
```

### C.4 Evaluate

The evaluations behind `paper/inputs/` and the figures, on the retrained models:

```bash
python scripts/evaluate/closure_classifier.py --a reports/full --b reports/holdout_2p2h --out paper/inputs/closure/closure.json
python scripts/evaluate/ensemble_spread.py --members reports/full reports/full_s1 reports/full_s2 reports/full_s3 --out reports/ensemble
python scripts/evaluate/holdout_compare.py reports/holdout --a reports/eval_full_on_2p2h --b reports/eval_holdout_on_2p2h
```

Then draw the figures as in [B](#b-redraw-the-papers-figures-and-tables), pointing `--eval` and `--models` at
`reports/` instead of `weights/`.

---

## Reference

### The model

![model schematic](paper/schematic/model_schematic.png)

The model is trained in two stages, marked on the schematic: the **event stage** trains the set encoder, the event flag
heads, the hadron multiplicity head and the event flow; the **prong stage** adds the vertex-plane head and the prong set flow
and fine-tunes every parameter. A short third pass re-trains the vertex-plane head alone, conditioned on the sampled
vertex. See [`docs/model.md`](docs/model.md).

### Training sample

MINERvA groups its runs into *playlists*, run periods with fixed beam and detector conditions
([MINERvA playlists](https://minerva.fnal.gov/minerva-playlists/)). The model is trained on the open simulation of playlist
**ME1A**: the Medium Energy NuMI beam (~6 GeV flux peak) in forward-horn-current (neutrino-dominated) mode, with the water
and helium targets empty and 6+0 slip stacking. ME1A corresponds to 0.9 x 10^20 protons on target of data, about 10% of
MINERvA's medium-energy neutrino-mode exposure. XSuM-MINERvA is a surrogate for ME FHC 1A: it reproduces the detector
response of those conditions (see [Scope and limitations](#scope-and-limitations)).

### Inputs, outputs and uncertainties

The model takes the generator truth of one event: vertex position, target nucleus, and the list of final-state particles
after final-state interactions (species and three-momentum; hadrons above 10 MeV kinetic energy, neutrons included).
It returns whether the event is reconstructed, the muon momentum and direction, the reconstructed vertex, three
calorimetric energies, the number of hadron prongs and, per prong, direction, momentum under the pion and proton
hypotheses, proton score and the fit flags. The dead-time cut variable is not modelled; it is drawn from the open dataset's
distribution. Like MINERvA's reconstruction, the surrogate builds each event around a muon candidate: an event without
a muon is either not reconstructed, or another particle is reconstructed as the muon.

The output file has two trees like the open dataset: `Truth` with every input event and `MasterAnaDev` with the
reconstructed ones. Uncertainties: the surrogate is stochastic (its sampling spread is the learned detector resolution),
and the spread between the four ensemble members is the model uncertainty, carried as systematic universes with one
weight per reconstruction flag. The output branches and the recipe to propagate the universes are in
[`docs/surrogate_files.md`](docs/surrogate_files.md); the input file layout is in [`docs/generator_input.md`](docs/generator_input.md).

### Scope and limitations

XSuM-MINERvA is a surrogate for MINERvA's **ME FHC playlist 1A** simulation.

- **What it models.** Per event: whether the event is reconstructed, the MINOS match and muon charge sign, the muon
  momentum and direction, the reconstructed vertex, three calorimetric energies and the number of prongs. Per hadron
  prong: direction, momentum under the pion and proton hypotheses, proton score and the fit flags
  ([inputs and outputs](#inputs-outputs-and-uncertainties)).
- **Model uncertainty.** The ensemble universes cover the event-level quantities up to and including the prong count.
  The per-prong variables come from the nominal model only.
- **Normalisation.** The surrogate does not normalise predictions. Event weights and the sample's exposure come with the
  generator sample (`gen_weight`, `gen_pot`; see [`docs/generator_input.md`](docs/generator_input.md)).
- **Events without a muon.** As in the open dataset, these are either not reconstructed, or another particle is
  reconstructed as the muon.

> **TODO (Linyan):** the phase space the training data cover (E_nu, muon P and angle, multiplicity), and the behaviour
> outside it.

### Repository layout

| path | content |
|---|---|
| `xsum/` | the package: tuple I/O, input preparation, model, training, evaluation |
| `scripts/curate/` | slim the open-data tuples, build the truth file of the open dataset, merge and thin generator truth files |
| `scripts/train/` | the training stages and `train_all.sh`, the chain that produced the released weights |
| `scripts/infer/` | run a model on a truth file (in slices for large ones), write the ntuple, conform it to the full open-dataset schema |
| `scripts/evaluate/` | closure classifier, ensemble spread, 2p2h holdout comparison |
| `paper/` | the paper's figures and tables, regenerated from `paper/inputs/` and the weights |
| `weights/` | the deployed model (`full`), its three ensemble members (`full_s1..3`) and the 2p2h holdout model |
| `docs/` | output-file format and the uncertainty recipe (`surrogate_files.md`), the input format (`generator_input.md`), the model, tuple conventions |
| `tests/` | unit tests on a small example of the open dataset (`tests/data/`) |

### Troubleshooting

Common problems and their fixes.

| symptom | path | cause and fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'xsum'` from `pytest` | Quickstart | run `python -m pytest`, or `pip install -e .` once in the virtualenv |
| training or evaluation is very slow | B.2, C | no GPU was found, so it fell back to the CPU: fine for a test, but use a GPU for real runs. Check with `python -c "import torch; print(torch.cuda.is_available())"` |
| `cannot write Awkward Array type to ROOT file: option[...]` | A | the truth file has nullable columns: write them non-nullable ([A.1](#a1-write-a-truth-file)) |
| output has no `phys_n_dead_discr_pair_upstream_prim_track_proj` branch | A | the script was run outside the repository root, so `data/deadtime_marginal.json` was not found: run from the root, or pass `--deadtime /path/to/XSuM-MINERvA/data/deadtime_marginal.json` |
| `FileNotFoundError: paper/figures/...` | B | `mkdir -p paper/figures` |
| `pip install` tries to build `numpy` from source and fails | Quickstart | `numpy<2.1` may have no prebuilt package for your Python (seen with 3.13), and the build needs a recent compiler: make the environment with Python 3.11 instead, for example `conda create -p ./xsum-env python=3.11` |
| `slim_remote.py` prints nothing and no Parquet appears for a long time | C | normal for the first file, which is written only when complete; if a file is still missing after an hour, the storage pool serving it may be slow: stop the run and start it again, as finished files are skipped |
| `slim_remote.py` skips a file with `Pool unavailable` | C | that file's storage pool at Fermilab is offline for now: re-run later and only the missing files are fetched |
| `slim_remote.py` reports `FAILED` for every file | C | no route to `fndcadoor.fnal.gov:1095`: check xrootd access with `python -c "import uproot; print(uproot.open('$(head -1 configs/playlist1A_files.txt)').keys())"` |

### Support and contributing

> **TODO: where to get help.** A few ideas:
> - GitHub issues for problems, with what to include (command, error, versions).
> - A contact for the paper.
> - Whether contributions (e.g. converters for other generators) are welcome.
> - The [open data page](https://minerva.fnal.gov/opendata/) for questions about the MINERvA data itself.

### License

Copyright 2026 Fermi Forward Discovery Group, LLC. The code, the model weights and the example data in this repository
are released under the Apache License, Version 2.0 (see `LICENSE`). Fermilab is operated by Fermi Forward Discovery
Group, LLC under Contract No. 89243024CSC000002 with the U.S. Department of Energy.

### Citation

Paper in preparation; a citation will be added here. If you use the surrogate, please also cite the MINERvA open data
product, [doi:10.15484/3022562](https://doi.org/10.15484/3022562), and the MINERvA detector paper, Nucl. Instrum. Meth. A 743 (2014) 130.
