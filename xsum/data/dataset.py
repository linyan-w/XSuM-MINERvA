"""Torch dataset over the slimmed Parquet files: padded particle sets, context, and Tier 0/1 targets."""
from __future__ import annotations

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

from ..prep.particles import context_from_tuple, in_training_population, select_from_tuple

TIER1_FIELDS = ["MasterAnaDev_muon_Px", "MasterAnaDev_muon_Py", "MasterAnaDev_muon_Pz",
                "MasterAnaDev_recoil_E", "MasterAnaDev_recoil_passivecorrected", "MasterAnaDev_hadron_recoil",
                "recoil_energy_nonmuon_nonvtx100mm"]


def split_by_subrun(subrun: np.ndarray, fractions=(0.8, 0.1, 0.1), seed: int = 0) -> np.ndarray:
    """Assign each event to split 0/1/2 by its subrun, so overlay-correlated events stay together."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subrun)
    rng.shuffle(uniq)
    cuts = np.cumsum(np.array(fractions) / np.sum(fractions)) * len(uniq)
    split_of = np.zeros(len(uniq), dtype=np.int8)
    split_of[int(cuts[0]):] = 1
    split_of[int(cuts[1]):] = 2
    lut = dict(zip(uniq, split_of))
    return np.array([lut[s] for s in subrun], dtype=np.int8)


class EventDataset(Dataset):
    """One item = (particle classes [n_max], particle momenta [n_max,3], mask, context[5], targets dict).

    `reco` is the slimmed MasterAnaDev array (events that were reconstructed); `truth` the slimmed Truth array.
    Events in `truth` that are not in `reco` get reco_exists=0 and NaN Tier 1 targets.
    """

    def __init__(self, reco: ak.Array, truth: ak.Array, n_max: int = 64, ke_cut_mev: float = 10.0,
                 z_min: float = 4000.0):
        key_r = _event_key(reco); key_t = _event_key(truth)
        in_reco = np.isin(key_t, key_r)
        pop = in_training_population(truth, z_min=z_min)
        self.truth = truth[pop]
        self.reco_exists = in_reco[pop]
        # align reco rows to truth rows
        pos = {k: i for i, k in enumerate(key_r)}
        self.reco_index = np.array([pos.get(k, -1) for k in key_t[pop]])
        self.reco = reco
        self.n_max = n_max
        parts = select_from_tuple(self.truth, ke_cut_mev)
        self.cls = ak.to_numpy(ak.fill_none(ak.pad_none(parts["cls"], n_max, clip=True), 0)).astype(np.int64)
        mom = [ak.to_numpy(ak.fill_none(ak.pad_none(parts[c], n_max, clip=True), 0.0)) for c in ("px", "py", "pz")]
        self.mom = np.stack(mom, -1).astype(np.float32)
        self.mask = self.cls > 0
        self.ctx = context_from_tuple(self.truth).astype(np.float32)
        self.subrun = ak.to_numpy(self.truth["mc_subrun"])
        t1 = np.full((len(self.truth), len(TIER1_FIELDS)), np.nan, dtype=np.float32)
        ok = self.reco_index >= 0
        for j, f in enumerate(TIER1_FIELDS):
            t1[ok, j] = ak.to_numpy(reco[f])[self.reco_index[ok]]
        self.tier1 = t1
        minos = np.zeros(len(self.truth), bool); charge_neg = np.zeros(len(self.truth), bool)
        minos[ok] = ak.to_numpy(reco["MasterAnaDev_minos_trk_is_ok"])[self.reco_index[ok]]
        charge_neg[ok] = ak.to_numpy(reco["MasterAnaDev_muon_qp"])[self.reco_index[ok]] < 0
        self.tier0 = np.stack([self.reco_exists, minos, charge_neg], -1).astype(np.float32)
        nprong = np.full(len(self.truth), -1, dtype=np.int64)
        nprong[ok] = ak.to_numpy(reco["MasterAnaDev_hadron_number"])[self.reco_index[ok]]
        self.nprong = nprong

    def __len__(self):
        return len(self.truth)

    def __getitem__(self, i):
        return {
            "cls": torch.from_numpy(self.cls[i]), "mom": torch.from_numpy(self.mom[i]),
            "mask": torch.from_numpy(self.mask[i]), "ctx": torch.from_numpy(self.ctx[i]),
            "tier0": torch.from_numpy(self.tier0[i]), "tier1": torch.from_numpy(self.tier1[i]),
        }


def _event_key(arr: ak.Array) -> np.ndarray:
    run = ak.to_numpy(arr["mc_run"]).astype(np.int64)
    sub = ak.to_numpy(arr["mc_subrun"]).astype(np.int64)
    nth = ak.to_numpy(arr["mc_nthEvtInFile"]).astype(np.int64)
    return (run * 100_000 + sub) * 1_000_000 + nth
