import awkward as ak
import numpy as np
import uproot

from xsum.io.writer import write_ntuple
from xsum.prep.muon import decode_muon
from xsum.prep.prongs import decode, encode


def test_decode_muon_matches_tuple(reco):
    px, py, pz = (ak.to_numpy(reco[f"MasterAnaDev_muon_P{c}"]) for c in "xyz")
    ok = ak.to_numpy(reco["MasterAnaDev_minos_trk_is_ok"])
    qneg = ak.to_numpy(reco["MasterAnaDev_muon_qp"]) < 0
    out = decode_muon(px, py, pz, ok, qneg)
    assert np.allclose(out["MasterAnaDev_muon_P"], ak.to_numpy(reco["MasterAnaDev_muon_P"]), rtol=1e-6)
    # E is on-shell in the tuple only for MINOS-matched muons (see decode_muon)
    assert np.allclose(out["MasterAnaDev_muon_E"][ok], ak.to_numpy(reco["MasterAnaDev_muon_E"])[ok], rtol=1e-6)
    dth = np.abs(out["MasterAnaDev_muon_theta"] - ak.to_numpy(reco["MasterAnaDev_muon_theta"]))
    dph = np.abs(np.angle(np.exp(1j * (out["muon_phi"] - ak.to_numpy(reco["muon_phi"])))))
    print(f"\nmuon theta: median |diff| {np.median(dth):.2e} rad, max {dth.max():.2e}; phi median {np.median(dph):.2e}")
    assert dth.max() < 1e-3 and np.median(dph) < 5e-3
    assert (out["MasterAnaDev_nuHelicity"] == ak.to_numpy(reco["MasterAnaDev_nuHelicity"])).mean() > 0.999


def test_writer_round_trip(reco, reco_typenames, tmp_path):
    branches = decode(encode(reco), "flag")
    branches["MasterAnaDev_muon_P"] = ak.to_numpy(reco["MasterAnaDev_muon_P"])
    branches["MasterAnaDev_vtx"] = ak.to_numpy(reco["MasterAnaDev_vtx"])
    branches["MasterAnaDev_minos_trk_is_ok"] = ak.to_numpy(reco["MasterAnaDev_minos_trk_is_ok"])
    path = str(tmp_path / "pruned.root")
    write_ntuple(path, branches)
    t = uproot.open(path)["MasterAnaDev"]
    for k in branches:
        assert t[k].typename == reco_typenames[k], (k, t[k].typename, reco_typenames[k])
    for k in ("MasterAnaDev_sec_protons_P_fromdEdx",):
        assert t[k + "_sz"].typename == reco_typenames[k + "_sz"]
    back = t.arrays(list(branches))
    assert np.allclose(ak.to_numpy(back["MasterAnaDev_pion_P"]), branches["MasterAnaDev_pion_P"])
    assert ak.all(back["MasterAnaDev_sec_protons_P_fromdEdx"] == branches["MasterAnaDev_sec_protons_P_fromdEdx"])
