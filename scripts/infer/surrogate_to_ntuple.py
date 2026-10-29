#!/usr/bin/env python
"""Run the surrogate on truth events and write a pruned MasterAnaDev-style ntuple (docs/PROJECT.md 13.4).

Usage: scripts/infer/surrogate_to_ntuple.py MODEL_DIR TRUTH.parquet OUT.root [--n N] [--steps 64] [--seed 0]
Input: a slimmed Truth-tree Parquet (or any file with the same truth branches). Output tree `MasterAnaDev`
with the truth passthrough branches, the sampled Tier 0 flags, the muon/vertex/calorimetry blocks, the prong
multiplicity and, for an M3 model, the prong branches (pion/proton/sec_protons blocks) and a plane-snapped
vertex z. Only events the surrogate marks as reconstructed are written (like the real tuple).
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak, numpy as np, torch
from torch.utils.data import DataLoader

from xsum.data.compact import CompactDataset, collate
from xsum.io.writer import write_ntuple
from xsum.io.deadtime import sample_deadtime, VARIABLE as DEADTIME_VARIABLE, DEFAULT_TABLE as DEADTIME_TABLE
from xsum.prep.muon import decode_muon
from xsum.prep.particles import context_from_tuple, in_population, reference_momentum, select_from_tuple
from xsum.prep.prongs import decode as decode_prongs
from xsum.train.event_stage import to_dev
from xsum.data.compact import selection_from_config
from xsum.train import event_stage as m2, prong_stage as m3

PASSTHROUGH = ["mc_run", "mc_subrun", "mc_nthEvtInFile", "eventID", "mc_vtx", "mc_targetZ", "mc_targetA", "truth_targetID",
               "mc_nFSPart", "mc_FSPartPDG", "mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE", "mc_primFSLepton",
               "mc_incoming", "mc_current", "mc_intType"]


def truth_to_compact(truth, ke_cut_mev=10.0, keep_neutrons=True, population="ccnumu"):
    parts = select_from_tuple(truth, ke_cut_mev, keep_neutrons, population); n = ak.to_numpy(ak.num(parts["cls"])).astype(np.int32)
    mu_true = reference_momentum(truth, population, ke_cut_mev)
    N = len(truth)
    return {"offsets": np.concatenate([[0], np.cumsum(n)]).astype(np.int64),
            "cls": ak.to_numpy(ak.flatten(parts["cls"])).astype(np.int8),
            "mom": np.stack([ak.to_numpy(ak.flatten(parts[c])) for c in ("px", "py", "pz")], 1).astype(np.float32),
            "ctx": context_from_tuple(truth).astype(np.float32), "mu_true": mu_true,
            "reco_exists": np.zeros(N, bool), "tier0": np.zeros((N, 3), np.float32), "nprong": np.zeros(N, np.int8),
            "tier1": np.full((N, 11), np.nan, np.float32), "subrun": ak.to_numpy(truth["mc_subrun"]).astype(np.int32),
            "prongs": np.zeros((0, 9), np.float32), "p_offsets": np.zeros(N + 1, np.int64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("truth"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--steps", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadtime", default=DEADTIME_TABLE if pathlib.Path(DEADTIME_TABLE).exists() else None, help="empirical table of the dead-time cut variable (scripts/curate/deadtime_table.py); drawn per event by reconstructed vertex z, independently of the event; '' to skip")
    ap.add_argument("--ensemble", nargs="*", default=[], help="further members of an ensemble (model directories trained with other seeds, scripts/train/train_all.sh): the nominal sample draws each batch from a random member, and the epistemic columns are the spread between the members' common-noise draws instead of posterior draws")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    truth = ak.from_parquet(a.truth)
    ck = torch.load(pathlib.Path(a.model_dir) / "model.pt", map_location="cpu", weights_only=False); sel = selection_from_config(ck["config"])
    truth = truth[in_population(truth, sel["population"])]
    if a.n: truth = truth[:a.n]
    tier2 = ck["config"].get("tier2", False)
    d = truth_to_compact(truth, **sel)
    if tier2:
        model, tf, ptf = m3.load_model(pathlib.Path(a.model_dir) / "model.pt")
    else:
        model, tf = m2.load_model(pathlib.Path(a.model_dir) / "model.pt"); ptf = None
    ds = CompactDataset(d, np.arange(len(truth)), tf, a.seed, prong_tf=ptf)  # ptf=None -> Tier 1 only
    loader = lambda: DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4)
    members = [model]
    for md in a.ensemble:  # ensemble members share the split and the transforms (same data, same transform seed); checked on the scalings
        m_, tf_, _ = m3.load_model(pathlib.Path(md) / "model.pt"); assert np.allclose(tf_.loc, tf.loc) and np.allclose(tf_.scale, tf.scale), f"{md}: transform differs from {a.model_dir}"; members.append(m_)
    if a.ensemble: print(f"ensemble of {len(members)} members: nominal sample per batch from a random member, epistemic columns from the spread between members", flush=True)
    MEMBER = []

    def generate(seed, prongs=True, model=model, mixture=False):
        """One full generation pass; a fixed seed gives the same base noise, so repeated passes differ only through the weights.
        prongs=False skips the prong-set flow (the epistemic draws use only event-level outputs and the vertex class).
        mixture: each batch is drawn from a random ensemble member (recorded in MEMBER)."""
        S, PR, PM, VC, P0, PN = [], [], [], [], [], []; rng = np.random.default_rng(seed + 77)
        for i, b in enumerate(loader()):
            torch.manual_seed(seed * 100003 + i)  # per-batch seed: the event-level draw is identical between passes with and without prongs and across members (common noise)
            m_ = members[rng.integers(len(members))] if mixture else model
            if mixture: MEMBER.append(np.full(len(b["ctx"]), members.index(m_), np.int32))
            s = m_.sample(to_dev(b, "cuda"), a.steps, prongs=prongs)
            S.append(torch.cat([s["exist"][:, None].float(), s["minos"][:, None].float(), s["charge"][:, None].float(), s["nprong"][:, None].float(), s["x1"]], 1).cpu())
            P0.append(s["p0"].cpu()); PN.append(s["pn"].cpu())
            if tier2: PR.append(s["prongs"].cpu()); PM.append(s["pmask"].cpu()); VC.append(s["vclass"].cpu())
        return torch.cat(S).numpy(), PR, PM, VC, torch.cat(P0).numpy(), torch.cat(PN).numpy()

    S, PR, PM, VC, P0, _ = generate(a.seed, mixture=bool(a.ensemble))
    # ---- model uncertainty: one common-noise draw per ensemble member (the universes) and the between-member spread per event
    epi = None
    if a.ensemble:
        # one common-noise draw per member (event-level outputs only); the analytic flags become the between-member variances
        # of the flag logits, of the multiplicity log-probabilities and of the model-space event vector
        draws = [generate(a.seed, prongs=False, model=m_) for m_ in members]
        lg = np.stack([np.log(np.clip(dr[4], 1e-6, 1 - 1e-6) / np.clip(1 - dr[4], 1e-6, None)) for dr in draws]); ln = np.stack([np.log(np.clip(dr[5], 1e-9, None)) for dr in draws]); xs = np.stack([dr[0][:, 4:] for dr in draws])
        epi = {"logit_var": lg.var(0, ddof=1), "card_var": ln.var(0, ddof=1).mean(1), "flow_var": xs.var(0, ddof=1), "draws": [dr[:5] for dr in draws]}
        a.epi_draws = len(members); print(f"epistemic uncertainties from the {len(members)} ensemble members (common noise)", flush=True)
    exist = S[:, 0] > 0
    y = tf.inverse(S[exist, 4:], d["mu_true"][exist], d["ctx"][exist])
    if tier2:  # snap the vertex z to a plane when the class head says so
        vc = torch.cat(VC).numpy()[exist]
        zs = tf.planes.z_for_class(vc, d["ctx"][exist, 2], y[:, 5]); y[:, 5] = np.where(vc > 0, zs, y[:, 5])
    minos, charge = S[exist, 1] > 0, S[exist, 2] > 0
    out = {k: truth[k][exist] for k in PASSTHROUGH + [c for c in truth.fields if c.startswith(("nuwro_", "gibuu_", "gen_", "ref_"))]}  # generator-specific extras pass through
    out.update(decode_muon(y[:, 0], y[:, 1], y[:, 2], minos, charge))
    vtx = np.zeros((exist.sum(), 4)); vtx[:, :3] = y[:, 3:6]
    out["MasterAnaDev_vtx"] = vtx; out["vtx"] = vtx.copy()
    if a.deadtime: out[DEADTIME_VARIABLE] = sample_deadtime(vtx[:, 2], np.random.default_rng(a.seed + 7), a.deadtime)  # overlay dead time: not modelled, drawn from the open dataset's distribution
    out["MasterAnaDev_recoil_E"] = y[:, 6]; out["MasterAnaDev_recoil_E_wide_window"] = y[:, 6]; out["MasterAnaDev_hadron_recoil_CCInc"] = y[:, 6]
    out["MasterAnaDev_recoil_passivecorrected"] = y[:, 7]; out["MasterAnaDev_hadron_recoil_default"] = y[:, 7]
    out["MasterAnaDev_hadron_recoil"] = y[:, 8]; out["recoil_energy_nonmuon_nonvtx100mm"] = y[:, 9]; out["nonvtx_iso_blobs_energy"] = y[:, 10]
    n = S[exist, 3].astype(np.int32)
    out["MasterAnaDev_hadron_number"] = n; out["n_prongs"] = n + 1; out["multiplicity"] = n + 1
    if tier2:  # prong branches via the canonical prong table (docs/PROJECT.md 13.3)
        Npad = max(p.shape[1] for p in PR)
        padp = lambda p, v=0.0: torch.nn.functional.pad(p, (0, 0, 0, Npad - p.shape[1]) if p.dim() == 3 else (0, Npad - p.shape[1]), value=v)
        pr = torch.cat([padp(p) for p in PR]).numpy()[exist]; pm = torch.cat([padp(p, False) for p in PM]).numpy()[exist]
        flat = ptf.inverse(pr[pm]); counts = pm.sum(1)
        rec = ak.unflatten(ak.zip({"theta": flat[:, 0], "phi": flat[:, 1], "has_pion_fit": flat[:, 2] > 0.5, "pi_P": flat[:, 3],
                                   "has_proton_fit": flat[:, 4] > 0.5, "p_P": flat[:, 5], "p_score1": flat[:, 6],
                                   "is_primary_proton": flat[:, 7], "is_exiting": flat[:, 8].astype(np.int32),
                                   "tm_pdg": np.zeros(len(flat), np.int32), "tm_fraction": np.zeros(len(flat))}), counts)
        # primary proton: highest is_primary score among prongs with a proton fit
        prim = ak.to_numpy(ak.fill_none(ak.pad_none(rec["is_primary_proton"], Npad, clip=True), -np.inf))
        best = np.zeros_like(prim, dtype=bool); has = np.isfinite(prim).any(1)
        best[np.where(has)[0], prim[has].argmax(1)] = True
        rec = ak.with_field(rec, ak.unflatten(best[pm], counts), "is_primary_proton")
        out.update(decode_prongs(rec, choose_primary="flag"))
        out["MasterAnaDev_hadron_number"] = n; out["n_prongs"] = n + 1
    out["surrogate_p_reco_exists"] = P0[exist, 0].astype(np.float64)
    if a.ensemble: out["surrogate_member"] = np.concatenate(MEMBER)[exist]
    if epi is not None:
        # analytic (model space): std of the flag logits, mean multiplicity-logit variance, event-flow flag (sum of 9 variances)
        out["surrogate_epi_logit_reco_exists"] = np.sqrt(epi["logit_var"][exist, 0]); out["surrogate_epi_logit_minos"] = np.sqrt(epi["logit_var"][exist, 1]); out["surrogate_epi_logit_charge"] = np.sqrt(epi["logit_var"][exist, 2])
        out["surrogate_epi_multiplicity_logits"] = np.sqrt(epi["card_var"][exist]); out["surrogate_epi_flow_flag"] = epi["flow_var"][exist].sum(1)
        # K-draw std in physical units for the written event-level variables, and of the probabilities
        Ys = []; Ps = []; Ns = []
        for Sk, PRk, PMk, VCk, P0k in epi["draws"]:
            yk = tf.inverse(Sk[exist, 4:], d["mu_true"][exist], d["ctx"][exist])
            if tier2:
                vck = torch.cat(VCk).numpy()[exist]; zk = tf.planes.z_for_class(vck, d["ctx"][exist, 2], yk[:, 5]); yk[:, 5] = np.where(vck > 0, zk, yk[:, 5])
            Ys.append(yk); Ps.append(P0k[exist]); Ns.append(Sk[exist, 3])
        Ys = np.array(Ys); sd = Ys.std(0)
        if a.ensemble:
            # universes: member k's event-level outputs for every written event (common noise), and the weight that turns the
            # nominal row set into member k's reconstruction probabilities (Bernoulli ratios for the reco, MINOS and charge
            # flags given the nominal outcomes); the analysis fills one histogram per universe with these values and weights
            from xsum.prep.frames import theta_phi_beam
            K = len(Ys); Pn = P0[exist]; Pk = np.array(Ps); eps = 1e-6
            w = np.clip(Pk[:, :, 0], eps, 1) / np.clip(Pn[None, :, 0], eps, 1)
            w *= np.where(minos[None, :], np.clip(Pk[:, :, 1], eps, 1) / np.clip(Pn[None, :, 1], eps, 1), np.clip(1 - Pk[:, :, 1], eps, 1) / np.clip(1 - Pn[None, :, 1], eps, 1))
            w *= np.where(minos[None, :], np.where(charge[None, :], np.clip(Pk[:, :, 2], eps, 1) / np.clip(Pn[None, :, 2], eps, 1), np.clip(1 - Pk[:, :, 2], eps, 1) / np.clip(1 - Pn[None, :, 2], eps, 1)), 1.0)
            out["surrogate_univ_weight"] = np.ascontiguousarray(w.T)
            Pu = np.linalg.norm(Ys[:, :, :3], axis=2); thu = np.array([theta_phi_beam(y[:, 0], y[:, 1], y[:, 2])[0] for y in Ys])
            for name, arr in (("muon_P", Pu), ("muon_theta", thu), ("muon_Px", Ys[:, :, 0]), ("muon_Py", Ys[:, :, 1]), ("muon_Pz", Ys[:, :, 2]), ("vtx_x", Ys[:, :, 3]), ("vtx_y", Ys[:, :, 4]), ("vtx_z", Ys[:, :, 5]),
                              ("recoil_E", Ys[:, :, 6]), ("recoil_nonvtx100", Ys[:, :, 9]), ("nonvtx_iso_blobs_E", Ys[:, :, 10])):
                out[f"surrogate_univ_{name}"] = np.ascontiguousarray(arr.T.astype(np.float64))
            out["surrogate_univ_n_prongs"] = np.ascontiguousarray((np.array(Ns).astype(np.int32) + 1).T); out["surrogate_univ_members"] = np.full(exist.sum(), K, np.int32)
        for j, c in enumerate("xyz"): out[f"surrogate_epi_muon_P{c}"] = sd[:, j]
        out["surrogate_epi_muon_P"] = np.linalg.norm(Ys[:, :, :3], axis=2).std(0)
        for j, c in enumerate("xyz"): out[f"surrogate_epi_vtx_{c}"] = sd[:, 3 + j]
        out["surrogate_epi_recoil_E"] = sd[:, 6]; out["surrogate_epi_recoil_nonvtx100"] = sd[:, 9]; out["surrogate_epi_nonvtx_iso_blobs_energy"] = sd[:, 10]
        Ps = np.array(Ps); out["surrogate_epi_p_reco_exists"] = Ps[:, :, 0].std(0); out["surrogate_epi_p_minos"] = Ps[:, :, 1].std(0); out["surrogate_epi_p_charge_neg"] = Ps[:, :, 2].std(0)
        out["surrogate_epi_n_prongs"] = np.array(Ns).std(0); out["surrogate_epi_draws"] = np.full(exist.sum(), a.epi_draws, dtype=np.int32)
    write_ntuple(a.out, out)
    print(f"{len(truth)} truth events -> {exist.sum()} reconstructed ({exist.mean()*100:.1f}%) -> {a.out}")


if __name__ == "__main__":
    main()
