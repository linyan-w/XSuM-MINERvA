"""Input pipeline (docs/PROJECT.md 13.1): generator final-state particles -> model input particle set.

Works on awkward arrays with one entry per event. Generator-agnostic: needs only pdg, px, py, pz, E.
"""
from __future__ import annotations

import awkward as ak
import numpy as np

from ..constants import MU_PLUS_CLASS, PDG_CLASS, PDG_MASS
from .frames import from_beam

DEFAULT_KE_CUT_MEV = 10.0   # adopted 2026-10-02 (was 50); see docs/PROJECT.md §18
DEFAULT_KEEP_NEUTRONS = True  # neutrons are input tokens (adopted 2026-10-02)
DROP_PDG_ABS = {12, 14, 16}                 # neutrinos (always); neutrons are dropped unless keep_neutrons
HADRON_CLASSES = {4, 5, 6, 7, 8, 9, 10, 12} # p, pi+, pi-, pi0, K+-, K0, hyperon, n: KE cut applies
OTHER_CLASS = 11
# Training populations: "ccnumu" (CC nu_mu, the v1 population) and "all" (every interaction in the training volume:
# CC and NC, all flavours; adopted for the all-events models, 2026-10-09). The all-events population gives mu+ its
# own input class.
POPULATIONS = ("ccnumu", "all")


def _isin(arr: ak.Array, values) -> ak.Array:
    """Elementwise membership test for a jagged array against a small set of values."""
    out = None
    for v in values:
        m = arr == v
        out = m if out is None else (out | m)
    return out


def pdg_to_class(pdg):
    """Vectorised PDG -> species class. Unknown -> OTHER_CLASS (11)."""
    flat = np.asarray(ak.flatten(pdg)) if isinstance(pdg, ak.Array) else np.asarray(pdg)
    out = np.full(flat.shape, OTHER_CLASS, dtype=np.int32)
    for p, c in PDG_CLASS.items():
        out[flat == p] = c
    if isinstance(pdg, ak.Array):
        return ak.unflatten(out, ak.num(pdg))
    return out


def pdg_to_mass(pdg):
    flat = np.asarray(ak.flatten(pdg)) if isinstance(pdg, ak.Array) else np.asarray(pdg)
    out = np.zeros(flat.shape, dtype=np.float64)
    for p, m in PDG_MASS.items():
        out[flat == p] = m
    if isinstance(pdg, ak.Array):
        return ak.unflatten(out, ak.num(pdg))
    return out


def select_particles(pdg, px, py, pz, E, ke_cut_mev: float = DEFAULT_KE_CUT_MEV, keep_neutrons: bool = DEFAULT_KEEP_NEUTRONS,
                     mu_plus: bool = False) -> ak.Array:
    """Apply the 13.1 selection and return a record array {cls, px, py, pz} per event.

    Steps: drop neutrinos, GENIE pseudo-particles (2000000101), nuclear remnants (pdg > 1e9) and, unless
    keep_neutrons, neutrons; drop hadrons (neutrons included when kept) with KE < ke_cut_mev; map pdg -> class.
    Leptons and photons are never KE-cut. With mu_plus, mu+ gets its own class instead of 'other'.
    """
    apdg = abs(pdg)
    keep = ~_isin(apdg, sorted(DROP_PDG_ABS)) & (pdg != 2000000101) & (apdg < 1_000_000_000)
    if not keep_neutrons:
        keep = keep & (pdg != 2112)
    cls = pdg_to_class(pdg)
    if mu_plus:
        cls = ak.where(pdg == -13, MU_PLUS_CLASS, cls)
    ke = E - pdg_to_mass(pdg)
    is_had = _isin(cls, sorted(HADRON_CLASSES))
    keep = keep & ~(is_had & (ke < ke_cut_mev))
    return ak.zip({"cls": cls[keep], "px": px[keep], "py": py[keep], "pz": pz[keep]})


def select_from_tuple(events: ak.Array, ke_cut_mev: float = DEFAULT_KE_CUT_MEV, keep_neutrons: bool = DEFAULT_KEEP_NEUTRONS,
                      population: str = "ccnumu") -> ak.Array:
    """Convenience wrapper on slimmed tuple arrays (mc_FSPart* branches)."""
    return select_particles(events["mc_FSPartPDG"], events["mc_FSPartPx"], events["mc_FSPartPy"],
                            events["mc_FSPartPz"], events["mc_FSPartE"], ke_cut_mev, keep_neutrons, mu_plus=population == "all")


def context_from_tuple(events: ak.Array) -> np.ndarray:
    """Context vector (13.2): vtx_x, vtx_y, vtx_z, target_Z, target_A. Shape (n_events, 5)."""
    vtx = ak.to_numpy(events["mc_vtx"])[:, :3]
    Z = ak.to_numpy(events["mc_targetZ"]).astype(np.float64)[:, None]
    A = ak.to_numpy(events["mc_targetA"]).astype(np.float64)[:, None]
    return np.concatenate([vtx, Z, A], axis=1)


def is_cc_numu(events: ak.Array) -> np.ndarray:
    return (ak.to_numpy(events["mc_current"]) == 1) & (ak.to_numpy(events["mc_incoming"]) == 14)


def in_training_population(events: ak.Array, z_min: float = 4000.0, z_max: float = 8700.0) -> np.ndarray:
    """v1 training population (docs/PROJECT.md 14.2): CC nu_mu with true vertex z in [z_min, z_max]."""
    z = ak.to_numpy(events["mc_vtx"])[:, 2]
    return is_cc_numu(events) & (z >= z_min) & (z <= z_max)


def in_population(events: ak.Array, population: str = "ccnumu", z_min: float = 4000.0, z_max: float = 8700.0) -> np.ndarray:
    """Training population: CC nu_mu ("ccnumu") or every interaction ("all"), true vertex z in [z_min, z_max]."""
    assert population in POPULATIONS, population
    z = ak.to_numpy(events["mc_vtx"])[:, 2]; inz = (z >= z_min) & (z <= z_max)
    return inz & is_cc_numu(events) if population == "ccnumu" else inz


# Fallback reference for events without a charged final-state particle: 1 GeV along the beam (detector frame)
BEAM_REF = np.array(from_beam(0.0, 0.0, 1000.0), dtype=np.float32)
CHARGED_LEPTONS, CHARGED_HADRONS = (13, 11), (2212, 211, 321)


def reference_momentum(events: ak.Array, population: str = "ccnumu", ke_cut_mev: float = DEFAULT_KE_CUT_MEV) -> np.ndarray:
    """Momentum (px, py, pz) [n, 3] the reconstructed muon is expressed against (log P ratio, angle differences).

    CC nu_mu: the leading true mu-. All events: the leading mu- if present, else the leading charged lepton (mu+, e+-),
    else the leading charged hadron (p, pi+-, K+-) above the input KE threshold, else 1 GeV along the beam. The order
    is fixed by the particle set alone, so the encoder sees which token the reference is; for CC nu_mu events it is
    the same muon as in the CC nu_mu models.
    """
    pdg = events["mc_FSPartPDG"]; apdg = abs(pdg)
    P = np.sqrt(events["mc_FSPartPx"] ** 2 + events["mc_FSPartPy"] ** 2 + events["mc_FSPartPz"] ** 2)

    def lead(m):
        j = ak.argmax(ak.where(m, P, -1.0), axis=1, keepdims=True)
        v = np.stack([ak.to_numpy(ak.flatten(ak.fill_none(events[f"mc_FSPart{c}"][j], 0.0))) for c in ("Px", "Py", "Pz")], 1)
        return v.astype(np.float32), ak.to_numpy(ak.any(m, axis=1))

    ref, has = lead(pdg == 13)
    if population == "ccnumu":
        return ref
    ke = events["mc_FSPartE"] - pdg_to_mass(pdg)
    for m in (_isin(apdg, CHARGED_LEPTONS) & (P > 0), _isin(apdg, CHARGED_HADRONS) & (ke >= ke_cut_mev) & (P > 0)):
        v, h = lead(m); take = ~has & h; ref[take] = v[take]; has |= h
    ref[~has] = BEAM_REF
    return ref
