"""Muon and event-level decode helpers (docs/PROJECT.md 13.4): model outputs -> tuple branches."""
from __future__ import annotations

import numpy as np

from ..constants import M_MU
from .frames import theta_phi_beam


def decode_muon(mu_px, mu_py, mu_pz, minos_ok, charge_neg) -> dict:
    """Muon block of the pruned ntuple from the model's (px, py, pz) and Tier 0 flags."""
    # muon_P is exactly |(Px,Py,Pz)| in the tuple. muon_E is on-shell for MINOS-matched muons; for unmatched
    # muons the tuple sets E = P in 82% of cases (a MasterAnaDev quirk). We always write the on-shell E.
    # muon_qp is the MINOS-track q/p (1/GeV, -9999.9 when unmatched); it is not reproduced, only its sign
    # enters via nuHelicity.
    P = np.sqrt(mu_px**2 + mu_py**2 + mu_pz**2)
    E = np.sqrt(P**2 + M_MU**2)
    minos_ok = np.asarray(minos_ok, dtype=bool)
    theta, phi = theta_phi_beam(mu_px, mu_py, mu_pz)
    out = {
        "MasterAnaDev_muon_Px": mu_px, "MasterAnaDev_muon_Py": mu_py, "MasterAnaDev_muon_Pz": mu_pz,
        "MasterAnaDev_muon_P": P, "MasterAnaDev_muon_E": E,
        "MasterAnaDev_muon_theta": theta, "muon_phi": phi,
        "MasterAnaDev_minos_trk_is_ok": minos_ok,
        "MasterAnaDev_nuHelicity": np.where(np.asarray(charge_neg, bool) | ~minos_ok, 1, 2).astype(np.int32),
    }
    return out
