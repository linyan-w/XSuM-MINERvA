"""Detector frame <-> beam frame. The NuMI beam points BEAM_ANGLE below the detector z axis (rotation about x).

MasterAnaDev angles (muon_theta, muon_phi, pion_theta/phi, proton_theta/phi, sec_protons_theta) are all
measured in the beam frame, while Px/Py/Pz branches are in the detector frame. Verified on the tuple.
"""
from __future__ import annotations

import numpy as np

from ..constants import BEAM_ANGLE_RAD


def to_beam(px, py, pz, angle=BEAM_ANGLE_RAD):
    c, s = np.cos(angle), np.sin(angle)
    return px, py * c - pz * s, py * s + pz * c


def from_beam(px, py, pz, angle=BEAM_ANGLE_RAD):
    return to_beam(px, py, pz, -angle)


def theta_phi_beam(px, py, pz):
    x, y, z = to_beam(px, py, pz)
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.arccos(z / r), np.arctan2(y, x)


def momentum_from_beam_angles(P, theta, phi):
    """Detector-frame (px, py, pz) from magnitude and beam-frame angles."""
    x = P * np.sin(theta) * np.cos(phi)
    y = P * np.sin(theta) * np.sin(phi)
    z = P * np.cos(theta)
    return from_beam(x, y, z)
