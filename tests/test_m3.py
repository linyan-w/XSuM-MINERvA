import numpy as np
import torch

from xsum.data.compact import collate
from xsum.data.prongs_tf import ProngTransform, VertexPlaneTable
from xsum.models.surrogate import Surrogate


def _fake_prongs(n, rng):
    theta = rng.uniform(0.05, 2.5, n); phi = rng.uniform(-np.pi, np.pi, n)
    kin = rng.random(n) < 0.77; pf = kin & (rng.random(n) < 0.85)
    P = np.stack([np.where(kin, theta, -9.0), np.where(kin, phi, -9.0), kin, np.where(kin, rng.lognormal(5.6, 0.5, n), -1.0),
                  pf, np.where(pf, rng.lognormal(6.7, 0.4, n), -1.0), np.where(pf, rng.random(n), -1.0),
                  pf & (rng.random(n) < 0.5), rng.random(n) < 0.2], 1).astype(np.float32)
    return P


def test_prong_transform_round_trip():
    rng = np.random.default_rng(0)
    P = _fake_prongs(4000, rng)
    tf = ProngTransform().fit(P)
    x = tf.forward(P, rng)
    assert x.shape == (4000, 10) and np.isfinite(x).all()
    Q = tf.inverse(x)
    kin = P[:, 2] > 0.5; pf = P[:, 4] > 0.5
    assert ((Q[:, 2] > 0.5) == kin).all() and ((Q[:, 4] > 0.5) == pf).all()
    assert np.allclose(Q[kin, 0], P[kin, 0], atol=1e-3) and np.allclose(np.angle(np.exp(1j * (Q[kin, 1] - P[kin, 1]))), 0, atol=1e-3)
    assert np.allclose(Q[kin, 3], P[kin, 3], rtol=1e-3) and np.allclose(Q[pf, 5], P[pf, 5], rtol=1e-3)
    assert np.allclose(Q[pf, 6], P[pf, 6], atol=2e-3)
    assert ((Q[:, 8] > 0.5) == (P[:, 8] > 0.5)).all()


def test_vertex_plane_table():
    rng = np.random.default_rng(1)
    planes = 6000 + 22.1 * np.arange(60)
    true_z = rng.uniform(6010, 7300, 20000)
    snap = rng.random(20000) < 0.7
    i = np.clip(np.searchsorted(planes, true_z) + rng.integers(-1, 2, 20000), 0, 59)
    reco_z = np.where(snap, planes[i], true_z + rng.normal(0, 30, 20000))
    tab = VertexPlaneTable().fit(reco_z, min_count=50)
    assert abs(len(tab.z) - 60) <= 2
    cls = tab.classify(reco_z, true_z)
    assert (cls[snap] > 0).mean() > 0.98 and (cls[~snap] == 0).mean() > 0.95
    zz = tab.z_for_class(cls, true_z, reco_z)
    assert np.allclose(zz[cls > 0], reco_z[cls > 0], atol=0.2)
    # far-snapped events get class 8 and decode to the plane nearest the continuous z
    sel = tab.nearest(true_z) + 6 < len(tab.z)
    far = tab.classify(tab.z[tab.nearest(true_z[sel]) + 6], true_z[sel])
    assert (far == 8).all()


def test_surrogate_tier2_forward_and_sample():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    B = 5; items = []
    for i in range(B):
        k = 1 + i; npr = i % 3
        items.append((np.random.randint(1, 12, k).astype(np.int8), (np.random.randn(k, 3) * 500).astype(np.float32),
                      np.array([0, 0, 6000, 6, 12], np.float32), np.array([1, 1, 1], np.float32), np.int8(npr),
                      np.random.randn(9).astype(np.float32), np.array([0, 0, 4000], np.float32), np.float32(1.0), np.zeros(2, np.float32),
                      np.random.randn(npr, 10).astype(np.float32), np.int64(i % 9), np.float32(0.3)))
    b = collate(items)
    assert b["prongs"].shape[1] == 2 and b["pmask"].sum() == sum(i % 3 for i in range(B))
    m = Surrogate(d_model=32, n_heads=4, n_layers=1, flow_hidden=64, flow_layers=2, tier2=True, prong_layers=1)
    L = m.losses(b)
    assert {"vtx", "prong"} <= set(L) and all(torch.isfinite(v) for v in L.values())
    s = m.sample(b, n_steps=3)
    assert s["prongs"].shape[0] == B and s["prongs"].shape[2] == 10 and s["vclass"].shape == (B,)
