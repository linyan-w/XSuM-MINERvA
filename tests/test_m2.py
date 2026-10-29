import numpy as np
import torch

from xsum.data.compact import Tier1Transform, collate
from xsum.models.surrogate import Surrogate


def _fake_raw(n, rng):
    mu_true = np.stack([rng.normal(0, 300, n), rng.normal(0, 300, n), rng.uniform(1000, 8000, n)], 1)
    ctx = np.stack([rng.uniform(-800, 800, n), rng.uniform(-800, 800, n), rng.uniform(4500, 8500, n), np.full(n, 6.0), np.full(n, 12.0)], 1)
    y = np.empty((n, 11))
    y[:, :3] = mu_true * rng.lognormal(0, 0.1, (n, 1))
    y[:, 3:6] = ctx[:, :3] + rng.normal(0, 20, (n, 3))
    y[:, 6:] = rng.lognormal(7, 1, (n, 5)); y[rng.random(n) < 0.3, 10] = 0.0
    return y, mu_true, ctx


def test_tier1_transform_round_trip():
    rng = np.random.default_rng(0)
    y, mu, ctx = _fake_raw(5000, rng)
    tf = Tier1Transform().fit(y, mu, ctx)
    x, valid = tf.forward(y, mu, ctx, rng)
    assert valid.all() and np.isfinite(x).all() and np.abs(x).max() <= tf.clip
    yb = tf.inverse(x, mu, ctx)
    assert x.shape[1] == 9 and yb.shape[1] == 11
    assert np.allclose(yb[:, :6], y[:, :6], rtol=1e-4, atol=1e-2)
    for c in (6, 9, 10):
        assert np.allclose(yb[:, c], y[:, c], rtol=1e-3, atol=1e-2)   # zeros come back as exact zeros
    assert ((y[:, 10] == 0) == (yb[:, 10] == 0)).all()
    # derived columns follow recoil_E times a z-dependent ratio
    assert np.isfinite(yb[:, 7:9]).all() and (yb[:, 7] > 0).all()


def test_transform_flags_corrupt_rows():
    rng = np.random.default_rng(1)
    y, mu, ctx = _fake_raw(200, rng)
    tf = Tier1Transform().fit(y, mu, ctx)
    y[3, 0] = np.nan; y[7, 4] = np.inf
    x, valid = tf.forward(y, mu, ctx, rng)
    assert (~valid).sum() == 2 and np.isfinite(x).all()


def test_surrogate_forward_and_sample():
    torch.manual_seed(0)
    B = 6
    items = []
    for i in range(B):
        k = 1 + i
        items.append((np.random.randint(1, 12, k).astype(np.int8), np.random.randn(k, 3).astype(np.float32) * 500,
                      np.array([0, 0, 6000, 6, 12], np.float32), np.array([1, 1, 1], np.float32), np.int8(i % 3),
                      np.random.randn(9).astype(np.float32), np.array([0, 0, 4000], np.float32), np.float32(1.0), np.zeros(2, np.float32)))
    b = collate(items)
    m = Surrogate(d_model=32, n_heads=4, n_layers=1, flow_hidden=64, flow_layers=2)
    L = m.losses(b)
    assert all(torch.isfinite(v) for v in L.values())
    s = m.sample(b, n_steps=4)
    assert s["x1"].shape == (B, 9) and s["nprong"].shape == (B,)
