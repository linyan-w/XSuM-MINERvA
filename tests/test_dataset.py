import numpy as np

from xsum.data.dataset import EventDataset, split_by_subrun


def test_dataset_builds(reco, truth):
    ds = EventDataset(reco, truth, n_max=64)
    assert len(ds) > 0
    item = ds[0]
    assert item["cls"].shape == (64,) and item["mom"].shape == (64, 3) and item["ctx"].shape == (5,)
    eff = ds.reco_exists.mean()
    print(f"\ntraining population {len(ds)} events, reco_exists fraction {eff:.3f}, "
          f"max particles {ds.mask.sum(1).max()}, mean {ds.mask.sum(1).mean():.2f}")
    assert 0.05 < eff < 0.95
    ok = ds.reco_exists
    assert np.isfinite(ds.tier1[ok]).all() and np.isnan(ds.tier1[~ok]).all()
    assert (ds.tier0[~ok] == 0).all()


def test_split_by_subrun():
    sub = np.repeat(np.arange(50), 10)
    s = split_by_subrun(sub, seed=1)
    assert set(np.unique(s)) <= {0, 1, 2}
    for k in np.unique(sub):
        assert len(np.unique(s[sub == k])) == 1
    frac = np.bincount(s, minlength=3) / len(s)
    assert abs(frac[0] - 0.8) < 0.1
