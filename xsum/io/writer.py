"""Write a pruned MasterAnaDev-style ntuple with uproot, matching the open-data tuple's types and counters."""
from __future__ import annotations

import awkward as ak
import numpy as np
import uproot


def _counter_name(name: str) -> str:
    return name + "_sz"


def write_ntuple(path: str, branches: dict, tree: str = "MasterAnaDev", counter_names: dict | None = None, extra_trees: dict | None = None) -> None:
    """branches: {name: numpy array (scalar / fixed-size) or awkward jagged array}.

    Jagged branches get a MINERvA-style `<name>_sz` int32 counter unless `counter_names` maps the branch to
    another counter name (each counter name may be used by one branch only). Types follow the input dtypes, so
    pass int32 for integer branches and bool for booleans where the tuple uses them.
    extra_trees: {tree name: branches} written to the same file after `tree` (e.g. the `Truth` tree of every generated event).
    """
    cn = counter_names or {}
    with uproot.recreate(path) as f:
        for name, br in [(tree, branches)] + list((extra_trees or {}).items()):
            data = {}
            for k, v in br.items():
                if isinstance(v, ak.Array):
                    data[k] = ak.fill_none(v, 0) if "?" in str(v.type) else v  # option types (e.g. from a Parquet round trip) are not writable
                else:
                    data[k] = np.asarray(v)
            f.mktree(name, {k: _type_of(v) for k, v in data.items()}, counter_name=lambda n_: cn.get(n_, _counter_name(n_)))
            f[name].extend(data)


def _type_of(v):
    if isinstance(v, ak.Array):
        return v.type
    v = np.asarray(v)
    if v.ndim == 1:
        return v.dtype
    return np.dtype((v.dtype, v.shape[1:]))
