"""Slim an open-data AnaTuple (ROOT) to Parquet, keeping only the branches the project uses.

Both trees are slimmed: `MasterAnaDev` (reconstructed events with truth attached) and `Truth`
(every generated event, the efficiency denominator). Reading is chunked so a 20 GB file fits in memory.
"""
from __future__ import annotations

import json
import pathlib
import awkward as ak
import uproot

from .branches import RECO_TREE_BRANCHES, TRUTH_TREE_BRANCHES

_TREE_BRANCHES = {"MasterAnaDev": RECO_TREE_BRANCHES, "Truth": TRUTH_TREE_BRANCHES}


def slim_tree(root_path: str, tree: str, out_path: str, step_size: str = "200 MB",
              entry_stop: int | None = None) -> int:
    """Write the selected branches of `tree` in `root_path` to a single Parquet file. Returns entries written.

    `root_path` may be a local path or an xrootd URL (root://...); uproot then streams only the needed baskets.
    """
    branches = _TREE_BRANCHES[tree]
    chunks = []
    n = 0
    with uproot.open(root_path) as f:
        t = f[tree]
        missing = [b for b in branches if b not in t.keys()]
        if missing:
            raise KeyError(f"{tree} is missing branches: {missing}")
        for arr in t.iterate(branches, step_size=step_size, entry_stop=entry_stop, library="ak"):
            chunks.append(arr)
            n += len(arr)
        typenames = {b: t[b].typename for b in branches}
        typenames.update({b + "_sz": t[b + "_sz"].typename for b in branches if b + "_sz" in t.keys()})
    out = ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(out, out_path)
    # keep the ROOT typenames next to the Parquet so the ntuple writer can be checked without the ROOT file
    pathlib.Path(out_path).with_suffix(".typenames.json").write_text(json.dumps(typenames, indent=0))
    return n


def slim_file(root_path: str, out_dir: str, entry_stop: int | None = None) -> dict:
    """Slim both trees of one AnaTuple into `<out_dir>/<stem>.{reco,truth}.parquet`."""
    stem = pathlib.Path(root_path).stem
    out = {}
    for tree, tag in (("MasterAnaDev", "reco"), ("Truth", "truth")):
        path = str(pathlib.Path(out_dir) / f"{stem}.{tag}.parquet")
        out[tag] = {"path": path, "entries": slim_tree(root_path, tree, path, entry_stop=entry_stop)}
    return out
