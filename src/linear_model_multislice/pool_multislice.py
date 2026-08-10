"""Reduce the per-slice DINOv3 cache (image_features_dinov3_perslice.npz)
into single-vector-per-case caches by mean-pooling, so they drop straight
into the existing src/linear_model/ pipeline (dataset.py reads case_ids +
features exactly like the base single-slice cache).

Produces, for each variant:
  - top-k mean : mean of the k largest-tumor-area slices  (option 1)
  - all mean   : mean of every tumor slice                (option 2)

Also emits a matching config/config_multislice_<variant>.yaml (a copy of
config.yaml with cache.image_features_file + output.dir overridden) so
src/linear_model/nested_cv.py can be pointed at each variant with just
--config, no code changes.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def pool(perslice_npz: Path, k: int | None) -> tuple[np.ndarray, np.ndarray]:
    """k=None -> mean all tumor slices; k=int -> mean of top-k by area."""
    d = np.load(perslice_npz, allow_pickle=True)
    case_ids = d["case_ids"].astype(str)
    areas = d["areas"]
    feats = d["features"]
    df = pd.DataFrame({"case_id": case_ids, "area": areas})
    df["idx"] = np.arange(len(df))

    out_ids, out_feats = [], []
    for cid, grp in df.groupby("case_id", sort=True):
        g = grp.sort_values("area", ascending=False)
        idx = g["idx"].to_numpy()
        if k is not None:
            idx = idx[:k]
        out_ids.append(cid)
        out_feats.append(feats[idx].mean(axis=0))
    return np.array(out_ids), np.stack(out_feats).astype(np.float32)


def write_config(base_cfg_path: Path, cache_file: str, out_dir: str, dest: Path):
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["cache"]["image_features_file"] = cache_file  # literal (no {backbone})
    cfg["output"]["dir"] = out_dir
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perslice", type=Path,
                        default=PROJECT_ROOT / "output" / "cache" / "image_features_dinov3_perslice.npz")
    parser.add_argument("--top-ks", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--base-config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    args = parser.parse_args()

    cache_dir = args.perslice.parent
    config_dir = PROJECT_ROOT / "config"

    variants = [(f"top{k}", k) for k in args.top_ks] + [("all", None)]
    for name, k in variants:
        ids, feats = pool(args.perslice, k)
        cache_file = f"image_features_dinov3_{name}.npz"
        np.savez(cache_dir / cache_file, case_ids=ids, features=feats)
        cfg_dest = config_dir / f"config_multislice_{name}.yaml"
        write_config(args.base_config, cache_file, f"output_multislice_{name}", cfg_dest)
        print(f"{name:6s}: {feats.shape[0]} cases x {feats.shape[1]} -> {cache_file} + {cfg_dest.name}")


if __name__ == "__main__":
    main()
