"""Emit config/config_bench_<backbone>.yaml: a copy of config.yaml with the
image cache + output dir pointed at a benchmarked backbone's embeddings, so
src/linear_model/nested_cv.py scores it with no code change.
"""
import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    backbone = sys.argv[1]
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    cfg["cache"]["image_features_file"] = f"image_features_{backbone}.npz"
    cfg["output"]["dir"] = f"output_bench_{backbone}"
    dest = PROJECT_ROOT / "config" / f"config_bench_{backbone}.yaml"
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {dest.name}")


if __name__ == "__main__":
    main()
