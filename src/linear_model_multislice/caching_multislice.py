"""Extract per-slice DINOv3 CLS tokens for EVERY tumor-bearing axial slice
of each case (not just the single max-area slice the base pipeline uses).

Motivation: the linear-head interpretability pass found the DINOv3 survival
signal is diffuse across the whole tumor ROI, yet src/linear_model/
caching_features.py feeds the head only the single largest-tumor slice.
This caches all slices so that top-k mean, all-slice mean, and attention
pooling can all be derived from one extraction (see pool_multislice.py for
1 & 2, nested_cv_attention.py for 3).

Per slice: crop that slice to its OWN tumor bounding box (+ margin),
percentile-normalize, resize, replicate to 3 channels (identical to the
base pipeline's best_slice_roi/to_model_input, just applied to every tumor
slice), run through frozen DINOv3 -> 384-d CLS token.

Output (long format, one row per slice) under cache.dir:
  image_features_dinov3_perslice.npz
    case_ids : (M,) str    -- case each slice row belongs to
    areas    : (M,) int    -- tumor voxel area of that slice (for top-k)
    features : (M, 384)     -- CLS token per slice
Run in the `dinov3` env: conda run -n dinov3 python caching_multislice.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml

from load_backbone import load_dinov3_backbone, select_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_case_ids(cfg: dict) -> list[str]:
    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    case_ids = set()
    for key in ("train_split", "valid_split", "test_split"):
        df = pd.read_csv(splits_dir / cfg["data"][key])
        case_ids.update(df["case_id"].tolist())
    return sorted(case_ids)


def slice_rois(image_path: Path, mask_path: Path, margin_px: int):
    """Yield (area, crop) for every axial slice with tumor. Each crop is
    normalized exactly like the base pipeline's best_slice_roi: crop to the
    slice's own tumor bbox + margin, then per-slice 1-99 percentile -> [0,1].
    """
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.uint8)

    areas = mask_arr.sum(axis=(1, 2))
    tumor_zs = np.where(areas > 0)[0]
    if len(tumor_zs) == 0:
        raise ValueError(f"Mask has no positive voxels: {mask_path}")

    out = []
    for z in tumor_zs:
        img_slice = img_arr[z]
        mask_slice = mask_arr[z]
        rows = np.any(mask_slice, axis=1)
        cols = np.any(mask_slice, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        H, W = img_slice.shape
        rmin = max(0, rmin - margin_px)
        rmax = min(H - 1, rmax + margin_px)
        cmin = max(0, cmin - margin_px)
        cmax = min(W - 1, cmax + margin_px)
        crop = img_slice[rmin:rmax + 1, cmin:cmax + 1]
        p1, p99 = np.percentile(img_slice, [1, 99])
        crop = np.clip((crop - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)
        out.append((int(areas[z]), crop))
    return out


def to_model_input(crop: np.ndarray, img_size: int) -> torch.Tensor:
    t = torch.from_numpy(crop)[None, None]
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--out-name", type=str, default="image_features_dinov3_perslice.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = PROJECT_ROOT / cfg["cache"]["dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / args.out_name
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path.name} exists (use --force)")
        return

    case_ids = collect_case_ids(cfg)
    print(f"{len(case_ids)} labeled cases")

    device = select_device(cfg["device"]["extraction_device"])
    print(f"Extracting per-slice DINOv3 features on {device}")
    model = load_dinov3_backbone(cfg["image"]["backbone"], PROJECT_ROOT / cfg["image"]["weights_path"], device)

    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    all_rows = pd.concat(
        [pd.read_csv(splits_dir / cfg["data"][k]) for k in ("train_split", "valid_split", "test_split")]
    ).drop_duplicates("case_id").set_index("case_id")

    img_size = cfg["image"]["img_size"]
    margin = cfg["image"]["roi_margin_px"]

    feat_rows, id_rows, area_rows = [], [], []
    batch_tensors, batch_meta = [], []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = model.forward_features(x)["x_norm_clstoken"]
        feat_rows.append(out.cpu().numpy())
        for cid, area in batch_meta:
            id_rows.append(cid)
            area_rows.append(area)
        batch_tensors.clear()
        batch_meta.clear()

    for i, case_id in enumerate(case_ids):
        row = all_rows.loc[case_id]
        rois = slice_rois(PROJECT_ROOT / row["image_path"], PROJECT_ROOT / row["mask_path"], margin)
        for area, crop in rois:
            batch_tensors.append(to_model_input(crop, img_size))
            batch_meta.append((case_id, area))
            if len(batch_tensors) >= args.batch_size:
                flush()
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(case_ids)} cases, {len(id_rows) + len(batch_meta)} slices so far")
    flush()

    features = np.concatenate(feat_rows, axis=0)
    np.savez(
        out_path,
        case_ids=np.array(id_rows),
        areas=np.array(area_rows, dtype=np.int64),
        features=features,
    )
    n_cases = len(set(id_rows))
    print(f"Wrote {out_path}: {features.shape[0]} slices from {n_cases} cases x {features.shape[1]} dims")
    print(f"  slices/case: min={pd.Series(id_rows).value_counts().min()} "
          f"max={pd.Series(id_rows).value_counts().max()} "
          f"mean={features.shape[0] / n_cases:.1f}")


if __name__ == "__main__":
    main()
