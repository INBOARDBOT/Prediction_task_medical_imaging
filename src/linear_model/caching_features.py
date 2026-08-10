"""Cache DINOv3 image embeddings and pruned/standardized radiomics features
for every labeled case, so the (cheap) CV training loop in training.py never
has to touch the GPU or re-read raw radiomics json files.

Image path: for each case, take the axial slice with the largest tumor mask
area, crop to the tumor bounding box (+ margin), percentile-normalize,
resize to the backbone's input size, replicate to 3 channels, and run it
through DINOv3 to get the CLS token embedding.

Radiomics path: take the feature list + train-set mean/std written by
src/feature_prunning/prune_features.py for a given event_type, and apply
that exact standardization to every case (train/valid/test alike) so the
saved cache is ready to feed the linear head directly.

Outputs (under cache.dir from config.yaml):
  - image_features_{backbone}.npz: case_ids (str array), features (N x D)
  - radiomics_features_{event_type}.csv: case_id + standardized feature cols
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


def best_slice_roi(image_path: Path, mask_path: Path, margin_px: int) -> np.ndarray:
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.uint8)

    areas = mask_arr.sum(axis=(1, 2))
    if areas.max() == 0:
        raise ValueError(f"Mask has no positive voxels: {mask_path}")
    best_z = int(np.argmax(areas))
    img_slice = img_arr[best_z]
    mask_slice = mask_arr[best_z]

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
    crop = np.clip((crop - p1) / (p99 - p1 + 1e-8), 0, 1)
    return crop.astype(np.float32)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_model_input(crop: np.ndarray, img_size: int) -> torch.Tensor:
    t = torch.from_numpy(crop)[None, None]
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t[0]


def cache_image_features(cfg: dict, case_ids: list[str], force: bool, batch_size: int) -> Path:
    cache_dir = PROJECT_ROOT / cfg["cache"]["dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone_name = cfg["image"]["backbone"]
    out_path = cache_dir / cfg["cache"]["image_features_file"].format(backbone=backbone_name)

    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} already exists (use --force to recompute)")
        return out_path

    device = select_device(cfg["device"]["extraction_device"])
    print(f"Extracting DINOv3 image features on {device}")
    model = load_dinov3_backbone(backbone_name, PROJECT_ROOT / cfg["image"]["weights_path"], device)

    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    all_rows = pd.concat(
        [pd.read_csv(splits_dir / cfg["data"][k]) for k in ("train_split", "valid_split", "test_split")]
    ).drop_duplicates("case_id").set_index("case_id")

    img_size = cfg["image"]["img_size"]
    margin = cfg["image"]["roi_margin_px"]

    all_features = []
    all_ids = []
    batch_tensors = []
    batch_ids = []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = model.forward_features(x)["x_norm_clstoken"]
        all_features.append(out.cpu().numpy())
        all_ids.extend(batch_ids)
        batch_tensors.clear()
        batch_ids.clear()

    for i, case_id in enumerate(case_ids):
        row = all_rows.loc[case_id]
        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]
        crop = best_slice_roi(image_path, mask_path, margin)
        batch_tensors.append(to_model_input(crop, img_size))
        batch_ids.append(case_id)
        if len(batch_tensors) >= batch_size:
            flush()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {len(case_ids)} cases processed")
    flush()

    features = np.concatenate(all_features, axis=0)
    np.savez(out_path, case_ids=np.array(all_ids), features=features)
    print(f"Wrote {out_path} ({features.shape[0]} cases x {features.shape[1]} dims)")
    return out_path


def cache_radiomics_features(cfg: dict, case_ids: list[str], event_type: str, force: bool) -> Path:
    cache_dir = PROJECT_ROOT / cfg["cache"]["dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / cfg["cache"]["radiomics_features_file"].format(event_type=event_type)

    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} already exists (use --force to recompute)")
        return out_path

    selected_path = (
        PROJECT_ROOT / cfg["radiomics"]["selected_features_dir"] / event_type / cfg["radiomics"]["selected_features_file"]
    )
    with open(selected_path) as f:
        selected = json.load(f)

    feature_specs = selected["features"]
    rows = []
    for case_id in case_ids:
        with open(PROJECT_ROOT / "data" / "radiomics" / f"{case_id}.json") as f:
            raw = json.load(f)
        row = {"case_id": case_id}
        for spec in feature_specs:
            name = spec["name"]
            row[name] = (raw[name] - spec["train_mean"]) / spec["train_std"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("case_id")
    df.to_csv(out_path)
    print(f"Wrote {out_path} ({df.shape[0]} cases x {df.shape[1]} features, event_type={event_type})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--event-types", nargs="+", default=None, help="Defaults to config's data.event_type.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-radiomics", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    case_ids = collect_case_ids(cfg)
    print(f"Found {len(case_ids)} labeled cases across train/valid/test splits")

    if not args.skip_image:
        cache_image_features(cfg, case_ids, args.force, args.batch_size)

    if not args.skip_radiomics:
        event_types = args.event_types or [cfg["data"]["event_type"]]
        for event_type in event_types:
            cache_radiomics_features(cfg, case_ids, event_type, args.force)


if __name__ == "__main__":
    main()
