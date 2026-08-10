"""Cache Merlin whole-volume image embeddings and pruned/standardized
radiomics features for every labeled case, so the (cheap) CV training loop in
training.py never has to touch the GPU or re-read raw radiomics json files.

Counterpart to src/linear_model/caching_features.py (DINOv3). Differences:

Image path: Merlin's native MONAI transform pipeline (RAS orientation,
1.5x1.5x3 mm resampling, HU window -1000..1000 -> [0,1]) then the frozen
Merlin image encoder -> 2048-d embedding. Two spatial modes, set by
image.roi_crop in the config:
  - roi_crop=False: the ENTIRE 3D volume is padded/cropped to 224x224x160
    (Merlin's as-trained usage, tumor mask unused).
  - roi_crop=True: the volume is cropped to the tumor mask's 3D bounding
    box (+ roi_margin_voxels) and resized to fill 224x224x160 -- the 3D
    analog of the DINOv3 path's tumor-cropped-then-resized axial slice.

Radiomics path: identical to the DINOv3 pipeline -- take the feature list +
train-set mean/std written by src/feature_prunning/prune_features.py for a
given event_type and apply that exact standardization to every case. The
resulting cache is backbone independent and shared with the DINOv3 pipeline.

Outputs (under cache.dir from config_merlin.yaml):
  - image_features_merlin.npz: case_ids (str array), features (N x 2048)
  - radiomics_features_{event_type}.csv: case_id + standardized feature cols
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from load_backbone import load_merlin_backbone, select_device

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


def build_image_transforms(cfg: dict):
    """Merlin's native preprocessing, parameterized from config. Uses the
    package default (merlin.data.monai_transforms.ImageTransforms) when the
    config matches, but rebuilt here so spacing / HU window / spatial size
    stay visible and tunable in config_merlin.yaml.

    Two modes, selected by image.roi_crop:
      - False (default): the whole volume is padded/center-cropped to
        spatial_size -- Merlin's as-trained usage, tumor mask unused.
      - True: the volume is cropped to the tumor mask's 3D bounding box
        (+ roi_margin_voxels margin) and *resized to fill* spatial_size,
        the 3D analog of the DINOv3 pipeline's tumor-cropped-then-resized
        axial slice. Intensity/spacing normalization is unchanged (still
        Merlin's HU window + resample), so ROI-vs-whole-volume is the only
        variable that differs from the whole-volume cache.
    """
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        Resized,
        ScaleIntensityRanged,
        Spacingd,
        SpatialPadd,
        ToTensord,
    )

    img = cfg["image"]
    size = list(img["spatial_size"])
    roi_crop = bool(img.get("roi_crop", False))
    keys = ["image", "label"] if roi_crop else ["image"]

    steps = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=tuple(img["spacing"]),
            mode=("bilinear", "nearest") if roi_crop else "bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=img["hu_min"], a_max=img["hu_max"],
            b_min=0.0, b_max=1.0, clip=True,
        ),
    ]

    if roi_crop:
        margin = int(img.get("roi_margin_voxels", 10))
        steps += [
            # Crop both image and mask to the tumor bounding box + margin
            # (source_key="label" -> foreground defined by mask > 0), then
            # resize the crop to fill Merlin's input so the tumor dominates
            # the field of view (cf. DINOv3's crop-then-resize-to-224).
            CropForegroundd(keys=["image", "label"], source_key="label", margin=margin),
            Resized(keys=["image"], spatial_size=size, mode="trilinear", align_corners=False),
            ToTensord(keys=["image"]),
        ]
    else:
        steps += [
            SpatialPadd(keys=["image"], spatial_size=size),
            CenterSpatialCropd(keys=["image"], roi_size=size),
            ToTensord(keys=["image"]),
        ]

    return Compose(steps)


def cache_image_features(cfg: dict, case_ids: list[str], force: bool, batch_size: int) -> Path:
    cache_dir = PROJECT_ROOT / cfg["cache"]["dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone_name = cfg["image"]["backbone"]
    out_path = cache_dir / cfg["cache"]["image_features_file"].format(backbone=backbone_name)

    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} already exists (use --force to recompute)")
        return out_path

    device = select_device(cfg["device"]["extraction_device"])
    print(f"Extracting Merlin image features on {device}")
    model = load_merlin_backbone(device)
    transforms = build_image_transforms(cfg)

    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    all_rows = pd.concat(
        [pd.read_csv(splits_dir / cfg["data"][k]) for k in ("train_split", "valid_split", "test_split")]
    ).drop_duplicates("case_id").set_index("case_id")

    all_features = []
    all_ids = []
    batch_tensors = []
    batch_ids = []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)  # (B, 1, 224, 224, 160)
        with torch.no_grad():
            out = model(x)
            emb = out.reshape(-1, out.shape[-1])  # Merlin returns (1, B, 2048) -> (B, 2048)
            if cfg["image"].get("embedding", "avgpool") == "contrastive":
                # Apply Merlin's 512-d contrastive projection head (a 1x1x1
                # Conv3d on the pooled features), which the ImageEmbedding
                # forward skips -- tests whether the image-text contrastive
                # space carries more survival signal than the raw 2048-d
                # average-pool embedding.
                ch = model.model.encode_image.i3_resnet.contrastive_head
                emb = ch(emb[:, :, None, None, None]).flatten(1)  # (B, 512)
        assert emb.shape[0] == len(batch_ids), f"embedding/batch mismatch: {emb.shape} vs {len(batch_ids)}"
        all_features.append(emb.cpu().numpy())
        all_ids.extend(batch_ids)
        batch_tensors.clear()
        batch_ids.clear()

    roi_crop = bool(cfg["image"].get("roi_crop", False))

    for i, case_id in enumerate(case_ids):
        row = all_rows.loc[case_id]
        image_path = PROJECT_ROOT / row["image_path"]
        sample = {"image": str(image_path)}
        if roi_crop:
            sample["label"] = str(PROJECT_ROOT / row["mask_path"])
        vol = transforms(sample)["image"]  # (1, 224, 224, 160)
        batch_tensors.append(vol)
        batch_ids.append(case_id)
        if len(batch_tensors) >= batch_size:
            flush()
        if (i + 1) % 25 == 0:
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
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config_merlin.yaml")
    parser.add_argument("--event-types", nargs="+", default=None, help="Defaults to config's data.event_type.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-radiomics", action="store_true")
    args = parser.parse_args()

    # merlin-vlm's HuggingFace download uses httpx, which chokes on a
    # socks:// ALL_PROXY; the http_proxy/https_proxy vars still work, so drop
    # the socks one if present.
    if os.environ.get("ALL_PROXY", "").startswith("socks"):
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

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
