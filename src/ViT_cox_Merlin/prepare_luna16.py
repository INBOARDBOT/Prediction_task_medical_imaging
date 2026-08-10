"""Build the radiomics-token pretraining corpus from LUNA16: for every
annotated lung nodule, take the axial slice through its center, crop
around it (diameter + margin), tokenize into the same 6x6 patch grid used
for NPC cases, and cache raw (unstandardized) per-patch pyradiomics
features. Standardization/zero-variance filtering happens once over the
whole corpus in pretrain_radiomics.py, not here (keeps this script
resumable across subsets without needing to see the whole corpus first).

Runs in the `radiomics` conda env (pyradiomics + SimpleITK + scipy, no
torch): `conda run -n radiomics python src/ViT_cox/prepare_luna16.py --subset subset0`
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from radiomics_tokenizer import build_extractor, extract_patch_tokens, reference_feature_names, resize_to_canvas

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LUNA_DIR = PROJECT_ROOT / "ViT_output" / "data" / "luna16"
CACHE_DIR = PROJECT_ROOT / "ViT_output" / "cache"

MARGIN_MM = 10.0  # extra context beyond the annotated nodule diameter


def world_to_voxel(image: sitk.Image, world_xyz: tuple[float, float, float]) -> tuple[int, int, int]:
    voxel = image.TransformPhysicalPointToIndex(world_xyz)
    return voxel  # (x, y, z) voxel indices


def crop_nodule(image: sitk.Image, coord_xyz: tuple[float, float, float], diameter_mm: float) -> np.ndarray | None:
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    spacing = image.GetSpacing()  # (x, y, z)
    vx, vy, vz = world_to_voxel(image, coord_xyz)

    if not (0 <= vz < arr.shape[0]):
        return None
    axial = arr[vz]  # (y, x)

    half_size_mm = diameter_mm / 2 + MARGIN_MM
    half_px_x = max(8, int(round(half_size_mm / spacing[0])))
    half_px_y = max(8, int(round(half_size_mm / spacing[1])))

    y0, y1 = vy - half_px_y, vy + half_px_y
    x0, x1 = vx - half_px_x, vx + half_px_x
    H, W = axial.shape
    y0c, y1c = max(0, y0), min(H, y1)
    x0c, x1c = max(0, x0), min(W, x1)
    if y1c - y0c < 8 or x1c - x0c < 8:
        return None

    crop = axial[y0c:y1c, x0c:x1c]
    return crop.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=str, required=True, help="e.g. subset0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subset_dir = LUNA_DIR / args.subset
    mhd_files = sorted(glob.glob(str(subset_dir / "*.mhd")))
    print(f"{args.subset}: {len(mhd_files)} scans")

    annotations = pd.read_csv(LUNA_DIR / "annotations.csv")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"luna16_tokens_{args.subset}.npz"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path.name} already exists (use --force to recompute)")
        return

    extractor = build_extractor()
    feature_names = reference_feature_names(extractor)

    all_tokens = []
    all_ids = []
    for i, mhd_path in enumerate(mhd_files):
        seriesuid = os.path.basename(mhd_path).replace(".mhd", "")
        rows = annotations[annotations["seriesuid"] == seriesuid]
        if rows.empty:
            continue

        image = sitk.ReadImage(mhd_path)
        for j, row in rows.reset_index(drop=True).iterrows():
            crop = crop_nodule(image, (row["coordX"], row["coordY"], row["coordZ"]), row["diameter_mm"])
            if crop is None:
                continue
            canvas = resize_to_canvas(crop)
            tokens = extract_patch_tokens(canvas, extractor, feature_names)
            all_tokens.append(tokens)
            all_ids.append(f"{seriesuid}_{j}")

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(mhd_files)} scans, {len(all_ids)} nodules so far")

    tokens_arr = np.stack(all_tokens, axis=0) if all_tokens else np.zeros((0, 36, len(feature_names)), dtype=np.float32)
    np.savez(out_path, tokens=tokens_arr, ids=np.array(all_ids), feature_names=np.array(feature_names))
    print(f"Wrote {out_path} ({tokens_arr.shape[0]} nodule crops x {tokens_arr.shape[1]} patches x {tokens_arr.shape[2]} features)")


if __name__ == "__main__":
    main()
