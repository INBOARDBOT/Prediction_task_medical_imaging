"""Build the radiomics-token pretraining corpus from a domain-matched
external NPC MRI dataset (277 primary NPC patients, T1/T2/CE-T1, Zenodo
10.5281/zenodo.13131827) instead of LUNA16 lung CT -- same disease, same
anatomy, same T1 sequence as our own cohort, so this should transfer much
better than an unrelated organ/modality.

Reuses the exact tumor-mask ROI-crop logic (best_slice_roi) already
validated on our own NPC cohort, just reading the T1WI DICOM series +
ROI-T1.nii mask instead of a single .nii.gz pair.

Runs in the `radiomics` conda env (pyradiomics + SimpleITK + scipy, no
torch): conda run -n radiomics python src/ViT_cox/prepare_npc_mri_pretrain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ViT_cox"))

from radiomics_tokenizer import (  # noqa: E402
    build_extractor,
    extract_patch_tokens,
    reference_feature_names,
    resize_to_canvas,
)

DATA_DIR = PROJECT_ROOT / "ViT_output" / "data" / "npc_mri" / "primary_data" / "data" / "MRI-Segments"
CACHE_DIR = PROJECT_ROOT / "ViT_output" / "cache"
ROI_MARGIN_PX = 16


def best_slice_roi_from_dicom(patient_dir: Path, margin_px: int) -> np.ndarray | None:
    t1_dir = patient_dir / "T1WI"
    roi_path = patient_dir / "ROI-T1.nii"
    if not t1_dir.exists() or not roi_path.exists():
        return None

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(t1_dir))
    if not dicom_names:
        return None
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    mask = sitk.ReadImage(str(roi_path))
    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
    if img_arr.shape != mask_arr.shape:
        return None

    areas = mask_arr.sum(axis=(1, 2))
    if areas.max() == 0:
        return None
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
    crop = img_slice[rmin : rmax + 1, cmin : cmax + 1]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    p1, p99 = np.percentile(img_slice, [1, 99])
    crop = np.clip((crop - p1) / (p99 - p1 + 1e-8), 0, 1)
    return crop.astype(np.float32)


def main():
    patient_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"{len(patient_dirs)} patients")

    extractor = build_extractor()
    feature_names = reference_feature_names(extractor)

    all_tokens = []
    all_ids = []
    for i, patient_dir in enumerate(patient_dirs):
        crop = best_slice_roi_from_dicom(patient_dir, ROI_MARGIN_PX)
        if crop is None:
            continue
        canvas = resize_to_canvas(crop)
        tokens = extract_patch_tokens(canvas, extractor, feature_names)
        all_tokens.append(tokens)
        all_ids.append(patient_dir.name)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(patient_dirs)} patients, {len(all_ids)} usable so far")

    tokens_arr = np.stack(all_tokens, axis=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "npc_mri_pretrain_tokens.npz"
    np.savez(out_path, tokens=tokens_arr, ids=np.array(all_ids), feature_names=np.array(feature_names))
    print(f"Wrote {out_path} ({tokens_arr.shape[0]} patients x {tokens_arr.shape[1]} patches x {tokens_arr.shape[2]} features)")


if __name__ == "__main__":
    main()
