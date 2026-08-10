"""Shared patch-grid radiomics tokenization (paper Fig. 4, p.15: "radiomics
embeddings were computed per patch in advance"). An ROI crop is resized to
a fixed canvas, divided into a non-overlapping GRID_SIZE x GRID_SIZE patch
grid, and each patch gets its own full pyradiomics feature vector (the
patch itself is the ROI -- an all-ones mask, same convention the paper
uses: "144 non-overlapping 32x32 patches per image").

Used identically to build the LUNA16 pretraining corpus and to tokenize
NPC cases for fine-tuning -- same canvas/grid/feature set in both, so the
pretrained shallow transformer's input format matches what it sees later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import zoom

CANVAS_SIZE = 192
GRID_SIZE = 6
PATCH_SIZE = CANVAS_SIZE // GRID_SIZE  # 32
N_PATCHES = GRID_SIZE * GRID_SIZE  # 36


def build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    return featureextractor.RadiomicsFeatureExtractor(minimumROIDimensions=2)


def resize_to_canvas(crop: np.ndarray, canvas_size: int = CANVAS_SIZE) -> np.ndarray:
    """No torch here -- this module (and the LUNA16/NPC data-prep scripts
    that import it) run in the `radiomics` conda env, which has
    scipy/pyradiomics but not torch; torch lives only in the `dinov3` env
    used for the transformer model/training scripts, which consume the
    token caches this module writes rather than importing it directly.
    """
    zoom_factors = (canvas_size / crop.shape[0], canvas_size / crop.shape[1])
    return zoom(crop.astype(np.float32), zoom_factors, order=1)


def best_slice_roi(image_path: Path, mask_path: Path, margin_px: int) -> np.ndarray:
    """Duplicated from src/linear_model/caching_features.py rather than
    imported -- that module imports torch at the top level, which isn't
    installed in the `radiomics` conda env this script runs in.
    """
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
    crop = img_slice[rmin : rmax + 1, cmin : cmax + 1]

    p1, p99 = np.percentile(img_slice, [1, 99])
    crop = np.clip((crop - p1) / (p99 - p1 + 1e-8), 0, 1)
    return crop.astype(np.float32)


def reference_feature_names(extractor: featureextractor.RadiomicsFeatureExtractor) -> list[str]:
    """Run the extractor once on a dummy patch to fix feature name order,
    since a small fraction of patches can fail to produce a given feature
    (e.g. a perfectly uniform region) and dict iteration order should not
    be trusted to vary consistently across calls.
    """
    rng = np.random.default_rng(0)
    patch = (rng.random((PATCH_SIZE, PATCH_SIZE)) * 500).astype(np.float32)
    mask = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    result = extractor.execute(sitk.GetImageFromArray(patch[None]), sitk.GetImageFromArray(mask[None]))
    return sorted(k for k in result if not k.startswith("diagnostics"))


def extract_patch_tokens(canvas: np.ndarray, extractor: featureextractor.RadiomicsFeatureExtractor, feature_names: list[str]) -> np.ndarray:
    """canvas: 2D array already resized to CANVAS_SIZE x CANVAS_SIZE.
    Returns (N_PATCHES, len(feature_names)) array, NaN where a feature
    failed to compute on that patch (rare, e.g. a fully-uniform crop
    corner) -- caller should impute before training.
    """
    tokens = np.full((N_PATCHES, len(feature_names)), np.nan, dtype=np.float32)
    idx = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            patch = canvas[i * PATCH_SIZE : (i + 1) * PATCH_SIZE, j * PATCH_SIZE : (j + 1) * PATCH_SIZE]
            mask = np.ones_like(patch, dtype=np.uint8)
            try:
                result = extractor.execute(sitk.GetImageFromArray(patch[None]), sitk.GetImageFromArray(mask[None]))
                for k, name in enumerate(feature_names):
                    if name in result:
                        tokens[idx, k] = float(result[name])
            except Exception:
                pass
            idx += 1
    return tokens
