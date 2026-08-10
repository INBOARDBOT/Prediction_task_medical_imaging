"""Tokenize the NPC cohort's tumor ROI crops into the same 6x6 patch-grid
radiomics format used for the LUNA16 pretraining corpus (same canvas size,
same patch grid, same full pyradiomics feature set) -- required so the
radiomics transformer pretrained on LUNA16 sees the same input format at
fine-tuning time. Reuses the exact ROI-crop logic (best_slice_roi) already
validated in src/linear_model/caching_features.py.

Runs in the `radiomics` conda env:
    conda run -n radiomics python src/ViT_cox/prepare_npc_tokens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ViT_cox"))

from radiomics_tokenizer import (  # noqa: E402
    best_slice_roi,
    build_extractor,
    extract_patch_tokens,
    reference_feature_names,
    resize_to_canvas,
)

CACHE_DIR = PROJECT_ROOT / "ViT_output" / "cache"
ROI_MARGIN_PX = 16


def main():
    complete_list = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv").set_index("case_id")

    extractor = build_extractor()
    feature_names = reference_feature_names(extractor)

    all_tokens = []
    all_ids = []
    for i, (case_id, row) in enumerate(complete_list.iterrows()):
        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]
        crop = best_slice_roi(image_path, mask_path, ROI_MARGIN_PX)
        canvas = resize_to_canvas(crop)
        tokens = extract_patch_tokens(canvas, extractor, feature_names)
        all_tokens.append(tokens)
        all_ids.append(case_id)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(complete_list)} cases")

    tokens_arr = np.stack(all_tokens, axis=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "npc_radiomics_tokens.npz"
    np.savez(out_path, tokens=tokens_arr, ids=np.array(all_ids), feature_names=np.array(feature_names))
    print(f"Wrote {out_path} ({tokens_arr.shape[0]} cases x {tokens_arr.shape[1]} patches x {tokens_arr.shape[2]} features)")


if __name__ == "__main__":
    main()
