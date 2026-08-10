# linear_model_Merlin

Same linear Cox head pipeline as `src/linear_model/` (simple `LinearCoxHead`
and `TwoBranchCoxHead`), but the frozen image backbone is the **Merlin** 3D CT
foundation model (`stanfordmimi/Merlin`, the `merlin-vlm` package) instead of
DINOv3.

- **Head is identical** — `head_model.py`, `dataset.py`, `training.py`,
  `nested_cv.py`, `baseline.py` are copied verbatim from `src/linear_model/`
  (they are backbone-agnostic; everything comes from the config).
- **What changed** — `load_backbone.py` loads `Merlin(ImageEmbedding=True)`,
  and `caching_features.py` runs the **whole 3D volume** through Merlin's
  native MONAI transforms (RAS, 1.5×1.5×3 mm, HU window −1000..1000 → [0,1],
  224×224×160) to a **2048-d** embedding. The tumor mask is not used (Merlin's
  as-trained usage). Contrast: DINOv3 uses a single tumor-cropped axial slice
  → 384-d CLS token.
- **Config** — `config/config_merlin.yaml`. Results go to `output_merlin/`;
  the image cache is `output/cache/image_features_merlin.npz`; the radiomics
  cache is shared with the DINOv3 pipeline (backbone-independent).
- **ROI-crop variant** — `config/config_merlin_roi.yaml` sets
  `image.roi_crop: true`: instead of the whole volume, each case is cropped
  to the tumor mask's 3D bounding box (+ `roi_margin_voxels`) and resized to
  fill 224×224×160 — the 3D analog of DINOv3's tumor-cropped slice, with
  Merlin's HU window/spacing unchanged. Writes a separate cache
  (`image_features_merlin_roi.npz`) and output dir (`output_merlin_roi/`).
  Run any script with `--config ../../config/config_merlin_roi.yaml`.
  Result: ROI helps (image 0.489→0.528 nested-CV) but stays at chance
  (p=0.24), still far below DINOv3's 0.643 — see `resume/linear_head/`.

## Environment

Run everything in the `MERLIN` conda env (not `dinov3`):

```bash
conda activate MERLIN
unset ALL_PROXY all_proxy   # merlin-vlm's HF download uses httpx, which
                            # rejects a socks:// proxy; http(s)_proxy still work.
                            # (caching_features.py also drops it automatically.)
```

`lifelines` and `nvitop` were pip-installed into `MERLIN` for the training
scripts and GPU auto-selection.

## Run order

```bash
cd src/linear_model_Merlin

# 1. Cache Merlin image embeddings (GPU) + radiomics features.
#    Merlin weights auto-download from HuggingFace on first use.
python caching_features.py

# 2. Train + evaluate the head (5-fold CV + final train/valid/test).
python training.py --input-mode radiomics
python training.py --input-mode image
python training.py --input-mode both      # TwoBranchCoxHead

# 3. Optional: label-permutation null + nested CV over the full cohort.
python baseline.py
python nested_cv.py
```

Add `--event-type {event,recurrence,metastasis,survival_status}` to any script
to switch endpoints.

## Caveat

The NPC cohort is **CT** (`data/NPC_pre/T1/`), so Merlin is used in its native
modality — the HU intensity window applies correctly (air at −1000 HU maps to
0). The remaining domain gap is anatomical: Merlin was trained on abdominal CT,
while this is head/neck NPC. That's a body-region shift, not a modality
mismatch.
