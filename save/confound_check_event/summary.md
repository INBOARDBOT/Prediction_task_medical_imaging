# Confound check -- event_type = event, image mode

## Why

`nested_cv_event/` found a real, statistically significant signal for
`image` mode (aggregated C-index 0.643, p=0.000 against a label-permutation
null). Before trusting that as tumor biology, worth ruling out that the
frozen DINOv3 embedding is actually picking up a technical/non-biological
shortcut: the pipeline crops to the tumor bounding box and upsamples to a
fixed resolution, so crop size (-> blur amount), slice position, and
intensity calibration are all baked into what the model sees, on top of
whatever real visual signal is in the tumor itself.

## Method

Recomputed the same aggregated out-of-fold risk score used for the nested
CV result (`src/linear_model/confound_check.py`, reuses
`nested_cv.nested_cv_oof_predictions`), for both `image` mode and
`radiomics` mode (the latter as a "known non-signal" comparison -- if a
correlate shows up equally strongly in both, it's more likely a shared,
uninteresting property of the cohort than something specific to what
`image` picked up). Sanity-check reproduction: 0.630 (image) / 0.472
(radiomics) vs. the original 0.643 / 0.476 -- small differences from
non-determinism in the linear layer's random initialization, same
magnitude and conclusion.

Checked six candidate confounds per case (native crop height/width/area,
axial slice position as a fraction of volume depth, count of tumor-bearing
slices, 3D tumor voxel count, and crop mean intensity) plus tumor volume
(`original_shape_MeshVolume`, the one plausible *real* confound -- tumor
size is a genuine, expected prognostic factor, so correlating with it
isn't itself suspicious, but the risk score should carry more than just
that).

## Results

Spearman correlation with predicted risk (371 cases):

| variable | image risk (rho, p) | radiomics risk (rho, p) |
|---|---|---|
| crop_height_px | +0.250, p<0.001 | +0.191, p<0.001 |
| crop_width_px | +0.158, p=0.002 | +0.219, p<0.001 |
| crop_area_px | +0.225, p<0.001 | +0.215, p<0.001 |
| z_position_frac | -0.049, p=0.344 | +0.020, p=0.695 |
| n_tumor_slices | +0.133, p=0.010 | +0.169, p=0.001 |
| tumor_voxels_3d | +0.225, p<0.001 | +0.271, p<0.001 |
| crop_mean_intensity | -0.134, p=0.010 | +0.215, p<0.001 |
| tumor_volume_mm3 | +0.230, p<0.001 | **+0.321, p<0.001** |

Bivariate CoxPH, `image_risk` + `tumor_volume` jointly predicting
event/time (both standardized):

| covariate | coef | HR | p |
|---|---|---|---|
| image_risk | 0.337 | 1.40 | **0.0006** |
| tumor_volume | 0.0002 | 1.0002 | 0.999 |

## Interpretation

- **No slice-position artifact**: correlation with `z_position_frac` is
  essentially zero for both risk scores (p=0.34, p=0.70). Ruled out.
- **Crop-size correlations exist but are similar magnitude for both
  modes**, and radiomics correlates *more* with tumor volume (rho=0.32)
  than image does (rho=0.23) -- despite radiomics having *no* predictive
  power (C-index ~0.47-0.48, never significant) while image does. If
  image's signal were just a repackaged size proxy, we'd expect it to
  correlate with volume at least as strongly as radiomics, not less. It
  doesn't, which argues against "image risk = volume in disguise."
- **The bivariate Cox model is the strongest evidence**: with both
  covariates in the same model, `image_risk` stays highly significant
  (p=0.0006, HR=1.40 per SD) while `tumor_volume`'s effect collapses to
  essentially nothing (p=0.999). Tumor size does not explain away the
  image signal -- if anything the reverse, image risk explains away what
  little volume signal there was.
- **One mild flag**: `crop_mean_intensity` correlates negatively with
  image risk (rho=-0.134, p=0.010) but positively with radiomics risk
  (rho=+0.215, p<0.001) -- opposite directions, both modest. Not alarming
  on its own, but intensity calibration differences across scans (if this
  cohort spans multiple scanners/protocols, which isn't recorded in the
  available metadata) remain a confound this check can't fully rule out,
  since there's no site/scanner field to test against directly.

## Bottom line

The image-mode signal for `event_type=event` survives this confound check:
no slice-position artifact, weaker (not stronger) volume correlation than
the non-predictive radiomics baseline, and it remains significant after
directly adjusting for tumor volume. This raises confidence that DINOv3 is
picking up something about tumor appearance beyond simple size, though a
scanner/protocol confound can't be excluded without acquisition metadata
this dataset doesn't have. Reasonable next step, per the earlier plan: try
fixing `both` mode (currently underperforming `image` alone) now that the
underlying `image` signal looks trustworthy.
