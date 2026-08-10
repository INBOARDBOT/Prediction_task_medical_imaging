"""Sanity-check the event_type=event / image-mode signal (nested-CV C-index
0.643, p=0.000) against plausible confounds before trusting it as a real
biological effect.

The image pipeline crops each case to its tumor bounding box then upsamples
to a fixed resolution -- so a few technical, non-biological quantities are
baked into what DINOv3 actually sees: how large the native crop was (and
therefore how much upsampling/blur it got), which axial slice was picked,
and the raw intensity calibration of that slice. If the predicted risk
score correlates strongly with one of those instead of/more than with
tumor volume (a real, expected prognostic factor), that's a red flag that
the model learned a technical shortcut rather than tumor appearance.

Checks:
  1. Recompute the same aggregated out-of-fold risk score used for the
     nested-CV result (image mode, event_type=event), plus the radiomics
     mode's risk score as a "known non-signal" comparison baseline.
  2. Recompute per-case nuisance variables: native crop height/width/area,
     axial slice position (fraction of volume depth), tumor voxel count
     (3D), and crop mean intensity -- none of these touch the DINOv3
     embedding at all, they're pulled straight from the image/mask.
  3. Spearman-correlate both risk scores against each nuisance variable and
     against tumor volume (from the raw radiomics json).
  4. Fit a bivariate CoxPH (image_risk + volume) to check whether the
     image risk score carries information beyond what tumor size already
     explains.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from lifelines import CoxPHFitter
from scipy.stats import spearmanr

from caching_features import best_slice_roi
from dataset import FeatureStore
from nested_cv import aggregate_and_score, nested_cv_oof_predictions
from training import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_nuisance_variables(case_ids: list[str], rows: pd.DataFrame, margin_px: int) -> pd.DataFrame:
    records = {}
    for case_id in case_ids:
        row = rows.loc[case_id]
        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.uint8)

        areas = mask_arr.sum(axis=(1, 2))
        best_z = int(np.argmax(areas))
        mask_slice = mask_arr[best_z]
        rows_mask = np.any(mask_slice, axis=1)
        cols_mask = np.any(mask_slice, axis=0)
        rmin, rmax = np.where(rows_mask)[0][[0, -1]]
        cmin, cmax = np.where(cols_mask)[0][[0, -1]]
        H, W = mask_slice.shape
        rmin_m = max(0, rmin - margin_px)
        rmax_m = min(H - 1, rmax + margin_px)
        cmin_m = max(0, cmin - margin_px)
        cmax_m = min(W - 1, cmax + margin_px)
        crop_h = rmax_m - rmin_m + 1
        crop_w = cmax_m - cmin_m + 1

        crop = best_slice_roi(image_path, mask_path, margin_px)

        records[case_id] = {
            "crop_height_px": crop_h,
            "crop_width_px": crop_w,
            "crop_area_px": crop_h * crop_w,
            "z_position_frac": best_z / max(1, mask_arr.shape[0] - 1),
            "n_tumor_slices": int((areas > 0).sum()),
            "tumor_voxels_3d": int(mask_arr.sum()),
            "crop_mean_intensity": float(crop.mean()),
        }
    return pd.DataFrame(records).T


def main():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg = load_config(config_path)
    event_type = "event"
    device = torch.device(cfg["training"]["device"])
    torch.manual_seed(cfg["training"]["seed"])

    feature_store = FeatureStore(cfg, event_type)
    all_ids = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv")["case_id"].to_numpy()

    n_repeats, n_folds, inner_valid_frac = 5, 5, 0.15

    print("Recomputing aggregated out-of-fold risk scores (image, radiomics)...")
    oof_image = nested_cv_oof_predictions(feature_store, all_ids, "image", cfg, device, n_repeats, n_folds, inner_valid_frac, cfg["training"]["seed"])
    risk_image, c_image = aggregate_and_score(oof_image, feature_store, all_ids)
    print(f"  image aggregated c-index (sanity check, should match 0.643): {c_image:.3f}")

    oof_radiomics = nested_cv_oof_predictions(feature_store, all_ids, "radiomics", cfg, device, n_repeats, n_folds, inner_valid_frac, cfg["training"]["seed"])
    risk_radiomics, c_radiomics = aggregate_and_score(oof_radiomics, feature_store, all_ids)
    print(f"  radiomics aggregated c-index (sanity check, should match 0.476): {c_radiomics:.3f}")

    print("Computing per-case nuisance variables (crop size, slice position, intensity)...")
    complete_rows = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv").set_index("case_id")
    nuisance = compute_nuisance_variables(list(all_ids), complete_rows, cfg["image"]["roi_margin_px"])

    tumor_volume = pd.Series(
        {cid: json.load(open(PROJECT_ROOT / "data" / "radiomics" / f"{cid}.json"))["original_shape_MeshVolume"] for cid in all_ids},
        name="tumor_volume_mm3",
    )

    df = nuisance.copy()
    df["tumor_volume_mm3"] = tumor_volume
    df["risk_image"] = risk_image
    df["risk_radiomics"] = risk_radiomics

    nuisance_cols = ["crop_height_px", "crop_width_px", "crop_area_px", "z_position_frac", "n_tumor_slices", "tumor_voxels_3d", "crop_mean_intensity", "tumor_volume_mm3"]

    print("\nSpearman correlations with predicted risk:")
    corr_rows = []
    for col in nuisance_cols:
        rho_img, p_img = spearmanr(df[col], df["risk_image"])
        rho_rad, p_rad = spearmanr(df[col], df["risk_radiomics"])
        corr_rows.append({"variable": col, "rho_image_risk": rho_img, "p_image_risk": p_img, "rho_radiomics_risk": rho_rad, "p_radiomics_risk": p_rad})
        print(f"  {col:22s} image: rho={rho_img:+.3f} (p={p_img:.3f})   radiomics: rho={rho_rad:+.3f} (p={p_rad:.3f})")
    corr_df = pd.DataFrame(corr_rows)

    labels = feature_store.labels.loc[all_ids]
    event = labels["event"].to_numpy(dtype=float)
    time = labels["time"].to_numpy(dtype=float)

    print("\nBivariate CoxPH: image_risk + tumor_volume -> event/time")
    cox_df = pd.DataFrame({
        "image_risk": (df["risk_image"] - df["risk_image"].mean()) / df["risk_image"].std(),
        "tumor_volume": (df["tumor_volume_mm3"] - df["tumor_volume_mm3"].mean()) / df["tumor_volume_mm3"].std(),
        "time": time,
        "event": event,
    })
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="time", event_col="event")
    print(cph.summary[["coef", "exp(coef)", "p"]])

    out_dir = PROJECT_ROOT / "save" / "confound_check_event"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = np.where(event == 1, "red", "grey")
    ax.scatter(df["tumor_volume_mm3"], df["risk_image"], c=colors, alpha=0.6, s=20)
    ax.set_xlabel("Tumor volume (mm^3, raw radiomics)")
    ax.set_ylabel("Image-mode aggregated risk score")
    ax.set_title("Image risk vs. tumor volume (red = death)")
    fig.tight_layout()
    fig.savefig(out_dir / "risk_vs_volume.png", dpi=150)
    plt.close(fig)

    corr_df.to_csv(out_dir / "nuisance_correlations.csv", index=False)
    bivariate_summary = cph.summary[["coef", "exp(coef)", "p"]].reset_index().rename(columns={"index": "covariate"})
    bivariate_summary.to_csv(out_dir / "bivariate_cox_image_vs_volume.csv", index=False)

    results = {
        "event_type": event_type,
        "sanity_check_image_cindex": c_image,
        "sanity_check_radiomics_cindex": c_radiomics,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out_dir}/nuisance_correlations.csv")
    print(f"Wrote {out_dir}/bivariate_cox_image_vs_volume.csv")
    print(f"Wrote {out_dir}/risk_vs_volume.png")


if __name__ == "__main__":
    main()
