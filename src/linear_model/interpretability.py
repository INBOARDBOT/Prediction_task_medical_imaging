"""Qualitative interpretability check for the event_type=event / image-mode
linear head: instead of relying only on the confound check (which rules out
a few specific technical shortcuts), visualize which parts of the tumor
crop the model's risk score is actually sensitive to.

Approach: DINOv3's final-layer patch tokens live in the same normalized
embedding space as the CLS token the linear head was trained on. Projecting
each patch token onto the trained weight vector (the same standardization +
dot product used for the real prediction) gives a per-patch "risk
contribution" map -- upsampled and overlaid on the crop, this is a cheap,
backprop-free saliency proxy (no gradient computation through the frozen
backbone needed).

Selects clear high-risk/died and low-risk/long-censored examples (by the
trained model's own predictions) and renders crop + heatmap side by side.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from caching_features import best_slice_roi, to_model_input
from dataset import FeatureStore
from load_backbone import load_dinov3_backbone, select_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def patch_importance_heatmap(model, image_path, mask_path, margin_px, img_size, w, img_mean, img_std, device):
    crop = best_slice_roi(image_path, mask_path, margin_px)
    x = to_model_input(crop, img_size).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"][0].cpu().numpy()  # (n_patches, 384)

    patch_std = (patch_tokens - img_mean) / img_std
    patch_scores = patch_std @ w  # (n_patches,)

    grid = int(np.sqrt(patch_scores.shape[0]))
    heatmap = patch_scores.reshape(grid, grid)

    heatmap_t = torch.from_numpy(heatmap).float()[None, None]
    heatmap_up = F.interpolate(heatmap_t, size=crop.shape, mode="bilinear", align_corners=False)[0, 0].numpy()

    return crop, heatmap_up


def main():
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "config.yaml"))
    feature_store = FeatureStore(cfg, "event")

    ckpt = torch.load(PROJECT_ROOT / "output" / "final_model_event_image.pt", map_location="cpu", weights_only=False)
    w = ckpt["state_dict"]["linear.weight"][0].numpy()

    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    train_ids = pd.read_csv(splits_dir / cfg["data"]["train_split"])["case_id"].tolist()
    # FeatureStore.image_features is already train-standardized; patch
    # tokens need the same (raw, pre-standardization) train mean/std, so
    # recompute it from the raw npz cache directly.
    cache_dir = PROJECT_ROOT / cfg["cache"]["dir"]
    backbone_name = cfg["image"]["backbone"]
    npz = np.load(cache_dir / cfg["cache"]["image_features_file"].format(backbone=backbone_name), allow_pickle=True)
    raw_features = pd.DataFrame(npz["features"], index=npz["case_ids"])
    raw_train = raw_features.loc[train_ids]
    img_mean = raw_train.mean(axis=0).to_numpy()
    img_std = raw_train.std(axis=0).replace(0, 1.0).to_numpy()

    all_ids = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv")["case_id"].tolist()
    X = feature_store.image_features.loc[all_ids].to_numpy()
    risk = X @ w
    labels = feature_store.labels.loc[all_ids]
    df = pd.DataFrame({"case_id": all_ids, "risk": risk, "event": labels["event"].to_numpy(), "time": labels["time"].to_numpy()}).set_index("case_id")

    high_risk_died = df[df["event"] == 1].sort_values("risk", ascending=False).head(3).index.tolist()
    low_risk_censored = df[(df["event"] == 0) & (df["time"] > 30)].sort_values("risk").head(3).index.tolist()
    examples = high_risk_died + low_risk_censored
    print("High risk / died:", high_risk_died)
    print("Low risk / long censored:", low_risk_censored)

    device = select_device(cfg["device"]["extraction_device"])
    model = load_dinov3_backbone(backbone_name, PROJECT_ROOT / cfg["image"]["weights_path"], device)

    complete_rows = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv").set_index("case_id")

    out_dir = PROJECT_ROOT / "save" / "interpretability_event"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops, heatmaps = [], []
    for case_id in examples:
        row = complete_rows.loc[case_id]
        crop, heatmap = patch_importance_heatmap(
            model, PROJECT_ROOT / row["image_path"], PROJECT_ROOT / row["mask_path"],
            cfg["image"]["roi_margin_px"], cfg["image"]["img_size"], w, img_mean, img_std, device,
        )
        crops.append(crop)
        heatmaps.append(heatmap)
        print(f"  {case_id}: heatmap mean={heatmap.mean():.4f}")

    # Shared color scale across all examples -- per-image normalization would
    # contrast-stretch each heatmap to its own range and hide whether
    # high-risk crops are systematically "more positive" than low-risk ones.
    vmax = max(np.abs(h).max() for h in heatmaps)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for i, case_id in enumerate(examples):
        ax = axes.flat[i]
        ax.imshow(crops[i], cmap="gray")
        im = ax.imshow(heatmaps[i], cmap="RdBu_r", alpha=0.45, vmin=-vmax, vmax=vmax)
        group = "died, high risk" if case_id in high_risk_died else "censored t>30mo, low risk"
        ax.set_title(f"{case_id} ({group})\nrisk={df.loc[case_id, 'risk']:.2f}, heatmap mean={heatmaps[i].mean():+.3f}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Patch-importance heatmap, shared color scale (red = raises predicted risk, blue = lowers it)", fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.6, label="patch risk contribution")
    fig.savefig(out_dir / "risk_heatmaps.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "example_cases.json", "w") as f:
        json.dump({"high_risk_died": high_risk_died, "low_risk_censored": low_risk_censored}, f, indent=2)

    print(f"Wrote {out_dir}/risk_heatmaps.png")


if __name__ == "__main__":
    main()
