"""Nested CV + label-permutation validation for the hybrid ViT-Cox model,
mirroring src/linear_model/nested_cv.py's protocol (the one that
distinguished the real event/image signal from noise): 5 repeats x 5-fold
stratified outer CV over the full 371-case cohort, inner train/valid split
per outer fold for early stopping, aggregated out-of-fold risk -> one
C-index over the whole cohort, then a label-permutation null to get an
empirical p-value.

Scope note: each hybrid-model training call is far heavier than the
linear head's (full transformer + contrastive loss), so this uses fewer
epochs (150 vs unbounded) and fewer permutations (30 vs 100) than the
linear-head study for tractability -- resolution on the empirical p-value
is coarser (~0.033) as a result.

The radiomics transformer's LUNA16 pretraining is reused as a fixed
initialization every fold (it never saw NPC data, so no leakage risk from
reusing it across folds); image-feature standardization follows the same
precedent set in src/linear_model/dataset.py (fit once on the original
stratified_train.csv, not refit per outer fold).

Image branch = cached **Merlin** 2048-d embeddings; everything else is
identical to src/ViT_cox/nested_cv.py.

Run in the `MERLIN` env: conda run -n MERLIN python src/ViT_cox_Merlin/nested_cv.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ViT_cox_Merlin"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "linear_model"))

from hybrid_model import HybridViTCox, nt_xent_loss  # noqa: E402
from head_model import cox_ph_loss  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_full_cohort(cfg: dict):
    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    train_ids = pd.read_csv(splits_dir / cfg["data"]["train_split"])["case_id"].tolist()
    all_ids = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv")["case_id"].tolist()
    all_rows = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv").set_index("case_id")

    event_type = cfg["data"]["event_type"]
    time_col = cfg["data"]["event_time_columns"][event_type]

    labels = {}
    for cid in all_ids:
        with open(PROJECT_ROOT / all_rows.loc[cid, "label_path"]) as f:
            label = json.load(f)
        labels[cid] = {"event": label[event_type], "time": label[time_col]}
    labels_df = pd.DataFrame(labels).T

    img_npz = np.load(PROJECT_ROOT / cfg["image"]["cache_file"], allow_pickle=True)
    image_features = pd.DataFrame(img_npz["features"], index=img_npz["case_ids"])
    train_img = image_features.loc[train_ids]
    img_mean, img_std = train_img.mean(axis=0), train_img.std(axis=0).replace(0, 1.0)
    image_features = (image_features - img_mean) / img_std

    rad_npz = np.load(PROJECT_ROOT / cfg["radiomics"]["tokens_cache"], allow_pickle=True)
    rad_ids = list(rad_npz["ids"])
    rad_tokens_raw = rad_npz["tokens"]
    id_to_idx = {cid: i for i, cid in enumerate(rad_ids)}

    ckpt_path = PROJECT_ROOT / cfg["radiomics"]["pretrained_checkpoint"]
    pretrained = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kept = np.array(pretrained["kept_feature_mask"])
    t_mean = np.array(pretrained["standardize_mean"], dtype=np.float64)
    t_std = np.array(pretrained["standardize_std"], dtype=np.float64)

    train_arr = np.stack([rad_tokens_raw[id_to_idx[cid]] for cid in train_ids]).astype(np.float64)
    col_median = np.nanmedian(train_arr.reshape(-1, train_arr.shape[-1]), axis=0)

    def get_radiomics_tensor(ids):
        arr = np.stack([rad_tokens_raw[id_to_idx[cid]] for cid in ids]).astype(np.float64)
        n, p, f = arr.shape
        flat = arr.reshape(-1, f).copy()
        nan_rows, nan_cols = np.where(np.isnan(flat))
        flat[nan_rows, nan_cols] = col_median[nan_cols]
        flat = flat[:, kept]
        flat = (flat - t_mean) / t_std
        return torch.tensor(flat.reshape(n, p, -1), dtype=torch.float32)

    image_t = torch.tensor(image_features.loc[all_ids].to_numpy(), dtype=torch.float32)
    radiomics_t = get_radiomics_tensor(all_ids)

    return {
        "ids": np.array(all_ids),
        "image": image_t,
        "radiomics": radiomics_t,
        "labels": labels_df.loc[all_ids],
        "pretrained": pretrained,
        "radiomics_feature_dim": radiomics_t.shape[-1],
    }


def train_one_fold(image_tr, rad_tr, event_tr, time_tr, image_va, rad_va, event_va, time_va, cfg, data, device):
    model = HybridViTCox(
        image_dim=cfg["image"]["embed_dim"],
        radiomics_feature_dim=data["radiomics_feature_dim"],
        radiomics_embed_dim=cfg["radiomics"]["embed_dim"],
        n_patches=cfg["radiomics"]["n_patches"],
    ).to(device)
    model.radiomics_encoder.load_state_dict(data["pretrained"]["encoder_state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    alpha, lambda_loss, temperature = cfg["model"]["alpha"], cfg["model"]["lambda_loss"], cfg["model"]["temperature"]

    image_tr, rad_tr, event_tr, time_tr = image_tr.to(device), rad_tr.to(device), event_tr.to(device), time_tr.to(device)
    image_va, rad_va, event_va, time_va = image_va.to(device), rad_va.to(device), event_va.to(device), time_va.to(device)

    best_valid_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(cfg["nested_cv"]["epochs"]):
        model.train()
        opt.zero_grad()
        out = model(image_tr, rad_tr, alpha)
        loss = (1 - lambda_loss) * cox_ph_loss(out["risk"], event_tr, time_tr) + lambda_loss * nt_xent_loss(out["z_img"], out["z_rad"], temperature)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_va = model(image_va, rad_va, alpha)
            loss_va = (1 - lambda_loss) * cox_ph_loss(out_va["risk"], event_va, time_va) + lambda_loss * nt_xent_loss(out_va["z_img"], out_va["z_rad"], temperature)

        if loss_va.item() < best_valid_loss - 1e-5:
            best_valid_loss, best_epoch = loss_va.item(), epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg["nested_cv"]["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def nested_cv_oof_predictions(data, cfg, device, n_repeats, n_folds, inner_valid_frac, seed):
    all_ids = data["ids"]
    labels = data["labels"].loc[all_ids]
    event_arr = labels["event"].to_numpy()
    id_to_pos = {cid: i for i, cid in enumerate(all_ids)}

    oof = np.full((n_repeats, len(all_ids)), np.nan)

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + repeat)
        for tr_idx, te_idx in skf.split(all_ids, event_arr):
            outer_train_ids = all_ids[tr_idx]
            outer_test_ids = all_ids[te_idx]
            outer_train_event = event_arr[tr_idx]

            inner_train_ids, inner_valid_ids = train_test_split(
                outer_train_ids, test_size=inner_valid_frac, stratify=outer_train_event, random_state=seed + repeat,
            )
            tr_pos = [id_to_pos[c] for c in inner_train_ids]
            va_pos = [id_to_pos[c] for c in inner_valid_ids]
            te_pos = [id_to_pos[c] for c in outer_test_ids]

            model = train_one_fold(
                data["image"][tr_pos], data["radiomics"][tr_pos], torch.tensor(labels["event"].to_numpy(dtype=np.float32)[tr_pos]), torch.tensor(labels["time"].to_numpy(dtype=np.float32)[tr_pos]),
                data["image"][va_pos], data["radiomics"][va_pos], torch.tensor(labels["event"].to_numpy(dtype=np.float32)[va_pos]), torch.tensor(labels["time"].to_numpy(dtype=np.float32)[va_pos]),
                cfg, data, device,
            )
            model.eval()
            with torch.no_grad():
                out_te = model(data["image"][te_pos].to(device), data["radiomics"][te_pos].to(device), cfg["model"]["alpha"])
            risk_te = out_te["risk"].cpu().numpy()
            for cid, r in zip(outer_test_ids, risk_te):
                oof[repeat, id_to_pos[cid]] = r

    return oof


def aggregate_and_score(oof, labels_df, all_ids):
    agg_risk = np.nanmean(oof, axis=0)
    labels = labels_df.loc[all_ids]
    event = labels["event"].to_numpy(dtype=float)
    time = labels["time"].to_numpy(dtype=float)
    c = concordance_index(time, -agg_risk, event)
    return agg_risk, c


def permute_labels(labels_df, ids, rng):
    shuffled = labels_df.copy()
    ids = list(ids)
    perm = rng.permutation(len(ids))
    shuffled.loc[ids, ["event", "time"]] = labels_df.loc[ids, ["event", "time"]].to_numpy()[perm]
    return shuffled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "vit_cox_merlin_config.yaml")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--inner-valid-frac", type=float, default=0.15)
    parser.add_argument("--n-permutations", type=int, default=30)
    parser.add_argument("--nested-epochs", type=int, default=150)
    parser.add_argument("--nested-patience", type=int, default=20)
    parser.add_argument("--lambda-loss", type=float, default=None, help="Overrides config's model.lambda_loss.")
    parser.add_argument("--alpha", type=float, default=None, help="Overrides config's model.alpha.")
    parser.add_argument("--temperature", type=float, default=None, help="Overrides config's model.temperature (NT-Xent).")
    parser.add_argument("--tag", type=str, default="", help="Suffix for output filenames, e.g. to avoid overwriting across sweep runs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["nested_cv"] = {"epochs": args.nested_epochs, "patience": args.nested_patience}
    if args.lambda_loss is not None:
        cfg["model"]["lambda_loss"] = args.lambda_loss
    if args.alpha is not None:
        cfg["model"]["alpha"] = args.alpha
    if args.temperature is not None:
        cfg["model"]["temperature"] = args.temperature
    device = torch.device(cfg["training"]["device"])
    torch.manual_seed(cfg["training"]["seed"])

    print("Loading full cohort...")
    data = load_full_cohort(cfg)
    print(f"{len(data['ids'])} cases, radiomics feature dim={data['radiomics_feature_dim']}")

    print(f"Running real nested CV ({args.n_repeats}x{args.n_folds}-fold)...")
    oof = nested_cv_oof_predictions(data, cfg, device, args.n_repeats, args.n_folds, args.inner_valid_frac, cfg["training"]["seed"])
    agg_risk, observed_c = aggregate_and_score(oof, data["labels"], data["ids"])
    print(f"Observed aggregated C-index: {observed_c:.3f}")

    print(f"Running {args.n_permutations} label-permutation null passes...")
    rng = np.random.default_rng(cfg["training"]["seed"])
    original_labels = data["labels"]
    null_cindices = []
    try:
        for i in range(args.n_permutations):
            data["labels"] = permute_labels(original_labels, data["ids"], rng)
            oof_null = nested_cv_oof_predictions(data, cfg, device, args.n_repeats, args.n_folds, args.inner_valid_frac, cfg["training"]["seed"] + 1000 + i)
            _, c_null = aggregate_and_score(oof_null, data["labels"], data["ids"])
            null_cindices.append(c_null)
            print(f"  permutation {i + 1}/{args.n_permutations}: c={c_null:.3f}")
    finally:
        data["labels"] = original_labels

    p_value = float(np.mean(np.array(null_cindices) >= observed_c))
    print(f"null mean={np.mean(null_cindices):.3f} std={np.std(null_cindices):.3f} empirical p={p_value:.3f}")

    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    plots_dir = out_dir / cfg["output"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(null_cindices, bins=15, color="lightgrey", edgecolor="black")
    ax.axvline(observed_c, color="red", linewidth=2, label=f"observed = {observed_c:.3f}")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Aggregated nested-CV concordance index")
    ax.set_ylabel("Count (label-permuted runs)")
    ax.set_title(f"Hybrid ViT-Cox nested-CV null (n={len(null_cindices)})\nempirical p = {p_value:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"nested_cv_null_hybrid{suffix}.png", dpi=150)
    plt.close(fig)

    results = {
        "n_cases": len(data["ids"]),
        "n_repeats": args.n_repeats,
        "n_folds": args.n_folds,
        "n_permutations": args.n_permutations,
        "nested_epochs": args.nested_epochs,
        "lambda_loss": cfg["model"]["lambda_loss"],
        "alpha": cfg["model"]["alpha"],
        "temperature": cfg["model"]["temperature"],
        "observed_cindex": observed_c,
        "null_mean": float(np.mean(null_cindices)),
        "null_std": float(np.std(null_cindices)),
        "empirical_p_value": p_value,
    }
    with open(out_dir / f"nested_cv_hybrid{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_dir}/nested_cv_hybrid{suffix}.json")
    print(f"Wrote {plots_dir}/nested_cv_null_hybrid{suffix}.png")


if __name__ == "__main__":
    main()
