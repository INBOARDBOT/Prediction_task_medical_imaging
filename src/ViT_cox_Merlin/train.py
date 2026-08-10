"""Train the Merlin hybrid ViT-Cox model (Fig. 4, p.15) on the NPC cohort:
image branch = cached **Merlin** whole-volume embeddings (2048-d), radiomics
branch = shallow transformer initialized from LUNA16 masked-token
pretraining. Single train/valid/test split (lighter first pass, per project
scope -- full nested-CV validation, like the linear-head study got, is a
follow-up once this architecture is confirmed to train sensibly).

Loss: L_total = (1-lambda_loss) * L_Cox + lambda_loss * L_contrastive
Risk: F = (1-alpha) * r_img + alpha * r_rad

Everything but the image cache is identical to src/ViT_cox/train.py.

Run in the `MERLIN` env: conda run -n MERLIN python src/ViT_cox_Merlin/train.py
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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ViT_cox_Merlin"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "linear_model"))

from hybrid_model import HybridViTCox, nt_xent_loss  # noqa: E402
from head_model import cox_ph_loss  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_concordance(risk: torch.Tensor, event: np.ndarray, time: np.ndarray) -> float:
    return concordance_index(time, -risk.detach().cpu().numpy(), event)


def load_data(cfg: dict):
    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    train_ids = pd.read_csv(splits_dir / cfg["data"]["train_split"])["case_id"].tolist()
    valid_ids = pd.read_csv(splits_dir / cfg["data"]["valid_split"])["case_id"].tolist()
    test_ids = pd.read_csv(splits_dir / cfg["data"]["test_split"])["case_id"].tolist()
    all_rows = pd.concat(
        [pd.read_csv(splits_dir / cfg["data"][k]) for k in ("train_split", "valid_split", "test_split")]
    ).drop_duplicates("case_id").set_index("case_id")

    event_type = cfg["data"]["event_type"]
    time_col = cfg["data"]["event_time_columns"][event_type]

    def load_labels(ids):
        rows = []
        for cid in ids:
            with open(PROJECT_ROOT / all_rows.loc[cid, "label_path"]) as f:
                label = json.load(f)
            rows.append({"event": label[event_type], "time": label[time_col]})
        df = pd.DataFrame(rows)
        return df["event"].to_numpy(dtype=float), df["time"].to_numpy(dtype=float)

    # Image CLS tokens -- reuse the linear-head study's cache directly.
    img_npz = np.load(PROJECT_ROOT / cfg["image"]["cache_file"], allow_pickle=True)
    image_features = pd.DataFrame(img_npz["features"], index=img_npz["case_ids"])
    train_img = image_features.loc[train_ids]
    img_mean, img_std = train_img.mean(axis=0), train_img.std(axis=0).replace(0, 1.0)
    image_features = (image_features - img_mean) / img_std

    # Radiomics tokens -- standardize with the SAME transform the
    # pretrained transformer was trained under (not a fresh NPC-train fit),
    # so the pretrained weights see inputs on the distribution they expect.
    rad_npz = np.load(PROJECT_ROOT / cfg["radiomics"]["tokens_cache"], allow_pickle=True)
    rad_ids = list(rad_npz["ids"])
    rad_tokens_raw = rad_npz["tokens"]  # (N, n_patches, n_raw_features)
    id_to_idx = {cid: i for i, cid in enumerate(rad_ids)}

    ckpt_path = PROJECT_ROOT / cfg["radiomics"]["pretrained_checkpoint"]
    pretrained = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else None

    # The radiomics-token transform (NaN-impute median, zero-variance kept
    # mask, standardization mean/std) must be fit ONCE and applied
    # identically to train/valid/test -- fitting it separately per split
    # (as an earlier version of this function did) gives each split a
    # different kept-feature count depending on which columns happen to be
    # constant in that particular subset, which breaks the model (a Linear
    # layer needs a fixed input dim) and leaks split-specific statistics
    # into what should be a fixed preprocessing step.
    # float64 throughout: some raw pyradiomics features have wildly
    # different magnitudes across columns (e.g. mean ~800 vs ~0.01), and
    # numpy's .std()/.mean() reduce in the array's own dtype by default --
    # in float32 this loses enough precision on a (9360, 107) array that
    # genuinely-constant columns can appear to have a spurious std just
    # above the 1e-8 threshold, silently keeping ~30 dead columns and (if
    # done per-split instead of once on train) giving mismatched feature
    # counts across splits.
    if pretrained is not None:
        kept = np.array(pretrained["kept_feature_mask"])
        transform_mean = np.array(pretrained["standardize_mean"], dtype=np.float64)
        transform_std = np.array(pretrained["standardize_std"], dtype=np.float64)
        train_arr = np.stack([rad_tokens_raw[id_to_idx[cid]] for cid in train_ids]).astype(np.float64)
        col_median = np.nanmedian(train_arr.reshape(-1, train_arr.shape[-1]), axis=0)
    else:
        train_arr = np.stack([rad_tokens_raw[id_to_idx[cid]] for cid in train_ids]).astype(np.float64)
        train_flat = train_arr.reshape(-1, train_arr.shape[-1])
        col_median = np.nanmedian(train_flat, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(train_flat))
        train_flat = train_flat.copy()
        train_flat[nan_rows, nan_cols] = col_median[nan_cols]
        kept = train_flat.std(axis=0) > 1e-8
        transform_mean = train_flat[:, kept].mean(axis=0)
        transform_std = train_flat[:, kept].std(axis=0)

    def get_radiomics_tensor(ids):
        arr = np.stack([rad_tokens_raw[id_to_idx[cid]] for cid in ids]).astype(np.float64)  # (n, patches, raw_features)
        n, p, f = arr.shape
        flat = arr.reshape(-1, f).copy()
        nan_rows, nan_cols = np.where(np.isnan(flat))
        flat[nan_rows, nan_cols] = col_median[nan_cols]
        flat = flat[:, kept]
        flat = (flat - transform_mean) / transform_std
        return torch.tensor(flat.reshape(n, p, -1), dtype=torch.float32)

    def get_image_tensor(ids):
        return torch.tensor(image_features.loc[ids].to_numpy(), dtype=torch.float32)

    data = {}
    for split_name, ids in [("train", train_ids), ("valid", valid_ids), ("test", test_ids)]:
        event, time = load_labels(ids)
        data[split_name] = {
            "ids": ids,
            "image": get_image_tensor(ids),
            "radiomics": get_radiomics_tensor(ids),
            "event": torch.tensor(event, dtype=torch.float32),
            "time": torch.tensor(time, dtype=torch.float32),
            "event_np": event,
            "time_np": time,
        }

    return data, pretrained, event_type


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "vit_cox_merlin_config.yaml")
    parser.add_argument("--lambda-loss", type=float, default=None, help="Overrides config's model.lambda_loss.")
    parser.add_argument("--alpha", type=float, default=None, help="Overrides config's model.alpha.")
    parser.add_argument("--temperature", type=float, default=None, help="Overrides config's model.temperature (NT-Xent).")
    parser.add_argument("--tag", type=str, default="", help="Suffix for output filenames, e.g. to avoid overwriting across sweep runs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"])
    torch.manual_seed(cfg["training"]["seed"])

    print("Loading data...")
    data, pretrained, event_type = load_data(cfg)
    print(f"train={len(data['train']['ids'])} valid={len(data['valid']['ids'])} test={len(data['test']['ids'])}")

    radiomics_feature_dim = data["train"]["radiomics"].shape[-1]
    print(f"radiomics feature dim (post NaN-impute/zero-var filter): {radiomics_feature_dim}")

    model = HybridViTCox(
        image_dim=cfg["image"]["embed_dim"],
        radiomics_feature_dim=radiomics_feature_dim,
        radiomics_embed_dim=cfg["radiomics"]["embed_dim"],
        n_patches=cfg["radiomics"]["n_patches"],
    ).to(device)

    if pretrained is not None:
        model.radiomics_encoder.load_state_dict(pretrained["encoder_state_dict"])
        print(f"Initialized radiomics encoder from {cfg['radiomics']['pretrained_checkpoint']} ({pretrained['n_pretrain_samples']} pretrain samples)")
    else:
        print("No pretrained radiomics checkpoint found -- training radiomics encoder from scratch.")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    alpha = args.alpha if args.alpha is not None else cfg["model"]["alpha"]
    lambda_loss = args.lambda_loss if args.lambda_loss is not None else cfg["model"]["lambda_loss"]
    temperature = args.temperature if args.temperature is not None else cfg["model"]["temperature"]

    for split in data.values():
        split["image"] = split["image"].to(device)
        split["radiomics"] = split["radiomics"].to(device)
        split["event"] = split["event"].to(device)
        split["time"] = split["time"].to(device)

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        opt.zero_grad()
        out = model(data["train"]["image"], data["train"]["radiomics"], alpha)
        loss_cox = cox_ph_loss(out["risk"], data["train"]["event"], data["train"]["time"])
        loss_cl = nt_xent_loss(out["z_img"], out["z_rad"], temperature)
        loss = (1 - lambda_loss) * loss_cox + lambda_loss * loss_cl
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_va = model(data["valid"]["image"], data["valid"]["radiomics"], alpha)
            loss_cox_va = cox_ph_loss(out_va["risk"], data["valid"]["event"], data["valid"]["time"])
            loss_cl_va = nt_xent_loss(out_va["z_img"], out_va["z_rad"], temperature)
            loss_va = (1 - lambda_loss) * loss_cox_va + lambda_loss * loss_cl_va

        history.append((loss.item(), loss_va.item(), loss_cox.item(), loss_cl.item()))
        if (epoch + 1) % 20 == 0 or epoch == 0:
            c_tr = compute_concordance(out["risk"], data["train"]["event_np"], data["train"]["time_np"])
            print(f"  epoch {epoch + 1}: train_loss={loss.item():.4f} (cox={loss_cox.item():.4f} cl={loss_cl.item():.4f}) valid_loss={loss_va.item():.4f} train_c={c_tr:.3f}")

        if loss_va.item() < best_valid_loss - 1e-5:
            best_valid_loss = loss_va.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= cfg["training"]["patience"]:
            print(f"  early stopping at epoch {epoch + 1} (best={best_epoch + 1})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out_te = model(data["test"]["image"], data["test"]["radiomics"], alpha)
    test_c = compute_concordance(out_te["risk"], data["test"]["event_np"], data["test"]["time_np"])
    print(f"Test c-index: {test_c:.3f} (best_epoch={best_epoch + 1})")

    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    plots_dir = out_dir / cfg["output"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    history_arr = np.array(history)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history_arr[:, 0], label="train total loss")
    ax.plot(history_arr[:, 1], label="valid total loss")
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("(1-lambda) L_Cox + lambda L_CL")
    ax.legend()
    ax.set_title(f"Hybrid ViT-Cox training curve (event_type={event_type})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"train_loss_curve_{event_type}{suffix}.png", dpi=150)
    plt.close(fig)

    risk_te = out_te["risk"].cpu().numpy()
    event_te, time_te = data["test"]["event_np"], data["test"]["time_np"]
    median = np.median(risk_te)
    high, low = risk_te >= median, risk_te < median
    fig, ax = plt.subplots(figsize=(6, 5))
    KaplanMeierFitter().fit(time_te[high], event_observed=event_te[high], label=f"High risk (n={int(high.sum())})").plot_survival_function(ax=ax)
    KaplanMeierFitter().fit(time_te[low], event_observed=event_te[low], label=f"Low risk (n={int(low.sum())})").plot_survival_function(ax=ax)
    result = logrank_test(time_te[high], time_te[low], event_observed_A=event_te[high], event_observed_B=event_te[low])
    ax.set_title(f"Hybrid model test set KM by risk\nlogrank p={result.p_value:.4f}")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    fig.tight_layout()
    fig.savefig(plots_dir / f"km_by_risk_{event_type}{suffix}.png", dpi=150)
    plt.close(fig)

    checkpoint_path = out_dir / (cfg["output"]["checkpoint_file"].format(event_type=event_type).replace(".pt", f"{suffix}.pt"))
    torch.save({"state_dict": model.state_dict(), "event_type": event_type, "alpha": alpha, "lambda_loss": lambda_loss}, checkpoint_path)

    metrics = {
        "event_type": event_type,
        "n_train": len(data["train"]["ids"]),
        "n_valid": len(data["valid"]["ids"]),
        "n_test": len(data["test"]["ids"]),
        "alpha": alpha,
        "lambda_loss": lambda_loss,
        "best_epoch": best_epoch + 1,
        "test_cindex": test_c,
        "test_logrank_p": float(result.p_value),
        "used_pretrained_radiomics": pretrained is not None,
        "contrastive_loss_final": float(history[-1][3]) if len(history[-1]) > 3 else None,
        "contrastive_loss_initial": float(history[0][3]) if len(history[0]) > 3 else None,
    }
    metrics_path = out_dir / (cfg["output"]["metrics_file"].format(event_type=event_type).replace(".json", f"{suffix}.json"))
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
