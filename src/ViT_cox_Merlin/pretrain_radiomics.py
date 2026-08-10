"""Self-supervised masked-patch pretraining of the shallow (1-block)
radiomics transformer on the LUNA16 lung-nodule CT corpus. The paper never
pretrains this branch separately (it's always trained jointly with the
Cox + contrastive losses on the same final dataset) -- this is an
extension: give the shallow transformer a sensible initialization from a
much larger, label-agnostic CT corpus before it ever sees the small NPC
cohort, using a BERT/MAE-style objective since LUNA16 has no survival
labels to train Cox loss against.

Standardization (mean/std, zero-variance feature filtering) is fit ONCE
here on the LUNA16 corpus and saved alongside the checkpoint, so
fine-tuning on NPC tokens can apply the identical transform the pretrained
weights were trained under.

Run in the `dinov3` env: conda run -n dinov3 python src/ViT_cox/pretrain_radiomics.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from radiomics_transformer import MaskedRadiomicsPretrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "ViT_output" / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "ViT_output" / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "ViT_output" / "plots"


def load_corpus(token_files: list[str]) -> tuple[np.ndarray, list[str]]:
    all_tokens = []
    feature_names = None
    for name in token_files:
        path = CACHE_DIR / name
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        data = np.load(path, allow_pickle=True)
        all_tokens.append(data["tokens"])
        if feature_names is None:
            feature_names = list(data["feature_names"])
        print(f"  {name}: {data['tokens'].shape[0]} crops")
    tokens = np.concatenate(all_tokens, axis=0)
    return tokens, feature_names


def preprocess(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Impute per-feature NaN with the feature's corpus median, drop
    zero-variance features, standardize. Returns (clean_tokens, mean, std,
    kept_feature_mask) so the exact same transform can be replayed on NPC
    tokens at fine-tuning time.
    """
    # float64 throughout: raw pyradiomics features span wildly different
    # magnitudes across columns, and numpy's .std() reducing in float32 on
    # an array this size loses enough precision that genuinely-constant
    # columns can appear to have a spurious std just above the 1e-8
    # threshold (confirmed empirically: 45 "kept" in float32 vs the
    # correct 16 in float64 on the NPC token corpus).
    n, p, f = tokens.shape
    flat = tokens.reshape(-1, f).astype(np.float64)

    col_median = np.nanmedian(flat, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(flat))
    flat[nan_rows, nan_cols] = col_median[nan_cols]

    std = flat.std(axis=0)
    kept = std > 1e-8
    flat = flat[:, kept]
    mean = flat.mean(axis=0)
    std = flat[:, :].std(axis=0)
    flat = (flat - mean) / std

    clean = flat.reshape(n, p, -1)
    return clean, mean, std, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--token-files", nargs="+", default=["npc_mri_pretrain_tokens.npz"],
        help="Cache files under ViT_output/cache/ to pretrain on. Default: the domain-matched external "
             "NPC MRI corpus (277 patients, T1/tumor-mask, Zenodo 10.5281/zenodo.13131827), swapped in for "
             "the earlier LUNA16 lung-CT corpus (pass --token-files luna16_tokens_subset0.npz ... to revert).",
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading pretraining corpus: {args.token_files}")
    tokens, feature_names = load_corpus(args.token_files)
    print(f"Total: {tokens.shape[0]} crops x {tokens.shape[1]} patches x {tokens.shape[2]} raw features")

    clean, mean, std, kept = preprocess(tokens)
    feature_dim = clean.shape[-1]
    print(f"After NaN-impute + zero-variance filter: {feature_dim} features kept")

    rng = np.random.default_rng(args.seed)
    n = clean.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TensorDataset(torch.tensor(clean[train_idx], dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(clean[val_idx], dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MaskedRadiomicsPretrainer(feature_dim, embed_dim=args.embed_dim, n_patches=clean.shape[1], mask_ratio=args.mask_ratio).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred, masked = model(batch)
            loss = ((pred[masked] - batch[masked]) ** 2).mean()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                pred, masked = model(batch)
                val_losses.append(((pred[masked] - batch[masked]) ** 2).mean().item())

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        history.append((train_loss, val_loss))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{args.epochs}: train_mse={train_loss:.4f} val_mse={val_loss:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "encoder_state_dict": model.encoder.state_dict(),
        "feature_dim": feature_dim,
        "embed_dim": args.embed_dim,
        "n_patches": clean.shape[1],
        "feature_names": feature_names,
        "kept_feature_mask": kept.tolist(),
        "standardize_mean": mean.tolist(),
        "standardize_std": std.tolist(),
        "n_pretrain_samples": int(n),
        "token_files": args.token_files,
    }
    torch.save(checkpoint, CHECKPOINT_DIR / "radiomics_transformer_pretrained.pt")

    history_arr = np.array(history)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history_arr[:, 0], label="train MSE")
    ax.plot(history_arr[:, 1], label="val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked-token reconstruction MSE")
    ax.legend()
    ax.set_title(f"Radiomics transformer pretraining ({n} crops from {', '.join(args.token_files)})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pretrain_loss_curve.png", dpi=150)
    plt.close(fig)

    with open(CHECKPOINT_DIR / "pretrain_metrics.json", "w") as f:
        json.dump({"final_train_mse": float(history_arr[-1, 0]), "final_val_mse": float(history_arr[-1, 1]), "n_samples": int(n), "feature_dim": feature_dim}, f, indent=2)

    print(f"Wrote {CHECKPOINT_DIR}/radiomics_transformer_pretrained.pt")
    print(f"Wrote {PLOTS_DIR}/pretrain_loss_curve.png")


if __name__ == "__main__":
    main()
