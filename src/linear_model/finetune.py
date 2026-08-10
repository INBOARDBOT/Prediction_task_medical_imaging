"""Fine-tune the last DINOv3 transformer block (of 12; ~8% of backbone
params) jointly with the linear Cox head, instead of using the fully
frozen embedding cache -- the biggest untried lever now that the frozen
signal (event_type=event, image mode) is confirmed real and confound-checked.

Scope, deliberately narrowed given cost/risk:
  - image mode only (radiomics has no established signal to amplify).
  - Single train/valid/test split, NOT the full nested-CV protocol used to
    validate the frozen result -- fine-tuning inside 25 outer folds would
    be prohibitively expensive here. Treat this as a preliminary/exploratory
    comparison, not a re-validation at the same rigor as the frozen result.
  - img_size=224 (DINOv3's native pretraining resolution) instead of the
    768 used for the cached frozen embeddings -- both cheaper (196 vs 2304
    patch tokens/image) and more principled for fine-tuning (768 is
    resolution the backbone never saw during pretraining).
  - Only the last transformer block + final norm are unfrozen; the other
    11 blocks stay frozen (PyTorch's autograd automatically skips storing
    gradients for frozen upstream parameters, so this is efficient without
    manual forward-pass surgery).
  - Backbone params use a much lower learning rate than the head, and a
    separate weight_decay, since a giant lr on a near-28M-param frozen
    representation with 260 training cases would overfit almost instantly.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from lifelines.utils import concordance_index

from caching_features import best_slice_roi, to_model_input
from head_model import cox_ph_loss
from load_backbone import load_dinov3_backbone, select_device
from training import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FineTunedImageCoxHead(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, n_unfrozen_blocks: int = 1):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # n_unfrozen_blocks=0 is the frozen control -- blocks[-0:] would
        # slice ALL blocks (Python has no negative zero), so guard it
        # explicitly rather than unfreezing everything by accident.
        if n_unfrozen_blocks > 0:
            for block in self.backbone.blocks[-n_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            for p in self.backbone.norm.parameters():
                p.requires_grad_(True)
        self.linear = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.backbone.forward_features(x)["x_norm_clstoken"]
        return self.linear(cls).squeeze(-1)

    def backbone_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]


def load_case_tensors(case_ids: list[str], rows: pd.DataFrame, margin_px: int, img_size: int) -> torch.Tensor:
    tensors = []
    for case_id in case_ids:
        row = rows.loc[case_id]
        crop = best_slice_roi(PROJECT_ROOT / row["image_path"], PROJECT_ROOT / row["mask_path"], margin_px)
        tensors.append(to_model_input(crop, img_size))
    return torch.stack(tensors)


def forward_in_batches(model: FineTunedImageCoxHead, X: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    outputs = []
    for i in range(0, X.shape[0], batch_size):
        outputs.append(model(X[i : i + batch_size].to(device)))
    return torch.cat(outputs)


def compute_concordance(risk: torch.Tensor, event: np.ndarray, time: np.ndarray) -> float:
    return concordance_index(time, -risk.detach().cpu().numpy(), event)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-weight-decay", type=float, default=0.01)
    parser.add_argument("--head-weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None, help="Overrides config's training.seed.")
    parser.add_argument("--tag", type=str, default="", help="Suffix for output filenames, e.g. to avoid overwriting across seeds.")
    parser.add_argument("--n-unfrozen-blocks", type=int, default=1, help="0 = frozen control (isolates the img_size effect from the fine-tuning effect).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(cfg["device"]["extraction_device"])
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    torch.manual_seed(seed)
    print(f"Fine-tuning last DINOv3 block, event_type=event, image mode, device={device}, img_size={args.img_size}")

    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    train_ids = pd.read_csv(splits_dir / cfg["data"]["train_split"])["case_id"].tolist()
    valid_ids = pd.read_csv(splits_dir / cfg["data"]["valid_split"])["case_id"].tolist()
    test_ids = pd.read_csv(splits_dir / cfg["data"]["test_split"])["case_id"].tolist()
    all_rows = pd.concat(
        [pd.read_csv(splits_dir / cfg["data"][k]) for k in ("train_split", "valid_split", "test_split")]
    ).drop_duplicates("case_id").set_index("case_id")

    def load_labels(ids):
        rows = []
        for cid in ids:
            with open(PROJECT_ROOT / all_rows.loc[cid, "label_path"]) as f:
                label = json.load(f)
            rows.append({"event": label["event"], "time": label["time_months"]})
        df = pd.DataFrame(rows)
        return df["event"].to_numpy(dtype=float), df["time"].to_numpy(dtype=float)

    event_tr, time_tr = load_labels(train_ids)
    event_va, time_va = load_labels(valid_ids)
    event_te, time_te = load_labels(test_ids)

    print("Loading + preprocessing raw crops (train/valid/test)...")
    margin_px = cfg["image"]["roi_margin_px"]
    X_tr = load_case_tensors(train_ids, all_rows, margin_px, args.img_size)
    X_va = load_case_tensors(valid_ids, all_rows, margin_px, args.img_size)
    X_te = load_case_tensors(test_ids, all_rows, margin_px, args.img_size)
    print(f"  train={X_tr.shape} valid={X_va.shape} test={X_te.shape}")

    backbone = load_dinov3_backbone(cfg["image"]["backbone"], PROJECT_ROOT / cfg["image"]["weights_path"], device)
    model = FineTunedImageCoxHead(backbone, cfg["image"]["embed_dim"], n_unfrozen_blocks=args.n_unfrozen_blocks).to(device)

    event_tr_t = torch.tensor(event_tr, device=device)
    time_tr_t = torch.tensor(time_tr, device=device)
    event_va_t = torch.tensor(event_va, device=device)
    time_va_t = torch.tensor(time_va, device=device)

    opt = torch.optim.Adam(
        [
            {"params": model.backbone_parameters(), "lr": args.backbone_lr, "weight_decay": args.backbone_weight_decay},
            {"params": model.linear.parameters(), "lr": args.head_lr, "weight_decay": args.head_weight_decay},
        ]
    )

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        risk_tr = forward_in_batches(model, X_tr, args.batch_size, device)
        loss_tr = cox_ph_loss(risk_tr, event_tr_t, time_tr_t)
        loss_tr.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            risk_va = forward_in_batches(model, X_va, args.batch_size, device)
            loss_va = cox_ph_loss(risk_va, event_va_t, time_va_t)

        history.append((loss_tr.item(), loss_va.item()))
        print(f"  epoch {epoch + 1}/{args.epochs}: train_loss={loss_tr.item():.4f} valid_loss={loss_va.item():.4f}")

        if loss_va.item() < best_valid_loss - 1e-5:
            best_valid_loss = loss_va.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= args.patience:
            print(f"  early stopping at epoch {epoch + 1} (best={best_epoch + 1})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        risk_te = forward_in_batches(model, X_te, args.batch_size, device)
    test_c = compute_concordance(risk_te, event_te, time_te)
    print(f"Fine-tuned test c-index: {test_c:.3f} (best_epoch={best_epoch + 1})")

    out_dir = PROJECT_ROOT / "save" / "finetune_event"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    history_arr = np.array(history)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history_arr[:, 0], label="train loss")
    ax.plot(history_arr[:, 1], label="valid loss")
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"best epoch ({best_epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cox partial NLL")
    ax.legend()
    ax.set_title(f"Fine-tuning loss curve (last DINOv3 block + head, seed={seed})")
    fig.tight_layout()
    fig.savefig(out_dir / f"finetune_loss_curve{suffix}.png", dpi=150)
    plt.close(fig)

    results = {
        "event_type": "event",
        "input_mode": "image",
        "seed": seed,
        "img_size": args.img_size,
        "n_unfrozen_blocks": args.n_unfrozen_blocks,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "best_epoch": best_epoch + 1,
        "test_cindex_finetuned": test_c,
        "frozen_baseline_single_split_test_cindex": 0.588,
        "frozen_baseline_nested_cv_cindex": 0.643,
    }
    with open(out_dir / f"results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_dir}/results{suffix}.json")
    print(f"Wrote {out_dir}/finetune_loss_curve{suffix}.png")


if __name__ == "__main__":
    main()
