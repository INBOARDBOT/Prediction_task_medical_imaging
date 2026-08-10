"""5-fold CV diagnostic + final train/valid/test pipeline for the linear Cox
head over radiomics and/or DINOv3 features (input_mode: radiomics/image/both).

  1. 5-fold CV on the train split only (stratified by event) -- a
     robustness/diagnostic report, not the deployed model.
  2. A single final model trained on the full train split, early-stopped on
     the valid split, evaluated once on the test split.
  3. Plots: KM curves by predicted risk group (test set), CV concordance
     index summary, and the final model's train/valid loss curves.

Reads only from the caches written by caching_features.py.
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
import yaml
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

from dataset import FeatureStore
from head_model import LinearCoxHead, TwoBranchCoxHead, cox_ph_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_concordance(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> float:
    # lifelines expects higher score = better survival, opposite of our risk
    # convention (higher risk = shorter expected survival), hence the sign flip.
    return concordance_index(time.numpy(), -risk.detach().cpu().numpy(), event.numpy())


def build_model_and_optimizer(input_mode: str, feature_store: FeatureStore, cfg: dict, device: torch.device):
    """"both" mode gets TwoBranchCoxHead with independent per-modality
    dropout/weight_decay (radiomics has no established signal on its own
    and needs much heavier regularization than image does -- see
    head_model.TwoBranchCoxHead). Single-modality modes keep the plain
    LinearCoxHead + one global weight_decay.
    """
    lr = cfg["training"]["lr"]

    if input_mode == "both":
        radiomics_dim = feature_store.input_dim("radiomics")
        image_dim = feature_store.input_dim("image")
        model = TwoBranchCoxHead(
            radiomics_dim, image_dim,
            radiomics_dropout=cfg["model"]["radiomics_dropout"],
            image_dropout=cfg["model"]["image_dropout"],
        ).to(device)
        opt = torch.optim.Adam(
            [
                {"params": model.radiomics_linear.parameters(), "weight_decay": cfg["model"]["radiomics_weight_decay"]},
                {"params": model.image_linear.parameters(), "weight_decay": cfg["model"]["image_weight_decay"]},
            ],
            lr=lr,
        )
    else:
        input_dim = feature_store.input_dim(input_mode)
        model = LinearCoxHead(input_dim, dropout=cfg["model"]["dropout"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])

    return model, opt


def train_head(
    model, opt,
    X_train, event_train, time_train,
    X_valid, event_valid, time_valid,
    epochs, patience, device,
):
    """Full-batch Adam training with early stopping on valid Cox loss.
    Returns the model restored to its best-valid-loss state, the per-epoch
    (train_loss, valid_loss) history, and the best epoch index.
    """
    X_train, event_train, time_train = X_train.to(device), event_train.to(device), time_train.to(device)
    X_valid, event_valid, time_valid = X_valid.to(device), event_valid.to(device), time_valid.to(device)

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        risk_train = model(X_train)
        loss_train = cox_ph_loss(risk_train, event_train, time_train)
        loss_train.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            risk_valid = model(X_valid)
            loss_valid = cox_ph_loss(risk_valid, event_valid, time_valid)

        history.append((loss_train.item(), loss_valid.item()))

        if loss_valid.item() < best_valid_loss - 1e-5:
            best_valid_loss = loss_valid.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def run_cv(feature_store: FeatureStore, train_ids: list[str], input_mode: str, cfg: dict, device: torch.device) -> list[float]:
    train_labels = feature_store.labels.loc[train_ids]
    event_arr = train_labels["event"].to_numpy()
    train_ids_arr = np.array(train_ids)

    skf = StratifiedKFold(
        n_splits=cfg["training"]["n_folds"], shuffle=True, random_state=cfg["training"]["seed"]
    )

    fold_cindices = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_ids_arr, event_arr)):
        fold_train_ids = train_ids_arr[tr_idx].tolist()
        fold_valid_ids = train_ids_arr[va_idx].tolist()

        X_tr, e_tr, t_tr = feature_store.build(fold_train_ids, input_mode)
        X_va, e_va, t_va = feature_store.build(fold_valid_ids, input_mode)

        model, opt = build_model_and_optimizer(input_mode, feature_store, cfg, device)
        model, _, best_epoch = train_head(
            model, opt, X_tr, e_tr, t_tr, X_va, e_va, t_va,
            cfg["training"]["epochs"], cfg["training"]["patience"], device,
        )
        model.eval()
        with torch.no_grad():
            risk_va = model(X_va.to(device))
        c = compute_concordance(risk_va, e_va, t_va)
        fold_cindices.append(c)
        print(f"  fold {fold_idx + 1}/{cfg['training']['n_folds']}: c-index={c:.3f} (best_epoch={best_epoch})")

    return fold_cindices


def run_final(feature_store: FeatureStore, train_ids, valid_ids, test_ids, input_mode: str, cfg: dict, device: torch.device):
    X_tr, e_tr, t_tr = feature_store.build(train_ids, input_mode)
    X_va, e_va, t_va = feature_store.build(valid_ids, input_mode)
    X_te, e_te, t_te = feature_store.build(test_ids, input_mode)

    model, opt = build_model_and_optimizer(input_mode, feature_store, cfg, device)
    model, history, best_epoch = train_head(
        model, opt, X_tr, e_tr, t_tr, X_va, e_va, t_va,
        cfg["training"]["epochs"], cfg["training"]["patience"], device,
    )

    model.eval()
    with torch.no_grad():
        risk_te = model(X_te.to(device))
    test_c = compute_concordance(risk_te, e_te, t_te)
    print(f"Final model: best_epoch={best_epoch}, test c-index={test_c:.3f}")

    return model, history, risk_te.cpu().numpy(), e_te.numpy(), t_te.numpy(), test_c


def plot_km_by_risk(risk: np.ndarray, event: np.ndarray, time: np.ndarray, out_path: Path) -> float:
    median = np.median(risk)
    high = risk >= median
    low = ~high

    fig, ax = plt.subplots(figsize=(6, 5))
    KaplanMeierFitter().fit(time[high], event_observed=event[high], label=f"High risk (n={int(high.sum())})").plot_survival_function(ax=ax)
    KaplanMeierFitter().fit(time[low], event_observed=event[low], label=f"Low risk (n={int(low.sum())})").plot_survival_function(ax=ax)

    result = logrank_test(time[high], time[low], event_observed_A=event[high], event_observed_B=event[low])
    ax.set_title(f"Test set KM by predicted risk (median split)\nlogrank p={result.p_value:.4f}")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(result.p_value)


def plot_cv_cindices(fold_cindices: list[float], out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(fold_cindices, showmeans=True)
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(1, 0.02, size=len(fold_cindices)), fold_cindices, color="black", zorder=3)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Concordance index")
    ax.set_xticks([1])
    ax.set_xticklabels(["5-fold CV (train split)"])
    ax.set_title(f"CV concordance index: {np.mean(fold_cindices):.3f} +/- {np.std(fold_cindices):.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(history: list[tuple[float, float]], out_path: Path):
    history = np.array(history)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history[:, 0], label="train loss")
    ax.plot(history[:, 1], label="valid loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cox partial NLL")
    ax.legend()
    ax.set_title("Final model training curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--event-type", type=str, default=None, help="Overrides config's data.event_type.")
    parser.add_argument("--input-mode", type=str, default=None, choices=["radiomics", "image", "both"], help="Overrides config's model.input_mode.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    event_type = args.event_type or cfg["data"]["event_type"]
    input_mode = args.input_mode or cfg["model"]["input_mode"]
    device = torch.device(cfg["training"]["device"])
    torch.manual_seed(cfg["training"]["seed"])

    print(f"event_type={event_type} input_mode={input_mode} device={device}")

    feature_store = FeatureStore(cfg, event_type)
    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    train_ids = pd.read_csv(splits_dir / cfg["data"]["train_split"])["case_id"].tolist()
    valid_ids = pd.read_csv(splits_dir / cfg["data"]["valid_split"])["case_id"].tolist()
    test_ids = pd.read_csv(splits_dir / cfg["data"]["test_split"])["case_id"].tolist()

    print(f"5-fold CV on train split ({len(train_ids)} cases)...")
    fold_cindices = run_cv(feature_store, train_ids, input_mode, cfg, device)
    print(f"CV c-index: {np.mean(fold_cindices):.3f} +/- {np.std(fold_cindices):.3f}")

    print("Training final model (train + early-stop on valid)...")
    model, history, risk_te, event_te, time_te, test_c = run_final(
        feature_store, train_ids, valid_ids, test_ids, input_mode, cfg, device
    )

    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    plots_dir = out_dir / cfg["output"]["plots_dir"] / event_type / input_mode
    plots_dir.mkdir(parents=True, exist_ok=True)

    logrank_p = plot_km_by_risk(risk_te, event_te, time_te, plots_dir / "km_by_risk.png")
    plot_cv_cindices(fold_cindices, plots_dir / "cv_cindex.png")
    plot_loss_curves(history, plots_dir / "loss_curve.png")

    checkpoint_path = out_dir / cfg["output"]["checkpoint_file"].format(event_type=event_type, input_mode=input_mode)
    if input_mode == "both":
        arch_info = {
            "architecture": "two_branch",
            "radiomics_dim": feature_store.input_dim("radiomics"),
            "image_dim": feature_store.input_dim("image"),
        }
    else:
        arch_info = {"architecture": "linear", "input_dim": feature_store.input_dim(input_mode)}
    torch.save(
        {
            "state_dict": model.state_dict(),
            "event_type": event_type,
            "input_mode": input_mode,
            **arch_info,
        },
        checkpoint_path,
    )

    metrics_path = out_dir / cfg["output"]["metrics_file"].format(event_type=event_type, input_mode=input_mode)
    metrics = {
        "event_type": event_type,
        "input_mode": input_mode,
        "n_train": len(train_ids),
        "n_valid": len(valid_ids),
        "n_test": len(test_ids),
        "cv_cindices": fold_cindices,
        "cv_cindex_mean": float(np.mean(fold_cindices)),
        "cv_cindex_std": float(np.std(fold_cindices)),
        "test_cindex": test_c,
        "test_logrank_p": logrank_p,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
