"""Nested cross-validation over the FULL labeled cohort (data/splits/
complete_list.csv, 371 cases) to get a higher-power, lower-variance
estimate of whether any input_mode carries real prognostic signal -- as
opposed to the single fixed 55-case test split used in baseline.py, which
is underpowered to reliably detect a modest true effect.

For each of --n-repeats x --n-folds outer folds (stratified by event):
the outer-train portion is further split into an inner-train/inner-valid
pair (stratified, --inner-valid-frac) purely for early stopping; the model
never sees the outer-test fold during training or stopping. Every case in
the cohort ends up with --n-repeats out-of-fold risk predictions (one per
repeat, since each repeat re-shuffles the folds), averaged into a single
aggregated risk score per case. The concordance index over all 371 cases'
aggregated risk is the headline metric.

A label-permutation null (same discipline as baseline.py, but over the
whole cohort as a single pool since there's no fixed split here) is run on
this whole nested procedure to check whether the aggregated result clears
chance.
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
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split

from dataset import FeatureStore
from training import build_model_and_optimizer, load_config, train_head

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def nested_cv_oof_predictions(
    feature_store: FeatureStore,
    all_ids: np.ndarray,
    input_mode: str,
    cfg: dict,
    device: torch.device,
    n_repeats: int,
    n_folds: int,
    inner_valid_frac: float,
    seed: int,
) -> np.ndarray:
    """Returns per-case out-of-fold risk predictions, shape (n_repeats, n_cases)."""
    labels = feature_store.labels.loc[all_ids]
    event_arr = labels["event"].to_numpy()
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}

    oof = np.full((n_repeats, len(all_ids)), np.nan)

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + repeat)
        for tr_idx, te_idx in skf.split(all_ids, event_arr):
            outer_train_ids = all_ids[tr_idx]
            outer_test_ids = all_ids[te_idx]
            outer_train_event = event_arr[tr_idx]

            inner_train_ids, inner_valid_ids = train_test_split(
                outer_train_ids, test_size=inner_valid_frac, stratify=outer_train_event,
                random_state=seed + repeat,
            )

            X_tr, e_tr, t_tr = feature_store.build(list(inner_train_ids), input_mode)
            X_va, e_va, t_va = feature_store.build(list(inner_valid_ids), input_mode)
            X_te, e_te, t_te = feature_store.build(list(outer_test_ids), input_mode)

            model, opt = build_model_and_optimizer(input_mode, feature_store, cfg, device)
            model, _, _ = train_head(
                model, opt, X_tr, e_tr, t_tr, X_va, e_va, t_va,
                cfg["training"]["epochs"], cfg["training"]["patience"], device,
            )
            model.eval()
            with torch.no_grad():
                risk_te = model(X_te.to(device)).cpu().numpy()

            for cid, r in zip(outer_test_ids, risk_te):
                oof[repeat, id_to_idx[cid]] = r

    return oof


def aggregate_and_score(oof: np.ndarray, feature_store: FeatureStore, all_ids: np.ndarray) -> tuple[np.ndarray, float]:
    agg_risk = np.nanmean(oof, axis=0)
    labels = feature_store.labels.loc[all_ids]
    event = labels["event"].to_numpy(dtype=float)
    time = labels["time"].to_numpy(dtype=float)
    c = concordance_index(time, -agg_risk, event)
    return agg_risk, c


def permute_labels(labels: pd.DataFrame, ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    shuffled = labels.copy()
    ids = list(ids)
    perm = rng.permutation(len(ids))
    shuffled.loc[ids, ["event", "time"]] = labels.loc[ids, ["event", "time"]].to_numpy()[perm]
    return shuffled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--event-type", type=str, default=None)
    parser.add_argument("--input-modes", nargs="+", default=["radiomics", "image", "both"])
    parser.add_argument("--complete-list", type=Path, default=PROJECT_ROOT / "data" / "splits" / "complete_list.csv")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--inner-valid-frac", type=float, default=0.15)
    parser.add_argument("--n-permutations", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    event_type = args.event_type or cfg["data"]["event_type"]
    device = torch.device(cfg["training"]["device"])
    torch.manual_seed(cfg["training"]["seed"])

    feature_store = FeatureStore(cfg, event_type)
    all_ids = pd.read_csv(args.complete_list)["case_id"].to_numpy()
    print(f"Nested CV over {len(all_ids)} cases, event_type={event_type}, {args.n_repeats}x{args.n_folds}-fold")

    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    plots_dir = out_dir / cfg["output"]["plots_dir"] / event_type / "nested_cv"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "event_type": event_type,
        "n_cases": len(all_ids),
        "n_repeats": args.n_repeats,
        "n_folds": args.n_folds,
        "n_permutations": args.n_permutations,
        "modes": {},
    }

    for input_mode in args.input_modes:
        print(f"[{input_mode}] running real nested CV...")
        oof = nested_cv_oof_predictions(
            feature_store, all_ids, input_mode, cfg, device,
            args.n_repeats, args.n_folds, args.inner_valid_frac, cfg["training"]["seed"],
        )
        agg_risk, observed_c = aggregate_and_score(oof, feature_store, all_ids)
        print(f"  aggregated nested-CV c-index over {len(all_ids)} cases: {observed_c:.3f}")

        print(f"[{input_mode}] running {args.n_permutations} label-permutation null nested-CV passes...")
        rng = np.random.default_rng(cfg["training"]["seed"])
        original_labels = feature_store.labels
        null_cindices = []
        try:
            for i in range(args.n_permutations):
                feature_store.labels = permute_labels(original_labels, all_ids, rng)
                oof_null = nested_cv_oof_predictions(
                    feature_store, all_ids, input_mode, cfg, device,
                    args.n_repeats, args.n_folds, args.inner_valid_frac, cfg["training"]["seed"] + 1000 + i,
                )
                _, c_null = aggregate_and_score(oof_null, feature_store, all_ids)
                null_cindices.append(c_null)
                if (i + 1) % 10 == 0:
                    print(f"    permutation {i + 1}/{args.n_permutations}")
        finally:
            feature_store.labels = original_labels

        p_value = float(np.mean(np.array(null_cindices) >= observed_c))
        print(f"  null mean={np.mean(null_cindices):.3f} std={np.std(null_cindices):.3f}  empirical p={p_value:.3f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(null_cindices, bins=20, color="lightgrey", edgecolor="black")
        ax.axvline(observed_c, color="red", linewidth=2, label=f"observed ({input_mode}) = {observed_c:.3f}")
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Aggregated nested-CV concordance index")
        ax.set_ylabel("Count (label-permuted runs)")
        ax.set_title(f"Nested-CV label-permutation null (n={len(null_cindices)})\nempirical p = {p_value:.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"nested_null_{input_mode}.png", dpi=150)
        plt.close(fig)

        results["modes"][input_mode] = {
            "observed_cindex": observed_c,
            "null_mean": float(np.mean(null_cindices)),
            "null_std": float(np.std(null_cindices)),
            "empirical_p_value": p_value,
        }

    metrics_path = out_dir / f"nested_cv_{event_type}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {metrics_path}")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
