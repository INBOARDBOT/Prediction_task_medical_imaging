"""Baseline calibration for the linear Cox head.

  1. Label-permutation null: shuffle (event, time) pairs within each split
     (train/valid/test independently, so each split's censoring rate and
     size stay identical to the real experiment) and rerun the exact same
     train / early-stop-on-valid / test procedure many times. This gives
     the test-C-index band the pipeline would produce on pure noise at
     this sample size -- if the real (unshuffled) test C-index doesn't
     clear this band, it isn't distinguishable from chance.

  2. Single-covariate baseline: a classical CoxPH (lifelines) fit on tumor
     volume alone (original_shape_MeshVolume), the simplest clinically
     meaningful prognostic proxy, as a floor the fancier pipeline should
     beat.
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
from lifelines import CoxPHFitter

from dataset import FeatureStore
from training import compute_concordance, load_config, run_final  # noqa: F401  (compute_concordance kept for parity/reuse)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def permute_labels_within_splits(labels: pd.DataFrame, split_id_lists: list[list[str]], rng: np.random.Generator) -> pd.DataFrame:
    shuffled = labels.copy()
    for ids in split_id_lists:
        ids = list(ids)
        perm = rng.permutation(len(ids))
        shuffled.loc[ids, ["event", "time"]] = labels.loc[ids, ["event", "time"]].to_numpy()[perm]
    return shuffled


def run_permutation_baseline(
    feature_store: FeatureStore,
    train_ids: list[str], valid_ids: list[str], test_ids: list[str],
    input_mode: str, cfg: dict, device: torch.device,
    n_permutations: int, seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    original_labels = feature_store.labels

    null_test_cindices = []
    try:
        for i in range(n_permutations):
            feature_store.labels = permute_labels_within_splits(original_labels, [train_ids, valid_ids, test_ids], rng)
            _, _, _, _, _, test_c = run_final(feature_store, train_ids, valid_ids, test_ids, input_mode, cfg, device)
            null_test_cindices.append(test_c)
            if (i + 1) % 20 == 0:
                print(f"  permutation {i + 1}/{n_permutations}")
    finally:
        feature_store.labels = original_labels

    return null_test_cindices


def run_volume_baseline(train_ids: list[str], test_ids: list[str], event_type: str, time_column: str, penalizer: float = 0.1) -> float:
    feature_name = "original_shape_MeshVolume"

    def load(ids):
        rows = []
        for case_id in ids:
            with open(PROJECT_ROOT / "data" / "radiomics" / f"{case_id}.json") as f:
                raw = json.load(f)
            with open(PROJECT_ROOT / "data" / "labels" / f"{case_id}.json") as f:
                label = json.load(f)
            rows.append({
                "case_id": case_id,
                feature_name: raw[feature_name],
                "event": label[event_type],
                "time_months": label[time_column],
            })
        return pd.DataFrame(rows).set_index("case_id")

    train_df = load(train_ids)
    test_df = load(test_ids)

    mu, sigma = train_df[feature_name].mean(), train_df[feature_name].std()
    train_df[feature_name] = (train_df[feature_name] - mu) / sigma
    test_df[feature_name] = (test_df[feature_name] - mu) / sigma

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_df, duration_col="time_months", event_col="event")
    return float(cph.score(test_df, scoring_method="concordance_index"))


def plot_null_distribution(null_cindices: list[float], observed: float, p_value: float, input_mode: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(null_cindices, bins=20, color="lightgrey", edgecolor="black")
    ax.axvline(observed, color="red", linewidth=2, label=f"observed ({input_mode}) = {observed:.3f}")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Test concordance index")
    ax.set_ylabel("Count (label-permuted runs)")
    ax.set_title(f"Label-permutation null (n={len(null_cindices)})\nempirical p = {p_value:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config_merlin.yaml")
    parser.add_argument("--event-type", type=str, default=None)
    parser.add_argument("--input-modes", nargs="+", default=["radiomics", "image", "both"])
    parser.add_argument("--n-permutations", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    event_type = args.event_type or cfg["data"]["event_type"]
    device = torch.device(cfg["training"]["device"])
    torch.manual_seed(cfg["training"]["seed"])

    feature_store = FeatureStore(cfg, event_type)
    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    train_ids = pd.read_csv(splits_dir / cfg["data"]["train_split"])["case_id"].tolist()
    valid_ids = pd.read_csv(splits_dir / cfg["data"]["valid_split"])["case_id"].tolist()
    test_ids = pd.read_csv(splits_dir / cfg["data"]["test_split"])["case_id"].tolist()

    print(f"Volume-only classical CoxPH baseline (event_type={event_type})...")
    volume_c = run_volume_baseline(train_ids, test_ids, event_type, cfg["data"]["event_time_columns"][event_type])
    print(f"  tumor volume alone: test c-index = {volume_c:.3f}")

    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    plots_dir = out_dir / cfg["output"]["plots_dir"] / event_type / "baseline"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "event_type": event_type,
        "volume_only_test_cindex": volume_c,
        "n_permutations": args.n_permutations,
        "modes": {},
    }

    for input_mode in args.input_modes:
        print(f"Observed model ({input_mode}): training real final model...")
        _, _, _, _, _, observed_c = run_final(feature_store, train_ids, valid_ids, test_ids, input_mode, cfg, device)
        print(f"  observed test c-index = {observed_c:.3f}")

        print(f"Running {args.n_permutations} label-permutation null runs ({input_mode})...")
        null_cindices = run_permutation_baseline(
            feature_store, train_ids, valid_ids, test_ids, input_mode, cfg, device, args.n_permutations, cfg["training"]["seed"]
        )
        p_value = float(np.mean(np.array(null_cindices) >= observed_c))
        print(f"  null mean={np.mean(null_cindices):.3f} std={np.std(null_cindices):.3f}  empirical p={p_value:.3f}")

        plot_null_distribution(null_cindices, observed_c, p_value, input_mode, plots_dir / f"null_{input_mode}.png")

        results["modes"][input_mode] = {
            "observed_test_cindex": observed_c,
            "null_mean": float(np.mean(null_cindices)),
            "null_std": float(np.std(null_cindices)),
            "null_p95": float(np.percentile(null_cindices, 95)),
            "empirical_p_value": p_value,
        }

    metrics_path = out_dir / f"baseline_{event_type}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {metrics_path}")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
