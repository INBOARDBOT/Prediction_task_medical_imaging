import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = Path(__file__).resolve().parent 



def load_feature_matrix(split_path: Path, event_type: str = "event", time_month: str = "time_months") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    split = pd.read_csv(split_path).set_index("case_id")

    feature_rows = {}
    label_rows = {}
    for case_id, row in split.iterrows():
        feat_path = PROJECT_ROOT / row["radiomics"]
        with open(feat_path) as f:
            feature_rows[case_id] = json.load(f)

        label_path = PROJECT_ROOT / row["label_path"]
        with open(label_path) as f:
            label = json.load(f)
        label_rows[case_id] = {event_type: label[event_type], time_month: label[time_month]}
    features = pd.DataFrame(feature_rows).T.loc[split.index]
    labels = pd.DataFrame(label_rows).T.loc[split.index]

    return features, labels[event_type], labels[time_month]


def univariate_cox_stats(
    features: pd.DataFrame,
    event: pd.Series,
    time: pd.Series,
    n_folds: int,
    seed: int,
    penalizer: float,
) -> pd.DataFrame:

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(features, event))

    stats = {}
    for name in features.columns:
        col = features[name]
        if col.std() == 0 or not np.isfinite(col.std()):
            stats[name] = {"concordance_index": np.nan, "concordance_index_std": np.nan, "p_value": np.nan, "fit_ok": False}
            continue

        fold_scores = []
        fit_ok = True
        for train_idx, test_idx in folds:
            train_col = col.iloc[train_idx]
            mu, sigma = train_col.mean(), train_col.std()
            if sigma == 0 or not np.isfinite(sigma):
                fit_ok = False
                break

            train_df = pd.DataFrame({
                name: (train_col - mu) / sigma,
                "time_months": time.iloc[train_idx].to_numpy(),
                "event": event.iloc[train_idx].to_numpy(),
            })
            test_df = pd.DataFrame({
                name: (col.iloc[test_idx] - mu) / sigma,
                "time_months": time.iloc[test_idx].to_numpy(),
                "event": event.iloc[test_idx].to_numpy(),
            })
            try:
                cph = CoxPHFitter(penalizer=penalizer)
                cph.fit(train_df, duration_col="time_months", event_col="event")
                fold_scores.append(cph.score(test_df, scoring_method="concordance_index"))
            except Exception:
                fit_ok = False
                break

        if not fit_ok or not fold_scores:
            stats[name] = {"concordance_index": np.nan, "concordance_index_std": np.nan, "p_value": np.nan, "fit_ok": False}
            continue

        mu_full, sigma_full = col.mean(), col.std()
        full_df = pd.DataFrame({
            name: (col - mu_full) / sigma_full,
            "time_months": time.to_numpy(),
            "event": event.to_numpy(),
        })
        try:
            cph_full = CoxPHFitter(penalizer=penalizer)
            cph_full.fit(full_df, duration_col="time_months", event_col="event")
            p_value = cph_full.summary.loc[name, "p"]
        except Exception:
            p_value = np.nan

        stats[name] = {
            "concordance_index": float(np.mean(fold_scores)),
            "concordance_index_std": float(np.std(fold_scores)),
            "p_value": p_value,
            "fit_ok": True,
        }

    return pd.DataFrame(stats).T


def cluster_redundant_features(features: pd.DataFrame, candidates: list[str], corr_threshold: float) -> dict[str, int]:
    if len(candidates) == 1:
        return {candidates[0]: 1}

    corr = features[candidates].corr(method="spearman").abs()
    distance = 1 - corr
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance.values, 0.0)

    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1 - corr_threshold, criterion="distance")
    return dict(zip(candidates, cluster_ids))


def select_features(
    features: pd.DataFrame,
    stats: pd.DataFrame,
    corr_threshold: float,
    max_features: int,
) -> pd.DataFrame:
    report = stats.copy()
    report["kept"] = False
    report["cluster_id"] = np.nan
    report["drop_reason"] = "fit_failed_or_zero_variance"

    fit_ok = report["fit_ok"] == True  # noqa: E712
    candidates = report.index[fit_ok].tolist()
    if not candidates:
        return report

    clusters = cluster_redundant_features(features, candidates, corr_threshold)
    report.loc[candidates, "cluster_id"] = [clusters[n] for n in candidates]
    report.loc[candidates, "drop_reason"] = "redundant"

    best_per_cluster = []
    for cluster_id in set(clusters.values()):
        members = [n for n, c in clusters.items() if c == cluster_id]
        best = report.loc[members, "concordance_index"].idxmax()
        best_per_cluster.append(best)
        for n in members:
            if n != best:
                report.loc[n, "drop_reason"] = f"redundant_with_{best}"

    ranked = report.loc[best_per_cluster].sort_values("concordance_index", ascending=False)
    final = ranked.index[:max_features].tolist()
    dropped_for_cap = ranked.index[max_features:].tolist()
    for n in dropped_for_cap:
        report.loc[n, "drop_reason"] = "cut_by_max_features"

    report.loc[final, "kept"] = True
    report.loc[final, "drop_reason"] = "kept"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=SPLITS_DIR / "stratified_train.csv",
        help="CSV with case_id, event, time_months to fit selection on (train split only).",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.85, help="|Spearman correlation| above which features are considered redundant.")
    parser.add_argument("--event_type", type=str, default="event", help="Column name for the event indicator in the split file.")
    parser.add_argument("--time_month", type=str, default="time_months", help="Column name for the time indicator in the split file.")
    parser.add_argument("--max-features", type=int, default=15, help="Cap on the number of selected features.")
    parser.add_argument("--n-folds", type=int, default=5, help="Stratified K-fold count for out-of-fold feature scoring.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the K-fold split.")
    parser.add_argument("--penalizer", type=float, default=0.1, help="L2 penalizer for CoxPHFitter; raise this if you still see ConvergenceWarning (e.g. with rare event types like recurrence/metastasis).")
    args = parser.parse_args()

    features, event, time = load_feature_matrix(args.split_file, args.event_type, args.time_month)
    print(f"Loaded {features.shape[0]} cases x {features.shape[1]} features from {args.split_file.name}")
    print(f"Event column '{args.event_type}': {int(event.sum())} / {len(event)} positive ({event.mean():.1%})")

    stats = univariate_cox_stats(features, event, time, args.n_folds, args.seed, args.penalizer)
    report = select_features(features, stats, args.corr_threshold, args.max_features)

    event_output_dir = OUTPUT_DIR / args.event_type
    event_output_dir.mkdir(parents=True, exist_ok=True)
    report_out = report.reset_index().rename(columns={"index": "feature"})
    report_out.to_csv(event_output_dir / "feature_report.csv", index=False)

    selected = report.index[report["kept"]].tolist()
    print(f"Selected {len(selected)} / {features.shape[1]} features")

    selected_payload = {
        "split_used": args.split_file.name,
        "event_type": args.event_type,
        "n_train_cases": int(features.shape[0]),
        "corr_threshold": args.corr_threshold,
        "max_features": args.max_features,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "features": [
            {   
                "name": name,
                "concordance_index": float(report.loc[name, "concordance_index"]),
                "concordance_index_std": float(report.loc[name, "concordance_index_std"]),
                "p_value": float(report.loc[name, "p_value"]),
                "train_mean": float(features[name].mean()),
                "train_std": float(features[name].std()),
            }
            for name in selected
        ],
    }
    with open(event_output_dir / "selected_features.json", "w") as f:
        json.dump(selected_payload, f, indent=2)

    print(f"Wrote {event_output_dir / 'selected_features.json'}")
    print(f"Wrote {event_output_dir / 'feature_report.csv'}")


if __name__ == "__main__":
    main()
