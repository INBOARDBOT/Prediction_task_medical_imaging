"""Nested-CV validation of the attention-pool Cox head (option 3), using the
identical protocol to src/linear_model/nested_cv.py (5x5 stratified outer CV
over the full 371-case cohort, inner train/valid split for early stopping,
aggregated out-of-fold risk -> one C-index, label-permutation null for an
empirical p-value) so its number is directly comparable to the single-slice
0.643 and the mean-pool variants.

Input is the per-slice cache from caching_multislice.py; image tokens are
standardized per-dim using statistics fit ONCE on the original
stratified_train.csv cases' slices (same discipline as the base pipeline's
image-feature standardization), not refit per fold.

Run in the `dinov3` env (only needs torch/lifelines/sklearn, no GPU):
  conda run -n dinov3 python nested_cv_attention.py
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
sys.path.insert(0, str(PROJECT_ROOT / "src" / "linear_model"))

from head_model import cox_ph_loss  # noqa: E402
from attention_head import AttentionPoolCoxHead  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class SliceStore:
    """Per-slice token store -> padded (tokens, mask) tensors per case list."""

    def __init__(self, cfg: dict, event_type: str, perslice_npz: Path):
        d = np.load(perslice_npz, allow_pickle=True)
        cids = d["case_ids"].astype(str)0
        .0.





























        
        feats = d["features"].astype(np.float32)
        self.by_case: dict[str, np.ndarray] = {}
        order = np.argsort(-d["areas"])  # largest tumor slice first, stable-ish
        cids_o, feats_o = cids[order], feats[order]
        for cid, f in zip(cids_o, feats_o):
            self.by_case.setdefault(cid, []).append(f)
        self.by_case = {c: np.stack(v) for c, v in self.by_case.items()}
        self.max_slices = max(v.shape[0] for v in self.by_case.values())
        self.dim = feats.shape[1]

        # Standardize per-dim using the original train split's slices only.
        splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
        train_ids = pd.read_csv(splits_dir / cfg["data"]["train_split"])["case_id"].tolist()
        train_slices = np.concatenate([self.by_case[c] for c in train_ids if c in self.by_case], axis=0)
        self.mean = train_slices.mean(axis=0)
        self.std = train_slices.std(axis=0)
        self.std[self.std == 0] = 1.0

        self.labels = self._load_labels(cfg, event_type)

    def _load_labels(self, cfg: dict, event_type: str) -> pd.DataFrame:
        all_rows = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv").set_index("case_id")
        time_col = cfg["data"]["event_time_columns"][event_type]
        rows = {}
        for cid, row in all_rows.iterrows():
            with open(PROJECT_ROOT / row["label_path"]) as f:
                label = json.load(f)
            rows[cid] = {"event": label[event_type], "time": label[time_col]}
        return pd.DataFrame(rows).T

    def build(self, case_ids):
        n, s, d = len(case_ids), self.max_slices, self.dim
        X = np.zeros((n, s, d), dtype=np.float32)
        mask = np.zeros((n, s), dtype=np.float32)
        for i, cid in enumerate(case_ids):
            toks = (self.by_case[cid] - self.mean) / self.std
            k = toks.shape[0]
            X[i, :k] = toks
            mask[i, :k] = 1.0
        lab = self.labels.loc[list(case_ids)]
        return (
            torch.tensor(X), torch.tensor(mask),
            torch.tensor(lab["event"].to_numpy(dtype=np.float32)),
            torch.tensor(lab["time"].to_numpy(dtype=np.float32)),
        )


def train_attention(store, tr_ids, va_ids, cfg, args, device):
    model = AttentionPoolCoxHead(store.dim, attn_dim=args.attn_dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    Xtr, mtr, etr, ttr = (t.to(device) for t in store.build(tr_ids))
    Xva, mva, eva, tva = (t.to(device) for t in store.build(va_ids))

    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = cox_ph_loss(model(Xtr, mtr), etr, ttr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = cox_ph_loss(model(Xva, mva), eva, tva)
        if vloss.item() < best_loss - 1e-5:
            best_loss, best_epoch = vloss.item(), epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def nested_oof(store, all_ids, cfg, args, device, seed):
    event_arr = store.labels.loc[list(all_ids)]["event"].to_numpy()
    id_to_pos = {c: i for i, c in enumerate(all_ids)}
    oof = np.full((args.n_repeats, len(all_ids)), np.nan)
    for repeat in range(args.n_repeats):
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed + repeat)
        for tr_idx, te_idx in skf.split(all_ids, event_arr):
            outer_train = all_ids[tr_idx]
            outer_test = all_ids[te_idx]
            inner_tr, inner_va = train_test_split(
                outer_train, test_size=args.inner_valid_frac,
                stratify=event_arr[tr_idx], random_state=seed + repeat,
            )
            model = train_attention(store, inner_tr, inner_va, cfg, args, device)
            model.eval()
            Xte, mte, _, _ = (t.to(device) for t in store.build(outer_test))
            with torch.no_grad():
                risk = model(Xte, mte).cpu().numpy()
            for cid, r in zip(outer_test, risk):
                oof[repeat, id_to_pos[cid]] = r
    return oof


def score(oof, store, all_ids):
    agg = np.nanmean(oof, axis=0)
    lab = store.labels.loc[list(all_ids)]
    return agg, concordance_index(lab["time"].to_numpy(float), -agg, lab["event"].to_numpy(float))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    p.add_argument("--perslice", type=Path, default=PROJECT_ROOT / "output" / "cache" / "image_features_dinov3_perslice.npz")
    p.add_argument("--event-type", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "output_multislice_attention")
    p.add_argument("--attn-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--inner-valid-frac", type=float, default=0.15)
    p.add_argument("--n-permutations", type=int, default=100)
    args = p.parse_args()

    cfg = load_config(args.config)
    event_type = args.event_type or cfg["data"]["event_type"]
    device = torch.device(cfg["training"]["device"])
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)

    store = SliceStore(cfg, event_type, args.perslice)
    all_ids = np.array(pd.read_csv(PROJECT_ROOT / "data" / "splits" / "complete_list.csv")["case_id"].tolist())
    print(f"Attention-pool nested CV: {len(all_ids)} cases, max_slices={store.max_slices}, "
          f"attn_dim={args.attn_dim}, dropout={args.dropout}")

    oof = nested_oof(store, all_ids, cfg, args, device, seed)
    _, observed_c = score(oof, store, all_ids)
    print(f"Observed aggregated C-index: {observed_c:.3f}")

    rng = np.random.default_rng(seed)
    original = store.labels
    null_c = []
    try:
        for i in range(args.n_permutations):
            perm = rng.permutation(len(all_ids))
            shuffled = original.copy()
            shuffled.loc[list(all_ids), ["event", "time"]] = original.loc[list(all_ids), ["event", "time"]].to_numpy()[perm]
            store.labels = shuffled
            oof_n = nested_oof(store, all_ids, cfg, args, device, seed + 1000 + i)
            _, c_n = score(oof_n, store, all_ids)
            null_c.append(c_n)
            if (i + 1) % 20 == 0:
                print(f"  permutation {i + 1}/{args.n_permutations}")
    finally:
        store.labels = original

    p_value = float(np.mean(np.array(null_c) >= observed_c))
    print(f"null mean={np.mean(null_c):.3f} std={np.std(null_c):.3f} empirical p={p_value:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(null_c, bins=20, color="lightgrey", edgecolor="black")
    ax.axvline(observed_c, color="red", linewidth=2, label=f"observed = {observed_c:.3f}")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Aggregated nested-CV concordance index")
    ax.set_ylabel("Count (label-permuted runs)")
    ax.set_title(f"Attention-pool nested-CV null (n={len(null_c)})\nempirical p = {p_value:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "nested_null_attention_image.png", dpi=150)
    plt.close(fig)

    results = {
        "event_type": event_type, "mode": "image_attention_pool",
        "attn_dim": args.attn_dim, "dropout": args.dropout,
        "n_repeats": args.n_repeats, "n_folds": args.n_folds, "n_permutations": args.n_permutations,
        "observed_cindex": observed_c, "null_mean": float(np.mean(null_c)),
        "null_std": float(np.std(null_c)), "empirical_p_value": p_value,
    }
    with open(args.out_dir / f"nested_cv_attention_{event_type}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out_dir}/nested_cv_attention_{event_type}.json")


if __name__ == "__main__":
    main()
