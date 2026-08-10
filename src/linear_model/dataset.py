"""Combine cached DINOv3 image embeddings + pruned radiomics features into
survival datasets, selectable by input_mode ("radiomics", "image", "both").
Reads only from the caches written by caching_features.py -- never touches
the GPU or raw radiomics json at train time.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_MODES = ("radiomics", "image", "both")


class FeatureStore:
    """Loads the image + radiomics caches once and builds (X, event, time)
    tensors for an arbitrary list of case_ids and a chosen input_mode.
    """

    def __init__(self, cfg: dict, event_type: str):
        self.cfg = cfg
        self.event_type = event_type

        cache_dir = PROJECT_ROOT / cfg["cache"]["dir"]
        backbone = cfg["image"]["backbone"]
        image_npz = np.load(cache_dir / cfg["cache"]["image_features_file"].format(backbone=backbone), allow_pickle=True)
        self.image_features = pd.DataFrame(
            image_npz["features"], index=image_npz["case_ids"]
        )

        # Standardize image features using train-split statistics only (same
        # discipline as the radiomics cache, which is already standardized
        # with train-fit mean/std from the pruning step) so neither input
        # type dominates the linear head purely by feature scale.
        splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
        train_ids = pd.read_csv(splits_dir / cfg["data"]["train_split"])["case_id"].tolist()
        train_image = self.image_features.loc[train_ids]
        img_mean = train_image.mean(axis=0)
        img_std = train_image.std(axis=0).replace(0, 1.0)
        self.image_features = (self.image_features - img_mean) / img_std

        radiomics_csv = cache_dir / cfg["cache"]["radiomics_features_file"].format(event_type=event_type)
        self.radiomics_features = pd.read_csv(radiomics_csv, index_col="case_id")

        self.labels = self._load_labels(cfg)

    def _load_labels(self, cfg: dict) -> pd.DataFrame:
        splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
        all_rows = pd.concat(
            [pd.read_csv(splits_dir / cfg["data"][k]) for k in ("train_split", "valid_split", "test_split")]
        ).drop_duplicates("case_id").set_index("case_id")

        event_type = self.event_type
        time_column = cfg["data"]["event_time_columns"][event_type]
        rows = {}
        for case_id, row in all_rows.iterrows():
            with open(PROJECT_ROOT / row["label_path"]) as f:
                label = json.load(f)
            rows[case_id] = {"event": label[event_type], "time": label[time_column]}
        return pd.DataFrame(rows).T

    def input_dim(self, input_mode: str) -> int:
        if input_mode == "radiomics":
            return self.radiomics_features.shape[1]
        if input_mode == "image":
            return self.image_features.shape[1]
        if input_mode == "both":
            return self.radiomics_features.shape[1] + self.image_features.shape[1]
        raise ValueError(f"Unknown input_mode '{input_mode}', expected one of {INPUT_MODES}")

    def build(self, case_ids: list[str], input_mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_mode == "radiomics":
            X = self.radiomics_features.loc[case_ids].to_numpy(dtype=np.float32)
        elif input_mode == "image":
            X = self.image_features.loc[case_ids].to_numpy(dtype=np.float32)
        elif input_mode == "both":
            rad = self.radiomics_features.loc[case_ids].to_numpy(dtype=np.float32)
            img = self.image_features.loc[case_ids].to_numpy(dtype=np.float32)
            X = np.concatenate([rad, img], axis=1)
        else:
            raise ValueError(f"Unknown input_mode '{input_mode}', expected one of {INPUT_MODES}")

        labels = self.labels.loc[case_ids]
        event = torch.tensor(labels["event"].to_numpy(dtype=np.float32))
        time = torch.tensor(labels["time"].to_numpy(dtype=np.float32))
        return torch.tensor(X), event, time


class SurvivalDataset(Dataset):
    def __init__(self, feature_store: FeatureStore, case_ids: list[str], input_mode: str):
        self.case_ids = list(case_ids)
        self.X, self.event, self.time = feature_store.build(self.case_ids, input_mode)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        return self.X[idx], self.event[idx], self.time[idx]
