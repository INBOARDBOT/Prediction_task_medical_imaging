"""CT-FM (project-lighter/ct_fm_feature_extractor, a SegResNet encoder
self-supervised on ~148k CT volumes) backbone for the benchmark.

Feeds each case's tumor-ROI 3D crop (same localization idea as the Merlin
ROI run) through CT-FM's native preprocessing (SPL orientation, HU
-1024..2048 -> [0,1]), then global-average-pools the encoder's deepest
512-channel feature map into one embedding per case.

Output: output/cache/image_features_ctfm_roi.npz. Run in the MERLIN env
(has monai + lighter_zoo): conda run -n MERLIN python extract_ctfm.py
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _drop_socks():
    for k in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(k, "").startswith("socks"):
            os.environ.pop(k, None)


def build_transforms(margin: int, size):
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        ScaleIntensityRanged, CropForegroundd, Resized, ToTensord,
    )
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="SPL"),  # CT-FM native
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2048, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="label", margin=margin),
        Resized(keys=["image"], spatial_size=size, mode="trilinear", align_corners=False),
        ToTensord(keys=["image"]),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    ap.add_argument("--size", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _drop_socks()
    import torch
    from lighter_zoo import SegResEncoder

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CT-FM on {device} ...")
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(device).eval()
    tf = build_transforms(args.margin, tuple(args.size))

    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    ids = sorted(set().union(*[set(pd.read_csv(splits_dir / cfg["data"][k])["case_id"]) for k in ("train_split", "valid_split", "test_split")]))
    if args.limit:
        ids = ids[:args.limit]
    rows = pd.concat([pd.read_csv(splits_dir / cfg["data"][k]) for k in ("train_split", "valid_split", "test_split")]).drop_duplicates("case_id").set_index("case_id")

    feats, out_ids, batch, bids = [], [], [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            out = model(x)
            deep = out[-1] if isinstance(out, (list, tuple)) else out  # (B,512,d,d,d)
            emb = deep.mean(dim=(2, 3, 4))
        feats.append(emb.float().cpu().numpy())
        out_ids.extend(bids)
        batch.clear()
        bids.clear()

    for i, cid in enumerate(ids):
        r = rows.loc[cid]
        vol = tf({"image": str(PROJECT_ROOT / r["image_path"]), "label": str(PROJECT_ROOT / r["mask_path"])})["image"]
        batch.append(vol)
        bids.append(cid)
        if len(batch) >= args.batch_size:
            flush()
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ids)}")
    flush()

    features = np.concatenate(feats, axis=0)
    if args.limit:
        print(f"[smoke] ctfm: {features.shape} (not saved)")
        return
    out_path = PROJECT_ROOT / cfg["cache"]["dir"] / "image_features_ctfm_roi.npz"
    np.savez(out_path, case_ids=np.array(out_ids), features=features)
    print(f"Wrote {out_path}: {features.shape[0]} cases x {features.shape[1]} dims")


if __name__ == "__main__":
    main()
