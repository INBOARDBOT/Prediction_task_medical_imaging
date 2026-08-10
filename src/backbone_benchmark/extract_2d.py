"""Backbone benchmark (2D models): embed each case's tumor-cropped max-area
axial slice -- the SAME input DINOv3 gets -- through a chosen 2D foundation
model, and cache one embedding per case so the existing linear-head
nested-CV can score it head-to-head against DINOv3's 0.643.

Each backbone uses its OWN HuggingFace image processor (native resize +
normalization), so every model sees the input distribution it was
pretrained on. Only the crop (tumor bbox + margin, percentile-normalized,
replicated to 3 channels) is shared, identical to src/linear_model/
caching_features.py's best_slice_roi.

Embedding per family:
  rad_dino     : DINOv2 ViT CLS token (pooler_output)     [medical, chest]
  clip         : CLIP image features (get_image_features)  [natural]
  pubmedclip   : PubMedCLIP image features                 [medical CLIP -- MedCLIP slot]
  sam1 / sam2  : image-encoder feature map, mean-pooled    [segmentation]

Output (under cache.dir): image_features_<name>.npz (case_ids, features).
Run in the `dinov3` env (transformers): conda run -n dinov3 python extract_2d.py --backbone rad_dino
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _drop_socks_proxy():
    for k in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(k, "").startswith("socks"):
            os.environ.pop(k, None)


def load_config(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def collect_case_ids(cfg: dict) -> list[str]:
    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    ids = set()
    for key in ("train_split", "valid_split", "test_split"):
        ids.update(pd.read_csv(splits_dir / cfg["data"][key])["case_id"].tolist())
    return sorted(ids)


def best_slice_crop(image_path: Path, mask_path: Path, margin_px: int) -> np.ndarray:
    """Identical to DINOv3's best_slice_roi: max-tumor-area slice, cropped to
    tumor bbox + margin, per-slice 1-99 percentile -> [0,1]."""
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.uint8)
    areas = mask.sum(axis=(1, 2))
    if areas.max() == 0:
        raise ValueError(f"empty mask: {mask_path}")
    z = int(np.argmax(areas))
    sl, ms = img[z], mask[z]
    rows, cols = np.any(ms, axis=1), np.any(ms, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = sl.shape
    rmin, rmax = max(0, rmin - margin_px), min(H - 1, rmax + margin_px)
    cmin, cmax = max(0, cmin - margin_px), min(W - 1, cmax + margin_px)
    crop = sl[rmin:rmax + 1, cmin:cmax + 1]
    p1, p99 = np.percentile(sl, [1, 99])
    return np.clip((crop - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)


def crop_to_pil(crop: np.ndarray):
    from PIL import Image
    rgb = (np.stack([crop] * 3, axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


# ----- backbone registry: name -> loader returning (processor, embed_fn) -----

def _load_rad_dino(device):
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino").to(device).eval()

    def embed(pixel_values):
        out = model(pixel_values=pixel_values)
        return out.pooler_output  # CLS token
    return proc, embed


def _load_clip_like(repo, device):
    # transformers 5.x get_image_features returns an output object, not a
    # tensor -- compute the projected image embedding explicitly (the true
    # CLIP image feature) so this is robust across versions.
    from transformers import CLIPModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(repo)
    model = CLIPModel.from_pretrained(repo).to(device).eval()

    def embed(pixel_values):
        vout = model.vision_model(pixel_values=pixel_values)
        return model.visual_projection(vout.pooler_output)
    return proc.image_processor, embed


def _load_sam1(device):
    from transformers import SamModel, SamProcessor
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    def embed(pixel_values):
        emb = model.get_image_embeddings(pixel_values)  # (B,256,64,64)
        return emb.mean(dim=(2, 3))
    return proc.image_processor, embed


def _load_sam2(device):
    # transformers >= 4.x SAM2 support; fall back across class/repo names.
    from transformers import AutoProcessor
    repo = "facebook/sam2-hiera-large"
    proc = AutoProcessor.from_pretrained(repo)
    try:
        from transformers import Sam2Model
        model = Sam2Model.from_pretrained(repo).to(device).eval()
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(repo).to(device).eval()

    def embed(pixel_values):
        vision = model.get_image_embeddings(pixel_values) if hasattr(model, "get_image_embeddings") \
            else model.vision_encoder(pixel_values).last_hidden_state
        v = vision[0] if isinstance(vision, (tuple, list)) else vision
        if hasattr(v, "last_hidden_state"):
            v = v.last_hidden_state
        while v.dim() > 2:
            v = v.mean(dim=-1) if v.dim() > 3 else v.mean(dim=1)
        return v
    return getattr(proc, "image_processor", proc), embed


REGISTRY = {
    "rad_dino": _load_rad_dino,
    "clip": lambda dev: _load_clip_like("openai/clip-vit-base-patch32", dev),
    "pubmedclip": lambda dev: _load_clip_like("flaviagiammarino/pubmed-clip-vit-base-patch32", dev),
    "sam1": _load_sam1,
    "sam2": _load_sam2,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=list(REGISTRY))
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases; >0 = smoke test on first N")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    _drop_socks_proxy()
    cfg = load_config(args.config)
    cache_dir = PROJECT_ROOT / cfg["cache"]["dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"image_features_{args.backbone}.npz"
    if out_path.exists() and not args.force and not args.limit:
        print(f"[skip] {out_path.name} exists (use --force)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading backbone '{args.backbone}' on {device} ...")
    proc, embed = REGISTRY[args.backbone](device)

    case_ids = collect_case_ids(cfg)
    if args.limit:
        case_ids = case_ids[:args.limit]
    splits_dir = PROJECT_ROOT / cfg["data"]["splits_dir"]
    rows = pd.concat([pd.read_csv(splits_dir / cfg["data"][k]) for k in ("train_split", "valid_split", "test_split")]) \
        .drop_duplicates("case_id").set_index("case_id")
    margin = cfg["image"]["roi_margin_px"]

    feats, ids, batch_pil, batch_ids = [], [], [], []

    def flush():
        if not batch_pil:
            return
        px = proc(images=batch_pil, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            e = embed(px)
        feats.append(e.float().cpu().numpy())
        ids.extend(batch_ids)
        batch_pil.clear()
        batch_ids.clear()

    for i, cid in enumerate(case_ids):
        r = rows.loc[cid]
        crop = best_slice_crop(PROJECT_ROOT / r["image_path"], PROJECT_ROOT / r["mask_path"], margin)
        batch_pil.append(crop_to_pil(crop))
        batch_ids.append(cid)
        if len(batch_pil) >= args.batch_size:
            flush()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(case_ids)}")
    flush()

    features = np.concatenate(feats, axis=0)
    if args.limit:
        print(f"[smoke] {args.backbone}: {features.shape} (not saved)")
        return
    np.savez(out_path, case_ids=np.array(ids), features=features)
    print(f"Wrote {out_path}: {features.shape[0]} cases x {features.shape[1]} dims")


if __name__ == "__main__":
    main()
