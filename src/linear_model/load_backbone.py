"""Load the local DINOv3 backbone and pick a GPU to run it on."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINOV3_REPO = PROJECT_ROOT / "dinov3"
if str(DINOV3_REPO) not in sys.path:
    sys.path.insert(0, str(DINOV3_REPO))


def select_device(preference: str = "auto") -> torch.device:
    """preference: "auto" picks the CUDA device with the most free memory
    (via nvitop); "cpu"/"cuda"/"cuda:N" are used as-is.
    """
    if preference != "auto":
        return torch.device(preference)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        from nvitop import Device

        devices = Device.all()
        best = max(devices, key=lambda d: d.memory_free())
        return torch.device(f"cuda:{best.index}")
    except Exception:
        return torch.device("cuda:0")


def load_dinov3_backbone(backbone: str, weights_path: Path, device: torch.device) -> torch.nn.Module:
    from dinov3.hub.backbones import (
        dinov3_vitb16,
        dinov3_vitl16,
        dinov3_vitl16plus,
        dinov3_vits16,
        dinov3_vits16plus,
    )

    builders = {
        "dinov3_vits16": dinov3_vits16,
        "dinov3_vits16plus": dinov3_vits16plus,
        "dinov3_vitb16": dinov3_vitb16,
        "dinov3_vitl16": dinov3_vitl16,
        "dinov3_vitl16plus": dinov3_vitl16plus,
    }
    if backbone not in builders:
        raise ValueError(f"Unknown backbone '{backbone}'. Available: {list(builders)}")

    model = builders[backbone](pretrained=True, weights=str(weights_path))
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
