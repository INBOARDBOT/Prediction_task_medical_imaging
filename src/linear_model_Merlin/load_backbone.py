

import torch


def select_device(preference: str = "auto") -> torch.device:
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


def load_merlin_backbone(device: torch.device) -> torch.nn.Module:

    from merlin import Merlin

    model = Merlin(ImageEmbedding=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
