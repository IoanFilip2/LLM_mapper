"""Device selection without changing global PyTorch behavior."""
def resolve_device(requested="auto"):
    import torch
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()}
    if requested == "auto" or requested is None:
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if requested not in available:
        raise ValueError("Device must be auto, cpu, mps, or cuda")
    if not available[requested]:
        raise RuntimeError(f"Requested {requested} device is unavailable")
    return requested


def synchronize(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
