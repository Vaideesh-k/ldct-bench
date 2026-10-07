"""Small helpers shared by the scripts: choosing the device and timing on it."""
import platform

import torch


def pick_device(preferred="cuda"):
    """cuda if available, else mps (Mac GPU), else cpu with a warning.
    preferred="cpu" forces the CPU."""
    if preferred != "cpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    print("WARNING: running on the CPU. This is fine for tests but far too slow for real training.")
    return torch.device("cpu")


def synchronize(device):
    """Wait until the GPU has really finished, so timings are honest
    (GPU work runs in the background otherwise)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def device_name(device):
    """Readable name for reports, e.g. 'NVIDIA GeForce RTX 4060 Laptop GPU'."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"
