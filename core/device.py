import torch

def get_device() -> torch.device:
    """
    Determine the best available hardware accelerator for PyTorch inference.
    Supports NVIDIA CUDA, Apple Silicon MPS, and CPU fallback.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")