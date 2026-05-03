import os
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_device(cfg_device):
    if cfg_device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
            return "cpu"
        except Exception:
            return "cpu"
    return cfg_device
