import os
import tempfile

import torch


def atomic_save_torch(state_dict, path: str):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_model_", suffix=".pth")
    os.close(fd)
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def load_torch(path: str, device):
    return torch.load(path, map_location=device)
