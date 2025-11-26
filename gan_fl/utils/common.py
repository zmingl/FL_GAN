import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_global_model(model, test_loader: DataLoader, device) -> float:
    m = type(model)()
    m.load_state_dict(model.state_dict())
    m.to(device)
    m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = m(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total