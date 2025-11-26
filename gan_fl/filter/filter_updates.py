import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans


def eval_model_on_syn(model, X_syn: torch.Tensor, y_syn: torch.Tensor,
                      device) -> float:
    m = copy.deepcopy(model).to(device)
    m.eval()
    with torch.no_grad():
        logits = m(X_syn.to(device))
        preds = logits.argmax(dim=1).cpu()
    acc = (preds == y_syn).float().mean().item()
    return acc


def filter_updates(
    models: Dict[int, torch.nn.Module],
    X_syn: torch.Tensor,
    y_syn: torch.Tensor,
    method: str = "adaptive",
    fixed_tau: float = 0.0,
    device=None,
) -> Tuple[List[int], Dict[int, float]]:
    metrics: Dict[int, float] = {}
    for cid, m in models.items():
        metrics[cid] = eval_model_on_syn(m, X_syn, y_syn, device)

    ids = list(models.keys())
    values = np.array([metrics[i] for i in ids], dtype=np.float32)

    accepted: List[int] = []

    if method == "fixed":
        tau = fixed_tau
        for cid in ids:
            if metrics[cid] > tau:
                accepted.append(cid)

    elif method == "adaptive":
        tau = float(values.mean())
        for cid in ids:
            if metrics[cid] > tau:
                accepted.append(cid)

    elif method == "cluster":
        scores = values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scores)
        counts = np.bincount(labels)
        benign_label = int(np.argmax(counts))
        for cid, lab in zip(ids, labels):
            if lab == benign_label:
                accepted.append(cid)
    else:
        raise ValueError(f"Unknown filter method: {method}")

    return accepted, metrics