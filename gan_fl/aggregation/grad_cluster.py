# gan_fl/aggregation/grad_cluster.py

from typing import Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans


def _flatten_state_diff(
    global_state: Dict[str, torch.Tensor],
    client_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Return a 1D vector of (client_param - global_param) for all float tensors.
    """
    diffs = []
    for name, g_param in global_state.items():
        c_param = client_state[name]

        # Only use float tensors
        if not torch.is_floating_point(g_param):
            continue

        diffs.append((c_param - g_param).reshape(-1))

    if not diffs:
        device = next(iter(global_state.values())).device
        return torch.zeros(1, device=device)

    return torch.cat(diffs)


def _fedavg_states(
    states: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    Weighted FedAvg over a list of state_dicts.
    """
    assert len(states) > 0
    assert len(states) == len(weights)

    total_weight = float(sum(weights))
    assert total_weight > 0.0

    avg_state: Dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        acc = torch.zeros_like(states[0][key])
        for s, w in zip(states, weights):
            acc = acc + s[key] * (w / total_weight)
        avg_state[key] = acc

    return avg_state


def gradient_cluster_aggregate(
    global_model,
    client_models: Dict[int, torch.nn.Module],
    accepted_ids: List[int],
    n_clusters: int = 2,
    random_state: int = 0,
):
    """
    Aggregation with gradient-based client clustering.

    Steps:
      1) Build update vectors (client - global) for each client.
      2) Run K-means on these vectors.
      3) Do FedAvg inside each cluster.
      4) Do another FedAvg over cluster models.
    """
    if not accepted_ids:
        return global_model

    global_state = global_model.state_dict()
    client_states = {
        cid: client_models[cid].state_dict() for cid in accepted_ids
    }

    # Build feature matrix X: [num_clients, dim]
    vectors = []
    for cid in accepted_ids:
        v = _flatten_state_diff(global_state, client_states[cid])
        vectors.append(v.detach().cpu().numpy())
    X = np.stack(vectors, axis=0)

    # Number of clusters cannot be larger than number of clients
    k = min(n_clusters, len(accepted_ids))

    # If only one cluster, this is just FedAvg
    if k <= 1:
        states = [client_states[cid] for cid in accepted_ids]
        weights = [1.0 for _ in accepted_ids]  # equal weights
        new_state = _fedavg_states(states, weights)
        global_model.load_state_dict(new_state)
        return global_model

    # K-means in update space
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X)

    # FedAvg inside each cluster
    cluster_states: List[Dict[str, torch.Tensor]] = []
    cluster_weights: List[float] = []

    for cluster_id in range(k):
        cids = [cid for cid, lab in zip(accepted_ids, labels) if lab == cluster_id]
        if not cids:
            continue

        states = [client_states[cid] for cid in cids]
        weights = [1.0 for _ in cids]  # equal weights
        cluster_states.append(_fedavg_states(states, weights))
        cluster_weights.append(len(cids))

    # If there is only one non-empty cluster, use it directly
    if len(cluster_states) == 1:
        new_state = cluster_states[0]
    else:
        # FedAvg over cluster models
        new_state = _fedavg_states(cluster_states, cluster_weights)

    global_model.load_state_dict(new_state)
    return global_model
