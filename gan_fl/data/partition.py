import numpy as np
from typing import List


def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float
) -> List[np.ndarray]:
    """
    Partition indices into num_clients shards using Dirichlet(alpha).
    """
    num_classes = len(np.unique(targets))
    idx_by_class = [np.where(targets == c)[0] for c in range(num_classes)]
    for idx in idx_by_class:
        np.random.shuffle(idx)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return [np.array(idx, dtype=int) for idx in client_indices]