import copy
from typing import Dict, List

import torch
import torch.nn as nn


def fedavg_aggregate(
    global_model: nn.Module,
    client_models: Dict[int, nn.Module],
    accepted_ids: List[int],
) -> nn.Module:
    if not accepted_ids:
        print("Warning: no accepted updates – keeping global model.")
        return global_model

    new_model = copy.deepcopy(global_model)
    with torch.no_grad():
        for p in new_model.parameters():
            p.zero_()

        for cid in accepted_ids:
            cm = client_models[cid]
            for p_new, p_c in zip(new_model.parameters(), cm.parameters()):
                p_new.add_(p_c.data / len(accepted_ids))

    return new_model
