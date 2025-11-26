import copy
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .client_config import ClientConfig


class FedClient:
    def __init__(self, client_id: int, dataset, indices,
                 config: ClientConfig, device):
        self.client_id = client_id
        self.config = config
        self.device = device

        self.dataset = Subset(dataset, indices.tolist())
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    def is_malicious(self) -> bool:
        return self.config.attack_type != "none"

    # ---------- attack helpers ----------

    def _poison_batch_label_flip(self, x, y) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, (y + 1) % 10

    def _poison_batch_backdoor(self, x, y):
        mask = (y == self.config.backdoor_source)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return x, y

        num_poison = max(1, int(self.config.backdoor_fraction * idx.numel()))
        poison_idx = idx[torch.randperm(idx.numel())[:num_poison]]

        x_poison = x.clone()
        v = self.config.backdoor_trigger_value
        x_poison[poison_idx, :, -3:, -3:] = v
        y_poison = y.clone()
        y_poison[poison_idx] = self.config.backdoor_target
        return x_poison, y_poison

    # ---------- Algorithm 2: local training ----------

    def local_train(self, global_model):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.config.lr,
                              momentum=0.9)

        # 数据层面攻击：LF / BK
        if self.config.attack_type in ["none", "lf", "bk"]:
            for _ in range(self.config.local_epochs):
                for x, y in self.loader:
                    x, y = x.to(self.device), y.to(self.device)

                    if self.config.attack_type == "lf":
                        x, y = self._poison_batch_label_flip(x, y)
                    elif self.config.attack_type == "bk":
                        x, y = self._poison_batch_backdoor(x, y)

                    opt.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    opt.step()
        else:
            # 参数层面攻击：RN / SF，先正常训练，再改参数
            for _ in range(self.config.local_epochs):
                for x, y in self.loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    opt.step()

            if self.config.attack_type == "rn":
                with torch.no_grad():
                    for p in model.parameters():
                        noise = torch.randn_like(p)
                        p.copy_(noise)
            elif self.config.attack_type == "sf":
                gamma = 10.0
                with torch.no_grad():
                    for p_local, p_global in zip(model.parameters(),
                                                 global_model.parameters()):
                        delta = p_local.data - p_global.data
                        p_local.data = p_global.data - gamma * delta

        return model.cpu()