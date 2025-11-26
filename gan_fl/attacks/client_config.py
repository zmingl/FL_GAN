from dataclasses import dataclass


@dataclass
class ClientConfig:
    local_epochs: int = 10
    batch_size: int = 128
    lr: float = 0.01
    attack_type: str = "none"   # none, rn, sf, lf, bk
    backdoor_target: int = 6
    backdoor_source: int = 0
    backdoor_trigger_value: float = 1.0
    backdoor_fraction: float = 0.1