# Robust Federated Learning with GAN-based Defense (MNIST demo)

This repo provides a PyTorch reproduction of the main pipeline in:

> Zafar et al. *Robust Federated Learning Against Poisoning Attacks:  
> A GAN-Based Defense Framework* (IEEE TDSC, 2025).

## Key components

- LeNet-5 classifier for MNIST
- cGAN where the discriminator is the current global model
- Synthetic validation set `D_syn` for authenticating client updates
- Four poisoning attacks: Random Noise, Sign Flipping, Label Flipping, Backdoor
- Threshold-based and clustering-based filtering (Algorithm 3)
- Full FL loop (Algorithm 1)

This is a research-oriented reference implementation, not an optimized production system.

## Environment

- Python 3.13 (repo uses a local `venv/`)
- PyTorch + torchvision, scikit-learn (see `requirements.txt`)

Create and activate the venv, then install deps:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run

Download MNIST automatically and run a small demo:

```bash
source venv/bin/activate
python main.py --rounds 3 --clients-per-round 10 --num-clients 100
```

To speed up debugging, reduce rounds/clients and generator steps:

```bash
python main.py --rounds 1 --clients-per-round 2 --num-clients 4 \
  --local-epochs 1 --g-steps 1 --g-batch-size 8 --q-per-class 4 \
  --batch-size 8 --data-dir ./data
```

## CLI options (`main.py`)

- Data/seed: `--data-dir` (default `./data`), `--seed` (42)
- Federated setup: `--num-clients` (100), `--clients-per-round` (10), `--rounds` (20)
- Local training: `--local-epochs` (10), `--batch-size` (128), `--lr` (0.01), `--dirichlet-alpha` (1.0 = IID, smaller → more non-IID)
- Attacks: `--malicious-frac` (0.2), `--attack-type` {none,rn,sf,lf,bk}, backdoor knobs `--backdoor-source/target/trigger-value/fraction`
- cGAN: `--noise-dim` (100), `--g-steps` (200), `--g-batch-size` (256), `--q-per-class` (200 per class in `D_syn`)
- Filtering: `--filter-method` {fixed,adaptive,cluster}, `--fixed-tau` (used when `fixed`)

## Code map

- `main.py` — parses args and launches training
- `gan_fl/trainer/federated_trainer.py` — end-to-end FL loop
- `gan_fl/data/` — MNIST loader and Dirichlet partition
- `gan_fl/models/` — LeNet-5 classifier, MNIST conditional generator
- `gan_fl/attacks/` — client config and poisoning logic (RN/SF/LF/BK)
- `gan_fl/gan/trainer.py` — generator training and synthetic data creation
- `gan_fl/filter/filter_updates.py` — client filtering via synthetic eval or clustering
- `gan_fl/aggregation/fedavg.py` — FedAvg aggregation of accepted clients
- `gan_fl/utils/common.py` — seeding, device selection, eval helper

## Notes

- Default device auto-selects CUDA if available; otherwise CPU.
- Filtering may reject all clients when synthetic data is weak; tune `g-steps`, `q-per-class`, or filter method/thresholds accordingly.
