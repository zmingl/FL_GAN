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
python main.py --rounds 4
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

## execution process
1. Randomly select a portion of clients
2. Local training on the client side (Algorithm 2)
3. Train cGAN using the global model
4. Generate D_syn
5. Client models are evaluated on D_syn
6. FilterUpdates (Algorithm 3)
7. Accepted clients participate in FedAvg
8. Update the global model

## running demo
```bash
python main.py \
    --rounds 5 \
    --num-clients 20 \
    --clients-per-round 5 \
    --local-epochs 1 \
    --g-steps 100 \
    --g-batch-size 128 \
    --q-per-class 50
```
## demo result
```
Using device: cpu
100.0%
100.0%
100.0%
100.0%
Total clients: 20, avg samples/client: 3000.0
Malicious clients: 4 / 20

=== Round 1/5 ===
Sampled clients: [7, 4, 3, 2, 13]
Synthetic dataset size: 500
Client synthetic accuracies:
  Client   2 [B] : 0.4000
  Client   3 [M] : 0.0560
  Client   4 [B] : 0.6000
  Client   7 [M] : 0.0260
  Client  13 [B] : 0.4460
Accepted clients: [4, 2, 13]
Global test accuracy: 0.1191

=== Round 2/5 ===
Sampled clients: [1, 0, 2, 6, 7]
Synthetic dataset size: 500
Client synthetic accuracies:
  Client   0 [M] : 0.0920
  Client   1 [B] : 0.5000
  Client   2 [B] : 0.5040
  Client   6 [B] : 0.6000
  Client   7 [M] : 0.0900
Accepted clients: [1, 2, 6]
Global test accuracy: 0.2933

=== Round 3/5 ===
Sampled clients: [16, 0, 17, 6, 13]
Synthetic dataset size: 500
Client synthetic accuracies:
  Client   0 [M] : 0.0080
  Client   6 [B] : 0.9000
  Client  13 [B] : 0.8460
  Client  16 [B] : 0.7600
  Client  17 [B] : 0.6900
Accepted clients: [16, 17, 6, 13]
Global test accuracy: 0.4804

=== Round 4/5 ===
Sampled clients: [7, 14, 8, 0, 5]
Synthetic dataset size: 500
Client synthetic accuracies:
  Client   0 [M] : 0.0060
  Client   5 [B] : 0.9000
  Client   7 [M] : 0.0780
  Client   8 [M] : 0.1180
  Client  14 [B] : 0.8000
Accepted clients: [14, 5]
Global test accuracy: 0.5613

=== Round 5/5 ===
Sampled clients: [13, 10, 8, 4, 6]
Synthetic dataset size: 500
Client synthetic accuracies:
  Client   4 [B] : 0.9000
  Client   6 [B] : 0.9020
  Client   8 [M] : 0.0160
  Client  10 [B] : 0.8000
  Client  13 [B] : 1.0000
Accepted clients: [13, 10, 4, 6]
Global test accuracy: 0.7455

Training finished.
Final global test accuracy: 0.7455
```