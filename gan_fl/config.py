import argparse


def build_argparser():
    p = argparse.ArgumentParser(
        description="GAN-based defense for Federated Learning (MNIST demo)"
    )
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--seed", type=int, default=42)

    # FL setup
    p.add_argument("--num-clients", type=int, default=100)
    p.add_argument("--clients-per-round", type=int, default=10)
    p.add_argument("--rounds", type=int, default=20)

    p.add_argument("--local-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--dirichlet-alpha", type=float, default=1.0,
                   help="1.0 for IID, 0.5 for non-IID")

    # Attacks
    p.add_argument("--malicious-frac", type=float, default=0.2,
                   help="fraction of malicious clients (0.1–0.3)")
    p.add_argument(
        "--attack-type",
        type=str,
        default="rn",
        choices=["none", "rn", "sf", "lf", "bk"],
        help="rn=Random Noise, sf=Sign Flipping, lf=Label Flipping, bk=Backdoor",
    )
    p.add_argument("--backdoor-source", type=int, default=0)
    p.add_argument("--backdoor-target", type=int, default=6)
    p.add_argument("--backdoor-trigger-value", type=float, default=1.0)
    p.add_argument("--backdoor-fraction", type=float, default=0.1)

    # cGAN
    p.add_argument("--noise-dim", type=int, default=100)
    p.add_argument("--g-steps", type=int, default=200,
                   help="generator updates per round")
    p.add_argument("--g-batch-size", type=int, default=256)
    p.add_argument("--q-per-class", type=int, default=200,
                   help="samples per class in D_syn")

    # Filtering
    p.add_argument(
        "--filter-method",
        type=str,
        default="adaptive",
        choices=["fixed", "adaptive", "cluster"],
    )
    p.add_argument("--fixed-tau", type=float, default=0.7,
                   help="used when filter-method=fixed")
    
    #Gradclustering
    p.add_argument(
        "--agg-method",
        type=str,
        default="fedavg",
        choices=["fedavg", "grad_cluster"],
        help="Aggregation: 'fedavg' or 'grad_cluster' (gradient-based clustering).",
    )

    return p