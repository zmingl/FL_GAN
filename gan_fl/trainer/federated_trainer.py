import random
import numpy as np

from gan_fl.utils import set_seed, get_device, eval_global_model
from gan_fl.data import load_mnist, dirichlet_partition
from gan_fl.models import LeNet5, GeneratorMNIST
from gan_fl.attacks import ClientConfig, FedClient
from gan_fl.gan import train_generator_cgan, generate_synthetic_dataset
from gan_fl.filter import filter_updates
from gan_fl.aggregation import fedavg_aggregate
from gan_fl.aggregation.grad_cluster import gradient_cluster_aggregate




def run_federated_training(args):
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    # 1) dataset
    train_ds, test_ds, test_loader = load_mnist(args.data_dir)
    num_classes = 10

    # 2) partition data
    N = args.num_clients
    train_targets = np.array(train_ds.targets)
    client_idx_list = dirichlet_partition(
        train_targets, num_clients=N, alpha=args.dirichlet_alpha
    )
    print(f"Total clients: {N}, avg samples/client: {len(train_ds) / N:.1f}")

    # 3) mark malicious
    num_malicious = int(args.malicious_frac * N)
    malicious_ids = set(random.sample(range(N), num_malicious))
    print(f"Malicious clients: {len(malicious_ids)} / {N}")

    clients = {}
    for cid in range(N):
        attack_type = args.attack_type if cid in malicious_ids else "none"
        cfg = ClientConfig(
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            attack_type=attack_type,
            backdoor_target=args.backdoor_target,
            backdoor_source=args.backdoor_source,
            backdoor_trigger_value=args.backdoor_trigger_value,
            backdoor_fraction=args.backdoor_fraction,
        )
        clients[cid] = FedClient(
            client_id=cid,
            dataset=train_ds,
            indices=client_idx_list[cid],
            config=cfg,
            device=device,
        )

    # 4) init global model & generator
    global_model = LeNet5(num_classes=num_classes)
    generator = GeneratorMNIST(noise_dim=args.noise_dim,
                               num_classes=num_classes)
    noise_dim = args.noise_dim

    # 5) FL rounds (Algorithm 1)
    for t in range(1, args.rounds + 1):
        print(f"\n=== Round {t}/{args.rounds} ===")

        sampled_ids = random.sample(range(N), args.clients_per_round)
        print("Sampled clients:", sampled_ids)

        # local training
        client_models = {}
        for cid in sampled_ids:
            m_local = clients[cid].local_train(global_model)
            client_models[cid] = m_local

        # train cGAN
        generator = train_generator_cgan(
            generator=generator,
            global_model=global_model,
            num_classes=num_classes,
            noise_dim=noise_dim,
            device=device,
            steps=args.g_steps,
            batch_size=args.g_batch_size,
        )

        # generate D_syn
        X_syn, y_syn = generate_synthetic_dataset(
            generator=generator,
            num_classes=num_classes,
            q_per_class=args.q_per_class,
            noise_dim=noise_dim,
            device=device,
        )
        print(f"Synthetic dataset size: {X_syn.size(0)}")

        # filter updates (Algorithm 3)
        accepted_ids, metrics = filter_updates(
            models=client_models,
            X_syn=X_syn,
            y_syn=y_syn,
            method=args.filter_method,
            fixed_tau=args.fixed_tau,
            device=device,
        )

        print("Client synthetic accuracies:")
        for cid in sorted(client_models.keys()):
            flag = "M" if clients[cid].is_malicious() else "B"
            print(f"  Client {cid:3d} [{flag}] : {metrics[cid]:.4f}")
        print("Accepted clients:", accepted_ids)

        # aggregate
        # aggregate
        if args.agg_method == "grad_cluster":
            # Gradient-based clustering + aggregation
            global_model = gradient_cluster_aggregate(
                global_model=global_model,
                client_models=client_models,
                accepted_ids=accepted_ids,
                n_clusters=2,
                random_state=getattr(args, "seed", 0),
            )
        else:
            # Standard FedAvg
            global_model = fedavg_aggregate(
                global_model=global_model,
                client_models=client_models,
                accepted_ids=accepted_ids,
            )


        # eval
        test_acc = eval_global_model(global_model, test_loader, device)
        print(f"Global test accuracy: {test_acc:.4f}")

    print("\nTraining finished.")
    final_acc = eval_global_model(global_model, test_loader, device)
    print(f"Final global test accuracy: {final_acc:.4f}")