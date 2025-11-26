import copy
import torch
import torch.nn.functional as F


def train_generator_cgan(
    generator,
    global_model,
    num_classes: int,
    noise_dim: int,
    device,
    steps: int = 200,
    batch_size: int = 256,
):
    generator.to(device)
    disc = copy.deepcopy(global_model).to(device)
    disc.eval()
    for p in disc.parameters():
        p.requires_grad = False

    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4,
                             betas=(0.5, 0.999))

    for _ in range(steps):
        z = torch.randn(batch_size, noise_dim, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        x_fake = generator(z, y)
        logits = disc(x_fake)
        loss_g = F.cross_entropy(logits, y)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    generator.cpu()
    return generator


def generate_synthetic_dataset(
    generator,
    num_classes: int,
    q_per_class: int,
    noise_dim: int,
    device,
):
    generator.to(device)
    generator.eval()
    xs, ys = [], []
    with torch.no_grad():
        for y_cls in range(num_classes):
            z = torch.randn(q_per_class, noise_dim, device=device)
            y = torch.full((q_per_class,), y_cls, dtype=torch.long,
                           device=device)
            x_fake = generator(z, y)
            xs.append(x_fake.cpu())
            ys.append(y.cpu())
    generator.cpu()
    X_syn = torch.cat(xs, dim=0)
    y_syn = torch.cat(ys, dim=0)
    return X_syn, y_syn
