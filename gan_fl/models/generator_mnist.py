import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorMNIST(nn.Module):
    def __init__(self, noise_dim: int = 100, num_classes: int = 10):
        super().__init__()
        self.noise_dim = noise_dim
        embed_dim = 50

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.fc1 = nn.Linear(noise_dim + embed_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1 * 28 * 28)

    def forward(self, z, y):
        y_emb = self.label_emb(y)
        x = torch.cat([z, y_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = x.view(-1, 1, 28, 28)
        return x