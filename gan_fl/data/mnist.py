from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(data_dir: str, batch_size_test: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST(root=data_dir, train=True,
                              download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False,
                             download=True, transform=transform)

    test_loader = DataLoader(test_ds, batch_size=batch_size_test,
                             shuffle=False)
    return train_ds, test_ds, test_loader