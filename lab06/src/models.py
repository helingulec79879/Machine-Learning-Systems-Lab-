"""Models for Lab 6.

The best model from Lab 3 is the CNN (best accuracy per parameter, see lab03/report.md §6).
It is copied here unchanged so that Lab 6 runs on its own:
Conv(1,16,3,p1) -> ReLU -> MaxPool2 -> Conv(16,32,3,p1) -> ReLU -> MaxPool2 -> Linear(1568,10), 20,490 parameters.
"""
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, num_classes))

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784 * in_channels, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.net(x)


MODELS = {"cnn": CNN, "mlp": MLP}


def build_model(model_cfg, in_channels):
    name = model_cfg["name"].lower()
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name](in_channels=in_channels, num_classes=model_cfg.get("num_classes", 10))


def model_size(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    return n_params, n_bytes / 2**20
