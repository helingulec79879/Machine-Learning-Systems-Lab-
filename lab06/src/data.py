"""Datasets and DataLoaders (reproducible split, seeded workers, per-epoch shuffle seed)."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
import torchvision
import torchvision.transforms as T

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 3),
    "fashionmnist": ((0.2860,), (0.3530,), 1),
    "mnist": ((0.1307,), (0.3081,), 1),
}
DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "fashionmnist": torchvision.datasets.FashionMNIST,
    "mnist": torchvision.datasets.MNIST,
}


def in_channels(cfg):
    return STATS[cfg["data"]["dataset"].lower()][2]


def _transforms(name, augment):
    mean, std, _ = STATS[name]
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not augment:
        return T.Compose(norm), T.Compose(norm)
    if name == "cifar10":
        aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    else:
        aug = [T.RandomCrop(28, padding=2)]
    return T.Compose(aug + norm), T.Compose(norm)


def build_datasets(cfg):
    d = cfg["data"]
    name = d["dataset"].lower()
    ds_cls = DATASETS[name]
    t_train, t_eval = _transforms(name, d.get("augment", True))

    # Two views of the same training split: augmented for training, clean for validation.
    full_train = ds_cls(d["root"], train=True, download=True, transform=t_train)
    full_val = ds_cls(d["root"], train=True, download=True, transform=t_eval)
    test = ds_cls(d["root"], train=False, download=True, transform=t_eval)

    g = torch.Generator().manual_seed(cfg["seed"])          # split depends only on the seed
    perm = torch.randperm(len(full_train), generator=g).tolist()
    val_idx, train_idx = perm[: d["val_size"]], perm[d["val_size"]:]
    frac = d.get("subset_fraction", 1.0) or 1.0
    if frac < 1.0:
        train_idx = train_idx[: int(len(train_idx) * frac)]
        val_idx = val_idx[: max(1, int(len(val_idx) * frac))]
    return Subset(full_train, train_idx), Subset(full_val, val_idx), test


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, batch_size, num_workers, shuffle, cfg, device, generator=None, drop_last=False):
    d = cfg["data"]
    # The sampler gets its own generator. (If it were passed as DataLoader(generator=...), the
    # loader would also draw the worker base seed from it on the first epoch of a process, which
    # shifts the shuffle order after a resume.)
    sampler = RandomSampler(ds, generator=generator) if shuffle else None
    kw = dict(
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=bool(d.get("pin_memory", True)) and device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    if num_workers > 0:
        kw["persistent_workers"] = bool(d.get("persistent_workers", True))
        kw["prefetch_factor"] = d.get("prefetch_factor", 2)
    return DataLoader(ds, **kw)


def build_loaders(cfg, device):
    """Returns train/val/test loaders and the shuffle generator.

    The training loop re-seeds `generator` with (seed + epoch) before each epoch, so the
    sample order of epoch k is identical whether training ran straight through or was resumed.
    """
    train_ds, val_ds, test_ds = build_datasets(cfg)
    nw = cfg["data"]["num_workers"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    train = make_loader(train_ds, cfg["train"]["batch_size"], nw, True, cfg, device, gen, drop_last=True)
    eb = cfg["data"].get("eval_batch_size", 512)
    val = make_loader(val_ds, eb, nw, False, cfg, device)
    test = make_loader(test_ds, eb, nw, False, cfg, device)
    return train, val, test, gen
