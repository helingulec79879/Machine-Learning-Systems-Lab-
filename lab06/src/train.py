"""Lab 6 – Task 1: reproducible, YAML-driven training pipeline.

Features
  * one YAML config object drives everything (CLI overrides with --set key=value)
  * seeding of python / numpy / torch / dataloader workers, fixed train/val split
  * checkpoint every epoch (model, optimizer, scheduler, RNG states, MLflow run id) -> exact resume
  * gradient accumulation (Task 6), simulated interruption (Task 7)
  * MLflow: params (whole config), per-epoch metrics, artifacts (config, curves, history, best model)

Usage (from lab06/)
  python src/train.py
  python src/train.py --set run_name=sgd optimizer.name=sgd optimizer.lr=0.05
  python src/train.py --set checkpoint.resume=auto          # continue the last run of run_name
"""
import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import mlflow

from data import build_loaders, in_channels
from models import build_model, model_size


# ----------------------------------------------------------------------------- config
def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for ov in overrides or ():
        key, val = ov.split("=", 1)
        set_by_path(cfg, key, yaml.safe_load(val))
    return cfg


def set_by_path(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ----------------------------------------------------------------------------- utils
def set_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(name="auto"):
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for queued GPU work, so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class PeakMemory:
    """Peak memory of the current epoch.

    CUDA: allocator peak. MPS: allocator memory, sampled. CPU: process RSS, sampled after the
    forward pass (when activations are alive) and after the step, reset every epoch.
    """

    def __init__(self, device):
        self.device, self.peak = device, 0
        self._proc = None
        if device.type == "cpu":
            import psutil
            self._proc = psutil.Process()

    def reset(self):
        self.peak = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def update(self):
        if self.device.type == "mps":
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
        elif self._proc is not None:
            self.peak = max(self.peak, self._proc.memory_info().rss)

    def peak_mb(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 2**20
        return self.peak / 2**20


def build_optimizer(params, c):
    name = c["name"].lower()
    wd = float(c.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(params, lr=float(c["lr"]), momentum=float(c.get("momentum", 0.9)),
                               nesterov=bool(c.get("nesterov", False)), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=float(c["lr"]), weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=float(c["lr"]), weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name}")


def build_scheduler(opt, c, epochs):
    name = (c.get("name") or "none").lower()
    S = torch.optim.lr_scheduler
    if name == "none":
        return None
    if name == "step":
        return S.StepLR(opt, step_size=int(c.get("step_size", 3)), gamma=float(c.get("gamma", 0.1)))
    if name == "cosine":
        return S.CosineAnnealingLR(opt, T_max=int(c.get("t_max") or epochs), eta_min=float(c.get("eta_min", 0.0)))
    if name == "plateau":
        return S.ReduceLROnPlateau(opt, mode="min", factor=float(c.get("factor", 0.5)),
                                   patience=int(c.get("patience", 1)))
    raise ValueError(f"Unknown scheduler {name}")


# ----------------------------------------------------------------------------- train / eval
def train_one_epoch(model, loader, opt, device, accum_steps=1, mem=None, max_steps=None):
    """One epoch with gradient accumulation.

    Each micro-batch loss is divided by the number of micro-batches in its group, so the summed
    gradient equals the gradient of the mean loss over the large (effective) batch.
    """
    model.train()
    n_batches = len(loader) if not max_steps else min(len(loader), max_steps)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n_seen, opt_steps = 0, 0
    opt.zero_grad(set_to_none=True)
    sync(device)
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        if mem is not None:
            mem.update()                                   # activations are alive here
        group_start = (i // accum_steps) * accum_steps
        group_size = min(accum_steps, n_batches - group_start)   # last group may be shorter
        (loss / group_size).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_steps += 1
        if mem is not None:
            mem.update()
        loss_sum += loss.detach() * x.size(0)          # stays on device: no per-step sync
        correct += (out.argmax(1) == y).sum()
        n_seen += x.size(0)
    sync(device)
    dt = time.perf_counter() - t0
    return {
        "train_loss": loss_sum.item() / n_seen,
        "train_acc": correct.item() / n_seen,
        "samples_per_sec": n_seen / dt,
        "epoch_time_s": dt,
        "optimizer_steps": opt_steps,
    }


@torch.no_grad()
def evaluate(model, loader, device, prefix="val"):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return {f"{prefix}_loss": loss_sum / n, f"{prefix}_acc": correct / n}


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, opt, sched, epoch, best_acc, run_id, cfg, history):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "best_acc": best_acc,
        "mlflow_run_id": run_id,
        "config": cfg,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)      # atomic: a crash while saving never corrupts last.pt


def load_checkpoint(path, model, opt, sched):
    state = torch.load(path, map_location="cpu", weights_only=False)   # own file, trusted
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    if sched is not None and state["scheduler"] is not None:
        sched.load_state_dict(state["scheduler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state


def plot_history(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], "o-", label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], "o-", label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], "o-", label="val")
    ax[1].set_title("accuracy"); ax[1].legend()
    ax[2].plot(ep, [h["lr"] for h in history], "o-")
    ax[2].set_title("learning rate")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main loop
def run(cfg):
    """Train according to `cfg`. Returns a dict with run_id, history and final metrics."""
    cfg = copy.deepcopy(cfg)
    set_seed(cfg["seed"], cfg.get("deterministic", False))
    device = get_device(cfg.get("device", "auto"))
    tr = cfg["train"]
    epochs, accum = int(tr["epochs"]), int(tr.get("grad_accum_steps", 1))

    train_loader, val_loader, test_loader, shuffle_gen = build_loaders(cfg, device)
    model = build_model(cfg["model"], in_channels(cfg)).to(device)
    opt = build_optimizer(model.parameters(), cfg["optimizer"])
    sched = build_scheduler(opt, cfg["scheduler"], epochs)
    n_params, size_mb = model_size(model)

    ckpt_dir = Path(cfg["checkpoint"]["dir"]) / cfg["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_acc, run_id, history = 1, 0.0, None, []

    resume = cfg["checkpoint"].get("resume")
    if resume:
        path = ckpt_dir / "last.pt" if resume == "auto" else Path(resume)
        if path.exists():
            state = load_checkpoint(path, model, opt, sched)
            start_epoch = state["epoch"] + 1
            best_acc, run_id, history = state["best_acc"], state["mlflow_run_id"], state["history"]
            print(f"[resume] {path} -> continuing at epoch {start_epoch} (MLflow run {run_id})")
        else:
            print(f"[resume] no checkpoint at {path}, starting from scratch")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else cfg["run_name"]) as active:
        run_id = active.info.run_id
        if start_epoch == 1:
            mlflow.log_params(flatten(cfg))
            mlflow.log_params({"model.n_params": n_params, "model.size_mb_fp32": round(size_mb, 3),
                               "device": str(device), "effective_batch_size": tr["batch_size"] * accum,
                               "torch_version": torch.__version__})
            mlflow.set_tags({str(k): str(v) for k, v in (cfg.get("tags") or {}).items()})
            mlflow.log_dict(cfg, "config.yaml")
        else:
            mlflow.set_tag(f"resumed_at_epoch_{start_epoch}", "true")

        print(f"device={device}  params={n_params:,}  size={size_mb:.2f} MB  "
              f"batch={tr['batch_size']}x{accum}  epochs={epochs}")
        mem = PeakMemory(device)
        interrupted = False
        for epoch in range(start_epoch, epochs + 1):
            shuffle_gen.manual_seed(cfg["seed"] + epoch)   # same order with or without resume
            mem.reset()
            lr = opt.param_groups[0]["lr"]
            m = train_one_epoch(model, train_loader, opt, device, accum, mem, tr.get("max_steps_per_epoch"))
            m.update(evaluate(model, val_loader, device, "val"))
            m["lr"] = lr
            m["peak_mem_mb"] = mem.peak_mb()
            if sched is not None:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(m["val_loss"])
                else:
                    sched.step()
            mlflow.log_metrics(m, step=epoch)
            history.append({"epoch": epoch, **m})

            if m["val_acc"] > best_acc:
                best_acc = m["val_acc"]
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
            save_checkpoint(ckpt_dir / "last.pt", model, opt, sched, epoch, best_acc, run_id, cfg, history)
            if cfg["checkpoint"].get("keep_every_epoch"):
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, opt, sched, epoch, best_acc,
                                run_id, cfg, history)

            print(f"epoch {epoch:3d}/{epochs} | loss {m['train_loss']:.4f} acc {m['train_acc']:.4f} | "
                  f"val_loss {m['val_loss']:.4f} val_acc {m['val_acc']:.4f} | lr {lr:.5f} | "
                  f"{m['samples_per_sec']:.0f} samples/s | {m['peak_mem_mb']:.0f} MB")

            if tr.get("stop_after_epoch") and epoch == int(tr["stop_after_epoch"]) and epoch < epochs:
                print(f"[interrupt] simulated interruption after epoch {epoch}; "
                      f"resume with --set checkpoint.resume=auto")
                mlflow.set_tag("interrupted_after_epoch", epoch)
                interrupted = True
                break

        result = {"run_id": run_id, "history": history, "best_val_acc": best_acc,
                  "n_params": n_params, "size_mb": size_mb, "interrupted": interrupted}
        if not interrupted:
            if (ckpt_dir / "best.pt").exists():
                model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
            test = evaluate(model, test_loader, device, "test")
            mlflow.log_metrics({**test, "best_val_acc": best_acc})
            result.update(test)
            # artifacts
            hist_path = ckpt_dir / "history.csv"
            with open(hist_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                w.writeheader(); w.writerows(history)
            plot_path = ckpt_dir / "curves.png"
            plot_history(history, plot_path, cfg["run_name"])
            mlflow.log_artifact(str(hist_path))
            mlflow.log_artifact(str(plot_path))
            if cfg["mlflow"].get("log_model_artifact", True):
                mlflow.log_artifact(str(ckpt_dir / "best.pt"), artifact_path="model")
            print(f"test_acc={test['test_acc']:.4f}  best_val_acc={best_acc:.4f}  run_id={run_id}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    print(json.dumps(cfg, indent=2))
    run(cfg)


if __name__ == "__main__":
    main()
