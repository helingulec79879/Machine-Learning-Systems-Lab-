"""Lab 6 – Tasks 3-7: studies. Every training run is an MLflow run tagged with its study name.

Run from lab06/:
    python src/experiments.py workers    # Task 3  num_workers [0,1,2,4,8]
    python src/experiments.py batch      # Task 4  batch size [32..512], 5 epochs
    python src/experiments.py sched      # Task 5  StepLR / Cosine / Plateau + SGD vs Adam, 10 epochs
    python src/experiments.py accum      # Task 6  batch 256 vs batch 64 x accumulation 4
    python src/experiments.py resume     # Task 7  interrupt at epoch 3, resume
    python src/experiments.py export     # table of all MLflow runs -> results/mlflow_runs.csv
    python src/experiments.py all

Add --quick for a fast smoke test (few steps per epoch) before the real runs.
Tables (.md/.csv) and plots (.png) are written to results/<study>/.
"""
import argparse
import csv
import time
from pathlib import Path

import mlflow

from data import build_loaders
from train import apply_overrides, get_device, load_config, run, set_seed

RESULTS = Path("results")


# ----------------------------------------------------------------------------- helpers
def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    return p


def write_table(rows, study, name="table"):
    out = RESULTS / study
    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:.1f}" if isinstance(v, float) else str(v))
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def train(base, study, run_name, quick, **overrides):
    ov = [f"{k}={v}" for k, v in overrides.items()]
    ov += [f"run_name={study}_{run_name}", f"tags={{study: {study}}}", "checkpoint.resume=null"]
    if quick:
        ov += ["train.max_steps_per_epoch=20", "data.subset_fraction=0.1"]
    cfg = apply_overrides(base, ov)
    print(f"\n===== {study}: {run_name} =====")
    return cfg, run(cfg)


def curves(results, study, metric_pairs, fname="curves.png"):
    p = plt()
    fig, axes = p.subplots(1, len(metric_pairs), figsize=(5.2 * len(metric_pairs), 3.8))
    axes = [axes] if len(metric_pairs) == 1 else axes
    for ax, (metric, title) in zip(axes, metric_pairs):
        for label, res in results.items():
            h = res["history"]
            ax.plot([e["epoch"] for e in h], [e[metric] for e in h], "o-", ms=3, label=label)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / study / fname, dpi=130); p.close(fig)


# ----------------------------------------------------------------------------- Task 3
def loader_throughput(cfg, max_steps=None):
    """Iterate the training DataLoader alone (no model): the ceiling the input pipeline can deliver."""
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])
    loader, _, _, gen = build_loaders(cfg, device)
    gen.manual_seed(cfg["seed"] + 1)
    n_steps = len(loader) if not max_steps else min(max_steps, len(loader))
    t0 = time.perf_counter()
    it = iter(loader)
    next(it)
    first = time.perf_counter() - t0
    n = 1
    t1 = time.perf_counter()
    for _ in range(n_steps - 1):
        x, _ = next(it)
        n += 1
    dt = time.perf_counter() - t1
    bs = cfg["train"]["batch_size"]
    return {"startup_s": first, "loader_samples_per_sec": (n - 1) * bs / max(dt, 1e-9)}


def study_workers(base, quick):
    study, rows, res = "workers", [], {}
    for nw in [0, 1, 2, 4, 8]:
        cfg, r = train(base, study, f"nw{nw}", quick, **{"data.num_workers": nw, "train.epochs": 1})
        lt = loader_throughput(cfg, 20 if quick else None)
        h = r["history"][-1]
        rows.append({"num_workers": nw,
                     "loader_only_samples_per_s": lt["loader_samples_per_sec"],
                     "train_samples_per_s": h["samples_per_sec"],
                     "epoch_time_s": h["epoch_time_s"],
                     "first_batch_latency_s": lt["startup_s"],
                     "train/loader_ratio": h["samples_per_sec"] / lt["loader_samples_per_sec"]})
        res[nw] = r
    write_table(rows, study)
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(11, 3.8))
    x = [r["num_workers"] for r in rows]
    ax[0].plot(x, [r["loader_only_samples_per_s"] for r in rows], "o-", label="DataLoader alone")
    ax[0].plot(x, [r["train_samples_per_s"] for r in rows], "s-", label="full training step")
    ax[0].set_xlabel("num_workers"); ax[0].set_ylabel("samples / s"); ax[0].set_xticks(x)
    ax[0].set_title("Throughput vs num_workers"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].bar([str(v) for v in x], [r["epoch_time_s"] for r in rows], color="#DD8452")
    ax[1].set_xlabel("num_workers"); ax[1].set_ylabel("s"); ax[1].set_title("Epoch time")
    fig.tight_layout(); fig.savefig(RESULTS / study / "workers.png", dpi=130); p.close(fig)
    print("Bottleneck rule: where train ≈ loader-only throughput the input pipeline is the bottleneck;"
          " where training stays flat while the loader keeps getting faster, compute is the bottleneck.")


# ----------------------------------------------------------------------------- Task 4
def study_batch(base, quick):
    study, rows, res = "batch", [], {}
    for bs in [32, 64, 128, 256, 512]:
        _, r = train(base, study, f"bs{bs}", quick, **{"train.batch_size": bs, "train.epochs": 5})
        h = r["history"]
        steady = h[1:] if len(h) > 1 else h               # epoch 1 includes warm-up
        rows.append({"batch_size": bs,
                     "throughput_samples_per_s": sum(e["samples_per_sec"] for e in steady) / len(steady),
                     "peak_memory_mb": max(e["peak_mem_mb"] for e in h),
                     "epoch_time_s": sum(e["epoch_time_s"] for e in steady) / len(steady),
                     "final_val_acc": h[-1]["val_acc"],
                     "test_acc": r["test_acc"],
                     "optimizer_steps_per_epoch": h[-1]["optimizer_steps"]})
        res[f"bs{bs}"] = r
    write_table(rows, study)
    p = plt()
    fig, ax = p.subplots(1, 3, figsize=(14, 3.8))
    x = [str(r["batch_size"]) for r in rows]
    ax[0].bar(x, [r["throughput_samples_per_s"] for r in rows]); ax[0].set_title("Throughput (samples/s)")
    ax[1].bar(x, [r["peak_memory_mb"] for r in rows], color="#DD8452"); ax[1].set_title("Peak memory (MB)")
    ax[2].bar(x, [100 * r["final_val_acc"] for r in rows], color="#55A868"); ax[2].set_title("Val accuracy after 5 epochs (%)")
    for a in ax:
        a.set_xlabel("batch size"); a.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(RESULTS / study / "batch.png", dpi=130); p.close(fig)
    curves(res, study, [("val_acc", "val accuracy"), ("train_loss", "train loss")])


# ----------------------------------------------------------------------------- Task 5
def study_sched(base, quick):
    study, rows, res = "sched", [], {}
    runs = {
        "sgd_step":    {"optimizer.name": "sgd", "optimizer.lr": 0.05, "scheduler.name": "step"},
        "sgd_cosine":  {"optimizer.name": "sgd", "optimizer.lr": 0.05, "scheduler.name": "cosine"},
        "sgd_plateau": {"optimizer.name": "sgd", "optimizer.lr": 0.05, "scheduler.name": "plateau"},
        "adam_cosine": {"optimizer.name": "adam", "optimizer.lr": 0.001, "scheduler.name": "cosine"},
    }
    for name, ov in runs.items():
        _, r = train(base, study, name, quick, **ov, **{"train.epochs": 10})
        h = r["history"]
        best = max(h, key=lambda e: e["val_acc"])
        rows.append({"run": name,
                     "final_val_acc": h[-1]["val_acc"], "best_val_acc": best["val_acc"],
                     "best_epoch": best["epoch"], "final_train_loss": h[-1]["train_loss"],
                     "final_val_loss": h[-1]["val_loss"], "test_acc": r["test_acc"],
                     "total_time_s": sum(e["epoch_time_s"] for e in h)})
        res[name] = r
    write_table(rows, study)
    curves({k: v for k, v in res.items() if k.startswith("sgd")}, study,
           [("lr", "learning rate"), ("val_acc", "val accuracy"), ("val_loss", "val loss")], "schedulers.png")
    curves({k: v for k, v in res.items() if k in ("sgd_cosine", "adam_cosine")}, study,
           [("train_loss", "train loss"), ("val_acc", "val accuracy")], "sgd_vs_adam.png")


# ----------------------------------------------------------------------------- Task 6
def study_accum(base, quick):
    study, rows, res = "accum", [], {}
    runs = {
        "bs256_acc1": {"train.batch_size": 256, "train.grad_accum_steps": 1},
        "bs64_acc4":  {"train.batch_size": 64, "train.grad_accum_steps": 4},
        "bs64_acc1":  {"train.batch_size": 64, "train.grad_accum_steps": 1},   # reference: no accumulation
    }
    for name, ov in runs.items():
        _, r = train(base, study, name, quick, **ov, **{"train.epochs": 5})
        h = r["history"]
        rows.append({"run": name, "effective_batch": ov["train.batch_size"] * ov["train.grad_accum_steps"],
                     "optimizer_steps_per_epoch": h[-1]["optimizer_steps"],
                     "final_train_loss": h[-1]["train_loss"], "final_val_acc": h[-1]["val_acc"],
                     "test_acc": r["test_acc"], "peak_memory_mb": max(e["peak_mem_mb"] for e in h),
                     "throughput_samples_per_s": sum(e["samples_per_sec"] for e in h) / len(h)})
        res[name] = r
    write_table(rows, study)
    curves(res, study, [("train_loss", "train loss"), ("val_loss", "val loss"), ("val_acc", "val accuracy")])
    a, b = res["bs256_acc1"]["history"], res["bs64_acc4"]["history"]
    diff = max(abs(x["val_acc"] - y["val_acc"]) for x, y in zip(a, b))
    print(f"max |val_acc(bs256) - val_acc(bs64x4)| over epochs = {diff:.4f}")


# ----------------------------------------------------------------------------- Task 7
def study_resume(base, quick, epochs=6, stop=3):
    study = "resume"
    _, ref = train(base, study, "reference", quick, **{"train.epochs": epochs})
    cfg, part1 = train(base, study, "interrupted", quick,
                       **{"train.epochs": epochs, "train.stop_after_epoch": stop})
    assert part1["interrupted"], "run was not interrupted"
    print("\n===== resuming from checkpoint =====")
    cfg2 = apply_overrides(cfg, ["checkpoint.resume=auto", "train.stop_after_epoch=null"])
    resumed = run(cfg2)
    assert resumed["run_id"] == part1["run_id"], "resume must continue the same MLflow run"

    rows = []
    for e_ref, e_res in zip(ref["history"], resumed["history"]):
        rows.append({"epoch": e_ref["epoch"], "phase": "before" if e_ref["epoch"] <= stop else "after resume",
                     "ref_train_loss": e_ref["train_loss"], "resumed_train_loss": e_res["train_loss"],
                     "ref_val_acc": e_ref["val_acc"], "resumed_val_acc": e_res["val_acc"],
                     "ref_lr": e_ref["lr"], "resumed_lr": e_res["lr"]})
    write_table(rows, study)
    p = plt()
    fig, ax = p.subplots(1, 3, figsize=(14, 3.8))
    for a, (m, t) in zip(ax, [("train_loss", "train loss"), ("val_acc", "val accuracy"), ("lr", "learning rate")]):
        a.plot([e["epoch"] for e in ref["history"]], [e[m] for e in ref["history"]], "o-", label="uninterrupted", lw=3, alpha=0.5)
        a.plot([e["epoch"] for e in resumed["history"]], [e[m] for e in resumed["history"]], "x--", label="interrupted + resumed")
        a.axvline(stop + 0.5, color="red", ls=":", label=f"interrupt after epoch {stop}")
        a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / study / "resume.png", dpi=130); p.close(fig)


# ----------------------------------------------------------------------------- MLflow export
def study_export(base, quick):
    """Table of every MLflow run of the experiment (graded evidence: MLflow runs)."""
    mlflow.set_tracking_uri(base["mlflow"]["tracking_uri"])
    df = mlflow.search_runs(experiment_names=[base["experiment_name"]], order_by=["start_time ASC"])
    if df.empty:
        print("no MLflow runs found"); return
    keep = ["run_id", "tags.mlflow.runName", "tags.study", "status", "params.optimizer.name", "params.optimizer.lr",
            "params.scheduler.name", "params.train.batch_size", "params.train.grad_accum_steps",
            "params.data.num_workers", "params.train.epochs", "metrics.val_acc", "metrics.best_val_acc",
            "metrics.test_acc", "metrics.samples_per_sec", "metrics.peak_mem_mb"]
    df = df[[c for c in keep if c in df.columns]]
    df.columns = [c.split(".", 1)[1] if c.startswith(("params.", "metrics.", "tags.")) else c for c in df.columns]
    df = df.rename(columns={"mlflow.runName": "run_name"})
    out = RESULTS / "mlflow_runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} MLflow runs -> {out}")


STUDIES = {"workers": study_workers, "batch": study_batch, "sched": study_sched,
           "accum": study_accum, "resume": study_resume, "export": study_export}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=list(STUDIES) + ["all"])
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--quick", action="store_true", help="tiny smoke test")
    a = p.parse_args()
    base = load_config(a.config, a.set)
    if a.quick:
        RESULTS = Path("results_quick")   # module-level name, so the helpers above see it
    for name in (STUDIES if a.study == "all" else [a.study]):
        STUDIES[name](base, a.quick)
