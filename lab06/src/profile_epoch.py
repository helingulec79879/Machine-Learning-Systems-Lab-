"""Lab 6 – Task 2: profile one training epoch with torch.profiler.

Splits the epoch into four labelled regions with record_function:
    data_loading   = next(batch) from the DataLoader + host->device copy
    forward        = model(x) + loss
    backward       = loss.backward()
    optimizer_step = opt.step() + zero_grad()
The device is synchronised at the end of each region; otherwise asynchronous GPU/MPS kernels
would be charged to whichever later call happens to wait for them.

Outputs (results/profile/ and an MLflow run tagged study=profile):
    profile_breakdown.csv / .md   share of time per region
    profile_breakdown.png         bar chart
    profiler_table.txt            top operators from torch.profiler
    trace.json                    open in chrome://tracing or https://ui.perfetto.dev

Usage (from lab06/): python src/profile_epoch.py [--set data.num_workers=0]
"""
import argparse
import csv
import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from data import build_loaders, in_channels
from models import build_model
from train import build_optimizer, flatten, get_device, load_config, set_seed, sync

REGIONS = ["data_loading", "forward", "backward", "optimizer_step"]


def profile_epoch(cfg, out_dir="results/profile", max_steps=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    train_loader, _, _, gen = build_loaders(cfg, device)
    gen.manual_seed(cfg["seed"] + 1)
    model = build_model(cfg["model"], in_channels(cfg)).to(device)
    opt = build_optimizer(model.parameters(), cfg["optimizer"])
    model.train()

    n_steps = len(train_loader) if not max_steps else min(max_steps, len(train_loader))
    wall = {r: 0.0 for r in REGIONS}
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    it = iter(train_loader)
    sync(device)
    t_epoch = time.perf_counter()
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(n_steps):
            t = time.perf_counter()
            with record_function("data_loading"):
                x, y = next(it)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                sync(device)
            t1 = time.perf_counter(); wall["data_loading"] += t1 - t

            with record_function("forward"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                sync(device)
            t2 = time.perf_counter(); wall["forward"] += t2 - t1

            with record_function("backward"):
                loss.backward()
                sync(device)
            t3 = time.perf_counter(); wall["backward"] += t3 - t2

            with record_function("optimizer_step"):
                opt.step()
                opt.zero_grad(set_to_none=True)
                sync(device)
            wall["optimizer_step"] += time.perf_counter() - t3
    epoch_time = time.perf_counter() - t_epoch

    # profiler view of the same regions (CPU wall time of each record_function range, in µs)
    prof_us = {r: 0.0 for r in REGIONS}
    for evt in prof.key_averages():
        if evt.key in prof_us:
            prof_us[evt.key] = evt.cpu_time_total

    total_wall = sum(wall.values())
    total_prof = sum(prof_us.values()) or 1.0
    rows = []
    for r in REGIONS:
        rows.append({
            "region": r,
            "time_s": round(wall[r], 3),
            "share_pct": round(100 * wall[r] / total_wall, 1),
            "profiler_time_s": round(prof_us[r] / 1e6, 3),
            "profiler_share_pct": round(100 * prof_us[r] / total_prof, 1),
        })

    with open(out / "profile_breakdown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    md = ["| Region | Time (s) | Share (%) | Profiler time (s) | Profiler share (%) |",
          "|---|---:|---:|---:|---:|"]
    md += [f"| {r['region']} | {r['time_s']} | {r['share_pct']} | {r['profiler_time_s']} | {r['profiler_share_pct']} |"
           for r in rows]
    md.append(f"\nDevice: {device}, steps: {n_steps}, batch: {cfg['train']['batch_size']}, "
              f"num_workers: {cfg['data']['num_workers']}, epoch time (incl. profiler overhead): {epoch_time:.1f} s")
    (out / "profile_breakdown.md").write_text("\n".join(md) + "\n")

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    (out / "profiler_table.txt").write_text(prof.key_averages().table(sort_by=sort_key, row_limit=30))
    prof.export_chrome_trace(str(out / "trace.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(REGIONS[::-1], [r["share_pct"] for r in rows][::-1], color="#4C72B0")
    for i, v in enumerate([r["share_pct"] for r in rows][::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("share of epoch time (%)")
    ax.set_title(f"One-epoch profile ({device}, workers={cfg['data']['num_workers']})")
    ax.set_xlim(0, 110)
    fig.tight_layout(); fig.savefig(out / "profile_breakdown.png", dpi=130); plt.close(fig)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name="profile_one_epoch"):
        mlflow.set_tag("study", "profile")
        mlflow.log_params(flatten(cfg))
        mlflow.log_metrics({f"share_{r['region']}_pct": r["share_pct"] for r in rows})
        mlflow.log_metric("profiled_epoch_time_s", epoch_time)
        for name in ["profile_breakdown.csv", "profile_breakdown.md", "profile_breakdown.png", "profiler_table.txt"]:
            mlflow.log_artifact(str(out / name))
        # trace.json can be large; it stays in results/profile/

    print("\n".join(md))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--max-steps", type=int, default=None, help="default: full epoch")
    a = p.parse_args()
    profile_epoch(load_config(a.config, a.set), max_steps=a.max_steps)
