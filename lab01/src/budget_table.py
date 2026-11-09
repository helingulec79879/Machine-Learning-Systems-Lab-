"""Step 4: compare measured cost with the Cloud / Edge / Mobile / TinyML budgets.

Two memory views are checked, because they answer different questions:
  * "process"  - peak RSS of the whole Python process during inference.
                 This is what the model costs if deployed as-is with Python +
                 scikit-learn.
  * "model"    - the model's own footprint: max(serialized size, Python-heap
                 peak during inference). This is a lower bound for a port to a
                 lean runtime (e.g. compiled C), which is what TinyML requires.

A platform "fits" when latency and model size are within budget and memory is
not above the platform's upper limit. Budgets with only a lower memory bound
(Cloud: >= 1 GB) fit any smaller footprint.

Outputs:
  results/deployment_fit.csv
  results/deployment_fit.png
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR

KB, MB_IN_KB = 1, 1024
# memory upper limit (KB), latency limit (ms), model-size limit (KB)
BUDGETS = {
    "Cloud": {"mem_kb": float("inf"), "lat_ms": 100, "size_kb": 500 * MB_IN_KB},
    "Edge": {"mem_kb": 1024 * MB_IN_KB, "lat_ms": 50, "size_kb": 50 * MB_IN_KB},
    "Mobile": {"mem_kb": 256 * MB_IN_KB, "lat_ms": 20, "size_kb": 10 * MB_IN_KB},
    "TinyML": {"mem_kb": 256, "lat_ms": 10, "size_kb": 100},
}


def load_cost():
    with (RESULTS_DIR / "system_cost.csv").open() as f:
        return list(csv.DictReader(f))


def check(row, view):
    lat = float(row["infer_latency_median_ms"])
    size_kb = float(row["model_size_kb"])
    if view == "process":
        mem_kb = float(row["infer_peak_rss_mb"]) * 1024
    else:
        mem_kb = max(size_kb, float(row["infer_tracemalloc_peak_kb"]))
    out = []
    for platform, b in BUDGETS.items():
        ok_mem = mem_kb <= b["mem_kb"]
        ok_lat = lat <= b["lat_ms"]
        ok_size = size_kb <= b["size_kb"]
        failed = [n for n, ok in (("memory", ok_mem), ("latency", ok_lat), ("size", ok_size)) if not ok]
        out.append({
            "model": row["model"], "memory_view": view, "platform": platform,
            "memory_kb": round(mem_kb, 1), "latency_ms": lat, "size_kb": size_kb,
            "fits": "YES" if not failed else "NO",
            "violated": ",".join(failed) or "-",
        })
    return out


def plot(cost):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [
        ("infer_latency_median_ms", "Single-sample latency (ms)", "lat_ms", 1),
        ("model_size_kb", "Model size (KB)", "size_kb", 1),
        ("infer_peak_rss_mb", "Inference peak RSS (KB)", "mem_kb", 1024),
    ]
    colors = {"Cloud": "#4c72b0", "Edge": "#55a868", "Mobile": "#dd8452", "TinyML": "#c44e52"}
    for ax, (col, title, bkey, scale) in zip(axes, metrics):
        names = [r["model"] for r in cost]
        vals = [float(r[col]) * scale for r in cost]
        ax.bar(names, vals, color="#8c8c8c")
        for platform, b in BUDGETS.items():
            if b[bkey] != float("inf"):
                ax.axhline(b[bkey], ls="--", lw=1, color=colors[platform], label=platform)
        ax.set_yscale("log")
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].legend(fontsize=8, title="budget limit")
    fig.suptitle("Measured cost vs. deployment budgets (log scale)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "deployment_fit.png", dpi=150)


def main():
    cost = load_cost()
    rows = [r for c in cost for view in ("process", "model") for r in check(c, view)]
    with (RESULTS_DIR / "deployment_fit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['model']:<20}{r['memory_view']:<9}{r['platform']:<8}{r['fits']:<5}{r['violated']}")
    plot(cost)
    print("saved results/deployment_fit.csv and results/deployment_fit.png")


if __name__ == "__main__":
    main()
