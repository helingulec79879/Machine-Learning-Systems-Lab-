"""Step 3: measure system cost of both baselines.

For each model:
  * training time      - 1 warm-up fit, then 5 timed fits, report the median
  * single-sample inference latency - 1 warm-up, then 100 timed predict() calls
                         on one test row, report the median
  * model size         - joblib.dump to results/, record bytes and KB
  * peak memory        - psutil RSS sampled in a background thread during
                         training and during inference; tracemalloc peak as a
                         second, Python-heap-only view

Memory is measured in a fresh child process per model so that one model's
allocations do not inflate the other's numbers.

Outputs:
  results/system_cost.csv   - one row per model (summary)
  results/timings_raw.json  - every individual timing run
"""
import csv
import json
import multiprocessing as mp
import statistics
import time
import tracemalloc
import warnings

import joblib

from common import RESULTS_DIR, PeakRSS, load_split, make_models, set_seeds

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100
MB = 1024 ** 2

warnings.filterwarnings("ignore")  # ConvergenceWarning is reported in step 2


def time_training(factory, X, y):
    factory().fit(X, y)  # warm-up
    runs = []
    for _ in range(N_TRAIN_RUNS):
        model = factory()
        t0 = time.perf_counter()
        model.fit(X, y)
        runs.append(time.perf_counter() - t0)
    return model, runs


def time_single_inference(model, x_one):
    model.predict(x_one)  # warm-up
    runs = []
    for _ in range(N_INFER_RUNS):
        t0 = time.perf_counter()
        model.predict(x_one)
        runs.append(time.perf_counter() - t0)
    return runs


def memory_worker(name, queue):
    """Runs in a fresh process: peak memory for one model's training and inference."""
    set_seeds()
    X_train, X_test, y_train, _ = load_split()
    factory = make_models()[name]
    x_one = X_test[:1]

    # training: RSS sampling
    with PeakRSS() as train_rss:
        model = factory().fit(X_train, y_train)
    # training: Python-heap peak (numpy buffers are traced too)
    tracemalloc.start()
    factory().fit(X_train, y_train)
    _, train_tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # inference: 100 single-sample predictions
    model.predict(x_one)
    with PeakRSS() as infer_rss:
        for _ in range(N_INFER_RUNS):
            model.predict(x_one)
    tracemalloc.start()
    for _ in range(N_INFER_RUNS):
        model.predict(x_one)
    _, infer_tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    queue.put({
        "train_peak_rss_mb": train_rss.peak / MB,
        "train_rss_increase_mb": train_rss.delta / MB,
        "train_tracemalloc_peak_kb": train_tm_peak / 1024,
        "infer_peak_rss_mb": infer_rss.peak / MB,
        "infer_rss_increase_mb": infer_rss.delta / MB,
        "infer_tracemalloc_peak_kb": infer_tm_peak / 1024,
    })


def measure_memory(name):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=memory_worker, args=(name, queue))
    proc.start()
    result = queue.get()
    proc.join()
    return result


def main():
    set_seeds()
    X_train, X_test, y_train, _ = load_split()
    x_one = X_test[:1]

    summary, raw = [], {}
    for name, factory in make_models().items():
        print(f"== {name}")
        model, train_runs = time_training(factory, X_train, y_train)
        infer_runs = time_single_inference(model, x_one)

        path = RESULTS_DIR / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_bytes = path.stat().st_size

        mem = measure_memory(name)

        row = {
            "model": name,
            "train_time_median_s": round(statistics.median(train_runs), 6),
            "train_time_min_s": round(min(train_runs), 6),
            "train_time_max_s": round(max(train_runs), 6),
            "infer_latency_median_ms": round(statistics.median(infer_runs) * 1e3, 4),
            "infer_latency_p95_ms": round(sorted(infer_runs)[94] * 1e3, 4),
            "model_size_bytes": size_bytes,
            "model_size_kb": round(size_bytes / 1024, 2),
            **{k: round(v, 2) for k, v in mem.items()},
        }
        summary.append(row)
        raw[name] = {"train_s": train_runs, "infer_s": infer_runs}
        for k, v in row.items():
            print(f"   {k:<28}{v}")

    with (RESULTS_DIR / "system_cost.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (RESULTS_DIR / "timings_raw.json").write_text(json.dumps(raw, indent=1))
    print("saved results/system_cost.csv and results/timings_raw.json")


if __name__ == "__main__":
    main()
