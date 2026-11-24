"""
Lab 2: A neural network from scratch (Reddi, Ch. 3)

Architecture: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax
Everything (forward, loss, backward, SGD) is implemented with NumPy only.
scikit-learn is used ONLY to load the dataset and split it.

Run (from the lab02/ folder):  python src/nn_scratch.py
Outputs go to lab02/results/
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
SEED = 42


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
def one_hot(y, k=10):
    out = np.zeros((y.shape[0], k))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0          # pixels 0..16 -> 0..1
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    return X_tr, X_te, one_hot(y_tr), one_hot(y_te), y_tr, y_te


# ---------------------------------------------------------------------------
# 2. Network design / initialisation
#    W1 (64,W)  b1 (W,)
#    W2 (W,W)   b2 (W,)
#    W3 (W,10)  b3 (10,)
#    Weights ~ N(0, 0.01)  (std = 0.01), biases = 0
# ---------------------------------------------------------------------------
def init_params(width, rng, std=0.01):
    return {
        "W1": rng.normal(0.0, std, (64, width)),    "b1": np.zeros(width),
        "W2": rng.normal(0.0, std, (width, width)), "b2": np.zeros(width),
        "W3": rng.normal(0.0, std, (width, 10)),    "b3": np.zeros(10),
    }


def n_params(p):
    return sum(v.size for v in p.values())


# ---------------------------------------------------------------------------
# 3. Forward pass (layer by layer)
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)                # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"];  a1 = relu(z1)
    z2 = a1 @ p["W2"] + p["b2"]; a2 = relu(z2)
    z3 = a2 @ p["W3"] + p["b3"]; yhat = softmax(z3)
    cache = (X, z1, a1, z2, a2, yhat)
    return yhat, cache


# ---------------------------------------------------------------------------
# 4. Cross-entropy loss (mean over batch)
# ---------------------------------------------------------------------------
def cross_entropy(yhat, Y):
    return -np.mean(np.sum(Y * np.log(yhat + 1e-12), axis=1))


# ---------------------------------------------------------------------------
# 5. Backward pass (hand-derived chain rule)
#    dL/dz3 = (yhat - Y) / N               (softmax + CE combined)
#    dW3 = a2^T dz3      db3 = sum(dz3)
#    da2 = dz3 W3^T      dz2 = da2 * 1[z2 > 0]
#    dW2 = a1^T dz2      db2 = sum(dz2)
#    da1 = dz2 W2^T      dz1 = da1 * 1[z1 > 0]
#    dW1 = X^T  dz1      db1 = sum(dz1)
# ---------------------------------------------------------------------------
def backward(p, cache, Y):
    X, z1, a1, z2, a2, yhat = cache
    N = X.shape[0]
    dz3 = (yhat - Y) / N
    g = {"W3": a2.T @ dz3, "b3": dz3.sum(axis=0)}
    dz2 = (dz3 @ p["W3"].T) * (z2 > 0)
    g["W2"] = a1.T @ dz2;  g["b2"] = dz2.sum(axis=0)
    dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
    g["W1"] = X.T @ dz1;   g["b1"] = dz1.sum(axis=0)
    return g


def loss_fn(p, X, Y):
    return cross_entropy(forward(p, X)[0], Y)


def gradient_check(p, X, Y, n_checks=10, eps=1e-5, rng=None, precision=np.longdouble):
    """Nudge n random weights by +-eps, compare central difference to analytic.

    * Analytic gradient: our float64 backward pass (the thing being tested).
    * Numerical gradient: central difference (L(w+eps) - L(w-eps)) / (2 eps).
      With N(0, 0.01) init the gradients are ~1e-5..1e-7 while L ~ 2.3, so in
      float64 the round-off of the difference (~1e-16 * 2.3 / 1e-5 ~ 2e-11)
      alone gives relative errors ~1e-6. We therefore evaluate the *numerical*
      side in extended precision (np.longdouble, 64-bit mantissa). Pass
      precision=np.float64 to see the plain float64 result.
    * Weights are drawn among those with a non-zero analytic gradient (dead-ReLU
      units / always-zero pixels give 0 == 0, which tests nothing).
    * rel_error = |a - n| / max(|a|, |n|)   (the stricter of the usual forms)
    """
    _, cache = forward(p, X)
    g = backward(p, cache, Y)
    pp = {k: v.astype(precision) for k, v in p.items()}
    Xp, Yp = X.astype(precision), Y.astype(precision)
    e = precision(eps)
    weight_keys = ["W1", "W2", "W3"]
    rows = []
    for _ in range(n_checks):
        k = weight_keys[rng.integers(len(weight_keys))]
        nz = np.argwhere(g[k] != 0)
        idx = tuple(int(i) for i in nz[rng.integers(len(nz))])
        old = pp[k][idx]
        pp[k][idx] = old + e; lp = loss_fn(pp, Xp, Yp)
        pp[k][idx] = old - e; lm = loss_fn(pp, Xp, Yp)
        pp[k][idx] = old
        num = float((lp - lm) / (2 * e))
        ana = float(g[k][idx])
        rel = abs(ana - num) / max(abs(ana), abs(num), 1e-300)
        rows.append({"param": k, "index": [int(i) for i in idx],
                     "analytic": float(ana), "numerical": float(num),
                     "rel_error": float(rel), "pass": bool(rel < 1e-6)})
    return rows


# ---------------------------------------------------------------------------
# 6. Training loop: shuffle, mini-batch 32, 50 epochs, per-epoch logging
# ---------------------------------------------------------------------------
def accuracy(p, X, y):
    return float(np.mean(forward(p, X)[0].argmax(axis=1) == y))


def train(width, lr, data, epochs=50, batch=32, log=True):
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = data
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    N = X_tr.shape[0]
    for ep in range(1, epochs + 1):
        perm = rng.permutation(N)
        for s in range(0, N, batch):
            bi = perm[s:s + batch]
            _, cache = forward(p, X_tr[bi])
            g = backward(p, cache, Y_tr[bi])
            for k in p:
                p[k] -= lr * g[k]
        hist["train_loss"].append(loss_fn(p, X_tr, Y_tr))
        hist["test_loss"].append(loss_fn(p, X_te, Y_te))
        hist["train_acc"].append(accuracy(p, X_tr, y_tr))
        hist["test_acc"].append(accuracy(p, X_te, y_te))
        if log:
            print(f"  [W={width:>2} lr={lr:<5}] epoch {ep:2d}  "
                  f"train_loss={hist['train_loss'][-1]:.4f}  "
                  f"test_loss={hist['test_loss'][-1]:.4f}  "
                  f"test_acc={hist['test_acc'][-1]:.4f}")
    return p, hist


# ---------------------------------------------------------------------------
# 7. Timing: forward vs backward, median of 100 runs
# ---------------------------------------------------------------------------
def time_passes(p, X, Y, runs=100):
    fwd, bwd = [], []
    for _ in range(5):                                   # warm-up
        _, c = forward(p, X); backward(p, c, Y)
    for _ in range(runs):
        t0 = time.perf_counter(); _, cache = forward(p, X)
        t1 = time.perf_counter(); backward(p, cache, Y)
        t2 = time.perf_counter()
        fwd.append(t1 - t0); bwd.append(t2 - t1)
    return float(np.median(fwd)), float(np.median(bwd))


def matmul_flops(width, batch):
    """Multiply-adds in the matmuls. Forward: 3 matmuls. Backward: 5
    (3 weight grads + 2 input grads; no dX needed for the first layer)."""
    shapes = [(64, width), (width, width), (width, 10)]
    fwd = sum(batch * i * o for i, o in shapes)
    bwd = 2 * fwd - batch * 64 * width
    return fwd, bwd


# ---------------------------------------------------------------------------
def main():
    data = load_data()
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = data
    report = {}

    print("=== 1. Dataset ===")
    print(f"train X {X_tr.shape}, Y {Y_tr.shape} | test X {X_te.shape}, Y {Y_te.shape}")
    print(f"X range [{X_tr.min():.2f}, {X_tr.max():.2f}]")
    print("train class counts:", np.bincount(y_tr).tolist())
    print("test  class counts:", np.bincount(y_te).tolist())
    report["dataset"] = {"train": list(X_tr.shape), "test": list(X_te.shape),
                         "train_counts": np.bincount(y_tr).tolist(),
                         "test_counts": np.bincount(y_te).tolist()}

    print("\n=== 2. Design ===")
    report["design"] = {}
    for W in (16, 64):
        p = init_params(W, np.random.default_rng(SEED))
        shapes = {k: list(v.shape) for k, v in p.items()}
        print(f"W={W}: {shapes}  total params = {n_params(p)}")
        report["design"][W] = {"shapes": shapes, "n_params": n_params(p)}

    print("\n=== 3./4. Forward pass and initial loss ===")
    report["sanity"] = {}
    for W in (16, 64):
        p = init_params(W, np.random.default_rng(SEED))
        yhat, _ = forward(p, X_tr[:32])
        assert yhat.shape == (32, 10)
        max_dev = float(np.abs(yhat.sum(axis=1) - 1).max())
        assert max_dev < 1e-6
        L0 = loss_fn(p, X_tr, Y_tr)
        print(f"W={W}: output {yhat.shape}, max |row_sum-1| = {max_dev:.2e}, "
              f"initial loss = {L0:.4f} (ln 10 = {np.log(10):.4f})")
        report["sanity"][W] = {"output_shape": list(yhat.shape),
                               "max_rowsum_dev": max_dev, "initial_loss": L0}

    print("\n=== 5. Gradient check (10 random weights, eps=1e-5) ===")
    report["grad_check"] = {}
    for W in (16, 64):
        p = init_params(W, np.random.default_rng(SEED))
        Xb, Yb = X_tr[:16], Y_tr[:16]
        rows_ext = gradient_check(p, Xb, Yb, rng=np.random.default_rng(7))
        rows_f64 = gradient_check(p, Xb, Yb, rng=np.random.default_rng(7),
                                  precision=np.float64)
        for tag, rows in (("numerical_in_longdouble", rows_ext),
                          ("numerical_in_float64", rows_f64)):
            n_pass = sum(r["pass"] for r in rows)
            print(f"W={W} [{tag}]  passed {n_pass}/10")
            for r in rows:
                print(f"   {r['param']}{tuple(r['index'])!s:<10} analytic={r['analytic']: .6e} "
                      f"numerical={r['numerical']: .6e} rel_err={r['rel_error']:.2e} "
                      f"{'PASS' if r['pass'] else 'FAIL'}")
            report["grad_check"][f"W{W}_{tag}"] = {"passed": n_pass, "rows": rows}

    print("\n=== 6. Training sweep ===")
    runs = {}
    for W in (16, 64):
        for lr in (0.001, 0.01, 0.1):
            _, h = train(W, lr, data)
            runs[(W, lr)] = h

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {0.001: "#4C72B0", 0.01: "#DD8452", 0.1: "#55A868"}
    for ax, W in zip(axes, (16, 64)):
        for lr in (0.001, 0.01, 0.1):
            h = runs[(W, lr)]
            ep = np.arange(1, 51)
            ax.plot(ep, h["train_loss"], color=colors[lr], lw=2,
                    label=f"lr={lr} train")
            ax.plot(ep, h["test_loss"], color=colors[lr], lw=1.5, ls="--",
                    label=f"lr={lr} test")
        ax.axhline(np.log(10), color="grey", lw=0.8, ls=":")
        ax.text(50, np.log(10) + 0.03, "ln 10", ha="right", color="grey", fontsize=9)
        ax.set_title(f"Hidden width W = {W}")
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].legend(fontsize=9)
    fig.suptitle("Loss curves: 64→W→ReLU→W→ReLU→10, SGD, batch 32, 50 epochs")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "loss_curves.png"), dpi=150)
    print("saved results/loss_curves.png")

    print("\nFinal results (epoch 50):")
    print(f"{'W':>3} {'lr':>6} {'train_loss':>10} {'test_loss':>10} {'train_acc':>9} {'test_acc':>8}")
    report["training"] = []
    for (W, lr), h in runs.items():
        print(f"{W:>3} {lr:>6} {h['train_loss'][-1]:>10.4f} {h['test_loss'][-1]:>10.4f} "
              f"{h['train_acc'][-1]:>9.4f} {h['test_acc'][-1]:>8.4f}")
        report["training"].append({"width": W, "lr": lr,
                                   "final_train_loss": h["train_loss"][-1],
                                   "final_test_loss": h["test_loss"][-1],
                                   "final_train_acc": h["train_acc"][-1],
                                   "final_test_acc": h["test_acc"][-1],
                                   "history": h})

    print("\n=== 7. Timing (median of 100 runs) ===")
    report["timing"] = []
    for W in (16, 64):
        p = init_params(W, np.random.default_rng(SEED))
        for B in (32, X_tr.shape[0]):
            f, b = time_passes(p, X_tr[:B], Y_tr[:B])
            ff, bf = matmul_flops(W, B)
            print(f"W={W:>2} batch={B:>4}: forward {f*1e6:8.1f} us | backward {b*1e6:8.1f} us "
                  f"| bwd/fwd = {b/f:.2f} | matmul MACs fwd {ff:,} bwd {bf:,} (ratio {bf/ff:.2f})")
            report["timing"].append({"width": W, "batch": B, "forward_us": f * 1e6,
                                     "backward_us": b * 1e6, "ratio": b / f,
                                     "fwd_macs": ff, "bwd_macs": bf})

    with open(os.path.join(RESULTS, "results.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("saved results/results.json")


if __name__ == "__main__":
    main()
