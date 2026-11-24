"""
Lab 5: Frameworks and automatic differentiation  -  Tasks 3(test), 4, 5, 6.

Run:   python lab5.py            (all tasks)
       python lab5.py 4 5        (selected tasks)
Outputs are printed and also written to ./results/
Requires: numpy, scikit-learn, torch (matplotlib optional for the plot).
"""
import json, math, os, platform, statistics, sys, time
import numpy as np

from engine import Value, vsum, zero_grad

RESULTS = {}
os.makedirs("results", exist_ok=True)


def median_time(fn, runs=5):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


# ====================================================================== #
# Task 3 sanity check: gradient ACCUMULATION when a node is reused        #
# ====================================================================== #
def task3_accumulation_test():
    print("\n=== Task 3: accumulation test ===")
    x = Value(3.0)
    y = x * x + x            # x used 3 times -> dy/dx = 2x + 1 = 7
    y.backward()
    print(f"y = x*x + x at x=3 -> dy/dx = {x.grad} (expected 7.0)")
    assert abs(x.grad - 7.0) < 1e-12
    a = Value(2.0); b = a + a  # db/da = 2 (would be 1 if we overwrote)
    b.backward()
    print(f"b = a + a          -> db/da = {a.grad} (expected 2.0)")
    assert a.grad == 2.0
    RESULTS["task3"] = {"x*x+x grad": x.grad, "a+a grad": a.grad, "passed": True}


# ====================================================================== #
# Task 4: verify against PyTorch autograd and numerical gradients        #
# ====================================================================== #
def f_value(a, b, c):
    return vsum([((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c)])


def f_float(a, b, c):
    return float(sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(a, b, c)))


def task4():
    import torch
    print("\n=== Task 4: ((a*b + c)**2).relu().sum() ===")
    rng = np.random.default_rng(42)
    n = 10
    A, B, C = (rng.standard_normal(n) for _ in range(3))

    # our engine
    a = [Value(v) for v in A]; b = [Value(v) for v in B]; c = [Value(v) for v in C]
    out = f_value(a, b, c); out.backward()
    ours = {"a": np.array([v.grad for v in a]),
            "b": np.array([v.grad for v in b]),
            "c": np.array([v.grad for v in c])}

    # PyTorch autograd (float64 so that the comparison is precise)
    ta, tb, tc = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (A, B, C))
    tout = ((ta * tb + tc) ** 2).relu().sum(); tout.backward()
    torch_g = {"a": ta.grad.numpy(), "b": tb.grad.numpy(), "c": tc.grad.numpy()}

    # numerical central differences
    h = 1e-6
    num = {}
    for name, arr in (("a", A), ("b", B), ("c", C)):
        g = np.zeros(n)
        for i in range(n):
            p, m = arr.copy(), arr.copy(); p[i] += h; m[i] -= h
            args_p = {"a": A, "b": B, "c": C}; args_m = dict(args_p)
            args_p[name] = p; args_m[name] = m
            g[i] = (f_float(**args_p) - f_float(**args_m)) / (2 * h)
        num[name] = g

    rows = []
    print(f"forward: ours={out.data:.12f}  torch={tout.item():.12f}")
    print(f"{'input':6}{'max|ours-torch|':>18}{'max|ours-numeric|':>20}{'max|torch-numeric|':>20}")
    worst = 0.0
    for k in "abc":
        d1 = np.max(np.abs(ours[k] - torch_g[k]))
        d2 = np.max(np.abs(ours[k] - num[k]))
        d3 = np.max(np.abs(torch_g[k] - num[k]))
        worst = max(worst, d1, d2)
        rows.append({"input": k, "ours_vs_torch": d1, "ours_vs_numeric": d2, "torch_vs_numeric": d3})
        print(f"{k:6}{d1:18.3e}{d2:20.3e}{d3:20.3e}")
    ok = worst < 1e-6
    print(f"max absolute difference = {worst:.3e}  ->  {'PASS' if ok else 'FAIL'} (< 1e-6)")
    print("first 3 grads of a  ours :", np.round(ours['a'][:3], 8))
    print("first 3 grads of a  torch:", np.round(torch_g['a'][:3], 8))
    RESULTS["task4"] = {"forward_ours": out.data, "forward_torch": tout.item(),
                        "rows": rows, "max_abs_diff": worst, "pass": bool(ok)}


# ====================================================================== #
# Task 5: tiny model on make_moons, 2 -> 8 -> 1, sigmoid, BCE            #
# ====================================================================== #
N_SAMPLES, EPOCHS, LR, HIDDEN = 100, 100, 0.5, 8


def moons_data():
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=0)
    X = (X - X.mean(0)) / X.std(0)
    return X, y.astype(float)


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((HIDDEN, 2)) * math.sqrt(2 / 2)     # He init (ReLU)
    b1 = np.zeros(HIDDEN)
    W2 = rng.standard_normal((1, HIDDEN)) * math.sqrt(1 / HIDDEN)
    b2 = np.zeros(1)
    return W1, b1, W2, b2


class ValueMLP:
    """2 -> 8 (ReLU) -> 1 (sigmoid), built from our Value objects."""
    def __init__(self, W1, b1, W2, b2):
        self.W1 = [[Value(w) for w in row] for row in W1]
        self.b1 = [Value(v) for v in b1]
        self.W2 = [Value(w) for w in W2[0]]
        self.b2 = Value(b2[0])

    def parameters(self):
        return [w for row in self.W1 for w in row] + self.b1 + self.W2 + [self.b2]

    def __call__(self, x):
        h = [(vsum([w * xi for w, xi in zip(row, x)]) + b).relu()
             for row, b in zip(self.W1, self.b1)]
        return (vsum([w * hi for w, hi in zip(self.W2, h)]) + self.b2).sigmoid()


def bce(p, y, eps=1e-12):
    return -(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps)))


def train_ours(X, y, params, epochs=EPOCHS):
    model = ValueMLP(*params)
    P = model.parameters()
    losses = []
    for _ in range(epochs):
        zero_grad(P)
        terms = []
        for xi, yi in zip(X, y):
            p = model(xi)
            terms.append(-(yi * p.log() + (1 - yi) * (1 - p).log()))
        loss = vsum(terms) * (1.0 / len(X))
        loss.backward()
        for q in P:
            q.data -= LR * q.grad
        losses.append(loss.data)
    preds = np.array([model(xi).data for xi in X])
    return losses, float(((preds > 0.5) == y).mean())


def train_torch(X, y, params, epochs=EPOCHS, dtype=None):
    import torch
    dtype = dtype or torch.float64
    W1, b1, W2, b2 = params
    model = torch.nn.Sequential(torch.nn.Linear(2, HIDDEN), torch.nn.ReLU(),
                                torch.nn.Linear(HIDDEN, 1), torch.nn.Sigmoid()).to(dtype)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1)); model[0].bias.copy_(torch.tensor(b1))
        model[2].weight.copy_(torch.tensor(W2)); model[2].bias.copy_(torch.tensor(b2))
    Xt = torch.tensor(X, dtype=dtype); yt = torch.tensor(y, dtype=dtype).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    lossf = torch.nn.BCELoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward(); opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = ((model(Xt) > 0.5).to(dtype) == yt).to(dtype).mean().item()
    return losses, acc, model


def task5():
    import torch
    torch.set_num_threads(1)   # fair: our engine is single-threaded Python
    print("\n=== Task 5: make_moons, 2->8->1, sigmoid, BCE ===")
    X, y = moons_data()
    params = init_params(0)

    lo, acc_o = train_ours(X, y, params)
    lt, acc_t, model = train_torch(X, y, params)
    loss_diff = max(abs(a - b) for a, b in zip(lo, lt))
    print(f"loss epoch 1   ours={lo[0]:.6f} torch={lt[0]:.6f}")
    print(f"loss epoch {EPOCHS} ours={lo[-1]:.6f} torch={lt[-1]:.6f}")
    print(f"max |loss_ours - loss_torch| over all epochs = {loss_diff:.2e}")
    print(f"train accuracy ours={acc_o:.3f} torch={acc_t:.3f}")

    t_ours, ts_o = median_time(lambda: train_ours(X, y, params), runs=5)
    t_torch, ts_t = median_time(lambda: train_torch(X, y, params), runs=5)
    print(f"training time ({EPOCHS} epochs, {N_SAMPLES} samples), median of 5 runs:")
    print(f"  our engine : {t_ours*1000:9.1f} ms   runs={[round(t*1000,1) for t in ts_o]}")
    print(f"  PyTorch    : {t_torch*1000:9.1f} ms   runs={[round(t*1000,1) for t in ts_t]}")
    print(f"  speed-up of PyTorch: {t_ours/t_torch:.1f}x")

    RESULTS["task5"] = {"epochs": EPOCHS, "n_samples": N_SAMPLES, "lr": LR,
                        "loss_first": [lo[0], lt[0]], "loss_last": [lo[-1], lt[-1]],
                        "max_loss_diff": loss_diff, "acc": [acc_o, acc_t],
                        "median_ms_ours": t_ours * 1000, "median_ms_torch": t_torch * 1000,
                        "runs_ms_ours": [t * 1000 for t in ts_o],
                        "runs_ms_torch": [t * 1000 for t in ts_t],
                        "speedup": t_ours / t_torch}
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(lo, label="our engine"); ax[0].plot(lt, "--", label="PyTorch")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss"); ax[0].legend()
        ax[0].set_title("Training loss (identical init)")
        xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float64)
        with torch.no_grad():
            zz = model(grid).numpy().reshape(xx.shape)
        ax[1].contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
        ax[1].scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=20)
        ax[1].set_title("Decision boundary"); fig.tight_layout()
        fig.savefig("results/task5_moons.png", dpi=120); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)


# ====================================================================== #
# Task 6: graph inspection, eager vs compiled, tensor layout             #
# ====================================================================== #
def bench(fn, warmup=10, iters=50, reps=5):
    """median (over reps) of mean time per call, after warm-up."""
    import torch
    for _ in range(warmup):
        fn()
    meds = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        meds.append((time.perf_counter() - t0) / iters)
    return statistics.median(meds)


def task6():
    import torch
    import torch.fx
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print("\n=== Task 6: PyTorch graph, eager vs compiled, tensor layout ===")
    out = {}

    class SmallNet(torch.nn.Module):
        def __init__(self, d=256):
            super().__init__()
            self.fc1 = torch.nn.Linear(d, d)
            self.fc2 = torch.nn.Linear(d, d)
            self.ln = torch.nn.LayerNorm(d)

        def forward(self, x):
            h = torch.nn.functional.gelu(self.fc1(x))
            h = self.ln(h * torch.sigmoid(h) + x)   # chain of element-wise ops
            return self.fc2(h).relu()

    torch.manual_seed(0)
    model = SmallNet().eval()
    x = torch.randn(64, 256)

    # --- 6a. inspect the graph ------------------------------------------
    print("\n-- torch.fx graph (static view of forward) --")
    gm = torch.fx.symbolic_trace(model)
    print(gm.graph)
    out["fx_nodes"] = [f"{n.op}:{n.target}" for n in gm.graph.nodes]

    print("-- autograd graph (dynamic, built during eager forward) --")
    xr = x.clone().requires_grad_(True)
    y = model(xr).sum()
    seen, frontier, lines = set(), [(y.grad_fn, 0)], []
    while frontier:
        fnode, d = frontier.pop()
        if fnode is None or fnode in seen:
            continue
        seen.add(fnode); lines.append("  " * d + type(fnode).__name__)
        frontier.extend((nf, d + 1) for nf, _ in fnode.next_functions)
    print("\n".join(lines))
    out["autograd_nodes"] = lines

    print("-- tensor properties --")
    for name, t in (("input x", x), ("fc1.weight", model.fc1.weight),
                    ("fc1.weight.T", model.fc1.weight.t()), ("x[:, ::2]", x[:, ::2])):
        print(f"{name:13} shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()} "
              f"contiguous={t.is_contiguous()} device={t.device}")
    out["tensor_props"] = {"x": [list(x.shape), str(x.dtype), list(x.stride()), x.is_contiguous()]}

    # --- 6b. eager vs compiled -------------------------------------------
    eager = lambda inp: model(inp)
    compiled, mode = None, None
    t0 = time.perf_counter()
    try:
        cm = torch.compile(model)
        with torch.no_grad():
            cm(x)                       # first call triggers compilation
        compiled, mode = cm, "torch.compile (inductor)"
    except Exception as e:
        print(f"torch.compile unsupported here ({type(e).__name__}: {str(e)[:120]}) -> using torch.jit.trace")
        t0 = time.perf_counter()
        with torch.no_grad():
            compiled = torch.jit.trace(model, x)
            compiled = torch.jit.freeze(compiled.eval())
            compiled(x)
        mode = "torch.jit.trace (+freeze)"
    compile_s = time.perf_counter() - t0
    print(f"\ncompiled with {mode}; first call (compile) took {compile_s:.2f} s")
    if "trace" in mode:
        print(compiled.graph)

    rows = []
    with torch.no_grad():
        for bs in (1, 64, 1024, 8192):
            xb = torch.randn(bs, 256)
            if "compile" in mode:
                compiled(xb)  # warm-up / possible recompile for new shape
            te = bench(lambda: eager(xb))
            tc = bench(lambda: compiled(xb))
            assert torch.allclose(eager(xb), compiled(xb), atol=1e-4, rtol=1e-4)
            rows.append({"batch": bs, "eager_ms": te * 1e3, "compiled_ms": tc * 1e3,
                         "speedup": te / tc})
    print(f"\n{'batch':>6}{'eager ms':>12}{'compiled ms':>14}{'speed-up':>10}")
    for r in rows:
        print(f"{r['batch']:6d}{r['eager_ms']:12.4f}{r['compiled_ms']:14.4f}{r['speedup']:10.2f}x")
    out.update({"compile_mode": mode, "compile_seconds": compile_s, "eager_vs_compiled": rows})

    # training step too (forward + backward), batch 1024
    try:
        tm = SmallNet(); opt = torch.optim.SGD(tm.parameters(), lr=1e-3)
        ctm = torch.compile(tm) if "compile" in mode else None
        xb = torch.randn(1024, 256)
        def step(m):
            opt.zero_grad(); m(xb).sum().backward(); opt.step()
        te = bench(lambda: step(tm), iters=20)
        if ctm is not None:
            tc = bench(lambda: step(ctm), iters=20)
            print(f"training step b=1024: eager {te*1e3:.3f} ms, compiled {tc*1e3:.3f} ms, x{te/tc:.2f}")
            out["train_step"] = {"eager_ms": te * 1e3, "compiled_ms": tc * 1e3}
    except Exception as e:
        print("train-step compile skipped:", type(e).__name__)

    # --- 6c. non-contiguous views ----------------------------------------
    print("\n-- contiguous vs non-contiguous (4096x4096 float32) --")
    big = torch.randn(4096, 4096)
    views = {"contiguous": big,
             "transposed .t()": big.t(),
             "strided [:, ::2] ": big[:, ::2],
             ".t().contiguous()": big.t().contiguous()}
    for k, v in views.items():
        print(f"{k:20} stride={v.stride()} contiguous={v.is_contiguous()}")
    ops = {
        "elementwise x*2+1": lambda t: t * 2 + 1,
        "row sum sum(dim=1)": lambda t: t.sum(dim=1),
        "exp()": lambda t: t.exp(),
    }
    nc_rows = []
    print(f"{'op':22}{'contig ms':>11}{'.t() ms':>10}{'ratio':>8}")
    for name, op in ops.items():
        tc_ = bench(lambda: op(big), warmup=3, iters=10)
        tn_ = bench(lambda: op(big.t()), warmup=3, iters=10)
        nc_rows.append({"op": name, "contig_ms": tc_ * 1e3, "noncontig_ms": tn_ * 1e3, "ratio": tn_ / tc_})
        print(f"{name:22}{tc_*1e3:11.2f}{tn_*1e3:10.2f}{tn_/tc_:8.2f}x")
    # strided slice (half the elements) vs a contiguous tensor of the same size
    half_c = torch.randn(4096, 2048); half_s = big[:, ::2]
    tc_ = bench(lambda: half_c * 2 + 1, warmup=3, iters=10)
    tn_ = bench(lambda: half_s * 2 + 1, warmup=3, iters=10)
    nc_rows.append({"op": "x*2+1 on [:, ::2] (same numel)", "contig_ms": tc_ * 1e3,
                    "noncontig_ms": tn_ * 1e3, "ratio": tn_ / tc_})
    print(f"{'x*2+1 on [:, ::2]':22}{tc_*1e3:11.2f}{tn_*1e3:10.2f}{tn_/tc_:8.2f}x")
    tcopy = bench(lambda: big.t().contiguous(), warmup=3, iters=10)
    print(f"cost of making .t() contiguous (copy): {tcopy*1e3:.2f} ms")
    out["noncontig"] = nc_rows; out["contiguous_copy_ms"] = tcopy * 1e3
    RESULTS["task6"] = out


if __name__ == "__main__":
    wanted = set(sys.argv[1:]) or {"3", "4", "5", "6"}
    try:
        import torch
        RESULTS["env"] = {"python": platform.python_version(), "torch": torch.__version__,
                          "platform": platform.platform(), "cpu_count": os.cpu_count()}
        print("Environment:", RESULTS["env"])
    except ImportError:
        print("PyTorch not installed - only Task 3 can run.")
    if "3" in wanted: task3_accumulation_test()
    if "4" in wanted: task4()
    if "5" in wanted: task5()
    if "6" in wanted: task6()
    with open("results/results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nSaved results/results.json")
