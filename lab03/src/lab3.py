"""Lab 3: Architectures and their costs  —  run from lab03/:  python src/lab3.py
Produces: results/results.json, results/*.png, report.md
"""
import json, os, statistics, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from flops_hand import HAND, enc_layer_flops

LAB = Path(__file__).resolve().parent.parent      # lab03/
RES = LAB / "results"; DATA = LAB.parent / "data"

SEED, BATCH, EPOCHS, LR = 0, 128, 5, 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAT_DEVICE = torch.device("cpu")      # latency measured on CPU for a repeatable protocol
torch.set_num_threads(max(1, os.cpu_count() // 2))


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- 1. data
def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    train, val = random_split(full, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED))
    mk = lambda ds, sh: DataLoader(ds, batch_size=BATCH, shuffle=sh, num_workers=2)
    return mk(train, True), mk(val, False), mk(test, False)


# ---------------------------------------------------------------- 2. models
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
    def forward(self, x): return self.net(x)


class PatchTransformer(nn.Module):
    """28x28 -> 16 patches of 7x7 -> Linear(49,64) -> 2 encoder layers -> mean -> Linear(64,10) (as in the lab spec)."""
    def __init__(self, d=64, heads=4, ff=128, layers=2, p=7):
        super().__init__()
        self.p = p
        self.proj = nn.Linear(p * p, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 10)

    def forward(self, x):                                   # x: (B,1,28,28)
        B, p = x.shape[0], self.p
        x = x.unfold(2, p, p).unfold(3, p, p)               # (B,1,4,4,7,7)
        x = x.reshape(B, -1, p * p)                         # (B,16,49)
        x = self.proj(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchTransformer}


# ---------------------------------------------------------------- 3. cost
def count_params(model):
    try:
        from torchinfo import summary
        s = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        return s.total_params, str(s)
    except ImportError:
        return sum(p.numel() for p in model.parameters()), "(torchinfo not installed)"


def thop_profile(model):
    """thop returns MACs (not FLOPs). Also collect per-layer numbers when supported."""
    try:
        from thop import profile
    except ImportError:
        return None, {}
    m = model.to("cpu").eval()
    x = torch.randn(1, 1, 28, 28)
    try:
        macs, _, info = profile(m, inputs=(x,), verbose=False, ret_layer_info=True)
    except TypeError:                                       # old thop without ret_layer_info
        macs, _ = profile(m, inputs=(x,), verbose=False); info = {}
    flat = {}
    def walk(d, prefix=""):
        for k, v in d.items():
            name = f"{prefix}.{k}" if prefix else k
            flat[name] = float(v[0])
            if len(v) > 2 and isinstance(v[2], dict): walk(v[2], name)
    walk(info)
    return float(macs), flat


@torch.no_grad()
def latency_ms(model, batch, runs, warmup=10):
    m = model.to(LAT_DEVICE).eval()
    x = torch.randn(batch, 1, 28, 28, device=LAT_DEVICE)
    for _ in range(warmup): m(x)
    times = []
    for _ in range(runs):
        if LAT_DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); m(x)
        if LAT_DEVICE.type == "cuda": torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


# ---------------------------------------------------------------- 4. train
def evaluate(model, loader):
    model.eval(); correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item(); n += y.numel()
    return correct / n


def train(model, tr, va):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(); hist = []
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * y.numel()
        va_acc = evaluate(model, va)
        hist.append({"epoch": ep, "train_loss": tot / len(tr.dataset), "val_acc": va_acc})
        print(f"   epoch {ep}: loss={hist[-1]['train_loss']:.4f}  val_acc={va_acc:.4f}")
    return hist


# ---------------------------------------------------------------- 5. plots
def plot(results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RES.mkdir(exist_ok=True)
    colors = {"MLP": "#2a6fdb", "CNN": "#e07a1f", "Attention": "#2b9e5a"}
    for key, xlabel, fname in [
        ("params", "Parameters", "acc_vs_params.png"),
        ("lat_b1_ms", "Latency, batch 1 (ms, median of 100)", "acc_vs_latency.png"),
        ("lat_b128_ms", "Latency, batch 128 (ms, median of 20)", "acc_vs_latency_b128.png")]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for name, r in results.items():
            ax.scatter(r[key], r["test_acc"] * 100, s=90, color=colors[name], zorder=3)
            ax.annotate(name, (r[key], r["test_acc"] * 100), textcoords="offset points",
                        xytext=(7, 5), color=colors[name])
        ax.set_xlabel(xlabel); ax.set_ylabel("Test accuracy (%)")
        if key == "params": ax.set_xscale("log")
        ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(RES / fname, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    set_seed()
    tr, va, te = get_loaders()
    results = {}
    for name, cls in MODELS.items():
        print(f"\n=== {name} ===")
        set_seed(); model = cls()
        params, summ = count_params(model)
        hp, hf = HAND[name]
        macs, layer_macs = thop_profile(model)
        lat1 = latency_ms(model, 1, 100); lat128 = latency_ms(model, 128, 20)
        print(f"   params={params:,} (hand {hp:,})  hand FLOPs={sum(hf.values()):,}  "
              f"thop MACs={macs}  lat b1={lat1:.3f}ms b128={lat128:.3f}ms")
        hist = train(model, tr, va)
        acc = evaluate(model, te)
        print(f"   TEST ACC = {acc:.4f}")
        results[name] = dict(params=params, hand_params=hp, hand_flops=sum(hf.values()),
                             hand_layers=hf, thop_macs=macs, thop_flops=None if macs is None else 2 * macs,
                             thop_layers=layer_macs, lat_b1_ms=lat1, lat_b128_ms=lat128,
                             throughput_b128=128 / (lat128 / 1e3), test_acc=acc, history=hist,
                             summary=summ)
    RES.mkdir(exist_ok=True)
    json.dump(results, open(RES / "results.json", "w"), indent=2)
    plot(results)
    write_report(results)
    print("\nWrote results/results.json, results/*.png, report.md")


def fmt(v, d=0):
    return "n/a" if v is None else (f"{v:,.{d}f}" if isinstance(v, float) else f"{v:,}")


def write_report(R):
    tpl = open(Path(__file__).parent / "report_template.md").read()
    rows = "\n".join(
        f"| {n} | {fmt(r['params'])} | {fmt(r['hand_flops'])} | {fmt(r['thop_macs'])} | "
        f"{fmt(r['thop_flops'])} | {r['lat_b1_ms']:.3f} | {r['lat_b128_ms']:.3f} | "
        f"{r['throughput_b128']:,.0f} | {r['test_acc']*100:.2f} |" for n, r in R.items())

    def layer(model, thop_key_hint):
        L = R[model]["thop_layers"]
        hits = [v for k, v in L.items() if k.endswith(thop_key_hint)]
        return fmt(2 * hits[0]) if hits else "n/a"
    def ffn_thop():
        L = R["Attention"]["thop_layers"]
        hits = [v for k, v in L.items() if k.endswith(("layers.0.linear1", "layers.0.linear2"))]
        return fmt(int(2 * sum(hits))) if hits else "n/a"
    enc0 = sum(enc_layer_flops().values()) - enc_layer_flops()["FFN"]
    layer_rows = "\n".join([
        f"| Linear (MLP fc1, 784→256) | {fmt(HAND['MLP'][1]['fc1 784->256'])} | {layer('MLP', 'net.1')} |",
        f"| Conv2d (CNN conv2, 16→32 @14×14) | {fmt(HAND['CNN'][1]['conv2 16->32 @14x14'])} | {layer('CNN', 'net.3')} |",
        f"| Self-attention (Attn layer 0, QKV+scores+AV+out) | {fmt(enc0)} | {layer('Attention', 'layers.0.self_attn')} |",
        f"| FFN (Attn layer 0, 64→128→64) | {fmt(enc_layer_flops()['FFN'])} | "
        f"{ffn_thop()} |",
    ])
    hist = "\n".join(f"| {n} | " + " | ".join(f"{h['val_acc']*100:.2f}" for h in r["history"]) + " |"
                     for n, r in R.items())
    out = (tpl.replace("{{MAIN_TABLE}}", rows).replace("{{LAYER_TABLE}}", layer_rows)
              .replace("{{HISTORY}}", hist)
              .replace("{{DEVICE}}", f"train on {DEVICE}, latency on {LAT_DEVICE}, "
                                     f"{torch.get_num_threads()} threads, torch {torch.__version__}"))
    open(LAB / "report.md", "w").write(out)


if __name__ == "__main__":
    main()
