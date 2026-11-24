# Lab 5 — Frameworks and automatic differentiation

Reading: Reddi, Ch. 7 (§7.3, §7.4, §7.6, §§7.7–7.8).

Code: `src/engine.py` (Tasks 1–3, scalar autograd engine), `src/lab5.py` (Tasks 3–6), `lab5.ipynb` (runs everything in order).
Run: `cd lab05 && python src/lab5.py` → prints all results and writes `results/results.json` and `results/task5_moons.png`.

## Tasks 1–3: `Value` engine

- `Value` stores `data`, `grad`, parents (`_prev`), the op name and `_backward`, a closure holding the **local derivative rule**.
- `+`, `*`, `**k`, `relu` (plus `exp`, `log`, `sigmoid` for Task 5) each create a new node during the forward pass, so the graph is built while computing.
- `backward()`: iterative topological sort → output gradient set to 1 → nodes visited in reverse topological order → every rule uses `+=`, so gradients are **accumulated**, never overwritten.
- Accumulation test: `y = x*x + x` at x = 3 gives dy/dx = 7 (correct), and `a + a` gives 2. Both pass.

## Task 4: verification on `((a*b + c)**2).relu().sum()`

10-element vectors (seed 42), float64. Gradients are compared with PyTorch autograd and with central differences (h = 1e-6). Pass criterion: max absolute difference < 1e-6.
Our engine matches the analytic gradient `2(ab+c)·b` with a difference of 0.0. The PyTorch comparison table is printed by `task4()`.
Note: `(a*b+c)**2` is never negative, so the ReLU does not change the value here; its gradient is 1 everywhere except at exactly 0.

## Task 5: make_moons, 2 → 8 (ReLU) → 1 (sigmoid), BCE

100 samples (noise 0.1, standardised), full-batch SGD, lr 0.5, 100 epochs. Both engines start from **identical weights**, so their loss curves should coincide.
Our engine: loss 0.731 → 0.155, training accuracy 93 %. The speed comparison (median of 5 full training runs; PyTorch with 1 thread) is printed by `task5()`.
The pure-Python engine is expected to be orders of magnitude slower, because every scalar is a Python object and every operation is a Python call, whereas PyTorch runs vectorised C++ kernels on whole tensors.

## Task 6: graph, eager vs compiled, tensor layout

Model: `Linear(256,256) → GELU → h*sigmoid(h) + x → LayerNorm → Linear → ReLU`.
- Graph inspection: `torch.fx.symbolic_trace` (static graph) and a walk over `grad_fn.next_functions` (the dynamic autograd graph built during the eager forward).
- Eager vs `torch.compile`, falling back to `torch.jit.trace` + `freeze` if compile is unsupported. Measured after warm-up for batch sizes 1, 64, 1024 and 8192, plus one training step. The compile time is reported separately.
- Tensor properties: shape, dtype, stride, `is_contiguous()` for the input, the weight, `weight.T` and `x[:, ::2]`.
- Non-contiguous views: `x*2+1`, `sum(dim=1)` and `exp()` on a 4096×4096 tensor vs its `.t()` view, a strided slice vs a contiguous tensor of the same size, and the cost of `.contiguous()`.

## When does compilation help?

Eager mode launches every operation separately from Python, and every intermediate result goes through memory. `torch.compile` captures the forward as one graph, fuses chains of element-wise ops into a few kernels and removes Python overhead. `jit.trace` records one run as a static graph.

**Compilation helps** when:
- the same model runs many times with the same shapes (serving, long training), so the one-off compile cost is paid back;
- the model has many small or element-wise, memory-bound ops that can be fused;
- batches are medium to large.

**Compilation does not help (or hurts)** when:
- the workload is a short script or debugging, where compile time dominates;
- the work is dominated by large matmuls, which already run in optimised BLAS;
- shapes change or the code has data-dependent control flow, which causes recompiles or graph breaks (`jit.trace` silently freezes one branch);
- the input is very small, so fixed overhead dominates.

**Non-contiguous tensors:** `.t()` and `[:, ::2]` are views with different strides, so their memory is accessed with jumps. That uses the cache poorly, so ops on them are slower. `.contiguous()` costs one copy but makes repeated later ops fast.
