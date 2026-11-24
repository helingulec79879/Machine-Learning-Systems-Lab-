# Lab 3 — Architectures and their costs

Fashion-MNIST, `ToTensor()` + `Normalize((0.2860,), (0.3530,))`, official 60k train split into 55k train / 5k validation (seed 0), official 10k test. Batch 128, Adam lr 1e-3, 5 epochs. Setup: {{DEVICE}}.

Code is in `src/` (`lab3.py` = data, models, cost measurement, training, plots; `flops_hand.py` = hand calculations). Handwritten FLOP/parameter calculations are in `handwritten/`.
Run `cd lab03 && python src/lab3.py` → writes `results/results.json`, the plots in `results/`, and regenerates this report.

## 1. Models

| Model | Definition | Params (hand) |
|---|---|---|
| MLP | Flatten 784 → Linear 256 → ReLU → Linear 128 → ReLU → Linear 10 | 235,146 |
| CNN | Conv(1,16,3,p1) → ReLU → MaxPool2 → Conv(16,32,3,p1) → ReLU → MaxPool2 → Linear(1568,10) | 20,490 |
| Attention | 16 patches of 7×7 → Linear(49,64) → 2 × TransformerEncoderLayer(d=64, h=4, ff=128) → mean pool → Linear(64,10) | 70,794 |

**Note on the 100k–150k target.** The architectures as specified do not land in that range: MLP has 235k parameters (fc1 alone is 200,960), the CNN has only 20k (convolutions share weights), and the attention model has 71k. I kept the specified architectures exactly, because changing them would break the hand-calculation in §3 against the lab spec; the plots use a log x-axis for parameters so the 10× spread stays readable. As specified there is no positional embedding, so the encoder plus mean pooling is permutation-invariant over patches: the model cannot tell where a patch came from, which limits its accuracy.

Parameter breakdown:
- MLP: 784·256+256 = 200,960; 256·128+128 = 32,896; 128·10+10 = 1,290.
- CNN: 3·3·1·16+16 = 160; 3·3·16·32+32 = 4,640; 1568·10+10 = 15,690.
- Attention: patch proj 49·64+64 = 3,200; per encoder layer: in_proj 64·192+192 = 12,480, out_proj 4,160, FFN 8,320 + 8,256, two LayerNorms 256 → 33,472; ×2 = 66,944; head 650.

## 2. Hand-calculated FLOPs (per image, 1 MAC = 2 FLOPs)

Handwritten version (on paper): [`handwritten/flops_hand_calculation.jpg`](handwritten/flops_hand_calculation.jpg)

<img src="handwritten/flops_hand_calculation.jpg" width="500" alt="Handwritten FLOP calculation">

| Model | Layer | Formula | FLOPs |
|---|---|---|---|
| MLP | fc1 | 2·784·256 | 401,408 |
| MLP | fc2 | 2·256·128 | 65,536 |
| MLP | fc3 | 2·128·10 | 2,560 |
| **MLP** | **total** | | **469,504** |
| CNN | conv1 | 2·3·3·1·16·28·28 | 225,792 |
| CNN | conv2 | 2·3·3·16·32·14·14 | 1,806,336 |
| CNN | fc | 2·1568·10 | 31,360 |
| **CNN** | **total** | | **2,063,488** |
| Attn | patch proj | 2·16·49·64 | 100,352 |
| Attn | QKV proj (per layer) | 2·seq·d·3d = 2·16·64·192 | 393,216 |
| Attn | scores QKᵀ (per layer) | 2·seq²·d = 2·256·64 | 32,768 |
| Attn | attn·V (per layer) | 2·seq²·d | 32,768 |
| Attn | out proj (per layer) | 2·16·64·64 | 131,072 |
| Attn | FFN (per layer) | 2·16·64·128 ·2 | 524,288 |
| Attn | one encoder layer | | 1,114,112 |
| Attn | head | 2·64·10 | 1,280 |
| **Attn** | **total** (2 layers) | | **2,329,856** |

Ignored as negligible: bias adds, ReLU, max-pool, softmax, LayerNorm, mean pooling (all linear in activation size).

## 3. Hand vs measured (thop)

Totals (thop reports MACs; ×2 to compare with FLOPs). Latency = median of 100 runs (batch 1) and 20 runs (batch 128) after 10 warm-up runs, `eval()` + `no_grad()`, CPU, fixed thread count.

| Model | Params (torchinfo) | Hand FLOPs | thop MACs | thop ×2 | Lat b1 (ms) | Lat b128 (ms) | Throughput b128 (img/s) | Test acc (%) |
|---|---|---|---|---|---|---|---|---|
{{MAIN_TABLE}}

One layer of each type:

| Layer | Hand FLOPs | thop ×2 |
|---|---|---|
{{LAYER_TABLE}}

**Explaining the gap.**
1. *MACs vs FLOPs.* thop counts multiply-accumulates, so its raw number is half of the Ch. 4 FLOP formula. After ×2, the Linear and Conv2d layers should match the hand numbers exactly (thop does not count the bias for these layers either).
2. *Attention is undercounted.* thop counts operations only for module types it has a rule for. `nn.MultiheadAttention` has no rule, and its `out_proj` is a `NonDynamicallyQuantizableLinear` (a subclass, so thop's exact-type lookup misses it). The QKV projection, QKᵀ and attn·V are computed inside `F.multi_head_attention_forward` as tensor ops, not as modules, so they are invisible. That is 589,824 FLOPs per layer (53% of an encoder layer) thop cannot see. In `eval()` PyTorch may also take the fused "fast path" for `TransformerEncoderLayer`, which bypasses `linear1`/`linear2` too — in that case thop sees only the patch projection and the head.
3. *Ops thop ignores.* Softmax, LayerNorm, ReLU, pooling and the `unfold`/reshape/mean in my `forward` are zero or tiny in thop; they are also excluded from my hand count, so they do not explain the gap but they do mean neither number is "total work".

Conclusion: for MLP/CNN, thop ×2 ≈ hand; for the transformer the hand formula is the trustworthy number, and profiler output must be checked against an analytical count.

## 4. Training (validation accuracy per epoch, %)

| Model | Ep 1 | Ep 2 | Ep 3 | Ep 4 | Ep 5 |
|---|---|---|---|---|---|
{{HISTORY}}

## 5. Trade-off plots

![Accuracy vs parameters](results/acc_vs_params.png)

![Accuracy vs latency (batch 1)](results/acc_vs_latency.png)

![Accuracy vs latency (batch 128)](results/acc_vs_latency_b128.png)

## 6. Which architecture is most cost-effective? (Ch. 4.6–4.7 access patterns)

The three models sit at very different points in the FLOPs-vs-parameters plane, and that is what the access-pattern view explains:

| | Params | FLOPs | FLOPs / param (reuse) | Dominant access pattern |
|---|---|---|---|---|
| MLP | 235k | 0.47M | ≈2 | Dense GEMV: every weight read once per image, used once |
| CNN | 20k | 2.06M | ≈100 | Sliding window: each 3×3 kernel reused at every spatial position |
| Attention | 71k | 2.33M | ≈32 | Token-parallel GEMMs + all-to-all seq² score matrix |

**MLP — memory-bound.** Each weight is touched once per sample (arithmetic intensity ≈ 2 FLOPs/param). At batch 1 the work is a matrix-vector product, so latency is set by streaming ~0.94 MB of fp32 weights, not by compute. It has the fewest FLOPs, so it is usually the fastest at batch 1, but it has the most parameters and its accuracy is limited because flattening throws away spatial locality — the model has to learn shift-tolerance from data. Batching turns GEMV into GEMM and amortises the weight reads, so its throughput scales well with batch 128.

**CNN — compute-bound, high reuse.** Weight sharing gives ~100 FLOPs per parameter: the whole model (80 KB) fits in L1/L2 cache and the kernels are reused across all 784 then 196 positions. The memory traffic is dominated by activations (16×28×28 feature maps), which have strong spatial locality and are consumed immediately by pooling. The locality bias matches images, so it gets the best accuracy with ~11× fewer parameters than the MLP. It costs ~4× the MLP's FLOPs, so at batch 1 it is typically somewhat slower on CPU, but it has by far the best accuracy per parameter and per byte moved.

**Attention — most FLOPs, irregular access.** Most of the cost is the token-wise GEMMs (QKV, out-proj, FFN = 96% of encoder FLOPs); the seq² score matrix is small here because seq = 16. But the block structure (reshape into heads, softmax, LayerNorm, residuals, transposes) issues many small kernels with data movement between them, so latency is higher than the FLOP count alone predicts — at batch 1 kernel-launch and memory overheads dominate. With only 16 patches and 5 epochs, the model lacks the CNN's built-in locality and needs more data/epochs to catch up; on 28×28 images it is the least cost-effective.

**Verdict.** Read the measured table above for the exact numbers; on this task the CNN is the most cost-effective architecture: best (or near-best) accuracy with the smallest parameter/memory footprint, and its high weight reuse keeps it efficient as batch size grows. The MLP is the choice only if batch-1 latency is the single constraint and a ~1–3 point accuracy drop is acceptable. The attention model pays for generality (global all-to-all mixing) that 28×28 grayscale images do not need.

## 7. Reproducibility

Seeds fixed (python, numpy, torch, split generator). Latency measured on CPU with a fixed thread count, 10 warm-up runs, median over 100 (batch 1) / 20 (batch 128) runs. Hand calculations live in `src/flops_hand.py` (`python src/flops_hand.py` prints them).
