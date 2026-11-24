"""Hand-calculated parameters and FLOPs (Reddi Ch. 4 formulas). No torch needed.

FLOP convention: 1 multiply-accumulate (MAC) = 2 FLOPs.
  Linear : 2 * in * out                     (per sample / per token)
  Conv2d : 2 * K * K * C_in * C_out * H_out * W_out
  Attn   : QKV 2*seq*d*3d, scores 2*seq^2*d, attn@V 2*seq^2*d, out-proj 2*seq*d*d
Bias adds, ReLU, pooling, softmax, LayerNorm are ignored (O(n), negligible).
"""

# ---------------- parameters ----------------
def lin_p(i, o): return i * o + o
def conv_p(k, ci, co): return k * k * ci * co + co

def mlp_params():
    return lin_p(784, 256) + lin_p(256, 128) + lin_p(128, 10)

def cnn_params():
    return conv_p(3, 1, 16) + conv_p(3, 16, 32) + lin_p(32 * 7 * 7, 10)

def enc_layer_params(d=64, ff=128):
    return (lin_p(d, 3 * d)      # in_proj (Q,K,V)
            + lin_p(d, d)        # out_proj
            + lin_p(d, ff) + lin_p(ff, d)  # FFN
            + 2 * 2 * d)         # 2 LayerNorms (gamma, beta)

def attn_params(seq=16, patch=49, d=64, ff=128, layers=2):
    return (lin_p(patch, d)                    # patch embedding
            + layers * enc_layer_params(d, ff) + lin_p(d, 10))

# ---------------- FLOPs ----------------
def lin_f(i, o, tokens=1): return 2 * i * o * tokens
def conv_f(k, ci, co, h, w): return 2 * k * k * ci * co * h * w

def mlp_flops():
    layers = {"fc1 784->256": lin_f(784, 256), "fc2 256->128": lin_f(256, 128),
              "fc3 128->10": lin_f(128, 10)}
    return layers

def cnn_flops():
    return {"conv1 1->16 @28x28": conv_f(3, 1, 16, 28, 28),
            "conv2 16->32 @14x14": conv_f(3, 16, 32, 14, 14),
            "fc 1568->10": lin_f(1568, 10)}

def enc_layer_flops(seq=16, d=64, ff=128):
    return {"QKV proj": 2 * seq * d * 3 * d,
            "scores QK^T": 2 * seq * seq * d,
            "attn @ V": 2 * seq * seq * d,
            "out proj": 2 * seq * d * d,
            "FFN": lin_f(d, ff, seq) + lin_f(ff, d, seq)}

def attn_flops(seq=16, d=64, ff=128, layers=2):
    out = {"patch proj 49->64": lin_f(49, d, seq)}
    for l in range(layers):
        for k, v in enc_layer_flops(seq, d, ff).items():
            out[f"enc{l} {k}"] = v
    out["head 64->10"] = lin_f(d, 10)
    return out

HAND = {
    "MLP": (mlp_params(), mlp_flops()),
    "CNN": (cnn_params(), cnn_flops()),
    "Attention": (attn_params(), attn_flops()),
}

if __name__ == "__main__":
    for name, (p, f) in HAND.items():
        print(f"\n{name}: params={p:,}  FLOPs={sum(f.values()):,}  MACs={sum(f.values())//2:,}")
        for k, v in f.items():
            print(f"   {k:28s} {v:>12,}")
    print("\nOne encoder layer total FLOPs:", f"{sum(enc_layer_flops().values()):,}")
