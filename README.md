# NNop.jl

<img src="assets/nnop-logo.svg" width="200"/>

[buildkite-img]: https://badge.buildkite.com/b30cae2b9773cfd3464e6dad35de6a4a7151a6cb161da14c33.svg?branch=master
[buildkite-img-amdgpu]: https://badge.buildkite.com/b30cae2b9773cfd3464e6dad35de6a4a7151a6cb161da14c33.svg?branch=master&step=AMDGPU%20-%20Julia%201.12
[buildkite-img-cuda]: https://badge.buildkite.com/b30cae2b9773cfd3464e6dad35de6a4a7151a6cb161da14c33.svg?branch=master&step=CUDA%20-%20Julia%201.12
[buildkite-url]: https://buildkite.com/julialang/nnop-dot-jl

Fused kernels (with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) integration):

- [Flash Attention](#flash-attention)
- [Softmax](#softmax)
- [RMS Norm](#rms-norm)
- [Layer Norm](#layer-norm)
- [Llama RoPE](#llama-rope)

|GPU Backend|CI Status|
|-|-|
|[AMDGPU](https://github.com/JuliaGPU/AMDGPU.jl)|[![][buildkite-img-amdgpu]][buildkite-url]|
|[CUDA](https://github.com/JuliaGPU/CUDA.jl)|[![][buildkite-img-cuda]][buildkite-url]|

## Benchmarking

See `benchmarks/main.jl` for comparison scripts between naїve & fused versions.

## Flash Attention

Implementation of [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).

```julia
E, L, H, B = 64, 4096, 4, 4
causal = false

q = ROCArray(rand(Float32, E, L, H, B))
k = ROCArray(rand(Float32, E, L, H, B))
v = ROCArray(rand(Float32, E, L, H, B))

o = NNop.flash_attention(q, k, v; causal)
∇ = Zygote.gradient(q, k, v) do q, k, v
    sum(NNop.flash_attention(q, k, v; causal))
end
```

### Features:

- Forward & backward passes.
- Arbitrary sequence length.
- FP32, FP16, BFP16 support.
- Variable sequence length.
- Causal masking.

### Fused pair-feature bias

`flash_attention` can build a head-wise pair bias on the fly from query/key features
without materializing either the pair feature tensor or the final bias tensor.

The bias for head `h`, query position `i`, key position `j`, batch `b` is:

```
bias[h, i, j, b] = sum(pair_proj[h, p] * phi_p(q_features[:, i, b], k_features[:, j, b]) for p in 1:d_pf)
```

where `phi` is a user-defined pair feature function. To define one, implement an
`isbits` struct and three methods:

```julia
struct MyPairOp{T}
    alpha::T
end

NNop.pair_feature_dim(::MyPairOp) = 2

@inline function NNop.pair_feature_tuple(
    op::MyPairOp{T},
    qvals::NTuple{2, T},
    kvals::NTuple{2, T},
) where T
    dx = qvals[1] - kvals[1]
    dy = qvals[2] - kvals[2]
    return (dx + dy, dx * dx + dy * dy)
end

@inline function NNop.pair_feature_tuple_pullback(
    op::MyPairOp{T},
    qvals::NTuple{2, T},
    kvals::NTuple{2, T},
    dphi::NTuple{2, T},
) where T
    dx = qvals[1] - kvals[1]
    dy = qvals[2] - kvals[2]
    dq = (dphi[1] + 2 * dx * dphi[2], dphi[1] + 2 * dy * dphi[2])
    dk = (-dq[1], -dq[2])
    return dq, dk
end
```

Then call `flash_attention` with the pair op:

```julia
using CUDA
using NNop

T = Float32
E, L, QH, KVH, B = 64, 128, 8, 2, 4

q = CUDA.randn(T, E, L, QH, B)
k = CUDA.randn(T, E, L, KVH, B)
v = CUDA.randn(T, E, L, KVH, B)

q_features = CUDA.randn(T, 2, L, B)
k_features = CUDA.randn(T, 2, L, B)
pair_proj = CUDA.randn(T, QH, 2)     # (QH, pair_feature_dim)

pair_op = MyPairOp{T}(0.5f0)
o = NNop.flash_attention(q, k, v, q_features, k_features, pair_proj, pair_op; causal=false)
```

#### Interface

- `pair_feature_dim(pair_op)` — number of pair features (compile-time constant via type parameter).
- `pair_feature_tuple(pair_op, qvals, kvals)` — returns `NTuple{d_pf, T}`.
  `qvals`/`kvals` are `NTuple{F, T}` where `F = size(q_features, 1)`.
- `pair_feature_tuple_pullback(pair_op, qvals, kvals, dphi)` — returns `(dq, dk)` tuples
  of length `F`. Must be mathematically consistent with `pair_feature_tuple`.
  Only required when `feature_grads=true`.

#### Constraints

- `pair_op` must be `isbits`. All methods must be GPU-safe (no allocation, no dynamic dispatch).
- `pair_proj` must be `(QH, d_pf)`, where `d_pf = NNop.pair_feature_dim(pair_op)`.
- `q_features` and `k_features` must be `(F, seq_len, batch)`. All `F` features are loaded
  densely into shared memory — select which features to pass by slicing the array.
- Gradients always flow through `q`, `k`, `v`, and `pair_proj`.
- Gradients for `q_features`/`k_features` are opt-in: pass `feature_grads=true`.
  This requires implementing `pair_feature_tuple_pullback`.
- Gradients do not flow through fields inside `pair_op`.

## Softmax

Implementation of [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).

```julia
x = ROCArray(rand(Float32, 8192, 1024))
y = NNop.online_softmax(x)
```

## RMS Norm

```julia
x = ROCArray(rand(Float32, 1024, 1024))
w = ROCArray(rand(Float32, 1024))
y = NNop.rms_norm(x, w)
∇ = Zygote.gradient(x, w) do x, w
    sum(NNop.rms_norm(x, w))
end
```

## Layer Norm

```julia
x = ROCArray(rand(Float32, 1024, 1024))
w = ROCArray(rand(Float32, 1024))
w = ROCArray(rand(Float32, 1024))
y = NNop.layer_norm(x, w)
∇ = Zygote.gradient(x, w, b) do x, w, b
    sum(NNop.layer_norm(x, w, b))
end
```

## Llama RoPE

```julia
E, L, B = 16, 1024, 1
QH, KH = 16, 16

emb = NNop.LlamaRotaryEmbedding(E)
position_ids = reshape(collect(0f0:Float32(L) - 1f0), :, 1)
position_ids = repeat(position_ids; inner=(1, B))

cos, sin = emb(position_ids)
cos = Adapt.adapt(kab, cos)
sin = Adapt.adapt(kab, sin)

q = Adapt.adapt(kab, ones(Float32, (E, L, QH, B)))
k = Adapt.adapt(kab, ones(Float32, (E, L, KH, B)))
q, k = NNop.llama_rope(q, k; cos, sin)
```
