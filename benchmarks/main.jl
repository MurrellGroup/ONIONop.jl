using NNop
using NNlib
using NNlib: ⊠, make_causal_mask, apply_attn_mask
using BenchmarkTools
using Zygote
using Random
using GPUArrays

import Adapt
import NNop.KernelAbstractions as KA

function naive_softmax(x; dims = 1)
    mx = maximum(x; dims)
    tmp = exp.(x .- mx)
    return tmp ./ sum(tmp; dims)
end

function att_padding_mask(kpadmask, other_dim; T = Float32)
    pm = T.(kpadmask)
    return NNop.CRC.@ignore_derivatives log.(
        reshape(pm, size(pm,1), 1, 1, size(pm,2)) .*
        (similar(pm, 1, other_dim, 1, size(pm,2)) .= 1)
    )
end

function naive_attention(
    q, k, v, pair = nothing;
    causal::Bool, kpad_mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
)
    kt = permutedims(k, (2, 1, 3, 4))
    a = (kt ⊠ q) .* Float32(inv(sqrt(size(q, 1))))
    if causal
        m = make_causal_mask(q)
        a = apply_attn_mask(a, m)
    end
    if !isnothing(kpad_mask)
        a = a .+ att_padding_mask(kpad_mask, size(q, 2))
    end
    if !isnothing(pair)
        a = a .+ permutedims(pair, (3, 2, 1, 4)) #When it comes in as H-QL-KL-B
    end
    return v ⊠ naive_softmax(a; dims=1)
end

function naive_rms_norm(x, w; offset::Float32 = 0f0, ϵ::Float32 = 1f-6)
    (w .+ offset) .* x ./ sqrt.(mean(x.^2; dims=1) .+ ϵ)
end

function naive_layer_norm(x, w, b; ϵ::Float32 = 1f-6)
    μ = mean(x; dims=1)
    σ² = var(x; mean=μ, dims=1, corrected=false)
    (x .- μ) ./ sqrt.(σ² .+ ϵ) .* w .+ b
end

function rotate_half(x)
    half_dim = size(x, 1) ÷ 2
    x1 = x[1:half_dim, :, :, :]
    x2 = x[(half_dim + 1):end, :, :, :]
    return vcat(-x2, x1)
end

function naive_llama_rope(q, k; cos, sin)
    cos = reshape(cos, size(cos)[1:2]..., 1, size(cos, 3))
    sin = reshape(sin, size(sin)[1:2]..., 1, size(sin, 3))
    q = q .* cos .+ rotate_half(q) .* sin
    k = k .* cos .+ rotate_half(k) .* sin
    return q, k
end

function test_layer_norm(kab)
    emb, n = 1024, 1024
    x = Adapt.adapt(kab, rand(Float32, emb, n))
    w = Adapt.adapt(kab, rand(Float32, emb))
    b = Adapt.adapt(kab, rand(Float32, emb))

    y1 = naive_layer_norm(x, w, b)
    y2 = NNop.layer_norm(x, w, b)
    @assert isapprox(y1, y2; atol=1f-6, rtol=1f-6)

    ∇n = Zygote.gradient(x, w, b) do x, w, b
        sum(naive_layer_norm(x, w, b))
    end
    ∇f = Zygote.gradient(x, w, b) do x, w, b
        sum(NNop.layer_norm(x, w, b))
    end

    @assert isapprox(∇n[1], ∇f[1]; atol=1f-6, rtol=1f-6)
    @assert isapprox(∇n[2], ∇f[2]; atol=1f-6, rtol=1f-6)
    @assert isapprox(∇n[3], ∇f[3]; atol=1f-6, rtol=1f-6)

    cache = GPUArrays.AllocCache()

    println("Naїve LayerNorm:")
    @btime GPUArrays.@cached $cache begin
        naive_layer_norm($x, $w, $b)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Fused LayerNorm:")
    @btime GPUArrays.@cached $cache begin
        NNop.layer_norm($x, $w, $b)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Naїve LayerNorm fwd + bwd:")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($x, $w, $b) do x, w, b
            sum(naive_layer_norm(x, w, b))
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Fused LayerNorm fwd + bwd:")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($x, $w, $b) do x, w, b
            sum(NNop.layer_norm(x, w, b))
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)
    return
end

function test_rms_norm(kab)
    emb, n = 1024, 1024
    x = Adapt.adapt(kab, rand(Float32, emb, n))
    w = Adapt.adapt(kab, rand(Float32, emb))

    y1 = naive_rms_norm(x, w; ϵ=1f-6)
    y2 = NNop.rms_norm(x, w; ϵ=1f-6)
    @assert y1 ≈ y2

    ∇n = Zygote.gradient(x, w) do x, w
        sum(naive_rms_norm(x, w; offset=1f0))
    end
    ∇f = Zygote.gradient(x, w) do x, w
        sum(NNop.rms_norm(x, w; offset=1f0))
    end
    @assert isapprox(∇n[1], ∇f[1]; atol=1f-6, rtol=1f-6)
    @assert isapprox(∇n[2], ∇f[2]; atol=1f-6, rtol=1f-6)

    cache = GPUArrays.AllocCache()

    println("Naїve RMS norm:")
    @btime GPUArrays.@cached $cache begin
        naive_rms_norm($x, $w)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Fused RMS norm:")
    @btime GPUArrays.@cached $cache begin
        NNop.rms_norm($x, $w)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Naїve RMS norm fwd + bwd:")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($x, $w) do x, w
            sum(naive_rms_norm(x, w))
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Fused RMS norm fwd + bwd:")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($x, $w) do x, w
            sum(NNop.rms_norm(x, w))
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)
    return
end

function test_llama_rope(kab)
    dim, n_heads, seq_len, batch = 64, 3, 1024, 4
    emb = NNop.LlamaRotaryEmbedding(dim)

    position_ids = reshape(collect(0f0:Float32(seq_len) - 1f0), :, 1)
    position_ids = repeat(position_ids; inner=(1, batch))

    cos, sin = emb(position_ids)
    cos = Adapt.adapt(kab, cos)
    sin = Adapt.adapt(kab, sin)
    q = Adapt.adapt(kab, rand(Float32, (dim, seq_len, n_heads, batch)))
    k = Adapt.adapt(kab, rand(Float32, (dim, seq_len, n_heads, batch)))

    q1, k1 = NNop.llama_rope(q, k; cos, sin)
    q2, k2 = naive_llama_rope(q, k; cos, sin)
    @assert q1 ≈ q2
    @assert k1 ≈ k2

    ∇1 = Zygote.gradient(q, k) do q, k
        qr, kr = NNop.llama_rope(q, k; cos, sin)
        sum(qr) + sum(kr)
    end
    ∇2 = Zygote.gradient(q, k) do q, k
        qr, kr = naive_llama_rope(q, k; cos, sin)
        sum(qr) + sum(kr)
    end
    @assert ∇1[1] ≈ ∇2[1]
    @assert ∇1[2] ≈ ∇2[2]

    GC.gc(false)
    GC.gc(true)

    cache = GPUArrays.AllocCache()

    println("Naїve Llama RoPE FWD:")
    @btime GPUArrays.@cached $cache begin
        naive_llama_rope($q, $k; cos=$cos, sin=$sin)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Fused Llama RoPe FWD:")
    @btime GPUArrays.@cached $cache begin
        NNop.llama_rope($q, $k; cos=$cos, sin=$sin)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Naїve Llama RoPE:")
    @btime GPUArrays.@cached $cache begin
        ∇ = Zygote.gradient($q, $k) do q, k
            qr, kr = naive_llama_rope(q, k; cos=$cos, sin=$sin)
            sum(qr) + sum(kr)
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Fused Llama RoPE FWD + BWD:")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($q, $k) do q, k
            qr, kr = NNop.llama_rope(q, k; cos=$cos, sin=$sin)
            sum(qr) + sum(kr)
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)
    return
end

function test_softmax(kab)
    for seq_len in (32, 33, 63, 255, 256, 512, 1024, 2048)
        x = Adapt.adapt(kab, rand(Float32, seq_len, 4))
        y1 = naive_softmax(x; dims=1)
        y2 = NNop.online_softmax(x)
        @assert y1 ≈ y2

        ∇1 = Zygote.gradient(x) do x
            sum(naive_softmax(x))
        end
        ∇2 = Zygote.gradient(x) do x
            sum(online_softmax(x))
        end
        @assert isapprox(∇1[1], ∇2[1]; atol=1f-6, rtol=1f-6)
    end

    x = Adapt.adapt(kab, rand(Float32, 8192, 1024))
    cache = GPUArrays.AllocCache()

    y1 = naive_softmax(x; dims=1)
    y2 = NNop.online_softmax(x)
    @assert y1 ≈ y2

    println("Naїve softmax:")
    @btime GPUArrays.@cached $cache begin
        naive_softmax($x; dims=1)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Online softmax:")
    @btime GPUArrays.@cached $cache begin
        NNop.online_softmax($x)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

struct BenchElementwiseProduct{F} end

NNop.pair_feature_dim(::BenchElementwiseProduct{F}) where F = F

@inline function NNop.pair_feature_tuple(
    ::BenchElementwiseProduct{F},
    qvals::NTuple{F, T},
    kvals::NTuple{F, T},
) where {F, T}
    return ntuple(i -> qvals[i] * kvals[i], Val(F))
end

@inline function NNop.pair_feature_tuple_pullback(
    ::BenchElementwiseProduct{F},
    qvals::NTuple{F, T},
    kvals::NTuple{F, T},
    dphi::NTuple{F, T},
) where {F, T}
    dq = ntuple(i -> dphi[i] * kvals[i], Val(F))
    dk = ntuple(i -> dphi[i] * qvals[i], Val(F))
    return dq, dk
end

function materialize_elementwise_product_pair_bias(q_features, k_features, pair_proj)
    # pair_proj: (QH, F), q_features/k_features: (F, L, B)
    # bias: (QH, QL, KL, B)
    pair = reshape(pair_proj, size(pair_proj, 1), size(pair_proj, 2), 1, 1, 1) .*
        reshape(q_features, 1, size(q_features, 1), size(q_features, 2), 1, size(q_features, 3)) .*
        reshape(k_features, 1, size(k_features, 1), 1, size(k_features, 2), size(k_features, 3))
    return dropdims(sum(pair; dims=2); dims=2)
end

function test_pair_feature_attention(kab)
    Random.seed!(0)
    T = Float32

    for (E, L, QH, KVH, B, F, causal) in [
        (64,  256,  8, 2, 4, 8, false),
        (64,  256,  8, 2, 4, 8, true),
        (64,  512,  8, 2, 4, 16, false),
        (64, 1024,  8, 2, 4, 16, false),
        (64, 2048,  4, 4, 2, 8, false),
    ]
        println("\n─── E=$E, L=$L, QH=$QH, KVH=$KVH, B=$B, F=$F, causal=$causal ───")

        q = Adapt.adapt(kab, randn(T, E, L, QH, B))
        k = Adapt.adapt(kab, randn(T, E, L, KVH, B))
        v = Adapt.adapt(kab, randn(T, E, L, KVH, B))
        q_features = Adapt.adapt(kab, randn(T, F, L, B))
        k_features = Adapt.adapt(kab, randn(T, F, L, B))
        pair_proj = Adapt.adapt(kab, randn(T, QH, F))
        pair_op = BenchElementwiseProduct{F}()

        # Materialized pair bias tensor size.
        pair_bytes = sizeof(T) * QH * L * L * B
        println("  Pair bias tensor: $(Base.format_bytes(pair_bytes))")

        # Warmup & correctness check.
        pair_mat = materialize_elementwise_product_pair_bias(q_features, k_features, pair_proj)
        o_mat = NNop.flash_attention(q, k, v, pair_mat; causal)
        o_fused = NNop.flash_attention(q, k, v, q_features, k_features, pair_proj, pair_op; causal)
        @assert isapprox(o_mat, o_fused; atol=1e-2, rtol=1e-2) "Forward mismatch"

        GC.gc(false); GC.gc(true)
        cache = GPUArrays.AllocCache()

        println("  Materialized FWD:")
        @btime GPUArrays.@cached $cache begin
            pair = materialize_elementwise_product_pair_bias($q_features, $k_features, $pair_proj)
            NNop.flash_attention($q, $k, $v, pair; causal=$causal)
            KA.synchronize($kab)
        end
        println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
        GPUArrays.unsafe_free!(cache)

        println("  Fused FWD:")
        @btime GPUArrays.@cached $cache begin
            NNop.flash_attention($q, $k, $v, $q_features, $k_features, $pair_proj, $pair_op; causal=$causal)
            KA.synchronize($kab)
        end
        println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
        GPUArrays.unsafe_free!(cache)

        println("  Materialized FWD + BWD:")
        @btime GPUArrays.@cached $cache begin
            Zygote.gradient($q, $k, $v, $q_features, $k_features, $pair_proj) do q, k, v, qf, kf, pp
                pair = materialize_elementwise_product_pair_bias(qf, kf, pp)
                sum(NNop.flash_attention(q, k, v, pair; causal=$causal))
            end
            KA.synchronize($kab)
        end
        println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
        GPUArrays.unsafe_free!(cache)

        println("  Fused FWD + BWD:")
        @btime GPUArrays.@cached $cache begin
            Zygote.gradient($q, $k, $v, $q_features, $k_features, $pair_proj) do q, k, v, qf, kf, pp
                sum(NNop.flash_attention(q, k, v, qf, kf, pp, $pair_op; causal=$causal))
            end
            KA.synchronize($kab)
        end
        println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
        GPUArrays.unsafe_free!(cache)
    end
    return
end

struct BenchDirLeakyReLUDist{T}
    negative_slope::T
    eps::T
end

NNop.pair_feature_dim(::BenchDirLeakyReLUDist) = 9

@inline function _bench_lrelu(slope::T, delta::T) where T
    return ifelse(delta ≥ zero(T), delta, slope * delta)
end

@inline function NNop.pair_feature_tuple(
    op::BenchDirLeakyReLUDist{T},
    qvals::NTuple{4, T},
    kvals::NTuple{4, T},
) where T
    d1 = qvals[1] - kvals[1]
    d2 = qvals[2] - kvals[2]
    d3 = qvals[3] - kvals[3]
    d4 = qvals[4] - kvals[4]
    s = op.negative_slope
    r = sqrt(d1 * d1 + d2 * d2 + d3 * d3 + op.eps)
    return (
        _bench_lrelu(s, d1), _bench_lrelu(s, -d1),
        _bench_lrelu(s, d2), _bench_lrelu(s, -d2),
        _bench_lrelu(s, d3), _bench_lrelu(s, -d3),
        _bench_lrelu(s, d4), _bench_lrelu(s, -d4),
        r,
    )
end

function materialize_dir_lrelu_dist_pair_bias(q_features, k_features, pair_proj, negative_slope, eps)
    lrelu(x) = ifelse.(x .>= zero(eltype(x)), x, negative_slope .* x)
    deltas = map(1:4) do f
        reshape(q_features[f, :, :], 1, size(q_features, 2), 1, size(q_features, 3)) .-
            reshape(k_features[f, :, :], 1, 1, size(k_features, 2), size(k_features, 3))
    end
    r = sqrt.(deltas[1] .^ 2 .+ deltas[2] .^ 2 .+ deltas[3] .^ 2 .+ eps)
    features = [
        lrelu(deltas[1]), lrelu(-deltas[1]),
        lrelu(deltas[2]), lrelu(-deltas[2]),
        lrelu(deltas[3]), lrelu(-deltas[3]),
        lrelu(deltas[4]), lrelu(-deltas[4]),
        r,
    ]
    return sum(reshape(pair_proj[:, p], :, 1, 1, 1) .* features[p] for p in 1:9)
end

function test_pair_feature_dir_lrelu_dist(kab)
    Random.seed!(0)
    T = Float32
    E, L, QH, KVH, B = 64, 4096, 4, 4, 4
    F = 4          # input feature dim
    PD = 9         # pair feature dim (2*F + 1)
    negative_slope = T(0.1)
    eps = T(1e-4)
    causal = false

    println("\n═══ Directional LeakyReLU + Distance pair features ═══")
    println("  E=$E, L=$L, QH=$QH, KVH=$KVH, B=$B, F=$F, PD=$PD, causal=$causal")

    q = Adapt.adapt(kab, randn(T, E, L, QH, B))
    k = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    v = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    q_features = Adapt.adapt(kab, randn(T, F, L, B))
    k_features = Adapt.adapt(kab, randn(T, F, L, B))
    pair_proj = Adapt.adapt(kab, randn(T, QH, PD))
    pair_op = BenchDirLeakyReLUDist{T}(negative_slope, eps)

    pair_bytes = sizeof(T) * QH * L * L * B
    println("  Materialized pair bias tensor: $(Base.format_bytes(pair_bytes))")

    # Warmup + correctness.
    pair_mat = materialize_dir_lrelu_dist_pair_bias(q_features, k_features, pair_proj, negative_slope, eps)
    o_mat = NNop.flash_attention(q, k, v, pair_mat; causal)
    o_fused = NNop.flash_attention(q, k, v, q_features, k_features, pair_proj, pair_op; causal)
    @assert isapprox(o_mat, o_fused; atol=1e-2, rtol=1e-2) "Forward mismatch"

    GC.gc(false); GC.gc(true)
    cache = GPUArrays.AllocCache()

    println("\n  Materialized FWD:")
    @btime GPUArrays.@cached $cache begin
        pair = materialize_dir_lrelu_dist_pair_bias($q_features, $k_features, $pair_proj, $negative_slope, $eps)
        NNop.flash_attention($q, $k, $v, pair; causal=$causal)
        KA.synchronize($kab)
    end
    println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("  Fused FWD:")
    @btime GPUArrays.@cached $cache begin
        NNop.flash_attention($q, $k, $v, $q_features, $k_features, $pair_proj, $pair_op; causal=$causal)
        KA.synchronize($kab)
    end
    println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("\n  Materialized FWD + BWD (all grads):")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($q, $k, $v, $q_features, $k_features, $pair_proj) do q, k, v, qf, kf, pp
            pair = materialize_dir_lrelu_dist_pair_bias(qf, kf, pp, $negative_slope, $eps)
            sum(NNop.flash_attention(q, k, v, pair; causal=$causal))
        end
        KA.synchronize($kab)
    end
    println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("  Fused FWD + BWD (proj grads only, feature_grads=false):")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($q, $k, $v, $q_features, $k_features, $pair_proj) do q, k, v, qf, kf, pp
            sum(NNop.flash_attention(q, k, v, qf, kf, pp, $pair_op; causal=$causal, feature_grads=false))
        end
        KA.synchronize($kab)
    end
    println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("  Fused FWD + BWD (all grads, feature_grads=true):")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($q, $k, $v, $q_features, $k_features, $pair_proj) do q, k, v, qf, kf, pp
            sum(NNop.flash_attention(q, k, v, qf, kf, pp, $pair_op; causal=$causal, feature_grads=true))
        end
        KA.synchronize($kab)
    end
    println("   - Peak memory: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

function test_flash_attention(kab)
    Random.seed!(0)
    T = Float16
    E, QL, KL, H, B = 64, 2048, 2048, 4, 4
    i = 0
    for causal in (false, true), use_padmask in (false, true), use_pair in (false, true)
        i += 1
        println("\n$i | Causal: $causal, use_padmask: $use_padmask, use_pair: $use_pair")
        q = Adapt.adapt(kab, randn(T, E, QL, H, B))
        k = Adapt.adapt(kab, randn(T, E, KL, H, B))
        v = Adapt.adapt(kab, randn(T, E, KL, H, B))

        pm = nothing
        if use_padmask
            pm = Adapt.adapt(kab, ones(Bool, KL, B))
            pm[end-10:end, end] .= false
        end

        pair = nothing
        if use_pair
            pair = Adapt.adapt(kab, randn(T, H, QL, KL, B))
        end

        on = naive_attention(q, k, v, pair; causal, kpad_mask=pm)
        o = NNop.flash_attention(q, k, v, pair; causal, kpad_mask=pm)
        @assert on ≈ o

        ∇1 = Zygote.gradient(q, k, v, pair) do q, k, v, pair
            sum(naive_attention(q, k, v, pair; causal, kpad_mask=pm))
        end
        ∇2 = Zygote.gradient(q, k, v, pair) do q, k, v, pair
            sum(NNop.flash_attention(q, k, v, pair; causal, kpad_mask=pm))
        end

        @assert isapprox(∇1[1], ∇2[1]; atol=1e-2, rtol=1e-2)
        @assert isapprox(∇1[2], ∇2[2]; atol=1e-2, rtol=1e-2)
        @assert isapprox(∇1[3], ∇2[3]; atol=1e-2, rtol=1e-2)
        !isnothing(pair) && @assert isapprox(∇1[4], ∇2[4]; atol=1e-2, rtol=1e-2)

        GC.gc(false)
        GC.gc(true)

        cache = GPUArrays.AllocCache()

        println("Naїve attention FWD:")
        @btime GPUArrays.@cached $cache begin
            naive_attention($q, $k, $v, $pair; causal=$causal, kpad_mask=$pm)
            KA.synchronize($kab)
        end
        println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
        GPUArrays.unsafe_free!(cache)

        println("Flash attention FWD:")
        @btime GPUArrays.@cached $cache begin
            NNop.flash_attention($q, $k, $v, $pair; causal=$causal, kpad_mask=$pm)
            KA.synchronize($kab)
        end
        println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
        GPUArrays.unsafe_free!(cache)

        println("Naїve attention FWD + BWD:")
        @btime GPUArrays.@cached $cache begin
            ∇ = Zygote.gradient($q, $k, $v, $pair) do q, k, v, pair
                sum(naive_attention(q, k, v, $pair; causal=$causal, kpad_mask=$pm))
            end
            KA.synchronize($kab)
        end
        println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
        GPUArrays.unsafe_free!(cache)

        println("Flash attention FWD + BWD:")
        @btime GPUArrays.@cached $cache begin
            Zygote.gradient($q, $k, $v, $pair) do q, k, v, pair
                sum(NNop.flash_attention(q, k, v, $pair; causal=$causal, kpad_mask=$pm))
            end
            KA.synchronize($kab)
        end
        println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
        GPUArrays.unsafe_free!(cache)
    end
    return
end
