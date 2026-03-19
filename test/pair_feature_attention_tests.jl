@testitem "Flash Attention Pair Features" setup=[TSCore] begin

import Adapt
import Zygote
using Test

# ── Test-local pair ops ───────────────────────────────────────────────

struct TestElementwiseProduct{F} end

NNop.pair_feature_dim(::TestElementwiseProduct{F}) where F = F

@inline function NNop.pair_feature_tuple(
    ::TestElementwiseProduct{F},
    qvals::NTuple{F, T},
    kvals::NTuple{F, T},
) where {F, T}
    return ntuple(i -> qvals[i] * kvals[i], Val(F))
end

@inline function NNop.pair_feature_tuple_pullback(
    ::TestElementwiseProduct{F},
    qvals::NTuple{F, T},
    kvals::NTuple{F, T},
    dphi::NTuple{F, T},
) where {F, T}
    dq = ntuple(i -> dphi[i] * kvals[i], Val(F))
    dk = ntuple(i -> dphi[i] * qvals[i], Val(F))
    return dq, dk
end

struct TestEuclideanDistance{T}
    gamma::T
    eps::T
end

NNop.pair_feature_dim(::TestEuclideanDistance) = 4

@inline function NNop.pair_feature_tuple(
    op::TestEuclideanDistance,
    qvals::NTuple{3, T},
    kvals::NTuple{3, T},
) where T
    dx = qvals[1] - kvals[1]
    dy = qvals[2] - kvals[2]
    dz = qvals[3] - kvals[3]
    r2 = dx * dx + dy * dy + dz * dz
    r = sqrt(r2 + op.eps)
    return (r, r2, exp(-op.gamma * r2), one(T))
end

@inline function NNop.pair_feature_tuple_pullback(
    op::TestEuclideanDistance{T},
    qvals::NTuple{3, T},
    kvals::NTuple{3, T},
    dphi::NTuple{4, T},
) where T
    dx = qvals[1] - kvals[1]
    dy = qvals[2] - kvals[2]
    dz = qvals[3] - kvals[3]
    r2 = dx * dx + dy * dy + dz * dz
    r = sqrt(r2 + op.eps)
    radial = dphi[1] / r + 2 * dphi[2] - 2 * op.gamma * dphi[3] * exp(-op.gamma * r2)
    dq = (radial * dx, radial * dy, radial * dz)
    dk = (-dq[1], -dq[2], -dq[3])
    return dq, dk
end

# 4 linear features → directional leaky relu on each + euc distance on first 3.
# phi = (lrelu(d1), lrelu(-d1), ..., lrelu(d4), lrelu(-d4), sqrt(d1²+d2²+d3²+eps))
# where d_f = q_f - k_f. Output dim = 2*4 + 1 = 9.
struct TestDirectionalLeakyReLUDistance{T}
    negative_slope::T
    eps::T
end

NNop.pair_feature_dim(::TestDirectionalLeakyReLUDistance) = 9

@inline function _lrelu(slope::T, delta::T) where T
    return ifelse(delta ≥ zero(T), delta, slope * delta)
end

@inline function NNop.pair_feature_tuple(
    op::TestDirectionalLeakyReLUDistance,
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
        _lrelu(s, d1), _lrelu(s, -d1),
        _lrelu(s, d2), _lrelu(s, -d2),
        _lrelu(s, d3), _lrelu(s, -d3),
        _lrelu(s, d4), _lrelu(s, -d4),
        r,
    )
end

@inline function NNop.pair_feature_tuple_pullback(
    op::TestDirectionalLeakyReLUDistance{T},
    qvals::NTuple{4, T},
    kvals::NTuple{4, T},
    dphi::NTuple{9, T},
) where T
    d1 = qvals[1] - kvals[1]
    d2 = qvals[2] - kvals[2]
    d3 = qvals[3] - kvals[3]
    d4 = qvals[4] - kvals[4]
    s = op.negative_slope

    # lrelu' wrt delta
    dp(d) = ifelse(d ≥ zero(T), one(T), s)
    # lrelu(-d)' wrt delta = -lrelu'(-d)
    dn(d) = -ifelse(-d ≥ zero(T), one(T), s)

    r = sqrt(d1 * d1 + d2 * d2 + d3 * d3 + op.eps)
    radial = dphi[9] / r

    dq1 = dphi[1] * dp(d1) + dphi[2] * dn(d1) + radial * d1
    dq2 = dphi[3] * dp(d2) + dphi[4] * dn(d2) + radial * d2
    dq3 = dphi[5] * dp(d3) + dphi[6] * dn(d3) + radial * d3
    dq4 = dphi[7] * dp(d4) + dphi[8] * dn(d4)

    dq = (dq1, dq2, dq3, dq4)
    dk = (-dq1, -dq2, -dq3, -dq4)
    return dq, dk
end

# ── Reference implementations ────────────────────────────────────────

function explicit_elementwise_product_pair_bias(q_features, k_features, pair_proj)
    pair = reshape(pair_proj, size(pair_proj, 1), size(pair_proj, 2), 1, 1, 1) .*
        reshape(q_features, 1, size(q_features, 1), size(q_features, 2), 1, size(q_features, 3)) .*
        reshape(k_features, 1, size(k_features, 1), 1, size(k_features, 2), size(k_features, 3))
    return dropdims(sum(pair; dims=2); dims=2)
end

function explicit_directional_lrelu_distance_pair_bias(q_features, k_features, pair_proj, negative_slope, eps)
    # q_features/k_features: (4, L, B)
    # For each of 4 features, compute delta = q_f - k_f, then lrelu(delta), lrelu(-delta)
    # Plus euclidean distance on features 1:3
    lrelu(x) = ifelse.(x .>= zero(eltype(x)), x, negative_slope .* x)

    deltas = map(1:4) do f
        reshape(q_features[f, :, :], 1, size(q_features, 2), 1, size(q_features, 3)) .-
            reshape(k_features[f, :, :], 1, 1, size(k_features, 2), size(k_features, 3))
    end

    r = sqrt.(deltas[1] .^ 2 .+ deltas[2] .^ 2 .+ deltas[3] .^ 2 .+ eps)

    # 9 features: pos1, neg1, pos2, neg2, pos3, neg3, pos4, neg4, r
    features = [
        lrelu(deltas[1]), lrelu(-deltas[1]),
        lrelu(deltas[2]), lrelu(-deltas[2]),
        lrelu(deltas[3]), lrelu(-deltas[3]),
        lrelu(deltas[4]), lrelu(-deltas[4]),
        r,
    ]

    bias = sum(reshape(pair_proj[:, p], :, 1, 1, 1) .* features[p] for p in 1:9)
    return bias
end

function explicit_distance_pair_bias(q_features, k_features, pair_proj, gamma, eps)
    dx = reshape(q_features[1, :, :], 1, size(q_features, 2), 1, size(q_features, 3)) .-
        reshape(k_features[1, :, :], 1, 1, size(k_features, 2), size(k_features, 3))
    dy = reshape(q_features[2, :, :], 1, size(q_features, 2), 1, size(q_features, 3)) .-
        reshape(k_features[2, :, :], 1, 1, size(k_features, 2), size(k_features, 3))
    dz = reshape(q_features[3, :, :], 1, size(q_features, 2), 1, size(q_features, 3)) .-
        reshape(k_features[3, :, :], 1, 1, size(k_features, 2), size(k_features, 3))

    r2 = dx .^ 2 .+ dy .^ 2 .+ dz .^ 2
    r = sqrt.(r2 .+ eps)

    return reshape(pair_proj[:, 1], :, 1, 1, 1) .* r .+
        reshape(pair_proj[:, 2], :, 1, 1, 1) .* r2 .+
        reshape(pair_proj[:, 3], :, 1, 1, 1) .* exp.(-gamma .* r2) .+
        reshape(pair_proj[:, 4], :, 1, 1, 1)
end

# ── Tests ─────────────────────────────────────────────────────────────

@testset "elementwise product causal=$causal, T=$T, E=$E, F=$F, L=$L" for causal in (
    false, true,
), T in (
    Float32,
), E in (
    32,
), F in (
    8,
), L in (
    63, 64,
)
    QH, KVH, B = 4, 2, 2
    q = Adapt.adapt(kab, randn(T, E, L, QH, B))
    k = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    v = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    q_features = Adapt.adapt(kab, randn(T, F, L, B))
    k_features = Adapt.adapt(kab, randn(T, F, L, B))
    pair_proj = Adapt.adapt(kab, randn(T, QH, F))
    pair_op = TestElementwiseProduct{F}()

    o1, ∇1 = Zygote.withgradient(q, k, v, q_features, k_features, pair_proj) do q, k, v, q_features, k_features, pair_proj
        pair = explicit_elementwise_product_pair_bias(q_features, k_features, pair_proj)
        sum(NNop.flash_attention(q, k, v, pair; causal, kpad_mask=nothing))
    end
    o2, ∇2 = Zygote.withgradient(q, k, v, q_features, k_features, pair_proj) do q, k, v, q_features, k_features, pair_proj
        sum(NNop.flash_attention(q, k, v, q_features, k_features, pair_proj, pair_op; causal, feature_grads=true, kpad_mask=nothing))
    end

    eps = 1e-2
    @test isapprox(o1, o2; atol=eps, rtol=eps)
    @test isapprox(∇1[1], ∇2[1]; atol=eps, rtol=eps)
    @test isapprox(∇1[2], ∇2[2]; atol=eps, rtol=eps)
    @test isapprox(∇1[3], ∇2[3]; atol=eps, rtol=eps)
    @test isapprox(∇1[4], ∇2[4]; atol=eps, rtol=eps)
    @test isapprox(∇1[5], ∇2[5]; atol=eps, rtol=eps)
    @test isapprox(∇1[6], ∇2[6]; atol=eps, rtol=eps)

    # feature_grads=false: pair_proj grad still correct, feature grads zero.
    o3, ∇3 = Zygote.withgradient(q, k, v, q_features, k_features, pair_proj) do q, k, v, q_features, k_features, pair_proj
        sum(NNop.flash_attention(q, k, v, q_features, k_features, pair_proj, pair_op; causal, feature_grads=false, kpad_mask=nothing))
    end
    @test isapprox(o1, o3; atol=eps, rtol=eps)
    @test isapprox(∇1[1], ∇3[1]; atol=eps, rtol=eps)   # dq
    @test isapprox(∇1[2], ∇3[2]; atol=eps, rtol=eps)   # dk
    @test isapprox(∇1[3], ∇3[3]; atol=eps, rtol=eps)   # dv
    @test isapprox(∇1[6], ∇3[6]; atol=eps, rtol=eps)   # dproj
    @test all(∇3[4] .== 0)                               # dq_features = 0
    @test all(∇3[5] .== 0)                               # dk_features = 0
end

@testset "euclidean distance causal=$causal, T=$T, E=$E, L=$L" for causal in (
    false, true,
), T in (
    Float32,
), E in (
    32,
), L in (
    63, 64,
)
    QH, KVH, B = 4, 2, 2
    gamma = T(0.7)
    eps = T(1e-4)

    q = Adapt.adapt(kab, randn(T, E, L, QH, B))
    k = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    v = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    q_features = Adapt.adapt(kab, randn(T, 3, L, B))
    k_features = Adapt.adapt(kab, randn(T, 3, L, B))
    pair_proj = Adapt.adapt(kab, randn(T, QH, 4))
    pair_op = TestEuclideanDistance(gamma, eps)

    o1, ∇1 = Zygote.withgradient(q, k, v, q_features, k_features, pair_proj) do q, k, v, q_features, k_features, pair_proj
        pair = explicit_distance_pair_bias(q_features, k_features, pair_proj, gamma, eps)
        sum(NNop.flash_attention(q, k, v, pair; causal, kpad_mask=nothing))
    end
    o2, ∇2 = Zygote.withgradient(q, k, v, q_features, k_features, pair_proj) do q, k, v, q_features, k_features, pair_proj
        sum(NNop.flash_attention(q, k, v, q_features, k_features, pair_proj, pair_op; causal, feature_grads=true, kpad_mask=nothing))
    end

    eps_cmp = 1e-2
    @test isapprox(o1, o2; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[1], ∇2[1]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[2], ∇2[2]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[3], ∇2[3]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[4], ∇2[4]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[5], ∇2[5]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[6], ∇2[6]; atol=eps_cmp, rtol=eps_cmp)
end

@testset "directional lrelu + distance causal=$causal, T=$T, E=$E, L=$L" for causal in (
    false, true,
), T in (
    Float32,
), E in (
    32,
), L in (
    63, 64,
)
    QH, KVH, B = 4, 2, 2
    negative_slope = T(0.1)
    eps = T(1e-4)

    q = Adapt.adapt(kab, randn(T, E, L, QH, B))
    k = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    v = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    q_features = Adapt.adapt(kab, randn(T, 4, L, B))
    k_features = Adapt.adapt(kab, randn(T, 4, L, B))
    pair_proj = Adapt.adapt(kab, randn(T, QH, 9))
    pair_op = TestDirectionalLeakyReLUDistance(negative_slope, eps)

    o1, ∇1 = Zygote.withgradient(q, k, v, q_features, k_features, pair_proj) do q, k, v, q_features, k_features, pair_proj
        pair = explicit_directional_lrelu_distance_pair_bias(q_features, k_features, pair_proj, negative_slope, eps)
        sum(NNop.flash_attention(q, k, v, pair; causal, kpad_mask=nothing))
    end
    o2, ∇2 = Zygote.withgradient(q, k, v, q_features, k_features, pair_proj) do q, k, v, q_features, k_features, pair_proj
        sum(NNop.flash_attention(q, k, v, q_features, k_features, pair_proj, pair_op; causal, feature_grads=true, kpad_mask=nothing))
    end

    eps_cmp = 1e-2
    @test isapprox(o1, o2; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[1], ∇2[1]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[2], ∇2[2]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[3], ∇2[3]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[4], ∇2[4]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[5], ∇2[5]; atol=eps_cmp, rtol=eps_cmp)
    @test isapprox(∇1[6], ∇2[6]; atol=eps_cmp, rtol=eps_cmp)
end

end
