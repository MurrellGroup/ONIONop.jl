# ── Pair-feature interface ────────────────────────────────────────────
#
# To fuse a custom pair bias into flash attention, define an `isbits` struct
# and implement these three methods. The kernel calls them per (q_i, k_j) pair
# on device, so they must be GPU-safe (no allocation, no dynamic dispatch).
#
# Gradients always flow through `q`, `k`, `v`, and `pair_proj`.
# Gradients for `q_features`/`k_features` are opt-in via `feature_grads=true`.
# Gradients do NOT flow through fields inside `pair_op`.

"""
    pair_feature_dim(pair_op) -> Int

Number of pair features produced by `pair_op`. Must be a compile-time constant
(use a type parameter, not a struct field).

Determines the second dimension of `pair_proj :: (QH, pair_feature_dim(pair_op))`.
"""
pair_feature_dim(pair_op) = error("`pair_feature_dim(::$(typeof(pair_op)))` is not implemented.")

"""
    pair_feature_tuple(pair_op, qvals::NTuple{F,T}, kvals::NTuple{F,T}) -> NTuple{d_pf, T}

Compute pair features from the q/k feature values. Called once per (i, j) pair
inside the kernel. `F = size(q_features, 1)` — all features are loaded densely.

`d_pf = pair_feature_dim(pair_op)`.
"""
@inline pair_feature_tuple(pair_op, qvals, kvals) =
    error("`pair_feature_tuple(::$(typeof(pair_op)), ...)` is not implemented.")

"""
    pair_feature_tuple_pullback(pair_op, qvals, kvals, dphi) -> (dq, dk)

VJP of [`pair_feature_tuple`](@ref). `dphi :: NTuple{d_pf, T}` is the
upstream cotangent. Returns `(dq, dk)` tuples of length `F`.

Must be mathematically consistent with `pair_feature_tuple`.

Only called when `feature_grads=true` is passed to `flash_attention`.
If you never need gradients w.r.t. `q_features`/`k_features`, you can
skip implementing this method.
"""
@inline pair_feature_tuple_pullback(pair_op, qvals, kvals, dphi) =
    error("`pair_feature_tuple_pullback(::$(typeof(pair_op)), ...)` is not implemented.")

# ── Tuple helpers ─────────────────────────────────────────────────────

@inline tuple_zero(::Type{T}, ::Val{N}) where {T, N} = ntuple(_ -> zero(T), Val(N))

@inline tuple_add(a::NTuple{N, T}, b::NTuple{N, T}) where {N, T} =
    ntuple(i -> a[i] + b[i], Val(N))

@inline tuple_from_feature_shm(src, idx, ::Val{N}) where {N} =
    ntuple(i -> src[i, idx], Val(N))

@inline tuple_from_proj(pair_proj, head_idx, ::Val{N}) where {N} =
    ntuple(i -> pair_proj[head_idx, i], Val(N))

@inline function tuple_dot(a::NTuple{N, T}, b::NTuple{N, T}) where {N, T}
    acc = zero(T)
    @inbounds for i in 1:N
        acc += a[i] * b[i]
    end
    return acc
end

@inline function scaled_proj_tuple(pair_proj_row::NTuple{N, T}, scale::T) where {N, T}
    return ntuple(i -> pair_proj_row[i] * scale, Val(N))
end

# ── Size checks ───────────────────────────────────────────────────────

function check_flash_attention_pair_feature_sizes(
    q::AbstractArray{T,4},
    k::AbstractArray{T,4},
    v::AbstractArray{T,4},
    q_features::AbstractArray{T,3},
    k_features::AbstractArray{T,3},
    pair_proj::AbstractMatrix{T},
    pair_op,
) where T <: Union{Float16, Float32}
    QE, QL, QH, B, KE, KL, KH, KB = check_flash_attention_sizes(q, k, v, nothing)
    FQ, QFL, QB = size(q_features)
    FK, KFL, KB2 = size(k_features)
    PD = pair_feature_dim(pair_op)

    isbitstype(typeof(pair_op)) || error("`pair_op` must be an isbits device functor.")
    QFL == QL || error("Query feature sequence length `$QFL` must match query sequence length `$QL`.")
    KFL == KL || error("Key feature sequence length `$KFL` must match key sequence length `$KL`.")
    QB == B || error("Query feature batch size `$QB` must match query batch size `$B`.")
    KB2 == B || error("Key feature batch size `$KB2` must match query batch size `$B`.")
    FQ == FK || error("Query feature dim `$FQ` must match key feature dim `$FK`.")
    size(pair_proj) == (QH, PD) ||
        error("Pair projection shape `$(size(pair_proj))` must be `(QH, d_pf) = ($(QH), $(PD))`.")

    return QE, QL, QH, B, KE, KL, KH, KB, PD
end

# ── Shared memory helpers ─────────────────────────────────────────────

function flash_attention_pair_feature_shmem_bwd(::Type{T};
    emb_dim::Int, feat_dim::Int, pair_dim::Int, groupsize::Int, qk_fp16::Bool,
)::Int where T
    return flash_attention_shmem_bwd(T; emb_dim, groupsize, qk_fp16) +
        sizeof(T) * groupsize * (2 * feat_dim + pair_dim)
end

function flash_attention_pair_feature_static_groupsize(
    ::Type{T};
    emb_dim::Int,
    coord_dim::Int,
    pair_dim::Int,
    target_shmem::UInt64,
) where T
    qk_fp16s = (true,)
    for qk_fp16 in qk_fp16s, groupsize in (256, 128, 64, 32, 16)
        shmem = flash_attention_pair_feature_shmem_bwd(
            T; emb_dim, feat_dim=coord_dim, pair_dim, groupsize, qk_fp16,
        )
        shmem ≤ target_shmem && return groupsize
    end
    error("Failed to find groupsize for pair-feature Flash Attention that satisfies Shared Memory constraint.")
end

# ── Forward kernel ────────────────────────────────────────────────────

@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_pair_feature_static_fwd!(
    cfg::Type{C}, cfg_out,
    o::AbstractArray{T, 4}, ms::AbstractArray{T, 3}, ls::AbstractArray{T, 3},
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
    q_features::AbstractArray{T, 3}, k_features::AbstractArray{T, 3},
    pair_proj::AbstractMatrix{T},
    pair_op,
    scale::T,
    kpad_mask::Maybe{AbstractMatrix{Bool}},
    ::Val{emb_dim}, ::Val{coord_dim}, ::Val{pair_dim}, ::Val{in_seq_bounds}, ::Val{causal},
) where {C, T, emb_dim, coord_dim, pair_dim, in_seq_bounds, causal}
    gsz = @groupsize()[1]
    n_warps = gsz ÷ 32
    kv_seq_tiles = cld(size(k, 2), gsz)
    n_q_per_kv = size(q, 3) ÷ size(k, 3)

    q_shm = @localmem T (gsz, emb_dim)
    k_shm = @localmem T (emb_dim, gsz)
    s_shm = @localmem T (gsz, gsz)
    o_shm = @localmem T (emb_dim, gsz)
    qf_shm = @localmem T (coord_dim, gsz)
    kf_shm = @localmem T (coord_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)
    q_offset = (gidx[1] - 1) * gsz
    in_q = in_seq_bounds || q_offset + tidx ≤ size(q, 2)

    q_head_idx = gidx[2]
    kv_head_idx = cld(q_head_idx, n_q_per_kv)
    proj_row = tuple_from_proj(pair_proj, q_head_idx, Val(pair_dim))

    @inline function sh_load_emb!(dest, src, offset, head_idx, mask::Bool, ::Val{tr}) where tr
        @unroll for i in 1:emb_dim
            x, y = tr ? (tidx, i) : (i, tidx)
            @inbounds dest[x, y] = mask ? src[i, tidx + offset, head_idx, gidx[3]] : zero(T)
        end
    end

    @inline function sh_load_features!(dest, src, offset, mask::Bool)
        @unroll for i in 1:coord_dim
            @inbounds dest[i, tidx] = mask ? src[i, tidx + offset, gidx[3]] : zero(T)
        end
    end

    sh_load_emb!(q_shm, q, q_offset, q_head_idx, in_q, Val{true}())
    sh_load_features!(qf_shm, q_features, q_offset, in_q)
    @unroll for i in 1:emb_dim
        o_shm[i, tidx] = zero(T)
    end
    @synchronize()

    l_i = zero(T)
    m_i = typemin(T)
    k_offset = 0
    end_iter = causal ? gidx[1] : kv_seq_tiles

    for _ in 1:end_iter
        in_k = in_seq_bounds || k_offset + tidx ≤ size(k, 2)
        sh_load_emb!(k_shm, k, k_offset, kv_head_idx, in_k, Val{false}())
        sh_load_features!(kf_shm, k_features, k_offset, in_k)
        @synchronize()

        C <: WMMATileConfig ?
            wmma!(s_shm, q_shm, k_shm, cfg, tidx, n_warps, d_frag -> d_frag .* scale, Val(false)) :
            mma!(s_shm, q_shm, k_shm, cfg, tidx, (res, c_shm, x, y) -> res * scale)
        @synchronize()

        qvals = tuple_from_feature_shm(qf_shm, tidx, Val(coord_dim))
        @unroll for i in 1:gsz
            in_seq_bounds || (in_q && k_offset + i ≤ size(k, 2)) || break
            kvals = tuple_from_feature_shm(kf_shm, i, Val(coord_dim))
            phi = pair_feature_tuple(pair_op, qvals, kvals)
            s_shm[tidx, i] += tuple_dot(proj_row, phi)
        end

        if causal
            @unroll for i in 1:gsz
                (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
                s_shm[tidx, i] = (tidx + q_offset ≥ i + k_offset) ? s_shm[tidx, i] : typemin(T)
            end
        end
        if !isnothing(kpad_mask)
            @unroll for i in 1:gsz
                (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
                valid = kpad_mask[k_offset + i, gidx[3]]
                s_shm[tidx, i] = valid ? s_shm[tidx, i] : typemin(T)
            end
        end

        m_ij = typemin(T)
        @unroll for i in 1:gsz
            (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
            m_ij = max(m_ij, s_shm[tidx, i])
        end

        l_ij = zero(T)
        @unroll for i in 1:gsz
            (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
            tmp = exp(s_shm[tidx, i] - m_ij)
            l_ij += tmp
            s_shm[tidx, i] = tmp
        end
        @synchronize()

        m_i_new = max(m_i, m_ij)
        α = exp(m_i - m_i_new)
        β = exp(m_ij - m_i_new)
        l_i_new = α * l_i + β * l_ij

        p_scale = β / l_i_new
        o_scale = l_i / l_i_new * α

        @unroll for i in 1:gsz
            s_shm[tidx, i] *= p_scale
        end
        @unroll for i in 1:emb_dim
            o_shm[i, tidx] *= o_scale
        end

        sh_load_emb!(k_shm, v, k_offset, kv_head_idx, in_k, Val{false}())
        @synchronize()
        C <: WMMATileConfig ?
            wmma!(o_shm, s_shm, k_shm, cfg_out, tidx, n_warps, identity, Val(true)) :
            mma!(o_shm, s_shm, k_shm, cfg_out, tidx, mma_acc_fn)
        @synchronize()

        m_i = m_i_new
        l_i = l_i_new
        k_offset += gsz
    end

    if in_seq_bounds || in_q
        @unroll for i in 1:emb_dim
            o[i, tidx + q_offset, q_head_idx, gidx[3]] = o_shm[i, tidx]
        end
        ms[tidx + q_offset, q_head_idx, gidx[3]] = m_i
        ls[tidx + q_offset, q_head_idx, gidx[3]] = l_i
    end
end

# ── Backward kernel ───────────────────────────────────────────────────

@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_pair_feature_static_bwd!(
    cfg::Type{C}, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
    dq::AbstractArray{T,4}, dk::AbstractArray{T,4}, dv::AbstractArray{T,4},
    dq_features_partial::AbstractArray{T,4}, dk_features_partial::AbstractArray{T,4}, dpair_proj::AbstractArray{T,3},
    Δ::AbstractArray{T,4}, δ::AbstractArray{T,3},
    o::AbstractArray{T,4}, ms::AbstractArray{T,3},
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    q_features::AbstractArray{T,3}, k_features::AbstractArray{T,3},
    pair_proj::AbstractMatrix{T},
    pair_op,
    scale::T,
    kpad_mask::Maybe{AbstractMatrix{Bool}},
    ::Val{emb_dim}, ::Val{coord_dim}, ::Val{pair_dim}, ::Val{in_seq_bounds}, ::Val{causal}, ::Val{is_gqa},
    ::Val{compute_feature_grads},
) where {C, T, emb_dim, coord_dim, pair_dim, in_seq_bounds, causal, is_gqa, compute_feature_grads}
    gsz = @groupsize()[1]
    n_warps = gsz ÷ 32
    q_seq_tiles = cld(size(q, 2), gsz)
    kv_seq_tiles = cld(size(k, 2), gsz)
    n_q_per_kv = size(q, 3) ÷ size(k, 3)

    q_shm = @localmem Float16 (gsz, emb_dim)
    k_shm = @localmem Float16 (emb_dim, gsz)
    s_shm = @localmem T       (gsz, gsz)
    Δ_shm = @localmem T       (emb_dim, gsz)
    d_shm = @localmem T       (emb_dim, gsz)
    qf_shm = @localmem T      (coord_dim, gsz)
    kf_shm = @localmem T      (coord_dim, gsz)
    dproj_shm = @localmem T   (pair_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)

    q_head_idx = gidx[1]
    kv_head_idx = cld(q_head_idx, n_q_per_kv)
    proj_row = tuple_from_proj(pair_proj, q_head_idx, Val(pair_dim))

    @inline function sh_load_emb!(dest, src, offset, head_idx, mask::Bool, ::Val{tr}) where tr
        @unroll for i in 1:emb_dim
            x, y = tr ? (tidx, i) : (i, tidx)
            @inbounds dest[x, y] = mask ? src[i, tidx + offset, head_idx, gidx[2]] : zero(T)
        end
    end

    @inline function sh_load_features!(dest, src, offset, mask::Bool)
        @unroll for i in 1:coord_dim
            @inbounds dest[i, tidx] = mask ? src[i, tidx + offset, gidx[2]] : zero(T)
        end
    end

    @inbounds for p in 1:pair_dim
        dproj_shm[p, tidx] = zero(T)
    end
    @synchronize()

    for start_n in 1:kv_seq_tiles
        lo_k = (start_n - 1) * gsz
        q_offset = causal ? lo_k : 0
        dk_acc = compute_feature_grads ? tuple_zero(T, Val(coord_dim)) : nothing

        in_k_ok = in_seq_bounds || tidx + lo_k ≤ size(k,2)
        sh_load_emb!(k_shm, k, lo_k, kv_head_idx, in_k_ok, Val(false))
        sh_load_features!(kf_shm, k_features, lo_k, in_k_ok)
        @synchronize()

        start_m = causal ? start_n : 1
        for _ in start_m:q_seq_tiles
            lo_q = q_offset

            in_q_ok = in_seq_bounds || tidx + lo_q ≤ size(q,2)
            sh_load_emb!(Δ_shm, Δ, lo_q, q_head_idx, in_q_ok, Val(false))
            sh_load_emb!(q_shm, q, lo_q, q_head_idx, in_q_ok, Val(true))
            sh_load_features!(qf_shm, q_features, lo_q, in_q_ok)
            @synchronize()

            C <: WMMATileConfig ?
                wmma!(s_shm, q_shm, k_shm, cfg, tidx, n_warps, d_frag -> d_frag .* scale, Val(false)) :
                mma!(s_shm, q_shm, k_shm, cfg, tidx, (res, c_shm, x, y) -> res * scale)
            @synchronize()

            qvals = tuple_from_feature_shm(qf_shm, tidx, Val(coord_dim))
            @unroll for j in 1:gsz
                in_seq_bounds || (in_q_ok && lo_k + j ≤ size(k, 2)) || break
                kvals = tuple_from_feature_shm(kf_shm, j, Val(coord_dim))
                phi = pair_feature_tuple(pair_op, qvals, kvals)
                s_shm[tidx, j] += tuple_dot(proj_row, phi)
            end

            if causal
                @unroll for j in 1:gsz
                    in_seq_bounds || j + lo_k ≤ size(k, 2) || break
                    s_shm[tidx, j] = tidx + lo_q ≥ j + lo_k ? s_shm[tidx, j] : typemin(T)
                end
            end
            if !isnothing(kpad_mask)
                @unroll for j in 1:gsz
                    in_seq_bounds || j + lo_k ≤ size(k, 2) || break
                    valid = kpad_mask[j + lo_k, gidx[2]]
                    s_shm[tidx, j] = valid ? s_shm[tidx, j] : typemin(T)
                end
            end

            in_ms = in_seq_bounds || tidx + lo_q ≤ size(ms,1)
            m_i = in_ms ? ms[tidx + lo_q, q_head_idx, gidx[2]] : typemax(T)
            @unroll for j in 1:gsz
                s_shm[tidx, j] = exp(s_shm[tidx, j] - m_i)
            end
            @synchronize()

            C <: WMMATileConfig ?
                wmma!(d_shm, Δ_shm, s_shm, cfg_dv, tidx, n_warps, identity, Val(false)) :
                mma!(d_shm, Δ_shm, s_shm, cfg_dv, tidx, mma_non_acc_fn)
            @synchronize()
            in_dv = in_seq_bounds || tidx + lo_k ≤ size(dv, 2)
            if in_dv
                @unroll for i in 1:emb_dim
                    if is_gqa
                        KA.@atomic dv[i, tidx + lo_k, kv_head_idx, gidx[2]] += d_shm[i, tidx]
                    else
                        dv[i, tidx + lo_k, kv_head_idx, gidx[2]] += d_shm[i, tidx]
                    end
                end
            end

            sh_load_emb!(d_shm, v, lo_k, kv_head_idx, in_dv, Val(false))
            @synchronize()

            s_shm_row = MVector{gsz, T}(undef)
            @unroll for j in 1:gsz
                s_shm_row[j] = s_shm[tidx, j]
            end

            C <: WMMATileConfig ?
                wmma!(s_shm, Δ_shm, d_shm, cfg_ds, tidx, n_warps, identity, Val(false)) :
                mma!(s_shm, Δ_shm, d_shm, cfg_ds, tidx, mma_non_acc_fn)
            @synchronize()

            d_i = (in_seq_bounds || tidx + lo_q ≤ size(δ, 1)) ?
                δ[tidx + lo_q, q_head_idx, gidx[2]] : zero(T)
            @unroll for j in 1:gsz
                s_shm[tidx, j] = s_shm_row[j] * (s_shm[tidx, j] - d_i) * scale
            end
            @synchronize()

            row = tidx + lo_q
            dq_acc = compute_feature_grads ? tuple_zero(T, Val(coord_dim)) : nothing
            @unroll for j in 1:gsz
                col = j + lo_k
                in_seq_bounds || col ≤ size(k_features, 2) || break
                if in_seq_bounds || row ≤ size(q_features, 2)
                    dbias = s_shm[tidx, j] / scale
                    kvals = tuple_from_feature_shm(kf_shm, j, Val(coord_dim))
                    phi = pair_feature_tuple(pair_op, qvals, kvals)
                    @inbounds for p in 1:pair_dim
                        dproj_shm[p, tidx] += dbias * phi[p]
                    end

                    if compute_feature_grads
                        dphi = scaled_proj_tuple(proj_row, dbias)
                        dq_tuple, _ = pair_feature_tuple_pullback(pair_op, qvals, kvals, dphi)
                        dq_acc = tuple_add(dq_acc, dq_tuple)
                    end
                end
            end
            if compute_feature_grads && (in_seq_bounds || row ≤ size(q_features, 2))
                @inbounds for c in 1:coord_dim
                    dq_features_partial[c, row, q_head_idx, gidx[2]] += dq_acc[c]
                end
            end
            if compute_feature_grads && in_k_ok
                kvals = tuple_from_feature_shm(kf_shm, tidx, Val(coord_dim))
                @unroll for i in 1:gsz
                    row_i = i + lo_q
                    in_seq_bounds || row_i ≤ size(q_features, 2) || break
                    qvals_i = tuple_from_feature_shm(qf_shm, i, Val(coord_dim))
                    dphi = scaled_proj_tuple(proj_row, s_shm[i, tidx] / scale)
                    _, dk_tuple = pair_feature_tuple_pullback(pair_op, qvals_i, kvals, dphi)
                    dk_acc = tuple_add(dk_acc, dk_tuple)
                end
            end

            C <: WMMATileConfig ?
                wmma!(d_shm, s_shm, q_shm, cfg_dk, tidx, n_warps, identity, Val(false)) :
                mma!(d_shm, s_shm, q_shm, cfg_dk, tidx, mma_non_acc_fn)
            @synchronize()
            if in_k_ok
                @unroll for i in 1:emb_dim
                    if is_gqa
                        KA.@atomic dk[i, tidx + lo_k, kv_head_idx, gidx[2]] += d_shm[i, tidx]
                    else
                        dk[i, tidx + lo_k, kv_head_idx, gidx[2]] += d_shm[i, tidx]
                    end
                end
            end

            in_dq = in_seq_bounds || tidx + lo_q ≤ size(dq, 2)
            sh_load_emb!(d_shm, dq, lo_q, q_head_idx, in_dq, Val(false))
            @synchronize()
            C <: WMMATileConfig ?
                wmma!(d_shm, s_shm, k_shm, cfg_dq, tidx, n_warps, identity, Val(true)) :
                mma!(d_shm, s_shm, k_shm, cfg_dq, tidx, mma_acc_fn)
            @synchronize()
            if in_dq
                @unroll for i in 1:emb_dim
                    dq[i, tidx + lo_q, q_head_idx, gidx[2]] = d_shm[i, tidx]
                end
            end

            q_offset += gsz
        end

        if compute_feature_grads && in_k_ok
            @inbounds for c in 1:coord_dim
                dk_features_partial[c, tidx + lo_k, q_head_idx, gidx[2]] = dk_acc[c]
            end
        end
    end

    @synchronize()
    if tidx ≤ pair_dim
        acc = zero(T)
        @inbounds for lane in 1:gsz
            acc += dproj_shm[tidx, lane]
        end
        dpair_proj[q_head_idx, tidx, gidx[2]] = acc
    end
end

# ── Reduction kernels ─────────────────────────────────────────────────

@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_pair_feature_reduce_dproj!(
    dproj::AbstractMatrix{T},
    dproj_partial::AbstractArray{T,3},
) where T
    idx = @index(Global, NTuple)
    head_idx, feat_idx = idx
    head_idx > size(dproj, 1) && return
    feat_idx > size(dproj, 2) && return

    acc = zero(T)
    @inbounds for b in 1:size(dproj_partial, 3)
        acc += dproj_partial[head_idx, feat_idx, b]
    end
    dproj[head_idx, feat_idx] = acc
end

@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_pair_feature_reduce_feature_grads!(
    dfeatures::AbstractArray{T,3},
    dfeatures_partial::AbstractArray{T,4},
) where T
    idx = @index(Global, NTuple)
    coord_idx, seq_idx, batch_idx = idx
    coord_idx > size(dfeatures_partial, 1) && return
    seq_idx > size(dfeatures_partial, 2) && return
    batch_idx > size(dfeatures_partial, 4) && return

    acc = zero(T)
    @inbounds for head_idx in 1:size(dfeatures_partial, 3)
        acc += dfeatures_partial[coord_idx, seq_idx, head_idx, batch_idx]
    end
    dfeatures[coord_idx, seq_idx, batch_idx] = acc
end

# ── Dispatch ──────────────────────────────────────────────────────────

function _flash_attention_pair_features(
    q::AbstractArray{T,4},
    k::AbstractArray{T,4},
    v::AbstractArray{T,4},
    q_features::AbstractArray{T,3},
    k_features::AbstractArray{T,3},
    pair_proj::AbstractMatrix{T},
    pair_op;
    causal::Bool,
    feature_grads::Bool = false,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
) where T <: Union{Float16, Float32}
    coord_dim = size(q_features, 1)
    return _flash_attention_pair_features_impl(
        q, k, v, q_features, k_features, pair_proj, pair_op,
        Val(coord_dim), Val(pair_feature_dim(pair_op));
        causal, feature_grads, kpad_mask,
    )
end

# ── Host forward ──────────────────────────────────────────────────────

function _flash_attention_pair_features_impl(
    q::AbstractArray{T,4},
    k::AbstractArray{T,4},
    v::AbstractArray{T,4},
    q_features::AbstractArray{T,3},
    k_features::AbstractArray{T,3},
    pair_proj::AbstractMatrix{T},
    pair_op,
    ::Val{coord_dim},
    ::Val{pair_dim};
    causal::Bool,
    feature_grads::Bool = false,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
) where {T <: Union{Float16, Float32}, coord_dim, pair_dim}
    QE, QL, QH, B, KE, KL, KH, KB, PD = check_flash_attention_pair_feature_sizes(
        q, k, v, q_features, k_features, pair_proj, pair_op,
    )

    kab = get_backend(q)

    target_shmem = shared_memory(kab, KA.device(kab))
    gsz = flash_attention_pair_feature_static_groupsize(T; emb_dim=QE, coord_dim, pair_dim, target_shmem)

    q_seq_tiles, kv_seq_tiles = cld.((QL, KL), gsz)
    threads = (gsz, 1, 1)
    ndrange = (gsz * q_seq_tiles, QH, B)
    in_bounds = QL % gsz == 0 && KL % gsz == 0
    scale = T(inv(sqrt(QE)))

    if supports_wmma(kab) && sizeof(T) == 2
        @assert QE % 16 == 0
        WM, WK, WN = 16, 16, 16

        BM, BK, BN = gsz, QE, gsz
        cfg = WMMATileConfig{BM, BK, BN, WM, WK, WN, false, false, false}
        BM, BK, BN = gsz, gsz, QE
        cfg_out = WMMATileConfig{BM, BK, BN, WM, WK, WN, false, true, true}
    else
        BM, BK, BN = gsz, QE, gsz
        TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
        cfg = FATileConfig{BM, BK, BN, TM, TN, false, false, false}

        BM, BK, BN = gsz, gsz, QE
        TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
        cfg_out = FATileConfig{BM, BK, BN, TM, TN, false, true, true}
    end

    o = similar(q)
    ms = KA.allocate(kab, eltype(o), (QL, QH, B))
    ls = KA.allocate(kab, eltype(o), (QL, QH, B))

    _flash_attention_pair_feature_static_fwd!(kab, threads)(
        cfg, cfg_out,
        o, ms, ls,
        q, k, v,
        q_features, k_features,
        pair_proj, pair_op,
        scale, kpad_mask,
        Val(QE), Val(coord_dim), Val(pair_dim), Val(in_bounds), Val(causal);
        ndrange,
    )
    return o, ms, ls
end

# ── Host backward ─────────────────────────────────────────────────────

function ∇flash_attention_pair_features(
    Δ::AbstractArray{T,4},
    o::AbstractArray{T,4}, ms::AbstractArray{T,3}, ls::AbstractArray{T,3},
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    q_features::AbstractArray{T,3},
    k_features::AbstractArray{T,3},
    pair_proj::AbstractMatrix{T},
    pair_op;
    causal::Bool,
    feature_grads::Bool = false,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
) where T <: Union{Float16, Float32}
    coord_dim = size(q_features, 1)
    return ∇flash_attention_pair_features_impl(
        Δ, o, ms, ls, q, k, v, q_features, k_features, pair_proj, pair_op,
        Val(coord_dim), Val(pair_feature_dim(pair_op));
        causal, feature_grads, kpad_mask,
    )
end

function ∇flash_attention_pair_features_impl(
    Δ::AbstractArray{T,4},
    o::AbstractArray{T,4}, ms::AbstractArray{T,3}, ls::AbstractArray{T,3},
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    q_features::AbstractArray{T,3},
    k_features::AbstractArray{T,3},
    pair_proj::AbstractMatrix{T},
    pair_op,
    ::Val{coord_dim},
    ::Val{pair_dim};
    causal::Bool,
    feature_grads::Bool = false,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
) where {T <: Union{Float16, Float32}, coord_dim, pair_dim}
    QE, QL, QH, B, KE, KL, KH, KB, PD = check_flash_attention_pair_feature_sizes(
        q, k, v, q_features, k_features, pair_proj, pair_op,
    )

    kab = get_backend(q)

    target_shmem = shared_memory(kab, KA.device(kab))
    gsz = flash_attention_pair_feature_static_groupsize(T; emb_dim=QE, coord_dim, pair_dim, target_shmem)

    q_tiles, k_tiles = cld.((QL, KL), gsz)
    in_bounds = QL % gsz == 0 && KL % gsz == 0
    scale = T(inv(sqrt(QE)))

    Δ_scaled = similar(Δ)
    δ = similar(ls)
    threads = (gsz, 1, 1)
    ndrange = (gsz * q_tiles, QH, B)
    _flash_attention_bwd_preprocess!(kab, threads)(
        Δ_scaled, δ, Δ, o, ls,
        Val(QE), Val(in_bounds); ndrange,
    )

    dq = KA.zeros(kab, T, size(q))
    dk = KA.zeros(kab, T, size(k))
    dv = KA.zeros(kab, T, size(v))
    dproj_partial = KA.zeros(kab, T, (QH, PD, B))

    # Feature grad partials: full-size when needed, tiny dummy otherwise.
    if feature_grads
        dqf_partial = KA.zeros(kab, T, (coord_dim, QL, QH, B))
        dkf_partial = KA.zeros(kab, T, (coord_dim, KL, QH, B))
    else
        dqf_partial = KA.zeros(kab, T, (1, 1, 1, 1))
        dkf_partial = KA.zeros(kab, T, (1, 1, 1, 1))
    end

    if supports_wmma(kab) && sizeof(T) == 2
        WM, WK, WN = 16, 16, 16

        BM, BK, BN = gsz, QE, gsz
        cfg = WMMATileConfig{BM, BK, BN, WM, WK, WN, false, false, false}
        BM, BK, BN = QE, gsz, gsz
        cfg_dv = WMMATileConfig{BM, BK, BN, WM, WK, WN, false, false, false}
        BM, BK, BN = gsz, gsz, QE
        cfg_dk = WMMATileConfig{BM, BK, BN, WM, WK, WN, true, false, true}
        cfg_dq = WMMATileConfig{BM, BK, BN, WM, WK, WN, false, true, true}
        BM, BK, BN = gsz, QE, gsz
        cfg_ds = WMMATileConfig{BM, BK, BN, WM, WK, WN, true, false, false}
    else
        BM, BK, BN = gsz, QE, gsz
        TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
        cfg = FATileConfig{BM, BK, BN, TM, TN, false, false, false}

        BM, BK, BN = QE, gsz, gsz
        TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
        cfg_dv = FATileConfig{BM, BK, BN, TM, TN, false, false, false}

        BM, BK, BN = gsz, gsz, QE
        TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
        cfg_dk = FATileConfig{BM, BK, BN, TM, TN, true, false, true}
        cfg_dq = FATileConfig{BM, BK, BN, TM, TN, false, true, true}

        BM, BK, BN = gsz, QE, gsz
        TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
        cfg_ds = FATileConfig{BM, BK, BN, TM, TN, true, false, false}
    end

    is_gqa = QH ÷ KH > 1

    threads = (gsz, 1)
    ndrange = (gsz * QH, B)
    _flash_attention_pair_feature_static_bwd!(kab, threads)(
        cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
        dq, dk, dv, dqf_partial, dkf_partial, dproj_partial,
        Δ_scaled, δ,
        o, ms,
        q, k, v,
        q_features, k_features,
        pair_proj, pair_op,
        scale, kpad_mask,
        Val(QE), Val(coord_dim), Val(pair_dim), Val(in_bounds), Val(causal), Val(is_gqa),
        Val(feature_grads);
        ndrange,
    )

    dproj = KA.zeros(kab, T, size(pair_proj))
    threads = (16, 16)
    ndrange = (QH, PD)
    _flash_attention_pair_feature_reduce_dproj!(kab, threads)(dproj, dproj_partial; ndrange)

    if feature_grads
        dqf = KA.zeros(kab, T, size(q_features))
        dkf = KA.zeros(kab, T, size(k_features))

        threads = (coord_dim, 64, 1)
        ndrange = (coord_dim, QL, B)
        _flash_attention_pair_feature_reduce_feature_grads!(kab, threads)(dqf, dqf_partial; ndrange)

        threads = (coord_dim, 64, 1)
        ndrange = (coord_dim, KL, B)
        _flash_attention_pair_feature_reduce_feature_grads!(kab, threads)(dkf, dkf_partial; ndrange)
    else
        dqf = KA.zeros(kab, T, size(q_features))
        dkf = KA.zeros(kab, T, size(k_features))
    end

    return dq, dk, dv, dqf, dkf, dproj
end

# ── Public API + rrule ────────────────────────────────────────────────

function flash_attention(
    q,
    k,
    v,
    q_features::AbstractArray{<:Real,3},
    k_features::AbstractArray{<:Real,3},
    pair_proj::AbstractMatrix{<:Real},
    pair_op;
    causal::Bool,
    feature_grads::Bool = false,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
)
    o = _flash_attention_pair_features(q, k, v, q_features, k_features, pair_proj, pair_op; causal, feature_grads, kpad_mask)
    within_gradient(q) && return o
    @assert length(o) == 3
    return o[1]
end

function CRC.rrule(::typeof(_flash_attention_pair_features),
    q,
    k,
    v,
    q_features::AbstractArray{<:Real,3},
    k_features::AbstractArray{<:Real,3},
    pair_proj::AbstractMatrix{<:Real},
    pair_op;
    causal::Bool,
    feature_grads::Bool = false,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
)
    o, ms, ls = _flash_attention_pair_features(
        q, k, v, q_features, k_features, pair_proj, pair_op; causal, feature_grads, kpad_mask,
    )

    function _pullback(Δ)
        dq, dk, dv, dqf, dkf, dproj = ∇flash_attention_pair_features(
            CRC.unthunk(Δ),
            o, ms, ls,
            q, k, v,
            q_features, k_features,
            pair_proj, pair_op;
            causal, feature_grads, kpad_mask,
        )
        return CRC.NoTangent(), dq, dk, dv, dqf, dkf, dproj, CRC.NoTangent()
    end
    return o, _pullback
end
