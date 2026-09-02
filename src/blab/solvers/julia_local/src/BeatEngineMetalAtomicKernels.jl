# Fused pair-atomic regular kernel: one thread per (test, trial) element pair
# on a 2-D thread grid, every Green's-function value evaluated once and used
# for all four operators, results scattered with Float32 atomics.
#
# This is the design hornlab-metal-bem measured as ~5x faster than its
# alternatives on Apple GPUs. Three things distinguish it from the colored
# pair-owned kernels: one dispatch instead of color_count^2, no second pass
# for the hypersingular operator, and no 64-bit integer division on the GPU
# (the pair is addressed by the 2-D grid position; the six trial quadrature
# points are hoisted out of the 36-point-pair loop). It trades determinism
# for that: results differ from the colored kernels by float32 summation
# order only.

using Metal: atomic_fetch_add_explicit, thread_position_in_grid_2d

@inline _metal_fast_cos(x::Float32) = Base.FastMath.cos_fast(x)
@inline _metal_fast_sin(x::Float32) = Base.FastMath.sin_fast(x)
@inline _metal_fast_rsqrt(x::Float32) = Metal.rsqrt_fast(x)

# Compile-time unrolled fold over the trial quadrature points:
# acc = term(...term(term(acc, 1), 2)..., N). The trial point, basis and
# weight are read from the (cached) rule and vertex arrays inside the term
# with constant indices instead of being hoisted into registers: this kernel
# is register-bound (the 3x3 double-layer and hypersingular accumulators
# alone are 36 floats), and hoisted trial data cost another ~42.
@inline _metal_trial_fold(acc, context, ::Val{0}, ::Val{R}) where {R} = acc
@inline function _metal_trial_fold(acc, context, ::Val{N}, ::Val{R}) where {N,R}
    acc = _metal_trial_fold(acc, context, Val(N - 1), Val(R))
    return _metal_trial_term(acc, context, Int32(N), Val(R))
end

@inline function _metal_trial_term(acc, context, trial_q::Int32, ::Val{R}) where {R}
    s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = acc
    x, y, z, test_weight_scale, k, inv_four_pi,
        test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        element_rule_points, rule_points, rule_weights, trial_index, face_count = context
    @inbounds xi = rule_points[trial_q]
    @inbounds eta = rule_points[trial_q + Int32(R)]
    rb1 = one(k) - xi - eta
    @inbounds trial_weight = rule_weights[trial_q]
    point_index = trial_index + face_count * (trial_q - Int32(1))
    @inbounds sx = element_rule_points[point_index]
    @inbounds sy = element_rule_points[point_index + face_count * Int32(R)]
    @inbounds sz = element_rule_points[point_index + face_count * Int32(2 * R)]
    Base.@fastmath begin
    dx = sx * trial_signs[1] - x
    dy = sy * trial_signs[2] - y
    dz = sz * trial_signs[3] - z
    radius2 = dx * dx + dy * dy + dz * dz
    if radius2 > zero(k)
        rb = SVector(rb1, xi, eta)
        # Fast-math intrinsics: this is what an Xcode-compiled Metal
        # shader gets by default, and what hornlab-metal-bem runs.
        inv_radius = _metal_fast_rsqrt(radius2)
        radius = radius2 * inv_radius
        phase = k * radius
        green_scale = inv_radius * inv_four_pi * (test_weight_scale * trial_weight)
        green_re = _metal_fast_cos(phase) * green_scale
        green_im = _metal_fast_sin(phase) * green_scale
        grad_re = -green_re * inv_radius - green_im * k
        grad_im = green_re * k - green_im * inv_radius
        test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
        trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
        s_re += green_re
        s_im += green_im
        a_re += grad_re * test_dot
        a_im += grad_im * test_dot
        d_re += rb * (grad_re * trial_dot)
        d_im += rb * (grad_im * trial_dot)
        h_re += rb * green_re
        h_im += rb * green_im
    end
    end
    return (s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im)
end

@inline function _metal_atomic_add_complex!(target_f32, complex_index, re, im)
    base = 2 * complex_index - 1
    atomic_fetch_add_explicit(pointer(target_f32, base), re)
    atomic_fetch_add_explicit(pointer(target_f32, base + 1), im)
    return nothing
end

# The pair arithmetic shared by the fused atomic kernel and the chunked
# gather kernel: every Green's-function value evaluated once for the four
# operators, accumulated in the rank-1 form. Returns the 3x1 single-layer and
# adjoint blocks and the 3x3 double-layer and hypersingular blocks
# (column-major, real and imaginary parts separately).
@inline function _metal_regular_pair_blocks(
    face_vertices,
    normals,
    areas,
    curls,
    rule_points,
    rule_weights,
    element_rule_points,
    test_index::Int32,
    trial_index::Int32,
    face_count::Int32,
    k,
    ::Val{R},
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
) where {R}
    @inbounds return _metal_regular_pair_blocks_inbounds(
        face_vertices, normals, areas, curls, rule_points, rule_weights, element_rule_points,
        test_index, trial_index, face_count, k, Val(R),
        trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
end

@inline function _metal_regular_pair_blocks_inbounds(
    face_vertices, normals, areas, curls, rule_points, rule_weights, element_rule_points,
    test_index::Int32, trial_index::Int32, face_count::Int32, k, ::Val{R},
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
) where {R}
    T = typeof(k)
    inv_four_pi = T(0.07957747154594767)
    k2 = k * k
    slp_re = zero(SVector{3,T})
    slp_im = zero(SVector{3,T})
    adj_re = zero(SVector{3,T})
    adj_im = zero(SVector{3,T})
    dlp_re = zero(SVector{9,T})
    dlp_im = zero(SVector{9,T})
    hyp_re = zero(SVector{9,T})
    hyp_im = zero(SVector{9,T})
    test_nx = normals[test_index]
    test_ny = normals[test_index + face_count]
    test_nz = normals[test_index + Int32(2) * face_count]
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + Int32(2) * face_count]
    normal_product = test_nx * trial_nx + test_ny * trial_ny + test_nz * trial_nz
    jac_scale = T(4) * areas[test_index] * areas[trial_index]
    trial_signs = SVector(trial_sign_x, trial_sign_y, trial_sign_z)

    # Rank-1 structure: every 3x3 block a pair contributes is
    # sum_a sum_b (test basis at a) x (trial basis at b) * scalar(a, b), so the
    # inner loop over b only needs 3-vector (and scalar) accumulators, and the
    # outer products are applied once per test point. That is 16 FMAs per
    # point pair instead of 48, and the hypersingular curl term needs only the
    # summed Green's function.
    g_total_re = zero(k)
    g_total_im = zero(k)
    test_q = Int32(1)
    while test_q <= Int32(R)
        test_xi = rule_points[test_q]
        test_eta = rule_points[test_q + Int32(R)]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        test_basis = SVector(tb1, tb2, tb3)
        point_index = test_index + face_count * (test_q - Int32(1))
        x = element_rule_points[point_index]
        y = element_rule_points[point_index + face_count * Int32(R)]
        z = element_rule_points[point_index + face_count * Int32(2 * R)]
        test_weight = rule_weights[test_q]

        s_re = zero(k)          # sum_b g w                 -> single layer, G0
        s_im = zero(k)
        a_re = zero(k)          # sum_b grad*test_dot w     -> adjoint double layer
        a_im = zero(k)
        d_re = zero(SVector{3,T})  # sum_b rb grad*trial_dot w -> double layer
        d_im = zero(SVector{3,T})
        h_re = zero(SVector{3,T})  # sum_b rb g w              -> hypersingular basis term
        h_im = zero(SVector{3,T})
        # The trial loop is unrolled at compile time (Val recursion) so the
        # hoisted trial points, basis values and weights are indexed by
        # constants and stay in registers; a runtime-indexed tuple in a while
        # loop is materialised in thread-private memory instead. No closure:
        # a captured-and-reassigned variable would be boxed.
        context = (x, y, z, test_weight * jac_scale, k, inv_four_pi,
            test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
            element_rule_points, rule_points, rule_weights, trial_index, face_count)
        s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = _metal_trial_fold(
            (s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im),
            context, Val(R), Val(R),
        )
        slp_re += test_basis * s_re
        slp_im += test_basis * s_im
        adj_re += test_basis * a_re
        adj_im += test_basis * a_im
        g_total_re += s_re
        g_total_im += s_im
        # Outer products, column-major: entry (row i, col j) at index i + 3 (j - 1).
        dlp_re += SVector(
            tb1 * d_re[1], tb2 * d_re[1], tb3 * d_re[1],
            tb1 * d_re[2], tb2 * d_re[2], tb3 * d_re[2],
            tb1 * d_re[3], tb2 * d_re[3], tb3 * d_re[3],
        )
        dlp_im += SVector(
            tb1 * d_im[1], tb2 * d_im[1], tb3 * d_im[1],
            tb1 * d_im[2], tb2 * d_im[2], tb3 * d_im[2],
            tb1 * d_im[3], tb2 * d_im[3], tb3 * d_im[3],
        )
        hyp_re += SVector(
            tb1 * h_re[1], tb2 * h_re[1], tb3 * h_re[1],
            tb1 * h_re[2], tb2 * h_re[2], tb3 * h_re[2],
            tb1 * h_re[3], tb2 * h_re[3], tb3 * h_re[3],
        )
        hyp_im += SVector(
            tb1 * h_im[1], tb2 * h_im[1], tb3 * h_im[1],
            tb1 * h_im[2], tb2 * h_im[2], tb3 * h_im[2],
            tb1 * h_im[3], tb2 * h_im[3], tb3 * h_im[3],
        )
        test_q += Int32(1)
    end
    # hyp so far holds the basis-weighted Green's sums; the hypersingular
    # block is curl_products * G0 - k^2 * (n.n') * (basis-weighted sums).
    k2n = k2 * normal_product
    # Computed after the loop so its nine values are not live registers during it.
    curl_products = _metal_pair_curl_products(
        curls,
        test_index,
        trial_index,
        face_count,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    hyp_re = curl_products * g_total_re - hyp_re * k2n
    hyp_im = curl_products * g_total_im - hyp_im * k2n
    return slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im
end

function _metal_regular_pair_atomic_kernel!(
    single_layer_f32,
    adjoint_double_layer_f32,
    double_layer_f32,
    hypersingular_f32,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    p1_dofs,
    element_dp0_dofs,
    rule_points,
    rule_weights,
    element_rule_points,
    element_list,
    element_count::Int32,
    k,
    p1_dof_count::Int32,
    face_count::Int32,
    ::Val{R},
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
    ::Val{scatter},
) where {R,scatter}
    position = thread_position_in_grid_2d()
    test_position = Int32(position.x)
    trial_position = Int32(position.y)
    (test_position > element_count || trial_position > element_count) && return nothing
    test_index = Int32(element_list[test_position])
    trial_index = Int32(element_list[trial_position])
    _metal_pair_is_skipped(
        faces,
        face_count,
        test_index,
        trial_index,
        pair_offsets,
        singular_trial_indices,
        skip_mode,
    ) && return nothing

    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im = _metal_regular_pair_blocks(
        face_vertices,
        normals,
        areas,
        curls,
        rule_points,
        rule_weights,
        element_rule_points,
        test_index,
        trial_index,
        face_count,
        k,
        Val(R),
        trial_sign_x,
        trial_sign_y,
        trial_sign_z,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )

    if !scatter
        # Timing diagnostic only: keep the arithmetic alive with one store per thread.
        single_layer_f32[1 + (test_index % Int32(2))] = sum(slp_re) + sum(adj_re) + sum(dlp_re) + sum(hyp_re) +
            sum(slp_im) + sum(adj_im) + sum(dlp_im) + sum(hyp_im)
        return nothing
    end
    dp0_column = Int(element_dp0_dofs[trial_index])
    p1_count = Int(p1_dof_count)
    local_row = 1
    while local_row <= 3
        row = Int(p1_dofs[test_index + Int32(local_row - 1) * face_count])
        operator_index = row + (dp0_column - 1) * p1_count
        _metal_atomic_add_complex!(single_layer_f32, operator_index, slp_re[local_row], slp_im[local_row])
        _metal_atomic_add_complex!(adjoint_double_layer_f32, operator_index, adj_re[local_row], adj_im[local_row])
        local_row += 1
    end
    local_column = 1
    while local_column <= 3
        column = Int(p1_dofs[trial_index + Int32(local_column - 1) * face_count])
        local_row = 1
        while local_row <= 3
            row = Int(p1_dofs[test_index + Int32(local_row - 1) * face_count])
            local_index = local_row + 3 * (local_column - 1)
            operator_index = row + (column - 1) * p1_count
            _metal_atomic_add_complex!(double_layer_f32, operator_index, dlp_re[local_index], dlp_im[local_index])
            _metal_atomic_add_complex!(hypersingular_f32, operator_index, hyp_re[local_index], hyp_im[local_index])
            local_row += 1
        end
        local_column += 1
    end
    return nothing
end

function _launch_metal_atomic_pair_kernels!(
    operators,
    cache::MetalRegularAssemblyCache,
    k,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    element_count = length(cache.element_indices)
    element_count == 0 && return nothing
    element_count <= typemax(Int32) || error("Metal atomic assembly supports at most $(typemax(Int32)) elements.")
    rule_count = cache.rule_count
    rule_count in (1, 3, 6) || error("Metal atomic assembly expects a 1-, 3-, or 6-point triangle rule; got $(rule_count).")
    tile_x, tile_y = _metal_atomic_tile()
    groups_x = cld(element_count, tile_x)
    groups_y = cld(element_count, tile_y)
    scatter = Val(get(ENV, "BLAB_METAL_ATOMIC_SCATTER", "1") != "0")
    Metal.@metal threads=(tile_x, tile_y) groups=(groups_x, groups_y) _metal_regular_pair_atomic_kernel!(
        reinterpret(Float32, operators.single_layer),
        reinterpret(Float32, operators.adjoint_double_layer),
        reinterpret(Float32, operators.double_layer),
        reinterpret(Float32, operators.hypersingular),
        cache.face_vertices,
        cache.normals,
        cache.areas,
        cache.faces,
        cache.curls,
        cache.p1_dofs,
        cache.element_dp0_dofs,
        cache.rule_points,
        cache.rule_weights,
        cache.element_rule_points,
        cache.color_elements,
        Int32(element_count),
        k,
        Int32(cache.p1_dof_count),
        Int32(cache.face_count),
        Val(rule_count),
        pair_offsets,
        singular_trial_indices,
        skip_mode,
        trial_sign_x,
        trial_sign_y,
        trial_sign_z,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
        scatter,
    )
    return nothing
end

# Threads per threadgroup as (test, trial); BLAB_METAL_ATOMIC_TILE=16x16 (default), 32x8, 8x32, ...
function _metal_atomic_tile()
    text = get(ENV, "BLAB_METAL_ATOMIC_TILE", "16x16")
    parts = split(lowercase(strip(text)), "x")
    length(parts) == 2 || error("BLAB_METAL_ATOMIC_TILE must look like 16x16; got $(text).")
    tile = (parse(Int, parts[1]), parse(Int, parts[2]))
    all(>(0), tile) && prod(tile) <= 1024 || error("BLAB_METAL_ATOMIC_TILE must have at most 1024 threads.")
    return tile
end

# Singular-block scatter: one thread per singular pair, atomically adding its
# compact 3x1 and 3x3 Duffy blocks into the operators. Replaces the two
# entry-owned gather kernels (which walk every dense entry and search the
# pair lists) with ~12 atomics per singular pair.
function _metal_singular_block_scatter_kernel!(
    single_layer_f32,
    adjoint_double_layer_f32,
    double_layer_f32,
    hypersingular_f32,
    slp_values,
    adjoint_values,
    dlp_values,
    hypersingular_values,
    test_indices,
    trial_indices,
    p1_dofs,
    element_dp0_dofs,
    pair_count,
    part_count,
    p1_dof_count,
    face_count,
)
    pair_position = _metal_global_linear_index()
    pair_position > pair_count && return nothing
    value_stride = pair_count * part_count
    test_index = Int(test_indices[pair_position])
    trial_index = Int(trial_indices[pair_position])
    dp0_column = Int(element_dp0_dofs[trial_index])
    local_row = 1
    while local_row <= 3
        row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
        operator_index = row + (dp0_column - 1) * p1_dof_count
        slp = zero(eltype(slp_values))
        adj = zero(eltype(adjoint_values))
        part = 1
        while part <= part_count
            value_index = pair_position + (part - 1) * pair_count + (local_row - 1) * value_stride
            slp += slp_values[value_index]
            adj += adjoint_values[value_index]
            part += 1
        end
        _metal_atomic_add_complex!(single_layer_f32, operator_index, real(slp), imag(slp))
        _metal_atomic_add_complex!(adjoint_double_layer_f32, operator_index, real(adj), imag(adj))
        local_row += 1
    end
    local_column = 1
    while local_column <= 3
        column = Int(p1_dofs[trial_index + (local_column - 1) * face_count])
        local_row = 1
        while local_row <= 3
            row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
            operator_index = row + (column - 1) * p1_dof_count
            component = (local_column - 1) * 3 + local_row - 1
            dlp = zero(eltype(dlp_values))
            hyp = zero(eltype(hypersingular_values))
            part = 1
            while part <= part_count
                value_index = pair_position + (part - 1) * pair_count + component * value_stride
                dlp += dlp_values[value_index]
                hyp += hypersingular_values[value_index]
                part += 1
            end
            _metal_atomic_add_complex!(double_layer_f32, operator_index, real(dlp), imag(dlp))
            _metal_atomic_add_complex!(hypersingular_f32, operator_index, real(hyp), imag(hyp))
            local_row += 1
        end
        local_column += 1
    end
    return nothing
end

# The Duffy pair arithmetic shared by the four-operator scatter kernel and the
# fused Burton-Miller scatter kernel: one thread's part of one singular pair,
# returning the 3x1 single-layer and adjoint blocks and the 3x3 double-layer
# and hypersingular blocks.
@inline function _metal_singular_pair_blocks(
    linear_index::Int32,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    normal_products,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    curls,
    k,
    face_count::Int32,
    pair_count::Int32,
    rule_point_count::Int32,
    part_count::Int32,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    pair_position = (linear_index - Int32(1)) % pair_count + Int32(1)
    part = (linear_index - Int32(1)) ÷ pair_count + Int32(1)
    T = typeof(k)
    @inbounds begin
        test_index = Int32(test_indices[pair_position])
        trial_index = Int32(trial_indices[pair_position])
        rule_index = Int32(rule_indices[pair_position])
        q_first = Int32(rule_offsets[rule_index])
        q_last = Int32(rule_offsets[rule_index + Int32(1)]) - Int32(1)
        per_part = cld(q_last - q_first + Int32(1), part_count)
        q = q_first + (part - Int32(1)) * per_part
        q_stop = min(q + per_part - Int32(1), q_last)
        jac_scale = jac_scales[pair_position]
        normal_product = normal_products[pair_position]
        test_nx = normals[test_index]
        test_ny = normals[test_index + face_count]
        test_nz = normals[test_index + Int32(2) * face_count]
        trial_nx = trial_sign_x * normals[trial_index]
        trial_ny = trial_sign_y * normals[trial_index + face_count]
        trial_nz = trial_sign_z * normals[trial_index + Int32(2) * face_count]
    end
    inv_four_pi = T(0.07957747154594767)
    slp_re = zero(SVector{3,T}); slp_im = zero(SVector{3,T})
    adj_re = zero(SVector{3,T}); adj_im = zero(SVector{3,T})
    dlp_re = zero(SVector{9,T}); dlp_im = zero(SVector{9,T})
    hb_re = zero(SVector{9,T}); hb_im = zero(SVector{9,T})
    g_total_re = zero(T); g_total_im = zero(T)
    while q <= q_stop
        @inbounds begin
            test_xi = rule_test_points[q]
            test_eta = rule_test_points[q + rule_point_count]
            trial_xi = rule_trial_points[q]
            trial_eta = rule_trial_points[q + rule_point_count]
            weight = rule_weights[q] * jac_scale
        end
        tb1 = one(k) - test_xi - test_eta
        rb1 = one(k) - trial_xi - trial_eta
        x, y, z = _metal_face_point(face_vertices, test_index, face_count, tb1, test_xi, test_eta)
        sx, sy, sz = _metal_face_point(face_vertices, trial_index, face_count, rb1, trial_xi, trial_eta)
        Base.@fastmath begin
            dx = sx * trial_sign_x - x
            dy = sy * trial_sign_y - y
            dz = sz * trial_sign_z - z
            radius2 = dx * dx + dy * dy + dz * dz
            if radius2 > zero(k)
                inv_radius = _metal_fast_rsqrt(radius2)
                radius = radius2 * inv_radius
                phase = k * radius
                green_scale = inv_radius * inv_four_pi * weight
                green_re = _metal_fast_cos(phase) * green_scale
                green_im = _metal_fast_sin(phase) * green_scale
                grad_re = -green_re * inv_radius - green_im * k
                grad_im = green_re * k - green_im * inv_radius
                test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
                tb = SVector(tb1, test_xi, test_eta)
                outer = SVector(
                    tb1 * rb1, test_xi * rb1, test_eta * rb1,
                    tb1 * trial_xi, test_xi * trial_xi, test_eta * trial_xi,
                    tb1 * trial_eta, test_xi * trial_eta, test_eta * trial_eta,
                )
                slp_re += tb * green_re
                slp_im += tb * green_im
                adj_re += tb * (grad_re * test_dot)
                adj_im += tb * (grad_im * test_dot)
                dlp_re += outer * (grad_re * trial_dot)
                dlp_im += outer * (grad_im * trial_dot)
                hb_re += outer * green_re
                hb_im += outer * green_im
                g_total_re += green_re
                g_total_im += green_im
            end
        end
        q += Int32(1)
    end
    curl_products = _metal_pair_curl_products(
        curls, test_index, trial_index, face_count,
        trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    k2n = k * k * normal_product
    hyp_re = curl_products * g_total_re - hb_re * k2n
    hyp_im = curl_products * g_total_im - hb_im * k2n
    return slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im
end

# Fused Duffy block kernel: one thread per (singular pair, part) evaluates the
# Green's function once per point pair for all four operators, accumulating
# in the same rank-1 form as the regular kernel. The separate slp/adjoint and
# dlp/hyp block kernels evaluated it four times per point pair (the dlp/hyp
# kernel ran one thread per trial basis function). A pair's Duffy rule
# (512-1536 point pairs at singular order 4) is split into `part_count`
# contiguous ranges so the launch has enough threads; the scatter kernel sums
# the parts.
function _metal_singular_fused_blocks_kernel!(
    slp_values,
    adjoint_values,
    dlp_values,
    hypersingular_values,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    normal_products,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    curls,
    k,
    face_count::Int32,
    pair_count::Int32,
    rule_point_count::Int32,
    part_count::Int32,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    linear_index = Int32(thread_position_in_grid_1d())
    linear_index > pair_count * part_count && return nothing
    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im = _metal_singular_pair_blocks(
        linear_index,
        test_indices, trial_indices, rule_indices, jac_scales, normal_products,
        rule_offsets, rule_test_points, rule_trial_points, rule_weights,
        face_vertices, normals, curls, k, face_count, pair_count, rule_point_count, part_count,
        trial_sign_x, trial_sign_y, trial_sign_z,
        trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    value_stride = pair_count * part_count
    @inbounds begin
        i = 1
        while i <= 3
            slp_values[linear_index + Int32(i - 1) * value_stride] = Complex(slp_re[i], slp_im[i])
            adjoint_values[linear_index + Int32(i - 1) * value_stride] = Complex(adj_re[i], adj_im[i])
            i += 1
        end
        i = 1
        while i <= 9
            dlp_values[linear_index + Int32(i - 1) * value_stride] = Complex(dlp_re[i], dlp_im[i])
            hypersingular_values[linear_index + Int32(i - 1) * value_stride] = Complex(hyp_re[i], hyp_im[i])
            i += 1
        end
    end
    return nothing
end

function _metal_singular_part_count()
    parts = parse(Int, get(ENV, "BLAB_METAL_SINGULAR_PARTS", "4"))
    1 <= parts <= 64 || error("BLAB_METAL_SINGULAR_PARTS must be 1..64; got $(parts).")
    return parts
end

function _launch_metal_singular_block_scatter_kernels!(
    operators,
    regular_cache::MetalRegularAssemblyCache,
    singular_cache::MetalSingularCorrectionCache,
    k,
    transform::SymmetryTransform=SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1),
)
    pair_count = singular_cache.pair_count
    pair_count == 0 && return nothing
    sx = typeof(k)(transform.signs[1])
    sy = typeof(k)(transform.signs[2])
    sz = typeof(k)(transform.signs[3])
    csx = typeof(k)(transform.determinant * transform.signs[1])
    csy = typeof(k)(transform.determinant * transform.signs[2])
    csz = typeof(k)(transform.determinant * transform.signs[3])
    rule_point_count = length(singular_cache.rule_weights)
    part_count = _metal_singular_part_count()
    value_count = pair_count * part_count
    slp_values = Metal.zeros(eltype(operators.single_layer), value_count, 3)
    adjoint_values = Metal.zeros(eltype(operators.adjoint_double_layer), value_count, 3)
    dlp_values = Metal.zeros(eltype(operators.double_layer), value_count, 9)
    hypersingular_values = Metal.zeros(eltype(operators.hypersingular), value_count, 9)
    _metal_launch(
        _metal_singular_fused_blocks_kernel!,
        value_count,
        slp_values, adjoint_values, dlp_values, hypersingular_values,
        singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
        singular_cache.jac_scales, singular_cache.normal_products, singular_cache.rule_offsets,
        singular_cache.rule_test_points, singular_cache.rule_trial_points, singular_cache.rule_weights,
        regular_cache.face_vertices, regular_cache.normals, regular_cache.curls,
        k, Int32(regular_cache.face_count), Int32(pair_count), Int32(rule_point_count), Int32(part_count),
        sx, sy, sz, csx, csy, csz,
    )
    _metal_launch(
        _metal_singular_block_scatter_kernel!,
        pair_count,
        reinterpret(Float32, operators.single_layer),
        reinterpret(Float32, operators.adjoint_double_layer),
        reinterpret(Float32, operators.double_layer),
        reinterpret(Float32, operators.hypersingular),
        slp_values, adjoint_values, dlp_values, hypersingular_values,
        singular_cache.test_indices, singular_cache.trial_indices,
        regular_cache.p1_dofs, regular_cache.element_dp0_dofs,
        pair_count, part_count, regular_cache.p1_dof_count, regular_cache.face_count,
    )
    Metal.synchronize()
    Metal.unsafe_free!(slp_values)
    Metal.unsafe_free!(adjoint_values)
    Metal.unsafe_free!(dlp_values)
    Metal.unsafe_free!(hypersingular_values)
    return nothing
end

function _launch_metal_regular_atomic_kernels!(operators, cache::MetalRegularAssemblyCache, k)
    return _launch_metal_atomic_pair_kernels!(
        operators,
        cache,
        k,
        cache.vertex_offsets,
        cache.incident_elements,
        Int32(0),
        one(k), one(k), one(k),
        one(k), one(k), one(k),
    )
end

function _launch_metal_symmetry_regular_atomic_kernels!(
    operators,
    cache::MetalRegularAssemblyCache,
    image_cache::MetalSingularCorrectionCache,
    transform::SymmetryTransform,
    k;
    skip_image_singular::Bool,
)
    sx = typeof(k)(transform.signs[1])
    sy = typeof(k)(transform.signs[2])
    sz = typeof(k)(transform.signs[3])
    csx = typeof(k)(transform.determinant * transform.signs[1])
    csy = typeof(k)(transform.determinant * transform.signs[2])
    csz = typeof(k)(transform.determinant * transform.signs[3])
    return _launch_metal_atomic_pair_kernels!(
        operators,
        cache,
        k,
        image_cache.pair_offsets,
        image_cache.trial_indices,
        skip_image_singular ? Int32(1) : Int32(2),
        sx, sy, sz,
        csx, csy, csz,
    )
end
