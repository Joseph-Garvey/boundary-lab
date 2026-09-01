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

@inline function _metal_atomic_add_complex!(target_f32, complex_index, re, im)
    base = 2 * complex_index - 1
    atomic_fetch_add_explicit(pointer(target_f32, base), re)
    atomic_fetch_add_explicit(pointer(target_f32, base + 1), im)
    return nothing
end

@inline function _metal_hoisted_points(face_vertices, element_index, face_count, rule_points, ::Val{R}) where {R}
    return ntuple(Val(R)) do q
        xi = rule_points[q]
        eta = rule_points[q + R]
        x, y, z = _metal_face_point(face_vertices, element_index, face_count, one(xi) - xi - eta, xi, eta)
        SVector(x, y, z)
    end
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
) where {R}
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
    curl_products = _metal_pair_curl_products(
        curls,
        test_index,
        trial_index,
        face_count,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    jac_scale = T(4) * areas[test_index] * areas[trial_index]
    trial_signs = SVector(trial_sign_x, trial_sign_y, trial_sign_z)
    trial_points = _metal_hoisted_points(face_vertices, trial_index, face_count, rule_points, Val(R))
    trial_weights = ntuple(q -> rule_weights[q], Val(R))
    trial_basis = ntuple(Val(R)) do q
        xi = rule_points[q]
        eta = rule_points[q + R]
        SVector(one(xi) - xi - eta, xi, eta)
    end

    test_q = Int32(1)
    while test_q <= Int32(R)
        test_xi = rule_points[test_q]
        test_eta = rule_points[test_q + Int32(R)]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        test_basis = SVector(tb1, tb2, tb3)
        x, y, z = _metal_face_point(face_vertices, test_index, face_count, tb1, tb2, tb3)
        test_weight = rule_weights[test_q]

        trial_q = 1
        while trial_q <= R
            source = trial_points[trial_q] .* trial_signs
            dx = source[1] - x
            dy = source[2] - y
            dz = source[3] - z
            radius2 = dx * dx + dy * dy + dz * dz
            if radius2 > zero(k)
                rb = trial_basis[trial_q]
                # Fast-math intrinsics: this is what an Xcode-compiled Metal
                # shader gets by default, and what hornlab-metal-bem runs.
                # The precise AIR sin/cos/sqrt/divide are software sequences on
                # Apple GPUs and dominate this loop when used.
                inv_radius = _metal_fast_rsqrt(radius2)
                radius = radius2 * inv_radius
                phase = k * radius
                green_scale = inv_radius * inv_four_pi
                green_re = _metal_fast_cos(phase) * green_scale
                green_im = _metal_fast_sin(phase) * green_scale
                weight = test_weight * trial_weights[trial_q] * jac_scale
                weighted_basis = test_basis * weight
                slp_re += weighted_basis * green_re
                slp_im += weighted_basis * green_im

                grad_re = -green_re * inv_radius - green_im * k
                grad_im = green_re * k - green_im * inv_radius
                test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                adj_re += weighted_basis * (grad_re * test_dot)
                adj_im += weighted_basis * (grad_im * test_dot)

                trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
                basis_products = SVector(
                    tb1 * rb[1], tb2 * rb[1], tb3 * rb[1],
                    tb1 * rb[2], tb2 * rb[2], tb3 * rb[2],
                    tb1 * rb[3], tb2 * rb[3], tb3 * rb[3],
                )
                dlp_re += basis_products * (grad_re * trial_dot * weight)
                dlp_im += basis_products * (grad_im * trial_dot * weight)
                hyper_factors = curl_products - basis_products * (k2 * normal_product)
                hyp_re += hyper_factors * (green_re * weight)
                hyp_im += hyper_factors * (green_im * weight)
            end
            trial_q += 1
        end
        test_q += Int32(1)
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
    tile = 16
    groups = cld(element_count, tile)
    Metal.@metal threads=(tile, tile) groups=(groups, groups) _metal_regular_pair_atomic_kernel!(
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
    )
    return nothing
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
    p1_dof_count,
    face_count,
)
    pair_position = _metal_global_linear_index()
    pair_position > pair_count && return nothing
    test_index = Int(test_indices[pair_position])
    trial_index = Int(trial_indices[pair_position])
    dp0_column = Int(element_dp0_dofs[trial_index])
    local_row = 1
    while local_row <= 3
        row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
        value_index = pair_position + (local_row - 1) * pair_count
        operator_index = row + (dp0_column - 1) * p1_dof_count
        slp = slp_values[value_index]
        adj = adjoint_values[value_index]
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
            value_index = pair_position + ((local_column - 1) * 3 + local_row - 1) * pair_count
            operator_index = row + (column - 1) * p1_dof_count
            dlp = dlp_values[value_index]
            hyp = hypersingular_values[value_index]
            _metal_atomic_add_complex!(double_layer_f32, operator_index, real(dlp), imag(dlp))
            _metal_atomic_add_complex!(hypersingular_f32, operator_index, real(hyp), imag(hyp))
            local_row += 1
        end
        local_column += 1
    end
    return nothing
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
    slp_values = Metal.zeros(eltype(operators.single_layer), pair_count, 3)
    adjoint_values = Metal.zeros(eltype(operators.adjoint_double_layer), pair_count, 3)
    dlp_values = Metal.zeros(eltype(operators.double_layer), pair_count, 9)
    hypersingular_values = Metal.zeros(eltype(operators.hypersingular), pair_count, 9)
    _metal_launch(
        _metal_singular_slp_adjoint_blocks_kernel!,
        pair_count,
        slp_values, adjoint_values,
        singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
        singular_cache.jac_scales, singular_cache.rule_offsets,
        singular_cache.rule_test_points, singular_cache.rule_trial_points, singular_cache.rule_weights,
        regular_cache.face_vertices, regular_cache.normals,
        k, regular_cache.face_count, pair_count, rule_point_count,
        sx, sy, sz,
    )
    _metal_launch(
        _metal_singular_dlp_hyp_blocks_kernel!,
        3 * pair_count,
        dlp_values, hypersingular_values,
        singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
        singular_cache.jac_scales, singular_cache.normal_products, singular_cache.rule_offsets,
        singular_cache.rule_test_points, singular_cache.rule_trial_points, singular_cache.rule_weights,
        regular_cache.face_vertices, regular_cache.normals, regular_cache.curls,
        k, regular_cache.face_count, pair_count, rule_point_count,
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
        pair_count, regular_cache.p1_dof_count, regular_cache.face_count,
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
