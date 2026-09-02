# Fused Burton-Miller exterior assembly for the CPU backend.
#
# Same change as the Metal path in BeatEngineMetalBurtonMiller.jl, and for the
# same reasons: the coupling eta = i/k is known at assembly time, so
#
#   lhs = 0.5 I_p1p1 - D + (i/k) H          rhs = (-S - (i/k)(K' + 0.5 I)) q
#
# can be formed per element pair instead of assembling S, K', D and H and
# combining them afterwards. The result is one N x N matrix and one right-hand
# side per drive rather than 6N^2 complex entries.
#
# What it is NOT worth: the per-quadrature-point arithmetic is unchanged. The
# combination is formed once per pair from the finished 3x1 and 3x3 blocks, and
# fusing the accumulators themselves would not remove a single complex FMA from
# the inner loop (both operators need the same two products per entry). The
# Metal probe measured the same thing on the GPU: 82-84% of the pair kernel is
# arithmetic, not stores. So this path buys memory and scatter traffic.
#
# The four-operator path stays for the coupled FEM/LEM solver and for any
# operator preconditioner. This is an exterior-only fast path.
#
# Every stage is test-element owned exactly as the four-operator assembly is,
# so the same colouring keeps the threaded writes race-free.

@inline function _beat_cpu_bm_scatter_blocks!(
    lhs,
    rhs,
    q_neumann,
    test_p1_dofs::NTuple{3,Int},
    trial_p1_dofs::NTuple{3,Int},
    dp0_dof::Int,
    single_block,
    adjoint_block,
    double_block,
    hyper_block,
    coupling::Complex{T},
    ::Val{subtract},
) where {T<:AbstractFloat,subtract}
    drive_count = size(q_neumann, 2)
    @inbounds for local_row in 1:3
        row = test_p1_dofs[local_row]
        coefficient = -single_block[local_row] - coupling * adjoint_block[local_row]
        subtract && (coefficient = -coefficient)
        for drive in 1:drive_count
            rhs[row, drive] += coefficient * q_neumann[dp0_dof, drive]
        end
        for local_col in 1:3
            column = trial_p1_dofs[local_col]
            value = -double_block[local_row, local_col] + coupling * hyper_block[local_row, local_col]
            lhs[row, column] += subtract ? -value : value
        end
    end
    return nothing
end

# The Duffy/tensor-product pair's four blocks. The four-operator path's
# `_beat_cpu_accumulate_pair!` scatters inside the quadrature loop instead of
# accumulating blocks, so this is a separate loop rather than an extraction;
# the arithmetic per point is identical.
function _beat_cpu_bm_pair_blocks(
    test_vertices::NTuple{3,SVector{3,T}},
    trial_vertices::NTuple{3,SVector{3,T}},
    test_normal::SVector{3,T},
    trial_normal::SVector{3,T},
    test_curls::NTuple{3,SVector{3,T}},
    trial_curls::NTuple{3,SVector{3,T}},
    normal_product::T,
    jac_scale::T,
    k::T,
    test_points,
    trial_points,
    weights,
) where {T<:AbstractFloat}
    single_block = zero(MVector{3,Complex{T}})
    adjoint_block = zero(MVector{3,Complex{T}})
    double_block = zero(MMatrix{3,3,Complex{T},9})
    hyper_block = zero(MMatrix{3,3,Complex{T},9})
    curl_products = MMatrix{3,3,T,9}(undef)
    for local_row in 1:3
        for local_col in 1:3
            curl_products[local_row, local_col] = dot(test_curls[local_row], trial_curls[local_col])
        end
    end
    k2 = k * k

    @inbounds for q in eachindex(weights)
        test_basis = p1_values(test_points[q])
        trial_basis = p1_values(trial_points[q])
        x = local_to_global(test_vertices, test_points[q])
        y = local_to_global(trial_vertices, trial_points[q])
        r_vec = y - x
        radius = norm(r_vec)
        radius == zero(T) && continue

        inv_radius = inv(radius)
        green = _beat_cpu_green(radius, inv_radius, k)
        grad_scale = green * Complex{T}(-inv_radius, k)
        trial_dot = dot(r_vec, trial_normal) * inv_radius
        test_dot = -dot(r_vec, test_normal) * inv_radius
        weight = weights[q] * jac_scale
        weighted_green = green * weight
        weighted_double = grad_scale * trial_dot * weight
        weighted_adjoint = grad_scale * test_dot * weight

        for local_row in 1:3
            test_value = test_basis[local_row]
            single_block[local_row] += test_value * weighted_green
            adjoint_block[local_row] += test_value * weighted_adjoint
            for local_col in 1:3
                basis_product = test_value * trial_basis[local_col]
                double_block[local_row, local_col] += basis_product * weighted_double
                hyper_block[local_row, local_col] += (
                    curl_products[local_row, local_col] - k2 * basis_product * normal_product
                ) * weighted_green
            end
        end
    end

    return single_block, adjoint_block, double_block, hyper_block
end

function _beat_cpu_bm_regular_test!(
    lhs, rhs, q_neumann, elements, test_index::Int, trial_indices, k::T,
    regular_quadrature, coupling::Complex{T},
) where {T<:AbstractFloat}
    test_data = elements[test_index]
    test_quad = regular_quadrature[test_index]
    for trial_index in trial_indices
        trial_data = elements[trial_index]
        elements_are_adjacent(test_data.face, trial_data.face) && continue
        single_block, adjoint_block, double_block, hyper_block = _beat_cpu_regular_pair_blocks(
            test_data, trial_data, test_quad, regular_quadrature[trial_index],
            dot(test_data.normal, trial_data.normal),
            T(4.0) * test_data.area * trial_data.area, k,
        )
        _beat_cpu_bm_scatter_blocks!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            single_block, adjoint_block, double_block, hyper_block, coupling, Val(false),
        )
    end
    return nothing
end

function _beat_cpu_bm_regular_image_test!(
    lhs, rhs, q_neumann, elements, image_elements, test_index::Int, trial_indices, k::T,
    regular_quadrature, image_quadrature, coupling::Complex{T},
) where {T<:AbstractFloat}
    test_data = elements[test_index]
    test_quad = regular_quadrature[test_index]
    for trial_index in trial_indices
        trial_data = image_elements[trial_index]
        single_block, adjoint_block, double_block, hyper_block = _beat_cpu_regular_pair_blocks(
            test_data, trial_data, test_quad, image_quadrature[trial_index],
            dot(test_data.normal, trial_data.normal),
            T(4.0) * test_data.area * trial_data.area, k,
        )
        _beat_cpu_bm_scatter_blocks!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            single_block, adjoint_block, double_block, hyper_block, coupling, Val(false),
        )
    end
    return nothing
end

function _beat_cpu_bm_singular_test!(
    lhs, rhs, q_neumann, elements, pairs, rules, k::T, coupling::Complex{T},
) where {T<:AbstractFloat}
    for pair in pairs
        test_data = elements[pair.test_index]
        trial_data = elements[pair.trial_index]
        duffy = rules[pair.rule_index]
        single_block, adjoint_block, double_block, hyper_block = _beat_cpu_bm_pair_blocks(
            test_data.vertices, trial_data.vertices, test_data.normal, trial_data.normal,
            test_data.curls, trial_data.curls, pair.normal_product, pair.jac_scale, k,
            duffy.test_points, duffy.trial_points, duffy.weights,
        )
        _beat_cpu_bm_scatter_blocks!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            single_block, adjoint_block, double_block, hyper_block, coupling, Val(false),
        )
    end
    return nothing
end

# Image-singular correction: the Duffy value minus the regular-rule value the
# image pass already added, combined with the same eta before the scatter. The
# four-operator path applies four separate deltas here; folding them is the
# most likely place for a silent sign error, which is why the equivalence gate
# runs the symmetry fixtures.
function _beat_cpu_bm_image_singular_delta_test!(
    lhs, rhs, q_neumann, elements, image_elements, pairs, rules, k::T,
    regular_quadrature, image_quadrature, coupling::Complex{T},
) where {T<:AbstractFloat}
    for pair in pairs
        test_data = elements[pair.test_index]
        trial_data = image_elements[pair.trial_index]
        duffy = rules[pair.rule_index]
        single_block, adjoint_block, double_block, hyper_block = _beat_cpu_bm_pair_blocks(
            test_data.vertices, trial_data.vertices, test_data.normal, trial_data.normal,
            test_data.curls, trial_data.curls, pair.normal_product, pair.jac_scale, k,
            duffy.test_points, duffy.trial_points, duffy.weights,
        )
        _beat_cpu_bm_scatter_blocks!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            single_block, adjoint_block, double_block, hyper_block, coupling, Val(false),
        )
        regular_single, regular_adjoint, regular_double, regular_hyper = _beat_cpu_regular_pair_blocks(
            test_data, trial_data, regular_quadrature[pair.test_index],
            image_quadrature[pair.trial_index], pair.normal_product,
            T(4.0) * test_data.area * trial_data.area, k,
        )
        _beat_cpu_bm_scatter_blocks!(
            lhs, rhs, q_neumann, test_data.p1_dofs, trial_data.p1_dofs, trial_data.dp0_dof,
            regular_single, regular_adjoint, regular_double, regular_hyper, coupling, Val(true),
        )
    end
    return nothing
end

"""
    assemble_burton_miller_neumann_system_cpu(mesh, p1_space, dp0_space, q_neumann, k, rule; ...)

Assemble the Burton-Miller Neumann system directly on the CPU, without forming
S, K', D or H. `q_neumann` is `dp0_dof_count x drive_count`; every drive's
right-hand side is accumulated in the same pass.

Returns `(matrix, rhs, ...)` with `matrix` `N x N` and `rhs` `N x drives`.
"""
function assemble_burton_miller_neumann_system_cpu(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_neumann,
    k::T,
    rule::TriangleRule{T};
    identity_p1_p1,
    identity_p1_dp0,
    skip_singular::Bool=false,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    threaded::Bool=true,
    timing=nothing,
    singular_cache=nothing,
    cpu_cache=nothing,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    symmetry_mode = normalized_symmetry_mode(symmetry_mode)
    if cpu_cache !== nothing
        cpu_cache.symmetry_mode == symmetry_mode || error("CPU assembly cache symmetry mode does not match the requested mode.")
        cpu_cache.singular_order == singular_order || error("CPU assembly cache singular order does not match the requested order.")
        cpu_cache.rule == rule || error("CPU assembly cache quadrature rule does not match the requested rule.")
    end
    q_host = q_neumann isa AbstractMatrix ? q_neumann : reshape(q_neumann, :, 1)
    size(q_host, 1) == dp0_space.global_dof_count ||
        error("Fused CPU Burton-Miller assembly needs one Neumann row per DP0 dof.")
    q_complex = Complex{T}.(q_host)
    drive_count = size(q_complex, 2)

    indices = cpu_cache === nothing ? collect(element_indices) : cpu_cache.indices
    p1_count = p1_space.global_dof_count
    lhs = zeros(Complex{T}, p1_count, p1_count)
    rhs = zeros(Complex{T}, p1_count, drive_count)
    coupling = Complex{T}(0, 1) / k
    elements = cpu_cache === nothing ? _beat_cpu_element_data(mesh, p1_space, dp0_space) : cpu_cache.elements
    regular_quadrature = cpu_cache === nothing ? _beat_cpu_regular_quadrature_data(mesh, rule) : cpu_cache.regular_quadrature
    adjacent_pairs = cpu_cache === nothing ? count_adjacent_pairs(mesh, indices) : cpu_cache.adjacent_pairs
    threaded_enabled = cpu_cache === nothing ? threaded && Threads.nthreads() > 1 : cpu_cache.threaded_enabled
    color_groups = if cpu_cache === nothing
        threaded_enabled ? _beat_cpu_element_color_groups(mesh, indices) : [indices]
    else
        cpu_cache.color_groups
    end
    image_transforms = cpu_cache === nothing ? collect(symmetry_image_transforms(symmetry_mode)) : cpu_cache.image_transforms

    regular_elapsed = @elapsed begin
        if threaded_enabled
            for group in color_groups
                Threads.@threads for group_index in eachindex(group)
                    _beat_cpu_bm_regular_test!(
                        lhs, rhs, q_complex, elements, group[group_index], indices, k,
                        regular_quadrature, coupling,
                    )
                end
            end
        else
            for test_index in indices
                _beat_cpu_bm_regular_test!(
                    lhs, rhs, q_complex, elements, test_index, indices, k,
                    regular_quadrature, coupling,
                )
            end
        end
        for (transform_index, transform) in enumerate(image_transforms)
            image_elements = cpu_cache === nothing ?
                _beat_cpu_reflect_element_data(elements, transform) :
                cpu_cache.image_elements[transform_index]
            image_quadrature = cpu_cache === nothing ?
                _beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform) :
                cpu_cache.image_quadrature[transform_index]
            if threaded_enabled
                for group in color_groups
                    Threads.@threads for group_index in eachindex(group)
                        _beat_cpu_bm_regular_image_test!(
                            lhs, rhs, q_complex, elements, image_elements, group[group_index],
                            indices, k, regular_quadrature, image_quadrature, coupling,
                        )
                    end
                end
            else
                for test_index in indices
                    _beat_cpu_bm_regular_image_test!(
                        lhs, rhs, q_complex, elements, image_elements, test_index,
                        indices, k, regular_quadrature, image_quadrature, coupling,
                    )
                end
            end
        end
    end
    timing !== nothing && (timing["fused_regular_cpu_scatter"] = regular_elapsed)

    singular_pairs = 0
    if !skip_singular
        cache = singular_cache === nothing ? build_singular_correction_cache(mesh, singular_order, indices) : singular_cache
        singular_elapsed = @elapsed begin
            if threaded_enabled
                for group in color_groups
                    Threads.@threads for group_index in eachindex(group)
                        _beat_cpu_bm_singular_test!(
                            lhs, rhs, q_complex, elements,
                            cache.pairs_by_test[group[group_index]], cache.rules, k, coupling,
                        )
                    end
                end
            else
                for test_index in indices
                    _beat_cpu_bm_singular_test!(
                        lhs, rhs, q_complex, elements,
                        cache.pairs_by_test[test_index], cache.rules, k, coupling,
                    )
                end
            end
        end
        timing !== nothing && (timing["fused_singular_cpu_scatter"] = singular_elapsed)
        singular_pairs = cache.pair_count
    end

    image_singular_pairs = 0
    image_singular_elapsed = @elapsed begin
        if !skip_singular
            for (transform_index, transform) in enumerate(image_transforms)
                image_cache = cpu_cache === nothing ?
                    _beat_cpu_image_singular_cache(mesh, singular_order, indices, transform) :
                    cpu_cache.image_singular_caches[transform_index]
                image_singular_pairs += image_cache.pair_count
                image_cache.pair_count == 0 && continue
                image_elements = cpu_cache === nothing ?
                    _beat_cpu_reflect_element_data(elements, transform) :
                    cpu_cache.image_elements[transform_index]
                image_quadrature = cpu_cache === nothing ?
                    _beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform) :
                    cpu_cache.image_quadrature[transform_index]
                if threaded_enabled
                    for group in color_groups
                        Threads.@threads for group_index in eachindex(group)
                            _beat_cpu_bm_image_singular_delta_test!(
                                lhs, rhs, q_complex, elements, image_elements,
                                image_cache.pairs_by_test[group[group_index]], image_cache.rules,
                                k, regular_quadrature, image_quadrature, coupling,
                            )
                        end
                    end
                else
                    for test_index in indices
                        _beat_cpu_bm_image_singular_delta_test!(
                            lhs, rhs, q_complex, elements, image_elements,
                            image_cache.pairs_by_test[test_index], image_cache.rules,
                            k, regular_quadrature, image_quadrature, coupling,
                        )
                    end
                end
            end
        end
    end
    timing !== nothing && (timing["fused_image_singular_cpu_scatter"] = image_singular_elapsed)

    # Row weights scale the operator part only: `assemble_l2_identity_matrix`
    # already applies them to both identity blocks.
    weights = p1_symmetry_orbit_weights(mesh, symmetry_mode)
    if symmetry_mode != :off
        lhs .*= reshape(Complex{T}.(weights), :, 1)
        rhs .*= reshape(Complex{T}.(weights), :, 1)
    end
    identity_elapsed = @elapsed begin
        lhs .+= Complex{T}(0.5) .* Complex{T}.(identity_p1_p1)
        rhs .+= (Complex{T}.(identity_p1_dp0) * q_complex) .* (-Complex{T}(0.5) * coupling)
    end
    timing !== nothing && (timing["fused_identity_cpu"] = identity_elapsed)

    return (
        matrix=lhs,
        rhs=rhs,
        regular_pairs=length(indices) * length(indices) - adjacent_pairs +
            length(image_transforms) * length(indices) * length(indices) - image_singular_pairs,
        singular_pairs=singular_pairs,
        image_singular_pairs=image_singular_pairs,
        drive_count=drive_count,
        on_gpu=false,
        assembly_mode=:cpu_fused_burton_miller,
    )
end

function solve_burton_miller_neumann_system_cpu(system)
    return lu!(copy(system.matrix)) \ system.rhs
end
