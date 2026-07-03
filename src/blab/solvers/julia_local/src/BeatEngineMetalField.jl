# Metal GPU field evaluation — 1:1 port of BeatEngineCudaField.jl.
#
# This is the phase-1 GPU compute path: the Galerkin representation-formula sweep over observation
# points (the O(point_count x source_count) hot loop). The host array packing is identical to the
# CUDA path; only the kernel launch + intrinsics differ.
#
# TODO(apple-hw): compile and validate on Apple Silicon. The Metal.jl thread-index intrinsics
# (thread_position_in_grid_1d, thread_position_in_threadgroup_1d, threadgroup_position_in_grid_1d,
# threads_per_threadgroup_1d), the MtlThreadGroupArray static size, and MtlThreadGroupBarrier must
# be confirmed against the installed Metal.jl version. Cross-check results against the CPU backend.

function _metal_field_arrays(cache::FieldEvaluationCache{T}) where {T}
    source_count = length(cache.source_points)
    source_points = Matrix{T}(undef, source_count, 3)
    source_normals = Matrix{T}(undef, source_count, 3)
    basis_values = Matrix{T}(undef, source_count, 3)
    source_weights = Vector{T}(undef, source_count)
    source_faces = Matrix{Int32}(undef, source_count, 3)
    source_elements = Vector{Int32}(undef, source_count)

    for source_index in 1:source_count
        point = cache.source_points[source_index]
        normal = cache.source_normals[source_index]
        basis = cache.basis_values[source_index]
        face = cache.source_faces[source_index]
        source_points[source_index, 1] = point[1]
        source_points[source_index, 2] = point[2]
        source_points[source_index, 3] = point[3]
        source_normals[source_index, 1] = normal[1]
        source_normals[source_index, 2] = normal[2]
        source_normals[source_index, 3] = normal[3]
        basis_values[source_index, 1] = basis[1]
        basis_values[source_index, 2] = basis[2]
        basis_values[source_index, 3] = basis[3]
        source_weights[source_index] = cache.source_weights[source_index]
        source_faces[source_index, 1] = Int32(face[1])
        source_faces[source_index, 2] = Int32(face[2])
        source_faces[source_index, 3] = Int32(face[3])
        source_elements[source_index] = Int32(cache.source_elements[source_index])
    end

    return source_points, source_normals, source_weights, source_faces, source_elements, basis_values
end

function build_metal_field_evaluation_cache(cache::FieldEvaluationCache{T}) where {T<:AbstractFloat}
    Metal.functional() || error("Metal field-evaluation cache requested, but Metal.functional() is false.")
    source_points, source_normals, source_weights, source_faces, source_elements, basis_values = _metal_field_arrays(cache)
    return MetalFieldEvaluationCache{T}(
        MtlArray(source_points),
        MtlArray(source_normals),
        MtlArray(source_weights),
        MtlArray(source_faces),
        MtlArray(source_elements),
        MtlArray(basis_values),
        length(source_weights),
    )
end

function build_metal_field_evaluation_cache(mesh::BoundaryMesh{T}, rule::TriangleRule{T}; symmetry_mode::Symbol=:off) where {T<:AbstractFloat}
    return build_metal_field_evaluation_cache(build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode))
end

function _metal_eval_point_arrays(eval_points, ::Type{T}) where {T}
    point_count = length(eval_points)
    points = Matrix{T}(undef, point_count, 3)
    for point_index in 1:point_count
        point = eval_points[point_index]
        points[point_index, 1] = T(point[1])
        points[point_index, 2] = T(point[2])
        points[point_index, 3] = T(point[3])
    end
    return points
end

function _metal_weighted_field_sources_kernel!(
    pressure_re,
    pressure_im,
    neumann_re,
    neumann_im,
    pressure,
    q_neumann,
    source_weights,
    source_faces,
    source_elements,
    basis_values,
    source_count,
)
    source_index = thread_position_in_grid_1d()
    stride = threads_per_threadgroup_1d() * threadgroups_per_grid_1d()

    while source_index <= source_count
        face1 = source_faces[source_index]
        face2 = source_faces[source_index + source_count]
        face3 = source_faces[source_index + 2 * source_count]
        basis1 = basis_values[source_index]
        basis2 = basis_values[source_index + source_count]
        basis3 = basis_values[source_index + 2 * source_count]
        weight = source_weights[source_index]

        p = (basis1 * pressure[face1] + basis2 * pressure[face2] + basis3 * pressure[face3]) * weight
        q = q_neumann[source_elements[source_index]] * weight
        pressure_re[source_index] = real(p)
        pressure_im[source_index] = imag(p)
        neumann_re[source_index] = real(q)
        neumann_im[source_index] = imag(q)

        source_index += stride
    end

    return nothing
end

function _metal_field_eval_kernel!(
    pot_re,
    pot_im,
    eval_points,
    source_points,
    source_normals,
    pressure_re,
    pressure_im,
    neumann_re,
    neumann_im,
    k,
    source_count,
    point_count,
)
    point_index = threadgroup_position_in_grid_1d()
    point_index > point_count && return nothing

    tid = thread_position_in_threadgroup_1d()
    threads = threads_per_threadgroup_1d()
    T = typeof(k)
    four_pi = T(12.566370614359172)
    scratch = MtlThreadGroupArray(T, 2 * METAL_DEFAULT_THREADS)
    local_re = zero(T)
    local_im = zero(T)

    x1 = eval_points[point_index]
    x2 = eval_points[point_index + point_count]
    x3 = eval_points[point_index + 2 * point_count]

    source_index = tid
    while source_index <= source_count
        y1 = source_points[source_index]
        y2 = source_points[source_index + source_count]
        y3 = source_points[source_index + 2 * source_count]
        r1 = y1 - x1
        r2 = y2 - x2
        r3 = y3 - x3
        radius2 = r1 * r1 + r2 * r2 + r3 * r3

        if radius2 > zero(T)
            radius = sqrt(radius2)
            phase = k * radius
            green_scale = inv(four_pi * radius)
            green_re = cos(phase) * green_scale
            green_im = sin(phase) * green_scale
            grad_scale_re = -inv(radius)
            grad_scale_im = k
            normal = (
                r1 * source_normals[source_index] +
                r2 * source_normals[source_index + source_count] +
                r3 * source_normals[source_index + 2 * source_count]
            ) / radius
            double_re = (green_re * grad_scale_re - green_im * grad_scale_im) * normal
            double_im = (green_re * grad_scale_im + green_im * grad_scale_re) * normal

            p_re = pressure_re[source_index]
            p_im = pressure_im[source_index]
            q_re = neumann_re[source_index]
            q_im = neumann_im[source_index]

            local_re += double_re * p_re - double_im * p_im - (green_re * q_re - green_im * q_im)
            local_im += double_re * p_im + double_im * p_re - (green_re * q_im + green_im * q_re)
        end

        source_index += threads
    end

    scratch[tid] = local_re
    scratch[tid + threads] = local_im
    MtlThreadGroupBarrier()

    offset = threads >>> 1
    while offset > 0
        if tid <= offset
            scratch[tid] += scratch[tid + offset]
            scratch[tid + threads] += scratch[tid + threads + offset]
        end
        MtlThreadGroupBarrier()
        offset >>>= 1
    end

    if tid == 1
        pot_re[point_index] = scratch[1]
        pot_im[point_index] = scratch[threads + 1]
    end

    return nothing
end

function evaluate_galerkin_field_metal(
    eval_points,
    mesh::BoundaryMesh{T},
    pressure,
    q_neumann,
    k::T,
    cache::MetalFieldEvaluationCache{T},
) where {T<:AbstractFloat}
    point_count = length(eval_points)
    point_count == 0 && return Complex{T}[]
    Metal.functional() || error("Metal field evaluation requested, but Metal.functional() is false.")

    d_eval_points = MtlArray(_metal_eval_point_arrays(eval_points, T))
    d_pressure = pressure isa MtlArray ? pressure : MtlArray(Complex{T}.(pressure))
    d_q_neumann = q_neumann isa MtlArray ? q_neumann : MtlArray(Complex{T}.(q_neumann))
    d_pressure_re = Metal.zeros(T, cache.source_count)
    d_pressure_im = Metal.zeros(T, cache.source_count)
    d_neumann_re = Metal.zeros(T, cache.source_count)
    d_neumann_im = Metal.zeros(T, cache.source_count)
    d_pot_re = Metal.zeros(T, point_count)
    d_pot_im = Metal.zeros(T, point_count)

    threads = METAL_DEFAULT_THREADS
    source_groups = max(cld(cache.source_count, threads), 1)
    Metal.@metal threads=threads groups=source_groups _metal_weighted_field_sources_kernel!(
        d_pressure_re,
        d_pressure_im,
        d_neumann_re,
        d_neumann_im,
        d_pressure,
        d_q_neumann,
        cache.source_weights,
        cache.source_faces,
        cache.source_elements,
        cache.basis_values,
        cache.source_count,
    )

    Metal.@metal threads=threads groups=point_count _metal_field_eval_kernel!(
        d_pot_re,
        d_pot_im,
        d_eval_points,
        cache.source_points,
        cache.source_normals,
        d_pressure_re,
        d_pressure_im,
        d_neumann_re,
        d_neumann_im,
        k,
        cache.source_count,
        point_count,
    )
    Metal.synchronize()

    result = Complex{T}.(Array(d_pot_re), Array(d_pot_im))

    _metal_free!(d_eval_points)
    pressure isa MtlArray || _metal_free!(d_pressure)
    q_neumann isa MtlArray || _metal_free!(d_q_neumann)
    _metal_free!(d_pressure_re)
    _metal_free!(d_pressure_im)
    _metal_free!(d_neumann_re)
    _metal_free!(d_neumann_im)
    _metal_free!(d_pot_re)
    _metal_free!(d_pot_im)

    return result
end
