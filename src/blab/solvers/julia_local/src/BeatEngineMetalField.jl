function _metal_field_arrays(cache::FieldEvaluationCache{T}) where {T<:AbstractFloat}
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
        for coordinate in 1:3
            source_points[source_index, coordinate] = point[coordinate]
            source_normals[source_index, coordinate] = normal[coordinate]
            basis_values[source_index, coordinate] = basis[coordinate]
            source_faces[source_index, coordinate] = Int32(face[coordinate])
        end
        source_weights[source_index] = cache.source_weights[source_index]
        source_elements[source_index] = Int32(cache.source_elements[source_index])
    end
    return source_points, source_normals, source_weights, source_faces, source_elements, basis_values
end

function build_metal_field_evaluation_cache(cache::FieldEvaluationCache{T}) where {T<:AbstractFloat}
    _require_metal!()
    source_points, source_normals, source_weights, source_faces, source_elements, basis_values =
        _metal_field_arrays(cache)
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

function build_metal_field_evaluation_cache(
    mesh::BoundaryMesh{T},
    rule::TriangleRule{T};
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    return build_metal_field_evaluation_cache(build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode))
end

function release_metal_field_evaluation_cache!(cache::MetalFieldEvaluationCache)
    Metal.unsafe_free!(cache.source_points)
    Metal.unsafe_free!(cache.source_normals)
    Metal.unsafe_free!(cache.source_weights)
    Metal.unsafe_free!(cache.source_faces)
    Metal.unsafe_free!(cache.source_elements)
    Metal.unsafe_free!(cache.basis_values)
    return nothing
end

function _metal_eval_point_arrays(eval_points, ::Type{T}) where {T<:AbstractFloat}
    points = Matrix{T}(undef, length(eval_points), 3)
    for point_index in eachindex(eval_points)
        point = eval_points[point_index]
        points[point_index, 1] = T(point[1])
        points[point_index, 2] = T(point[2])
        points[point_index, 3] = T(point[3])
    end
    return points
end

function _metal_weighted_field_sources_kernel!(
    weighted_pressure,
    weighted_neumann,
    pressure,
    q_neumann,
    source_weights,
    source_faces,
    source_elements,
    basis_values,
    source_count,
)
    source_index = _metal_global_linear_index()
    source_index > source_count && return nothing
    face1 = Int(source_faces[source_index])
    face2 = Int(source_faces[source_index + source_count])
    face3 = Int(source_faces[source_index + 2 * source_count])
    basis1 = basis_values[source_index]
    basis2 = basis_values[source_index + source_count]
    basis3 = basis_values[source_index + 2 * source_count]
    weight = source_weights[source_index]
    weighted_pressure[source_index] =
        (basis1 * pressure[face1] + basis2 * pressure[face2] + basis3 * pressure[face3]) * weight
    weighted_neumann[source_index] = q_neumann[Int(source_elements[source_index])] * weight
    return nothing
end

function _metal_field_eval_entries_kernel!(
    potentials,
    eval_points,
    source_points,
    source_normals,
    weighted_pressure,
    weighted_neumann,
    k,
    source_count,
    point_count,
)
    point_index = _metal_global_linear_index()
    point_index > point_count && return nothing
    four_pi = typeof(k)(12.566370614359172)
    x1 = eval_points[point_index]
    x2 = eval_points[point_index + point_count]
    x3 = eval_points[point_index + 2 * point_count]
    potential_re = zero(k)
    potential_im = zero(k)
    source_index = 1
    while source_index <= source_count
        r1 = source_points[source_index] - x1
        r2 = source_points[source_index + source_count] - x2
        r3 = source_points[source_index + 2 * source_count] - x3
        radius2 = r1 * r1 + r2 * r2 + r3 * r3
        if radius2 > zero(k)
            radius = sqrt(radius2)
            inv_radius = one(k) / radius
            phase = k * radius
            green_scale = inv_radius / four_pi
            green_re = cos(phase) * green_scale
            green_im = sin(phase) * green_scale
            normal_projection = (
                r1 * source_normals[source_index] +
                r2 * source_normals[source_index + source_count] +
                r3 * source_normals[source_index + 2 * source_count]
            ) * inv_radius
            double_re = (-green_re * inv_radius - green_im * k) * normal_projection
            double_im = (green_re * k - green_im * inv_radius) * normal_projection
            p = weighted_pressure[source_index]
            q = weighted_neumann[source_index]
            p_re = real(p)
            p_im = imag(p)
            q_re = real(q)
            q_im = imag(q)
            potential_re += double_re * p_re - double_im * p_im - green_re * q_re + green_im * q_im
            potential_im += double_re * p_im + double_im * p_re - green_re * q_im - green_im * q_re
        end
        source_index += 1
    end
    potentials[point_index] = Complex(potential_re, potential_im)
    return nothing
end

function evaluate_galerkin_field_metal(
    eval_points,
    mesh::BoundaryMesh{T},
    pressure,
    q_neumann,
    k::T,
    cache::MetalFieldEvaluationCache{T};
    return_device::Bool=false,
) where {T<:AbstractFloat}
    point_count = length(eval_points)
    point_count == 0 && return return_device ? MtlArray(Complex{T}[]) : Complex{T}[]
    _require_metal!()
    d_eval_points = MtlArray(_metal_eval_point_arrays(eval_points, T))
    pressure_on_device = pressure isa MtlArray
    neumann_on_device = q_neumann isa MtlArray
    d_pressure = pressure_on_device ? pressure : MtlArray(Complex{T}.(pressure))
    d_neumann = neumann_on_device ? q_neumann : MtlArray(Complex{T}.(q_neumann))
    d_weighted_pressure = Metal.zeros(Complex{T}, cache.source_count)
    d_weighted_neumann = Metal.zeros(Complex{T}, cache.source_count)
    d_potentials = Metal.zeros(Complex{T}, point_count)
    groupsize = 128
    _metal_launch(
        _metal_weighted_field_sources_kernel!,
        cache.source_count,
        d_weighted_pressure,
        d_weighted_neumann,
        d_pressure,
        d_neumann,
        cache.source_weights,
        cache.source_faces,
        cache.source_elements,
        cache.basis_values,
        cache.source_count;
        groupsize=groupsize,
    )
    _metal_launch(
        _metal_field_eval_entries_kernel!,
        point_count,
        d_potentials,
        d_eval_points,
        cache.source_points,
        cache.source_normals,
        d_weighted_pressure,
        d_weighted_neumann,
        k,
        cache.source_count,
        point_count;
        groupsize=groupsize,
    )
    Metal.synchronize()
    result = return_device ? d_potentials : Complex{T}.(Array(d_potentials))
    Metal.unsafe_free!(d_eval_points)
    pressure_on_device || Metal.unsafe_free!(d_pressure)
    neumann_on_device || Metal.unsafe_free!(d_neumann)
    Metal.unsafe_free!(d_weighted_pressure)
    Metal.unsafe_free!(d_weighted_neumann)
    return_device || Metal.unsafe_free!(d_potentials)
    return result
end
