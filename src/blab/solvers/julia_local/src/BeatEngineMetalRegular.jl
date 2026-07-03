# Host->device geometry/quadrature packing and the regular-assembly device cache.
#
# Packing mirrors _cuda_geometry_arrays / _cuda_rule_arrays exactly (column layouts the kernels
# index into). The device cache is built at session init in solver.jl and reused across
# frequencies. In the phase-1 hybrid the regular operators are assembled on the CPU, so this cache
# feeds only the (phase-2) GPU regular kernel; it is cheap (O(elements), not O(elements^2)) and is
# also a useful early check that Metal is functional.

function _metal_geometry_arrays(mesh::BoundaryMesh{T}) where {T}
    face_count = length(mesh.faces)
    face_vertices = Matrix{T}(undef, face_count, 9)
    normals = Matrix{T}(undef, face_count, 3)
    curls = Matrix{T}(undef, face_count, 9)
    faces = Matrix{Int32}(undef, face_count, 3)
    areas = Vector{T}(undef, face_count)

    for element_index in 1:face_count
        vertices = mesh.face_vertices[element_index]
        normal = mesh.normals[element_index]
        element_curls = surface_curls(vertices, normal)
        face = mesh.faces[element_index]
        areas[element_index] = mesh.areas[element_index]

        for i in 1:3
            face_vertices[element_index, 3 * (i - 1) + 1] = vertices[i][1]
            face_vertices[element_index, 3 * (i - 1) + 2] = vertices[i][2]
            face_vertices[element_index, 3 * (i - 1) + 3] = vertices[i][3]
            normals[element_index, i] = normal[i]
            faces[element_index, i] = Int32(face[i])
            curls[element_index, 3 * (i - 1) + 1] = element_curls[i][1]
            curls[element_index, 3 * (i - 1) + 2] = element_curls[i][2]
            curls[element_index, 3 * (i - 1) + 3] = element_curls[i][3]
        end
    end

    return face_vertices, normals, areas, faces, curls
end

function _metal_rule_arrays(rule::TriangleRule{T}) where {T}
    rule_count = length(rule.points)
    points = Matrix{T}(undef, rule_count, 2)
    weights = Vector{T}(undef, rule_count)
    for i in 1:rule_count
        points[i, 1] = rule.points[i][1]
        points[i, 2] = rule.points[i][2]
        weights[i] = rule.weights[i]
    end
    return points, weights
end

function build_metal_regular_assembly_cache(
    mesh::BoundaryMesh{T},
    rule::TriangleRule{T};
    element_indices=eachindex(mesh.faces),
) where {T<:AbstractFloat}
    Metal.functional() || error("Metal regular-pair assembly cache requested, but Metal.functional() is false.")
    indices = collect(element_indices)
    face_vertices, normals, areas, faces, curls = _metal_geometry_arrays(mesh)
    rule_points, rule_weights = _metal_rule_arrays(rule)

    return MetalRegularAssemblyCache{T}(
        MtlArray(face_vertices),
        MtlArray(normals),
        MtlArray(areas),
        MtlArray(faces),
        MtlArray(curls),
        MtlArray(rule_points),
        MtlArray(rule_weights),
        MtlArray(Int32.(indices)),
        MtlArray(Int32.(indices)),
        indices,
        length(mesh.faces),
        length(rule.points),
    )
end
