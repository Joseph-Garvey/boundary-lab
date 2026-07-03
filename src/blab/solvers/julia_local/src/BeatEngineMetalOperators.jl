# Operator-storage helpers for the Metal backend.
#
# The phase-1 hybrid returns host Complex{Float32} operator matrices with on_gpu=false, so the
# generic release_operator_storage!(operators) in BeatEngineCore.jl already no-ops correctly and
# the dense solve dispatches to solve_burton_miller_neumann_cpu. We therefore do NOT override
# release_operator_storage! here.

# Materialize a host Complex{T} matrix from device real/imag MtlArrays (used by the phase-2 GPU
# assembly path when copying operators back to host before the CPU/Accelerate solve).
function _metal_complex_cpu_matrix(real_part, imag_part, ::Type{T}) where {T}
    return Complex{T}.(Array(real_part), Array(imag_part))
end

# Best-effort free of a device array. MtlArrays are GC-managed; unsafe_free! reclaims eagerly when
# available. TODO(apple-hw): confirm Metal.unsafe_free! exists in the installed Metal.jl version.
@inline function _metal_free!(array)
    array === nothing && return nothing
    try
        if isdefined(Metal, :unsafe_free!)
            Metal.unsafe_free!(array)
        end
    catch
    end
    return nothing
end

function _metal_release_regular_cache!(cache::MetalRegularAssemblyCache)
    for field in (cache.face_vertices, cache.normals, cache.areas, cache.faces, cache.curls,
                  cache.rule_points, cache.rule_weights, cache.test_indices, cache.trial_indices)
        _metal_free!(field)
    end
    return nothing
end
