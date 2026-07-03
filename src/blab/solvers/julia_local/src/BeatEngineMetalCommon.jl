using Metal

# Device-resident geometry/quadrature cache for the (phase-2) GPU regular assembly. Mirrors
# CudaRegularAssemblyCache; arrays are MtlArrays. Held across frequencies so fixed mesh geometry is
# transferred once.
struct MetalRegularAssemblyCache{T}
    face_vertices
    normals
    areas
    faces
    curls
    rule_points
    rule_weights
    test_indices
    trial_indices
    element_indices::Vector{Int}
    face_count::Int
    rule_count::Int
end

# Device-resident field-evaluation cache. Mirrors CudaFieldEvaluationCache.
struct MetalFieldEvaluationCache{T}
    source_points
    source_normals
    source_weights
    source_faces
    source_elements
    basis_values
    source_count::Int
end

# Number of GPU threads per evaluation/assembly block. Apple GPUs prefer threadgroup sizes that are
# multiples of the 32-wide SIMD group; 256 matches the CUDA configuration and is a safe default.
# TODO(apple-hw): tune against maxTotalThreadsPerThreadgroup for the target device.
const METAL_DEFAULT_THREADS = 256

# TODO(apple-hw): Metal threadgroup-reduction helper. The intrinsic spellings below
# (thread_position_in_threadgroup_1d / MtlThreadGroupBarrier) follow current Metal.jl; confirm them
# against the installed Metal.jl version, including 1-based vs 0-based indexing.
@inline function _metal_threadgroup_sum!(scratch, value, tid::Integer, threads::Integer)
    scratch[tid] = value
    MtlThreadGroupBarrier()
    offset = threads >>> 1
    while offset > 0
        if tid <= offset
            scratch[tid] += scratch[tid + offset]
        end
        MtlThreadGroupBarrier()
        offset >>>= 1
    end
    return scratch[1]
end
