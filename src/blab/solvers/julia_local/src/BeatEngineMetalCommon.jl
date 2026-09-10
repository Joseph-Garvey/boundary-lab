# BEAT Engine Apple Metal backend: shared cache types and runtime controls.
#
# The Metal backend mirrors the ROCm backend file for file. FEM assembly and
# condensation stay on the CPU, the four dense Burton-Miller operators and the
# exterior field are assembled on the GPU, and the dense factorization runs on
# the CPU through LAPACK because Metal.jl provides no GPU LU. Production
# precision is Float32/ComplexF32 like every other BEAT backend, which is also
# the only precision Apple GPUs support.

using Metal: MtlArray, thread_position_in_grid_1d

const BEAT_METAL_OPERATOR_STORAGE_ENV = "BLAB_METAL_OPERATOR_STORAGE"

"""
    metal_operator_storage_mode()

Storage mode for the four dense Burton-Miller operator matrices.

Apple Silicon is a unified-memory device, so a shared-storage buffer can be
wrapped as a host `Array` with no copy at all; a private-storage buffer has to
be blitted through a staging buffer, which measured 3.5-8 GB/s and cost
1.1-1.5 s per frequency at 10,230 P1 dofs. Shared is therefore the default.
Set `BLAB_METAL_OPERATOR_STORAGE=private` to fall back to the copying path.
"""
function metal_operator_storage_mode()
    requested = lowercase(strip(get(ENV, BEAT_METAL_OPERATOR_STORAGE_ENV, "shared")))
    requested in ("shared", "private") ||
        error("$BEAT_METAL_OPERATOR_STORAGE_ENV must be shared or private; got $(repr(requested)).")
    return requested == "shared" ? Metal.SharedStorage : Metal.PrivateStorage
end

struct MetalSingularCorrectionCache{T}
    pair_offsets
    test_indices
    trial_indices
    rule_indices
    jac_scales
    normal_products
    rule_offsets
    rule_test_points
    rule_trial_points
    rule_weights
    pair_count::Int
    gather_tables::Ref{Any}   # MetalSingularGatherTables, built lazily by the deterministic write-back
end

# One deterministic write-back map. `entry_indices` lists every dense-matrix cell
# the singular correction touches, once, in ascending order. `contrib_offsets` is
# the CSR row start for each of them, and `contrib_values` holds the block-value
# index of each contribution at `part == 1`; the kernel reaches the remaining
# parts by adding `pair_count` repeatedly, so the map does not grow with the part
# split. One thread owns one cell, so the adds need no atomics and always happen
# in the same order.
struct MetalSingularGatherMap
    entry_indices
    contrib_offsets
    contrib_values
    contrib_columns   # nothing, or each contribution's DP0 column (fused right-hand side only)
    entry_count::Int
end

# The three maps a singular cache needs, built together in one host pass.
# `part_count` and the regular cache are part of the key: the value indices
# depend on the part split, and the destination indices depend on the mesh the
# regular cache was built for.
struct MetalSingularGatherTables
    p1_dp0::MetalSingularGatherMap   # single layer and adjoint double layer
    p1_p1::MetalSingularGatherMap    # double layer, hypersingular, and the fused left-hand side
    rhs::MetalSingularGatherMap      # the fused right-hand side
    part_count::Int
    regular_id::UInt
end

struct MetalRegularAssemblyCache{T,C}
    host_cache::C
    face_vertices
    normals
    areas
    faces
    curls
    rule_points
    rule_weights
    element_rule_points   # face_count x rule_count x 3: every element's regular quadrature points
    vertex_offsets
    incident_elements
    incident_local_indices
    dp0_elements
    p1_dofs
    element_dp0_dofs
    color_elements
    color_offsets::Vector{Int}
    element_indices::Vector{Int}
    face_count::Int
    p1_dof_count::Int
    dp0_dof_count::Int
    rule_count::Int
    symmetry_mode::Symbol
    image_transforms::Vector{SymmetryTransform}
    image_singular_caches::Vector{MetalSingularCorrectionCache{T}}
    image_singular_pair_count::Int
    gather_tables::Ref{Any}   # MetalGatherTables, built lazily by the pair_gather kernel mode
    fused_gather_tables::Ref{Any}   # MetalFusedGatherTables, built lazily by the fused Burton-Miller path
end

struct MetalFieldEvaluationCache{T}
    source_points
    source_normals
    source_weights
    source_faces
    source_elements
    basis_values
    source_count::Int
end

function _require_metal!()
    Metal.functional() || error("Metal solve requested, but Metal.functional() is false.")
    return nothing
end

# Apple GPUs execute in SIMD groups of 32. A threadgroup of 256 keeps eight SIMD
# groups resident per group, which suits the arithmetic-heavy regular kernels.
# Override with BLAB_METAL_KERNEL_GROUPSIZE when tuning a device.
function _metal_kernel_groupsize()
    groupsize = parse(Int, get(ENV, "BLAB_METAL_KERNEL_GROUPSIZE", "256"))
    groupsize in (32, 64, 128, 256, 512, 1024) ||
        error("BLAB_METAL_KERNEL_GROUPSIZE must be 32, 64, 128, 256, 512, or 1024; got $(groupsize).")
    return groupsize
end

function _normalized_metal_regular_kernel_mode(value=nothing)
    value === nothing && (value = get(ENV, "BLAB_METAL_REGULAR_KERNEL_MODE", "pair_gather"))
    mode = Symbol(lowercase(strip(String(value))))
    aliases = Dict(
        :pair => :pair_owned,
        :pair_owned => :pair_owned,
        :colored => :pair_owned,
        :colored_pair_owned => :pair_owned,
        :entry => :entry_owned,
        :entry_owned => :entry_owned,
        :atomic => :pair_atomic,
        :pair_atomic => :pair_atomic,
        :fused_atomic => :pair_atomic,
        :gather => :pair_gather,
        :pair_gather => :pair_gather,
        :chunked => :pair_gather,
        :chunked_pair_gather => :pair_gather,
    )
    normalized = get(aliases, mode, nothing)
    normalized === nothing && error(
        "BLAB_METAL_REGULAR_KERNEL_MODE must be pair_gather, pair_atomic, pair_owned, or entry_owned; got $(value).",
    )
    return normalized
end

# Singular corrections can run as device kernels (native) or be computed on the
# CPU and added to the device operators (host). The host mode exists so a kernel
# defect can be separated from a Duffy-rule defect on the same mesh.
function _normalized_metal_singular_mode(value=nothing)
    value === nothing && (value = get(ENV, "BLAB_METAL_SINGULAR_MODE", "native"))
    mode = Symbol(lowercase(strip(String(value))))
    aliases = Dict(
        :native => :native,
        :device => :native,
        :host => :host,
        :cpu => :host,
    )
    normalized = get(aliases, mode, nothing)
    normalized === nothing && error("BLAB_METAL_SINGULAR_MODE must be native or host; got $(value).")
    return normalized
end

# The singular blocks are evaluated once and then written into the operators
# either by an atomic scatter (one thread per pair, contended cells) or by a
# gather (one thread per touched cell, no contention). Only the gather is
# reproducible run to run, so it is the default; `scatter` is kept so the two can
# be compared on the same tree.
function _normalized_metal_singular_writeback(value=nothing)
    value === nothing && (value = get(ENV, "BLAB_METAL_SINGULAR_WRITEBACK", "gather"))
    mode = Symbol(lowercase(strip(String(value))))
    mode in (:gather, :scatter) ||
        error("BLAB_METAL_SINGULAR_WRITEBACK must be gather or scatter; got $(value).")
    return mode
end

@inline function _metal_global_linear_index()
    return Int(thread_position_in_grid_1d())
end

function _metal_launch(kernel, count::Integer, args...; groupsize::Integer=_metal_kernel_groupsize())
    count <= 0 && return nothing
    Metal.@metal threads=groupsize groups=cld(count, groupsize) kernel(args...)
    return nothing
end
