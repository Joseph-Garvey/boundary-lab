# Chunked pair-gather regular kernel: zero atomics.
#
# The fused pair-atomic kernel is bound by atomic throughput, not arithmetic:
# every (test, trial) element pair scatters 48 Float32 atomics (3x1 single
# layer and adjoint blocks, 3x3 double layer and hypersingular blocks, real
# and imaginary), and on an Apple GPU that is ~10x the cost of evaluating the
# Green's function for the pair. This mode replaces the scatter with a plain
# store and a gather:
#
#   1. The trial elements are processed in chunks of `chunk_size` columns.
#      A 2-D pair kernel writes each pair's 48 block values to a device buffer
#      laid out [test position, trial local, component] so that a tile of
#      consecutive test positions writes consecutive addresses.
#   2. A gather kernel per operator entry sums the buffer: one thread per
#      (P1 row, trial element) for the single-layer and adjoint operators, one
#      per (P1 row, chunk node) for the double-layer and hypersingular
#      operators, walking the row's incident test elements and the node's
#      incident chunk elements. Each entry has exactly one owner per launch,
#      so the accumulation is a plain read-modify-write.
#
# Summation order is fixed, so unlike the atomic kernel this mode is
# bit-reproducible run to run. Memory traffic is 192 bytes written and read
# per pair; the chunk size is chosen from BLAB_METAL_GATHER_BUDGET_MB (512).

using Metal: thread_position_in_grid_2d

const _metal_gather_stage_timing = Dict{String,Float64}()

struct MetalGatherTables
    chunk_size::Int
    chunk_count::Int
    elements              # MtlArray{Int32}: assembly position -> global element
    element_positions     # MtlArray{Int32}: global element -> assembly position (0 if absent)
    chunk_node_offsets::Vector{Int}   # host: chunk -> first position in chunk_nodes
    chunk_nodes           # MtlArray{Int32}: chunk-node position -> global P1 dof
    inc_offsets           # MtlArray{Int32}: chunk-node position -> first entry in inc_packed
    inc_packed            # MtlArray{Int32}: (trial local - 1) * 4 + local column
    blocks                # MtlArray{Float32}: 48 * element_count * chunk_size
end

const _METAL_GATHER_COMPONENTS = 48

function _metal_gather_chunk_size(element_count::Int)
    element_count <= 0 && return 1
    budget_mb = parse(Float64, get(ENV, "BLAB_METAL_GATHER_BUDGET_MB", "512"))
    per_column_bytes = element_count * _METAL_GATHER_COMPONENTS * sizeof(Float32)
    chunk = clamp(floor(Int, budget_mb * 1e6 / per_column_bytes), 1, element_count)
    override = strip(get(ENV, "BLAB_METAL_GATHER_CHUNK", ""))
    isempty(override) || (chunk = clamp(parse(Int, override), 1, element_count))
    # Buffer indices are Int32 on the device.
    while element_count * chunk * _METAL_GATHER_COMPONENTS >= typemax(Int32) && chunk > 1
        chunk = chunk ÷ 2
    end
    return chunk
end

function _metal_gather_chunk_count(cache::MetalRegularAssemblyCache)
    element_count = length(cache.element_indices)
    element_count == 0 && return 0
    tables = cache.gather_tables[]
    tables === nothing && return cld(element_count, _metal_gather_chunk_size(element_count))
    return tables.chunk_count
end

function _metal_gather_tables(cache::MetalRegularAssemblyCache)
    tables = cache.gather_tables[]
    tables === nothing || return tables
    indices = cache.element_indices
    element_count = length(indices)
    chunk_size = _metal_gather_chunk_size(element_count)
    chunk_count = cld(element_count, chunk_size)
    p1_dofs = Array(cache.p1_dofs)   # face_count x 3
    element_positions = zeros(Int32, cache.face_count)
    for (position, element_index) in enumerate(indices)
        element_positions[element_index] = Int32(position)
    end
    chunk_node_offsets = Vector{Int}(undef, chunk_count + 1)
    chunk_node_offsets[1] = 1
    chunk_nodes = Int32[]
    inc_offsets = Int32[1]
    inc_packed = Int32[]
    for chunk in 1:chunk_count
        start = (chunk - 1) * chunk_size + 1
        stop = min(chunk * chunk_size, element_count)
        node_incidence = Dict{Int32,Vector{Int32}}()
        for position in start:stop
            element_index = indices[position]
            trial_local = position - start + 1
            for local_column in 1:3
                node = p1_dofs[element_index, local_column]
                push!(get!(node_incidence, node, Int32[]), Int32((trial_local - 1) * 4 + local_column))
            end
        end
        for node in sort!(collect(keys(node_incidence)))
            push!(chunk_nodes, node)
            append!(inc_packed, node_incidence[node])
            push!(inc_offsets, Int32(length(inc_packed) + 1))
        end
        chunk_node_offsets[chunk + 1] = length(chunk_nodes) + 1
    end
    blocks = MtlArray{Float32}(undef, _METAL_GATHER_COMPONENTS * element_count * chunk_size)
    tables = MetalGatherTables(
        chunk_size,
        chunk_count,
        MtlArray(Int32.(indices)),
        MtlArray(element_positions),
        chunk_node_offsets,
        MtlArray(chunk_nodes),
        MtlArray(inc_offsets),
        MtlArray(inc_packed),
        blocks,
    )
    cache.gather_tables[] = tables
    return tables
end

function _release_metal_gather_tables!(cache::MetalRegularAssemblyCache)
    tables = cache.gather_tables[]
    tables === nothing && return nothing
    Metal.unsafe_free!(tables.elements)
    Metal.unsafe_free!(tables.element_positions)
    Metal.unsafe_free!(tables.chunk_nodes)
    Metal.unsafe_free!(tables.inc_offsets)
    Metal.unsafe_free!(tables.inc_packed)
    Metal.unsafe_free!(tables.blocks)
    cache.gather_tables[] = nothing
    return nothing
end

@inline function _metal_store_block!(blocks, base::Int32, stride::Int32, offset::Int32, values::SVector{N,T}) where {N,T}
    i = 1
    while i <= N
        @inbounds blocks[base + (offset + Int32(i - 1)) * stride] = values[i]
        i += 1
    end
    return nothing
end

# Component layout per pair: 0-2 S re, 3-5 S im, 6-8 K' re, 9-11 K' im,
# 12-20 D re, 21-29 D im, 30-38 H re, 39-47 H im (3x3 blocks column-major).
function _metal_regular_pair_blocks_kernel!(
    blocks,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    rule_points,
    rule_weights,
    element_rule_points,
    elements,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    pair_stride::Int32,
    k,
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
    trial_local = Int32(position.y)
    (test_position > element_count || trial_local > chunk_count) && return nothing
    @inbounds test_index = Int32(elements[test_position])
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    base = test_position + element_count * (trial_local - Int32(1))
    if _metal_pair_is_skipped(
        faces,
        face_count,
        test_index,
        trial_index,
        pair_offsets,
        singular_trial_indices,
        skip_mode,
    )
        component = Int32(0)
        while component < Int32(_METAL_GATHER_COMPONENTS)
            @inbounds blocks[base + component * pair_stride] = zero(eltype(blocks))
            component += Int32(1)
        end
        return nothing
    end
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
    _metal_store_block!(blocks, base, pair_stride, Int32(0), slp_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(3), slp_im)
    _metal_store_block!(blocks, base, pair_stride, Int32(6), adj_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(9), adj_im)
    _metal_store_block!(blocks, base, pair_stride, Int32(12), dlp_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(21), dlp_im)
    _metal_store_block!(blocks, base, pair_stride, Int32(30), hyp_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(39), hyp_im)
    return nothing
end

# One thread per (P1 row, trial element of the chunk): sums the 3x1 blocks of
# the row's incident test elements into S[row, dp0(trial)] and K'[row, dp0(trial)].
function _metal_gather_slp_adjoint_kernel!(
    single_layer,
    adjoint_double_layer,
    blocks,
    elements,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    element_dp0_dofs,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    pair_stride::Int32,
    p1_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    column_base = element_count * (trial_local - Int32(1))
    s_re = zero(eltype(blocks))
    s_im = zero(eltype(blocks))
    a_re = zero(eltype(blocks))
    a_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        pair = test_position + column_base
        @inbounds s_re += blocks[pair + (local_row - Int32(1)) * pair_stride]
        @inbounds s_im += blocks[pair + (local_row + Int32(2)) * pair_stride]
        @inbounds a_re += blocks[pair + (local_row + Int32(5)) * pair_stride]
        @inbounds a_im += blocks[pair + (local_row + Int32(8)) * pair_stride]
        incident_position += Int32(1)
    end
    @inbounds dp0_column = Int32(element_dp0_dofs[trial_index])
    operator_index = row + (dp0_column - Int32(1)) * p1_count
    @inbounds single_layer[operator_index] += Complex(s_re, s_im)
    @inbounds adjoint_double_layer[operator_index] += Complex(a_re, a_im)
    return nothing
end

# One thread per (P1 row, P1 node touched by the chunk): sums the 3x3 block
# entries of every (incident test element, incident chunk element) pair into
# D[row, node] and H[row, node].
function _metal_gather_dlp_hyp_kernel!(
    double_layer,
    hypersingular,
    blocks,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    chunk_nodes,
    inc_offsets,
    inc_packed,
    node_start::Int32,
    node_count::Int32,
    element_count::Int32,
    pair_stride::Int32,
    p1_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * node_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    node_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    node_position = node_start + node_local - Int32(1)
    @inbounds column = Int32(chunk_nodes[node_position])
    @inbounds chunk_first = Int32(inc_offsets[node_position])
    @inbounds chunk_stop = Int32(inc_offsets[node_position + Int32(1)]) - Int32(1)
    d_re = zero(eltype(blocks))
    d_im = zero(eltype(blocks))
    h_re = zero(eltype(blocks))
    h_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        chunk_position = chunk_first
        while chunk_position <= chunk_stop
            @inbounds packed = Int32(inc_packed[chunk_position])
            trial_local = (packed >> 2) + Int32(1)
            local_column = packed & Int32(3)
            pair = test_position + element_count * (trial_local - Int32(1))
            component = local_row + Int32(3) * (local_column - Int32(1))   # 1..9
            @inbounds d_re += blocks[pair + (component + Int32(11)) * pair_stride]
            @inbounds d_im += blocks[pair + (component + Int32(20)) * pair_stride]
            @inbounds h_re += blocks[pair + (component + Int32(29)) * pair_stride]
            @inbounds h_im += blocks[pair + (component + Int32(38)) * pair_stride]
            chunk_position += Int32(1)
        end
        incident_position += Int32(1)
    end
    operator_index = row + (column - Int32(1)) * p1_count
    @inbounds double_layer[operator_index] += Complex(d_re, d_im)
    @inbounds hypersingular[operator_index] += Complex(h_re, h_im)
    return nothing
end

@inline function _metal_gather_stage!(name::String, timed::Bool, start::Float64)
    timed || return start
    Metal.synchronize()
    now = time()
    _metal_gather_stage_timing[name] = get(_metal_gather_stage_timing, name, 0.0) + (now - start)
    return now
end

function _launch_metal_gather_pair_kernels!(
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
    rule_count = cache.rule_count
    rule_count in (1, 3, 6) || error("Metal gather assembly expects a 1-, 3-, or 6-point triangle rule; got $(rule_count).")
    tables = _metal_gather_tables(cache)
    chunk_size = tables.chunk_size
    pair_stride = Int32(element_count * chunk_size)
    tile_x, tile_y = _metal_atomic_tile()
    groupsize = _metal_kernel_groupsize()
    p1_count = Int32(cache.p1_dof_count)
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    timed && Metal.synchronize()
    stamp = time()
    for chunk in 1:tables.chunk_count
        chunk_start = (chunk - 1) * chunk_size + 1
        chunk_count = min(chunk_size, element_count - chunk_start + 1)
        Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _metal_regular_pair_blocks_kernel!(
            tables.blocks,
            cache.face_vertices,
            cache.normals,
            cache.areas,
            cache.faces,
            cache.curls,
            cache.rule_points,
            cache.rule_weights,
            cache.element_rule_points,
            tables.elements,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            pair_stride,
            k,
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
        stamp = _metal_gather_stage!("pairs", timed, stamp)
        _metal_launch(
            _metal_gather_slp_adjoint_kernel!,
            cache.p1_dof_count * chunk_count,
            operators.single_layer,
            operators.adjoint_double_layer,
            tables.blocks,
            tables.elements,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            cache.element_dp0_dofs,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("slp_adjoint", timed, stamp)
        node_start = tables.chunk_node_offsets[chunk]
        node_count = tables.chunk_node_offsets[chunk + 1] - node_start
        _metal_launch(
            _metal_gather_dlp_hyp_kernel!,
            cache.p1_dof_count * node_count,
            operators.double_layer,
            operators.hypersingular,
            tables.blocks,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            tables.chunk_nodes,
            tables.inc_offsets,
            tables.inc_packed,
            Int32(node_start),
            Int32(node_count),
            Int32(element_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("dlp_hyp", timed, stamp)
    end
    return nothing
end

function _launch_metal_regular_gather_kernels!(operators, cache::MetalRegularAssemblyCache, k)
    return _launch_metal_gather_pair_kernels!(
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

function _launch_metal_symmetry_regular_gather_kernels!(
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
    return _launch_metal_gather_pair_kernels!(
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
