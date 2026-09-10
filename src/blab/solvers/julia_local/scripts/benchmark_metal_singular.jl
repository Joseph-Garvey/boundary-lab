using LinearAlgebra
using Printf
using Metal

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const BEC = BeatEngineCore

function time_launch(launch!, label; repeats::Int=5)
    launch!()
    Metal.synchronize()
    best = Inf
    for _ in 1:repeats
        elapsed = @elapsed begin
            launch!()
            Metal.synchronize()
        end
        best = min(best, elapsed)
    end
    @printf("%-28s %8.4f s\n", label, best)
    return best
end

function benchmark_metal_singular()
    Metal.functional() || error("Metal.functional() is false.")

    mesh_name = get(ENV, "BLAB_BENCH_MESH", "sample_detailed.msh")
    regular_order = parse(Int, get(ENV, "BLAB_BENCH_REGULAR_ORDER", "2"))
    singular_order = parse(Int, get(ENV, "BLAB_BENCH_SINGULAR_ORDER", "2"))
    mesh_path = joinpath(@__DIR__, "..", "test_meshes", mesh_name)
    mesh = load_gmsh22_with_tags(mesh_path, Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, regular_order)
    k = Float32(2pi) * 1000.0f0 / 343.0f0

    regular_cache = build_metal_regular_assembly_cache(
        mesh, p1, dp0, rule;
        singular_order=singular_order,
        symmetry_mode=:off,
    )
    correction_cache = build_singular_correction_cache(mesh, singular_order)
    singular_cache = build_metal_singular_correction_cache(correction_cache, p1, dp0)

    pair_count = singular_cache.pair_count
    p1_dofs = regular_cache.p1_dof_count
    dp0_dofs = regular_cache.dp0_dof_count
    println("mesh=$(mesh_name) faces=$(length(mesh.faces)) p1_dofs=$(p1_dofs) dp0_dofs=$(dp0_dofs) q=$(regular_order) s=$(singular_order)")
    println("singular_pairs=$(pair_count) p1_entries=$(p1_dofs * p1_dofs) slp_entries=$(p1_dofs * dp0_dofs)")
    println("device=$(Metal.device())")
    flush(stdout)

    operators = (
        single_layer=Metal.zeros(ComplexF32, p1_dofs, dp0_dofs),
        double_layer=Metal.zeros(ComplexF32, p1_dofs, p1_dofs),
        adjoint_double_layer=Metal.zeros(ComplexF32, p1_dofs, dp0_dofs),
        hypersingular=Metal.zeros(ComplexF32, p1_dofs, p1_dofs),
    )
    slp_values = Metal.zeros(ComplexF32, pair_count, 3)
    adjoint_values = Metal.zeros(ComplexF32, pair_count, 3)
    dlp_values = Metal.zeros(ComplexF32, pair_count, 9)
    hypersingular_values = Metal.zeros(ComplexF32, pair_count, 9)

    groupsize = BEC._metal_kernel_groupsize()
    slp_entries = p1_dofs * dp0_dofs
    p1_entries = p1_dofs * p1_dofs

    total = time_launch("whole singular stage") do
        BEC._launch_metal_singular_block_gather_kernels!(operators, regular_cache, singular_cache, k)
    end

    blocks_slp = time_launch("blocks: slp+adjoint") do
        Metal.@metal threads=groupsize groups=cld(pair_count, groupsize) BEC._metal_singular_slp_adjoint_blocks_kernel!(
            slp_values, adjoint_values,
            singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
            singular_cache.jac_scales, singular_cache.rule_offsets,
            singular_cache.rule_test_points, singular_cache.rule_trial_points, singular_cache.rule_weights,
            regular_cache.face_vertices, regular_cache.normals,
            k, regular_cache.face_count, pair_count, 1.0f0, 1.0f0, 1.0f0,
        )
    end

    blocks_dlp = time_launch("blocks: dlp+hyp") do
        Metal.@metal threads=groupsize groups=cld(3 * pair_count, groupsize) BEC._metal_singular_dlp_hyp_blocks_kernel!(
            dlp_values, hypersingular_values,
            singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
            singular_cache.jac_scales, singular_cache.normal_products, singular_cache.rule_offsets,
            singular_cache.rule_test_points, singular_cache.rule_trial_points, singular_cache.rule_weights,
            regular_cache.face_vertices, regular_cache.normals, regular_cache.curls,
            k, regular_cache.face_count, pair_count,
            1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0,
        )
    end

    slp_gather = singular_cache.slp_gather
    dlp_gather = singular_cache.dlp_gather
    println("slp_gather_entries=$(slp_gather.entry_count) dlp_gather_entries=$(dlp_gather.entry_count)")

    gather_slp = time_launch("gather: slp+adjoint") do
        Metal.@metal threads=groupsize groups=cld(slp_gather.entry_count, groupsize) BEC._metal_singular_pair_gather_kernel!(
            operators.single_layer, operators.adjoint_double_layer,
            slp_values, adjoint_values,
            slp_gather.entry_indices, slp_gather.contrib_offsets, slp_gather.contrib_values,
            slp_gather.entry_count,
        )
    end

    gather_dlp = time_launch("gather: dlp+hyp") do
        Metal.@metal threads=groupsize groups=cld(dlp_gather.entry_count, groupsize) BEC._metal_singular_pair_gather_kernel!(
            operators.double_layer, operators.hypersingular,
            dlp_values, hypersingular_values,
            dlp_gather.entry_indices, dlp_gather.contrib_offsets, dlp_gather.contrib_values,
            dlp_gather.entry_count,
        )
    end

    println()
    parts = blocks_slp + blocks_dlp + gather_slp + gather_dlp
    @printf("blocks total  %8.4f s  (%.0f%% of parts)\n", blocks_slp + blocks_dlp, 100 * (blocks_slp + blocks_dlp) / parts)
    @printf("gather total  %8.4f s  (%.0f%% of parts)\n", gather_slp + gather_dlp, 100 * (gather_slp + gather_dlp) / parts)
    @printf("sum of parts  %8.4f s  vs whole stage %.4f s\n", parts, total)

    Metal.unsafe_free!(slp_values)
    Metal.unsafe_free!(adjoint_values)
    Metal.unsafe_free!(dlp_values)
    Metal.unsafe_free!(hypersingular_values)
    release_metal_singular_correction_cache!(singular_cache)
    release_metal_regular_assembly_cache!(regular_cache)
    return nothing
end

benchmark_metal_singular()
