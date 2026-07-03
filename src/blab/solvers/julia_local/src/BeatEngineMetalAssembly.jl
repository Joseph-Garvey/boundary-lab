# Metal hybrid operator assembly entry point.
#
# Called from assemble_regular_galerkin_operators(...; backend=:metal) in BeatEngineCore.jl with
# return_device=false. Phase 1 assembles the regular + singular Galerkin operators on the proven
# CPU path (assemble_regular_galerkin_operators_cpu) and returns host Complex{Float32} matrices with
# on_gpu=false, so the Burton-Miller dense solve dispatches to solve_burton_miller_neumann_cpu and
# runs on Apple Accelerate (AMX). Field evaluation (the GPU-accelerated part of this backend) is
# handled separately by evaluate_galerkin_field_metal.
#
# Phase 2 (TODO(apple-hw)): when _metal_regular_assembly_use_gpu() is enabled and the GPU regular
# kernel in BeatEngineMetalRegularKernels.jl is validated, assemble the regular operators on the GPU
# here, copy them to host, then apply host singular corrections before returning.

function assemble_regular_galerkin_operators_metal_regular(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool=true,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    cache=nothing,
    return_device::Bool=false,
    accelerator_quadrature::Bool=true,
    timing=nothing,
    singular_cache=nothing,
    metal_singular_cache=nothing,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    return_device && error("BEAT Engine Metal hybrid materializes operators on the host; return_device=true is unsupported.")

    if _metal_regular_assembly_use_gpu()
        # TODO(apple-hw): GPU regular assembly path (see BeatEngineMetalRegularKernels.jl).
        return _metal_assemble_regular_operators_gpu(
            mesh, p1_space, dp0_space, k, rule;
            skip_singular=skip_singular,
            singular_order=singular_order,
            element_indices=element_indices,
            cache=cache,
            timing=timing,
            singular_cache=singular_cache,
            metal_singular_cache=metal_singular_cache,
            symmetry_mode=symmetry_mode,
        )
    end

    operators = assemble_regular_galerkin_operators_cpu(
        mesh,
        p1_space,
        dp0_space,
        k,
        rule;
        skip_singular=skip_singular,
        singular_order=singular_order,
        element_indices=element_indices,
        threaded=true,
        timing=timing,
        singular_cache=singular_cache,
        symmetry_mode=symmetry_mode,
    )

    # Tag the diagnostics so result events report the Metal hybrid rather than the bare CPU path.
    return merge(
        operators,
        (
            regular_assembly_mode=:metal_hybrid_cpu_regular,
            regular_kernel_mode="metal_hybrid_cpu_regular_gpu_field",
        ),
    )
end
