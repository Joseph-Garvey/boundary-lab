# Phase-2 scaffold: GPU regular-pair Galerkin operator assembly on Metal.
#
# The phase-1 hybrid (BeatEngineMetalAssembly.jl) assembles the regular operators on the proven CPU
# path and only runs field evaluation on the GPU. Moving regular assembly onto the Metal GPU is a
# performance optimization, NOT a correctness requirement, and it involves the most numerically
# sensitive kernel in the engine. Because this repository is developed on Linux (no Apple GPU), that
# kernel cannot be compiled or validated here, so it is intentionally left as a guarded entry point
# with a precise porting specification rather than unvalidated math on a live path.
#
# PORTING SPEC (TODO(apple-hw)):
#   Source of truth: BeatEngineCudaRegular.jl `_cuda_regular_kernel!` (single-pair-per-thread) and
#   BeatEngineCudaAssembly.jl (the balanced-multipair launch + symmetry-image kernels).
#   1. Port `_cuda_regular_kernel!` to a Metal kernel:
#        - pair index: `pair = thread_position_in_grid_1d()`; stride =
#          `threads_per_threadgroup_1d() * threadgroups_per_grid_1d()`.
#        - geometry loads (face_vertices/normals/areas/curls) and the per-quadrature SLP/DLP/adjoint
#          DLP/hypersingular accumulation are pure fp32 scalar math and transcribe verbatim.
#        - scatter into the dense P1/DP0 dof matrices with `Metal.@atomic A[idx] += v` (P1 dofs are
#          shared between elements, so the atomic accumulation is required).
#        - apply the symmetry-image transforms exactly as `_launch_regular_symmetry_image_kernel!`.
#   2. Allocate slp/dlp/adj/hyp real+imag as `Metal.zeros(Float32, ...)`, launch with
#      `Metal.@metal threads=METAL_DEFAULT_THREADS groups=cld(total_pairs, METAL_DEFAULT_THREADS)`.
#   3. Copy the four operators back to host Complex{Float32} via `_metal_complex_cpu_matrix`, then
#      hand off to the shared host singular-correction + symmetry-row-weight path so the result is
#      bit-comparable to the CPU/CUDA operators before the dense solve.
#   4. Validate against the CPU backend on the julia_local/test_meshes fixtures (sample*.msh) within
#      a tight tolerance before enabling on the default path.

# Opt-in toggle for the phase-2 GPU regular path. Default off: phase-1 uses CPU regular assembly.
# TODO(apple-hw): flip to true only after the kernel above is implemented and validated on device.
_metal_regular_assembly_use_gpu() = false

function _metal_assemble_regular_operators_gpu(args...; kwargs...)
    error(
        "Metal GPU regular-pair assembly is a phase-2 optimization that has not been implemented or " *
        "validated on Apple hardware yet. The BEAT Engine Metal backend assembles regular operators " *
        "on the CPU and runs field evaluation on the GPU. See BeatEngineMetalRegularKernels.jl for " *
        "the porting specification.",
    )
end
