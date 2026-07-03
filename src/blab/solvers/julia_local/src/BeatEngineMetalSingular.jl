# Singular / near-singular corrections for the Metal backend.
#
# Phase 1 applies Duffy singular and image-singular corrections on the host, reusing the
# backend-agnostic host singular machinery (build_singular_correction_cache + the CPU accumulation
# helpers invoked by assemble_regular_galerkin_operators_cpu). No device-side singular cache is
# built, so build_metal_singular_correction_cache returns nothing.
#
# Phase 2 (TODO(apple-hw)): port the compact per-pair Duffy block kernel + atomic scatter from
# BeatEngineCudaSingular.jl once the GPU regular path (BeatEngineMetalRegularKernels.jl) is
# validated on device.

function build_metal_singular_correction_cache(args...; kwargs...)
    return nothing
end
