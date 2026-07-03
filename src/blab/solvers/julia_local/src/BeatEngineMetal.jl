# BEAT Engine Apple Metal backend (M-series GPU).
#
# This file group mirrors the CUDA backend (BeatEngineCuda*.jl) so the Metal path slots into the
# same dispatch in BeatEngineCore.jl / solver.jl. It implements a *hybrid* strategy that suits
# Apple Silicon:
#
#   * Field evaluation runs on the Metal GPU (BeatEngineMetalField.jl) — a 1:1 port of the CUDA
#     field kernels. This is the embarrassingly-parallel observation-point sweep.
#   * Operator assembly is materialized to host Complex{Float32} matrices and the Burton-Miller
#     dense solve runs on the CPU via Apple Accelerate (the AMX matrix coprocessor). Apple GPUs
#     are fp32-only and Metal has no mature complex dense LU, so this sidesteps the missing kernel
#     while keeping the O(N^3) solve on hardware built for it. Unified memory makes the GPU<->host
#     handoff cheap.
#   * Phase 1 keeps regular + singular operator assembly on the proven CPU path
#     (assemble_regular_galerkin_operators_cpu). The GPU regular-assembly kernels are scaffolded in
#     BeatEngineMetalRegularKernels.jl as a phase-2 optimization and are NOT on the default path.
#
# TODO(apple-hw): every Metal kernel and Metal.jl API call below must be compiled and validated on
# Apple Silicon. This repository is developed on Linux where Metal.jl cannot load, so the kernels
# here are written to mirror the validated CUDA path but have not been executed.

include(joinpath(@__DIR__, "BeatEngineMetalCommon.jl"))
include(joinpath(@__DIR__, "BeatEngineMetalRegular.jl"))
include(joinpath(@__DIR__, "BeatEngineMetalRegularKernels.jl"))
include(joinpath(@__DIR__, "BeatEngineMetalField.jl"))
include(joinpath(@__DIR__, "BeatEngineMetalSingular.jl"))
include(joinpath(@__DIR__, "BeatEngineMetalOperators.jl"))
include(joinpath(@__DIR__, "BeatEngineMetalAssembly.jl"))
