"""
The BEAT Engine worker entry point.

Nothing but loading and dispatch lives here. The engine is in `src/`, the
driver is in `BeatEngineDriver.jl`, and both are `include`d by a bundle
package under `julia_engine/` so that Julia can cache their native code in a
pkgimage. A script cannot be cached -- it is parsed, lowered and compiled from
source in every process -- and this file used to hold the whole driver, which
made the worker's own entry path the largest single item in a cold start.

The fallback below is not a nicety. Analysis scripts, an un-instantiated
checkout, and any environment whose bundle has not been resolved all take it,
and they behave exactly as they did before the bundles existed: same code,
same results, just compiled from source again.
"""

using JSON
using LinearAlgebra
using Printf
using Statistics
using StaticArrays

#: The bundle package for the accelerator this process is configured for. The
#: environment variable wins because that is what the Python wrapper sets when
#: it starts a worker; the project directory is the fallback the CLI relies on.
const BEAT_ENGINE_BUNDLE_NAME = let
    hint = lowercase(strip(get(ENV, "BLAB_BEAT_ENGINE_GPU_BACKEND", "")))
    if isempty(hint)
        active = Base.active_project()
        directory = active === nothing ? "" : lowercase(basename(dirname(active)))
        hint = directory == "julia_cuda" ? "cuda" :
            directory == "julia_rocm" ? "rocm" :
            directory == "julia_metal" ? "metal" : "cpu"
    end
    hint == "cuda" ? :BeatEngineCudaBundle :
        hint == "rocm" ? :BeatEngineRocmBundle :
        hint == "metal" ? :BeatEngineMetalBundle : :BeatEngineCpuBundle
end

const BEAT_ENGINE_BUNDLE = nothing

if BEAT_ENGINE_BUNDLE === nothing
    include(joinpath(@__DIR__, "src", "BeatEngineCore.jl"))
    @eval using .BeatEngineCore
    include(joinpath(@__DIR__, "BeatEngineDriver.jl"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if BEAT_ENGINE_BUNDLE === nothing
        main(ARGS)
    else
        BEAT_ENGINE_BUNDLE.main(ARGS)
    end
end
