# Build a Julia sysimage for the CUDA BEAT Engine.
#
# The engine is baked in through `BeatEngineCudaBundle`, and that is the whole
# reason this script is worth running. Before the bundles existed it could only
# name CUDA, JSON and StaticArrays: the engine and the worker driver were
# `include`d into `Main` by `solver.jl`, and PackageCompiler cannot bake code
# that is not in a package. So the image shortened the dependency load and left
# the larger half -- compiling 22,000 lines of engine and driver -- to be paid
# again in every process.
#
# The bundle's own `@compile_workload` already moves most of the first-solve
# JIT into its pkgimage, so this image is a second-order win on top of that: it
# collapses the remaining package loading into one mapped file. Build it, or
# do not (`BLAB_BUILD_SYSIMAGE=0`); the bundle is what carries the change.
#
# A sysimage, unlike a pkgimage, has no staleness check. Nothing here rebuilds
# it when an engine source changes, which is safe only because it is built once
# inside an image from the sources shipped in that image. Do not copy this
# pattern to a developer machine, where the sources move underneath it.

import Pkg

const PROJECT_DIR = abspath(joinpath(@__DIR__, "..", "src", "blab", "solvers", "julia_cuda"))
const SYSIMAGE_PATH = get(ENV, "BLAB_JULIA_SYSIMAGE", "/app/blab-beat-cuda.so")
const PRECOMPILE_FILE = abspath(joinpath(@__DIR__, "precompile-beat-cuda.jl"))
const CPU_TARGET = get(ENV, "BLAB_JULIA_CPU_TARGET", "generic,+aes")

Pkg.activate(mktempdir())
Pkg.add("PackageCompiler")

using PackageCompiler

PackageCompiler.create_sysimage(
    ["BeatEngineCudaBundle", "CUDA", "JSON", "StaticArrays"];
    project=PROJECT_DIR,
    sysimage_path=SYSIMAGE_PATH,
    precompile_execution_file=PRECOMPILE_FILE,
    cpu_target=CPU_TARGET,
)
