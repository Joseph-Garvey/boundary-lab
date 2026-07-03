# BEAT Engine Metal

The BEAT Engine Metal backend is an Apple Silicon (M-series) GPU-accelerated Julia solver path. It uses the same BEAT Engine request protocol, mesh handling, Burton-Miller formulation, symmetry model, and result stream described in [BEAT Engine Core](beat-engine-core.md), but targets the Apple GPU via [Metal.jl](https://github.com/JuliaGPU/Metal.jl) together with Apple's Accelerate framework.

The application exposes this path as `BEAT Engine (Metal)` / `beat_metal`. Accepted aliases are `metal`, `apple`, and `mps`.

## Why A Hybrid Design

Apple Silicon has three properties that shape this backend:

- **fp32-only GPU.** Apple GPUs do not implement IEEE double precision. This is not a blocker: the BEAT Engine already solves entirely in `Float32` / `Complex{Float32}`.
- **No mature complex dense LU on the GPU.** Neither Metal nor Metal.jl exposes a complex dense LU comparable to CUDA's CUSOLVER. Rather than hand-roll one, the Metal backend runs the Burton-Miller dense solve on the CPU.
- **Unified memory + the AMX matrix coprocessor.** CPU and GPU share the same physical memory, so moving assembled operators from the GPU to host arrays is inexpensive, and there is no discrete-VRAM ceiling. The CPU dense solve is routed through Apple's Accelerate framework, which dispatches BLAS/LAPACK to the AMX matrix coprocessor.

The backend therefore splits the work to match the hardware:

| Stage | Where it runs |
|-------|---------------|
| Regular + singular operator assembly | CPU (phase 1) → GPU (phase 2, see roadmap) |
| Field evaluation at observation points | **Metal GPU** |
| Burton-Miller dense solve | CPU via Apple Accelerate (AMX) |

Because the assembled operators are materialized to host `Complex{Float32}` matrices with `on_gpu=false`, the shared dispatch in `BeatEngineCore.jl` (`solve_burton_miller_neumann`) falls through to `solve_burton_miller_neumann_cpu`, which is exactly the Accelerate-backed path. Unified memory removes the quadratic-VRAM constraint that bounds mesh size on the CUDA backend (see the VRAM table in the top-level README): the Metal backend can address as much memory as the machine has.

## Metal Field Evaluation

`BeatEngineMetalField.jl` is a direct port of the CUDA field kernels (`BeatEngineCudaField.jl`). It performs the Galerkin representation-formula sweep over the concatenated horizontal, vertical, and optional spherical observation points:

- `build_metal_field_evaluation_cache` packs source points, normals, weights, faces, elements, and P1 basis values into `MtlArray`s, held resident across frequencies.
- A weighted-sources kernel precomputes per-source pressure and Neumann contributions.
- A block-per-observation-point kernel accumulates the single- and double-layer field contributions with a threadgroup reduction, and returns the host potential vector.

This is the embarrassingly-parallel part of the solve and is the phase-1 GPU acceleration.

## CPU Assembly And Accelerate Solve (Phase 1)

`BeatEngineMetalAssembly.jl` assembles the regular and singular Galerkin operators on the proven CPU path (`assemble_regular_galerkin_operators_cpu`), returning host matrices tagged with a `metal_hybrid_cpu_regular` assembly mode. Singular and image-singular corrections reuse the backend-agnostic host singular machinery (`build_singular_correction_cache`), so `BeatEngineMetalSingular.jl` builds no device-side singular cache in phase 1.

The dense solve then runs through `BeatEngineCpuSolve.jl`. When the `julia_metal` project is instantiated, `solver.jl` best-effort loads `AppleAccelerate`, which forwards Julia's LBT-backed BLAS/LAPACK to Accelerate and the AMX coprocessor. On environments without AppleAccelerate (the CUDA/CPU/ROCm projects), the load is a no-op and the standard OpenBLAS path is used.

## GPU Regular Assembly (Phase 2 Roadmap)

Moving regular-pair operator assembly onto the Metal GPU is a performance optimization, not a correctness requirement, and it is the most numerically sensitive kernel in the engine. It is scaffolded in `BeatEngineMetalRegularKernels.jl` with a precise porting specification derived from `BeatEngineCudaRegular.jl` and `BeatEngineCudaAssembly.jl`, and gated behind `_metal_regular_assembly_use_gpu()` (default `false`). The device geometry cache it needs is already built by `build_metal_regular_assembly_cache`.

## Requirements

- macOS on Apple Silicon (M1 or newer).
- [Julia](https://julialang.org/downloads/) available on `PATH`.
- The `julia_metal` project dependencies: `Metal`, `AppleAccelerate`, `JSON`, `StaticArrays`. Metal.jl only installs on macOS/aarch64.

To prepare the Julia environment, from the repository root run:

```bash
julia --project=src/blab/solvers/julia_metal -e "using Pkg; Pkg.instantiate()"
```

Then, in the GUI, open `Edit > Preferences` and set `BEM Solver` to `BEAT Engine (Metal)`. From the server, use `--solver beat_metal`.

## validate-on-apple-hardware

This backend was developed on Linux, where Metal.jl cannot load, so the Metal kernels have **not** been executed. Before relying on `beat_metal`, validate the following on Apple Silicon (each is marked `TODO(apple-hw)` in the source):

- Metal.jl thread-index intrinsics used in `BeatEngineMetalField.jl` (`thread_position_in_grid_1d`, `thread_position_in_threadgroup_1d`, `threadgroup_position_in_grid_1d`, `threads_per_threadgroup_1d`, `threadgroups_per_grid_1d`) and 1- vs 0-based indexing for the installed Metal.jl version.
- `MtlThreadGroupArray` static size and `MtlThreadGroupBarrier` availability.
- `Metal.functional()`, `Metal.zeros`, `MtlArray`, `Metal.synchronize`, and `Metal.unsafe_free!` names.
- `AppleAccelerate` load actually swaps the LAPACK backend (confirm via `LinearAlgebra.BLAS`/LBT).
- Field-evaluation numerics against the CPU backend on `src/blab/solvers/julia_local/test_meshes/sample*.msh` within a tight tolerance.
- Only then implement and validate the phase-2 GPU regular assembly kernel before flipping `_metal_regular_assembly_use_gpu()`.

A `benchmark_metal.jl` (modeled on `scripts/benchmark_cuda.jl` / `benchmark_cpu.jl`) should compare the Metal hybrid against `beat_cpu` (with and without Accelerate) and, where available, CUDA.

## Appendix: Apple Silicon Compute-Strategy Comparison

Three strategies were considered for the Apple Silicon port. This backend implements strategy 1 and gets strategy 3 for free.

| # | Strategy | Description | Status |
|---|----------|-------------|--------|
| 1 | **Metal.jl hybrid** | GPU field evaluation (and, in phase 2, GPU regular assembly) + CPU/Accelerate complex dense solve, inside the existing Julia BEAT Engine. Reuses all orchestration, protocol, symmetry, and result code. | **Implemented** (this backend). |
| 2 | **MLX backend** | A separate Python backend (parallel to `bempp_local`) built on Apple's MLX array framework — unified-memory arrays, `complex64`, `mlx` linalg solve, GPU via Metal. Does not extend the Julia engine; the assembly/field math would be re-expressed as MLX ops. | Future / benchmark candidate. |
| 3 | **Accelerate / AMX CPU** | Route the existing `beat_cpu` (and bempp) dense BLAS/LAPACK through Apple Accelerate so the solve uses the AMX coprocessor. | **Included** — it is exactly the phase-1 hybrid's dense-solve path (via `AppleAccelerate` in `solver.jl`). |

## Important Files

- `src/blab/solvers/julia_metal/Project.toml`: the Metal backend Julia environment (`Metal`, `AppleAccelerate`, `JSON`, `StaticArrays`).
- `src/blab/solvers/julia_local/src/BeatEngineMetal.jl`: include hub for the Metal implementation files.
- `src/blab/solvers/julia_local/src/BeatEngineMetalField.jl`: Metal GPU field-evaluation path (phase-1 GPU compute).
- `src/blab/solvers/julia_local/src/BeatEngineMetalAssembly.jl`: hybrid operator-assembly entry point (CPU regular + host singular in phase 1).
- `src/blab/solvers/julia_local/src/BeatEngineMetalRegular.jl`: host→device geometry packing and the regular-assembly device cache.
- `src/blab/solvers/julia_local/src/BeatEngineMetalRegularKernels.jl`: phase-2 GPU regular assembly scaffold and porting specification.
- `src/blab/solvers/julia_local/src/BeatEngineMetalCommon.jl` / `BeatEngineMetalOperators.jl` / `BeatEngineMetalSingular.jl`: cache structs and reduction helpers, host-materialize/free helpers, and the host singular-correction passthrough.
- `src/blab/solvers/julia_local/solver.jl`: `:metal` backend dispatch, device cache initialization, best-effort `AppleAccelerate` load, and device cleanup.
