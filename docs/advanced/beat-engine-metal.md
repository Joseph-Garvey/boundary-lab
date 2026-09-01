# BEAT Engine Apple Metal

BEAT Engine Apple Metal is Boundary Lab's local Apple Silicon GPU backend. It
uses the same mesh model, Burton-Miller formulation, symmetry rules, and result
protocol as the other BEAT Engine backends while moving the dense BEM operator
assembly and exterior-field evaluation to the GPU through Metal.jl.

The backend supports:

- exterior Burton-Miller BEM solves;
- coupled FEM-BEM-LEM physical-system solves;
- `off`, `x`, and `xy` symmetry;
- GPU-resident regular and singular operator assembly;
- GPU exterior-field evaluation for polar, spherical, and arbitrary observation
  points.

Production solves use `Float32` and `ComplexF32`, which is also the only
floating-point precision Apple GPUs provide. See [BEAT Engine
Core](beat-engine-core.md) for the shared boundary-integral formulation.

## Execution model

Boundary Lab prepares mesh topology, quadrature rules, symmetry transforms, and
frequency-independent cache data on the CPU. The Metal worker then:

1. allocates the single-layer, double-layer, adjoint double-layer, and
   hypersingular matrices as `MtlArray` objects;
2. evaluates regular Galerkin pairs with one fused kernel, one thread per
   element pair on a two-dimensional grid, every Green's-function value used
   for all four operators, scattered with Float32 atomics;
3. evaluates adjacent and coincident pairs with Duffy singular quadrature and
   gathers their compact correction blocks into the dense operators;
4. applies symmetry-image contributions and reduced-domain row weights;
5. copies the four operators to unified host memory, forms the Burton-Miller
   system, and factors it once per frequency with LAPACK on the CPU, reusing
   that factorization across every channel drive; and
6. evaluates the exterior field with Metal kernels.

### Coupled FEM-BEM-LEM

Coupled solves take the CPU backend's shape with the BEM stage moved to the
GPU: sparse FEM assembly and the UMFPACK interior Schur complement run on the
CPU, the four BEM operators are assembled on Metal and copied to the host,
and the retained coupled system is factored with the CPU dense LU. The
condensed formulation is the default, exactly as for the CPU backend, and the
monolithic formulation remains available for validation. Interior-FEM-only
solves have no BEM stage and run on the CPU path unchanged.

The dense factorization stays on the CPU because Metal.jl provides no GPU LU.
On Apple Silicon the copy from device to host is a memcpy within unified
memory, and the CPU LU runs through Accelerate-class BLAS. The backend's
dense-size ceiling is therefore the same as the CPU backend's.

Three regular-assembly kernels exist. The default, `pair_atomic`, is the
fused atomic kernel described above; it uses the fast-math AIR intrinsics
(`air.fast_sin`, `air.fast_cos`, `air.fast_rsqrt`), the same arithmetic an
Xcode-compiled Metal shader gets by default, 32-bit indices, and trial
quadrature points hoisted out of the point-pair loop. Its singular
corrections are scattered per pair with atomics too. `pair_owned` is the
ROCm backend's colored pair-owned design, deterministic because no two pairs
in one launch share a matrix entry; `entry_owned`, one thread per dense
matrix entry, is the correctness reference and does roughly nine times the
Green's-function work. On an M1 Max at 5k P1 dofs and quadrature order 4
they take 3.7 s, 6.8 s, and about 25 s per frequency; all three agree with
BEAT CPU to the same tolerances.

## Requirements

- An M-series Mac running macOS 14 or newer.
- Julia 1.10 to 1.12.
- The dedicated `src/blab/solvers/julia_metal` environment with Metal.jl.

To prepare the Julia environment from the repository root:

```bash
julia --project=src/blab/solvers/julia_metal -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Verify the runtime:

```bash
julia --project=src/blab/solvers/julia_metal -e 'using Metal; Metal.functional() || error("Metal unavailable"); Metal.versioninfo()'
```

## Selecting the backend

In application preferences, select **BEAT Engine (Apple Metal)**. The backend
identifier used by project and server workflows is `beat_metal`. The entry
is only offered on Apple Silicon macOS.

## Runtime controls

Normal application use does not require these environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `BLAB_METAL_ASSEMBLY_MODE` | `native` | Use `host_staged` to assemble operators on the CPU and upload them as a diagnostic fallback. |
| `BLAB_METAL_REGULAR_KERNEL_MODE` | `pair_atomic` | Use `pair_owned` for the deterministic colored kernels or `entry_owned` as the correctness reference. |
| `BLAB_METAL_SINGULAR_MODE` | `native` | Use `host` to compute the Duffy singular corrections on the CPU and add them to the device operators, separating kernel defects from rule defects. |
| `BLAB_METAL_KERNEL_GROUPSIZE` | `256` | Threads per threadgroup for the assembly kernels. |

## Verification

CPU-versus-Metal validation scripts:

| Script | Coverage |
|---|---|
| `validate_metal_exterior.jl` | Operators (both singular modes), boundary pressure, residual, and exterior field for an exterior solve. |
| `validate_metal_symmetry.jl` | X and XY reduced-domain assembly and solve parity, both singular modes. |
| `validate_metal_coupled.jl` | Coupled FEM-BEM-LEM assembly, condensation, solution, and field for the monolithic and condensed paths, prescribed-velocity and voltage excitations. |

For example:

```bash
julia -t auto --project=src/blab/solvers/julia_metal \
  src/blab/solvers/julia_local/scripts/validate_metal_exterior.jl
```

`BLAB_VALIDATE_MESH`, `BLAB_VALIDATE_REGULAR_ORDER`,
`BLAB_VALIDATE_SINGULAR_ORDER`, and `BLAB_VALIDATE_FREQUENCY_HZ` select the
fixture, quadrature orders, and frequency. The scripts exit with an error when
CPU-versus-Metal differences exceed their tolerances.

## Operational behavior

- A cold Julia worker compiles Metal kernels before its first solve. Steady-state
  solve time should be evaluated after warm-up.
- Frequency-independent caches remain resident for the worker's lifetime and are
  released when the worker exits.
- The default `pair_atomic` kernels are not bitwise reproducible run to run
  (atomic accumulation order); the differences are float32 summation noise.
  `pair_owned` and `entry_owned` are deterministic. Golden-file comparisons
  belong on those, tolerance comparisons on the default.
