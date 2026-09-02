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
2. evaluates regular Galerkin pairs in chunks of trial elements: one thread
   per element pair on a two-dimensional grid writes the pair's 3x1 and 3x3
   operator blocks to a device buffer, every Green's-function value used for
   all four operators, and gather kernels with one owner per matrix entry
   sum the buffer into the dense operators (no atomics, fixed summation
   order);
3. evaluates adjacent and coincident pairs with Duffy singular quadrature in
   one fused kernel per pair and scatters their compact correction blocks
   into the dense operators;
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

Four regular-assembly kernel modes exist. The default, `pair_gather`, is
the chunked pair-gather design described above. It exists because the
fused atomic kernel was bound by atomic throughput, not arithmetic: each
pair scatters 48 Float32 atomics (four operators, real and imaginary), and
on an M1 Max those cost as much as the Green's-function evaluations
themselves. Writing the blocks with plain stores and gathering them per
entry removes the atomics and makes the result bit-reproducible run to run.
The trial columns are processed in chunks sized from a device-memory budget
(`BLAB_METAL_GATHER_BUDGET_MB`, 512 MB by default, 192 bytes per pair per
chunk column). `pair_atomic` is the fused kernel with atomic scatter,
non-deterministic in float32 summation order; `pair_owned` is the ROCm
backend's colored pair-owned design, deterministic because no two pairs in
one launch share a matrix entry; `entry_owned`, one thread per dense matrix
entry, is the correctness reference and does roughly nine times the
Green's-function work.

All modes share one pair-arithmetic routine: the fast-math AIR intrinsics
(`air.fast_sin`, `air.fast_cos`, `air.fast_rsqrt`, the arithmetic an
Xcode-compiled Metal shader gets by default), 32-bit indices, a rank-1
accumulation (3-vector inner sums, outer products once per test point),
a compile-time unrolled trial loop, and per-element quadrature points
precomputed once per cache so a point costs three loads instead of nine
vertex loads and nine FMAs. The kernel is register-bound (the 3x3 double
layer and hypersingular accumulators alone are 36 floats), so trial data is
read from cached device arrays rather than hoisted into registers.

The singular corrections use one fused Duffy kernel per (pair, part) that
evaluates the Green's function once per point pair for all four operators;
a pair's rule (512 to 1536 point pairs at singular order 4) is split into
`BLAB_METAL_SINGULAR_PARTS` contiguous ranges and a scatter kernel sums the
parts with atomics (about 48 per pair, negligible).

On an M1 Max at 5,041 P1 dofs (10,078 faces), quadrature order 4, singular
order 4, one frequency: `pair_gather` assembles in about 1.06 s (pair
kernel 0.59 s, gathers 0.31 s, singular 0.12 s, allocation 0.04 s);
`pair_atomic` 1.9 s; the first port's colored `pair_owned` kernels 7.1 s;
`entry_owned` about 25 s. All modes agree with BEAT CPU to the same
tolerances. hornlab-metal-bem's P1 Galerkin kernel, which
assembles one operator with 18 atomics per pair, takes 0.42 s on the same
mesh.

A frequency sweep overlaps the GPU assembly of frequency i+1 with the CPU
factorization of frequency i on a second Julia thread
(`BLAB_METAL_PIPELINE=0` disables it); two operator sets are then resident
at once.

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
| `BLAB_METAL_REGULAR_KERNEL_MODE` | `pair_gather` | Use `pair_atomic` for the fused atomic kernel, `pair_owned` for the deterministic colored kernels, or `entry_owned` as the correctness reference. |
| `BLAB_METAL_SINGULAR_MODE` | `native` | Use `host` to compute the Duffy singular corrections on the CPU and add them to the device operators, separating kernel defects from rule defects. |
| `BLAB_METAL_KERNEL_GROUPSIZE` | `256` | Threads per threadgroup for the one-dimensional assembly kernels. |
| `BLAB_METAL_ATOMIC_TILE` | `16x16` | Threadgroup shape (test, trial) of the two-dimensional pair kernels. |
| `BLAB_METAL_GATHER_BUDGET_MB` | `512` | Device memory for the pair-block buffer; sets the trial chunk size of `pair_gather`. `BLAB_METAL_GATHER_CHUNK` overrides the chunk size directly. |
| `BLAB_METAL_GATHER_TIMING` | `0` | Set to `1` to synchronize after each `pair_gather` stage and report `metal_native_gather_*` timings (slower). |
| `BLAB_METAL_SINGULAR_PARTS` | `4` | Ranges each singular pair's Duffy rule is split into across threads. |
| `BLAB_METAL_PIPELINE` | `1` | Set to `0` to assemble and solve each sweep frequency sequentially instead of overlapping GPU assembly with the CPU factorization. |
| `BLAB_METAL_ATOMIC_SCATTER` | `1` | Diagnostic for `pair_atomic` only: `0` skips the atomic scatter to time the pair arithmetic (the operators are then wrong). |

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
