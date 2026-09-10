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

Exterior solves take the fused Burton-Miller path described below, which never
forms the four operators. The four-operator path described here is still what
coupled FEM-BEM-LEM solves, the `host_staged` assembly fallback and the `host`
singular mode use, and `BLAB_BEAT_FUSED_BM=0` selects it for exterior solves
too.

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
5. wraps the four operators as host arrays in place -- they live in Metal
   shared storage, so nothing is copied -- forms the Burton-Miller system, and
   factors it once per frequency with LAPACK on the CPU, reusing that
   factorization across every channel drive; and
6. evaluates the exterior field with Metal kernels.

### Coupled FEM-BEM-LEM

Coupled solves take the CPU backend's shape with the BEM stage moved to the
GPU: sparse FEM assembly and the UMFPACK interior Schur complement run on the
CPU, the four BEM operators are assembled on Metal and wrapped on the host,
and the retained coupled system is factored with the CPU dense LU. The
condensed formulation is the default, exactly as for the CPU backend, and the
monolithic formulation remains available for validation. Interior-FEM-only
solves have no BEM stage and run on the CPU path unchanged.

The dense factorization stays on the CPU because Metal.jl provides no GPU LU,
and the CPU LU runs through Accelerate-class BLAS. The backend's dense-size
ceiling is therefore the same as the CPU backend's.

There is no device-to-host transfer. The operators are allocated in Metal
shared storage and handed to the CPU as `unsafe_wrap`ped `Array`s over the same
buffers, so `metal_host_operators` costs nothing measurable. This is not what
a private-storage buffer does: `Array(::MtlArray)` on private storage blits
through a staging buffer at a measured 3.5-8 GB/s, which was 1.1-1.5 s per
frequency at 10,230 P1 dofs (five gigabytes of operators). Shared storage does
not slow the assembly kernels -- 4.34-4.41 s against 4.37-4.95 s private on the
same mesh -- and reading the operators back on the CPU is about 1.5x slower
per byte than reading a host copy, which costs about 0.19 s of matrix formation
and is far less than the transfer it removes. `BLAB_METAL_OPERATOR_STORAGE=private`
restores the copying path.

The host arrays alias device memory, so the operator storage is released once,
through whichever tuple the caller still holds: the host tuple returned by
`metal_host_operators` owns the buffers it wrapped, and freeing the device
tuple while those views are still live leaves them dangling.
### Fused Burton-Miller assembly (exterior solves)

The coupling eta = i/k is known at assembly time, so the exterior path forms

    lhs = 0.5 M_p1p1 - D + (i/k) H          rhs = (-S - (i/k)(K' + 0.5 M_p1dp0)) q

inside the assembly kernels and never allocates S, K', D or H. That is one
N x N matrix and one right-hand side per drive instead of 6N^2 complex entries,
a measured **6.0x** reduction in operator memory on every mesh tried, and
because the storage is O(N^2) it is sqrt(6) ~ 2.45x more dofs at the same peak.

Every channel's Neumann column is built before assembly and folded in during
the same pass, so one assembly still serves the whole channel set at a
frequency exactly as one factorization does.

The win is *not* the halved stores. The combination is applied to the pair's
3-vectors before the rank-1 expansion, not to four finished blocks afterwards:
one 3x3 expansion per test point instead of two, one 3x1 instead of two, and 24
live accumulator floats instead of 48. Measured on the pair kernel alone,
combining afterwards is 1.00-1.02x and combining before the expansion is
1.82-1.87x. Whole assembly is 2.12-2.77x faster over the ATH ladder from 1,974
to 10,230 dofs. The per-quadrature-point arithmetic is unchanged and cannot
change: D and H carry different geometric prefactors per entry, so both terms
are evaluated whatever they accumulate into. Only the expansion collapses, and
only because the hypersingular curl term carries no basis product and can be
summed as one scalar and expanded after the loop.

The singular correction is fused the same way and for the same reason. The
Duffy/Sauter-Schwab pair carried 50 live accumulator floats and expanded the
rank-1 outer product four times per quadrature point pair; combining inside the
loop and hoisting the loop-invariant curl term leaves 26 live floats and two
expansions. The singular block kernel is **2.7x** faster on both an M1 Max
1,209-dof symmetry-reduced mesh (20.7 to 7.7 ms) and a 4,552-dof full mesh
(84.1 to 30.9 ms), for a 14-17% shorter assembly and a 11-14% shorter
40-frequency sweep. Device memory high-water is unchanged to the byte: the
kernel writes the same `pair_count x part_count x 12` complex scratch through
the same scatter kernel.

`_metal_singular_pair_blocks` in `BeatEngineMetalAtomicKernels.jl` is the
four-operator path's singular pair and is deliberately untouched by the fusion,
so it stays an independent reference rather than a copy of the code under test.

`scripts/validate_metal_fused_burton_miller.jl` gates it by comparing the fused
system against the four-operator system on the same mesh, frequency and
quadrature, where the two differ only by float32 summation order.
`scripts/validate_metal_singular_summation.jl` bounds that summation-order
difference directly, per singular pair, against a Float64 evaluation of the
same algebra.

The Burton-Miller right-hand side is applied matrix-free. The operator
`-S - (i/k)(K' + 0.5 M)` is N x 2N complex -- 1.67 GB at 10,230 P1 dofs -- and
was materialised once per frequency only to be multiplied by a drive vector;
three matrix-vector products replace it. The left-hand side broadcasts the real
identity block directly instead of promoting it to a full complex copy. Between
them these were 0.84-1.16 s of a 6.3-6.7 s solve at 10,230 dofs, and about
2.5 GB of allocation per frequency.

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
`BLAB_METAL_SINGULAR_PARTS` contiguous ranges. A second kernel then sums the
parts and adds them to the operators. By default that is a gather: one thread
owns one dense-matrix cell and reads the list of block values belonging to it
from a map built on the host once per mesh, so no two threads share a cell and
the summation order is fixed. `BLAB_METAL_SINGULAR_WRITEBACK=scatter` selects
the older write-back instead, one thread per pair adding about 48 atomics, which
is not reproducible.

The map costs one host pass per mesh and is then reused for the worker's
lifetime, like the caches it hangs off. On an M1 Pro at `sample.msh` (2,776
faces, 37,198 singular pairs) it takes 269 ms to build and 3.0 MB of device
memory; at `sample_detailed.msh` (7,000 faces, 93,740 pairs) 637 ms and 7.7 MB.
In exchange the four-operator write-back drops from 2,249,760 atomic adds to
311,964 plain ones at the larger mesh, because a corrected cell is written once
instead of once per pair that touches it (3.2 pairs per cell for the single
layer and adjoint, 12.2 for the double layer and hypersingular).

Warm, the two write-backs are within a few percent of each other and the
singular stage is under a tenth of assembly, so this is not a speed change. A
single-frequency run on a fresh mesh is slightly slower for the map build; a
sweep amortizes it away.

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

## FEM static condensation

Coupled solves reduce the FEM interior onto the retained interface before
factoring. `beat_metal` is in both `PHYSICAL_SYSTEM_BACKEND_IDS` and
`CONDENSING_BACKEND_IDS`, so `system_solve.py` requests
`static_condensation: true` and this is the default path.

Condensation is CPU work on every backend. Metal uses
`_build_host_fem_condensation`, which partitions the FEM system, factors the
interior block, forms the Schur complement on the host, and hands the reduced
dense block to `_solve_host_condensed_excitations`. `condensation.backend`
reports `:cpu_hybrid`. This is the same shape as
`_build_rocm_hybrid_fem_condensation`, minus the upload.

### Why condensation is on by default

Measured on the curved-interface production fixture -- 19,492 FEM vertices,
94,265 tetrahedra, 5,103 BEM triangles, 1,318 retained interface vertices -- at
1 kHz, `q2/s2`, on an M1 Pro with eight Julia threads:

| | Monolithic | Condensed | Ratio |
| --- | ---: | ---: | ---: |
| System order | 23,433 | 5,259 | 4.5x smaller |
| Build | 96.689 s | 6.305 s | 15.3x |
| Solve | 2.799 s | 2.185 s | 1.28x |
| **Total** | **99.489 s** | **8.490 s** | **11.7x** |

The build gap is the dense coupled matrix: 23,433 squared in `ComplexF32` is
4.4 GB and its LU dominates everything else, against 221 MB condensed.
Agreement between the two formulations is 1.3e-4 on FEM pressure, 1.8e-4 on BEM
pressure, and 1.7e-4 on interface flux -- inside the 5e-4 Float32 gate used
across the coupled validations.

The condensation cost itself is reported in `condensation_timings`: partition
0.021 s, interior UMFPACK factorization 0.931 s, blocked Schur complement
1.799 s.

### Interior solver: UMFPACK by default

Condensation factors the FEM interior and then solves it against roughly one
right-hand side per retained interface node. Nearly all of the stage is those
triangular solves -- 98.6% of the Schur complement on `F2B_FLH`. Three settings
are available through `BLAB_METAL_FEM_CONDENSATION`:

| Value | Interior solver |
| --- | --- |
| `umfpack` (default) | SuiteSparse UMFPACK in `ComplexF64`, the same code the CPU and ROCm backends use. |
| `accelerate` | Apple Accelerate sparse LU in `ComplexF32`. |
| `accelerate_f64` | Apple Accelerate sparse LU in `ComplexF64`. |

Median over warm frequencies on `F2B_FLH`, M1 Pro, eight Julia threads:

| Stage | `umfpack` | `accelerate` | `accelerate_f64` |
| --- | ---: | ---: | ---: |
| `fem_condensation_factorization_s` | 0.326 s | **0.212 s** | 0.306 s |
| `fem_schur_extraction_s` | 1.070 s | **0.507 s** | 1.101 s |
| **`fem_condensation_s`** | **1.416 s** | **0.755 s** | 1.426 s |
| `assembly_s`, whole frequency | 2.683 s | **2.008 s** | 2.663 s |

`accelerate` is **1.87x** faster on condensation and 1.34x on the whole
assembly stage. `accelerate_f64` is a wash. The two columns together say the
important thing: **the speed-up is the precision drop, not a better solver.** In
equal precision Accelerate and UMFPACK are the same speed, so there is no
version of this that is both faster and equally accurate.

#### Why the default is `umfpack`

`accelerate` is not bit-equivalent to the default. Measured against the UMFPACK
result on `F2B_FLH` at 50/100/200 Hz, across diaphragm velocity, voice-coil
current, and all three exterior pressure sets:

| Metric | Worst observed |
| --- | ---: |
| Magnitude error | 0.016 dB |
| Phase error | 0.18 degrees |
| Relative norm, any output quantity | 3.2e-3 |

The physical error is negligible: 0.016 dB and 0.18 degrees are far below
anything audible, and below the repeatability of a real measurement. **But
3.2e-3 exceeds the 5e-4 relative-norm gate this project applies to its other
coupled validations**, so enabling it by default would quietly hold the Metal
coupled path to a looser numerical standard than every other backend.

That trade may well be worth taking -- 1.87x for 0.016 dB is a good deal -- but
it is a decision about which standard applies, not a performance tweak, so the
default stays conservative until that is settled. See [Options: speeding up FEM
static condensation on Apple
Metal](beat-engine-metal-condensation-options.md) for the full argument.

Note that `validate_metal_coupled.jl` does **not** discriminate between the two.
Its fixture is small enough that both solvers agree to the same 1e-6 figures and
the script passes under all three settings. The 3.2e-3 figure only appears on a
production-sized interior, so re-measure on a real model rather than on the
validation fixture.

#### Tuning the Accelerate path

`SparseSolve` cost grows **superlinearly** in the number of right-hand sides per
call, which makes the block size the dominant tuning parameter. Schur wall time
on the `F2B_FLH` interior by `BLAB_ACCELERATE_SCHUR_BLOCK`:

| Block | 8 | 16 | 32 | 64 | 256 | 1200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Schur | 0.790 s | **0.639 s** | 0.938 s | 1.502 s | 7.433 s | 29.104 s |

The default is 16. At 256 this path measures 6x *slower* than UMFPACK, so
**anyone re-measuring it must sweep the block size before drawing a
conclusion** -- a single bad block size inverts the result. Accelerate also does
not parallelise an individual solve, and at large blocks concurrent solves
serialise against each other; the win at block 16 comes from Julia-level
threading over narrow blocks, exactly as the UMFPACK path does.

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
| `BLAB_METAL_SINGULAR_WRITEBACK` | `gather` | How the singular blocks reach the operators. `gather` owns one dense cell per thread and is reproducible; `scatter` is the older one-thread-per-pair atomic write-back, kept for comparison. |
| `BLAB_METAL_OPERATOR_STORAGE` | `shared` | Use `private` to allocate the operator matrices in private storage and copy them to the host, the pre-2026-09-02 behavior. |
| `BLAB_METAL_PIPELINE` | `1` | Set to `0` to assemble and solve each sweep frequency sequentially instead of overlapping GPU assembly with the CPU factorization. |
| `BLAB_METAL_ATOMIC_SCATTER` | `1` | Diagnostic for `pair_atomic` only: `0` skips the atomic scatter to time the pair arithmetic (the operators are then wrong). |
| `BLAB_METAL_FEM_CONDENSATION` | `umfpack` | Interior solver for the coupled FEM condensation. `accelerate` is 1.87x faster and exceeds the 5e-4 gate; `accelerate_f64` is a wash. Falls back to `umfpack` when Accelerate's sparse library is absent. |
| `BLAB_ACCELERATE_SCHUR_BLOCK` | `16` | Right-hand sides per `SparseSolve` call. Cost is superlinear in this, so a bad value inverts the measurement. |
| `BLAB_COUPLED_STAGE_OVERLAP` | `auto` | `off` runs FEM condensation and BEM assembly in sequence instead of on separate threads. Needs more than one Julia thread. |
| `BLAB_BEAT_FUSED_BM` | `1` | Set to `0` to assemble the four operators and combine them on the host for exterior solves. Coupled solves, `host_staged` assembly and the `host` singular mode always take the four-operator path. |

The fused system is then solved by the adaptive dense solve described in
[BEAT Engine Core](beat-engine-core.md#adaptive-dense-solve) — dense LU or
diagonally preconditioned GMRES, chosen per solve. Metal has no GPU LU, so
both routes run on the host; shared storage means the host reads the assembled
matrix in place rather than copying it. Its environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `BLAB_BEAT_DENSE_SOLVE` | `auto` | Force `lu` or `gmres` instead of the cost model. |
| `BLAB_BEAT_GMRES_TOL` | `1e-6` | Tolerance on the true relative residual. |
| `BLAB_BEAT_GMRES_MAX_ITERATIONS` | `min(N, 1000)` | Iteration cap; reaching it reports non-convergence and falls back to the LU. |
| `BLAB_BEAT_GMRES_RESTART` | `0` | Restart length; `0` is unrestarted. |
| `BLAB_BEAT_GMRES_KRYLOV_PRECISION` | `f64` | `f32` reproduces the orthogonality-loss failure on demand. Not for production use. |
| `BLAB_BEAT_GMRES_REORTHOGONALIZE` | `dgks` | `always` or `never`; `never` is the failing variant, kept so the remedies can be compared. |
| `BLAB_BEAT_LU_GFLOPS` | `500` | Cost-model constants. Re-measure with `scripts/calibrate_dense_solve.jl` on any other machine. |
| `BLAB_BEAT_MATVEC_ENTRY_SECONDS` | `9.52e-11` | |
| `BLAB_BEAT_MATVEC_DOF_SECONDS` | `1.03e-6` | |
| `BLAB_BEAT_TRIANGULAR_GBPS` | `14` | |
| `BLAB_BEAT_GMRES_MODEL_ITERATIONS` | `210` | Expected iterations. A property of the operator, not the machine. |

## Verification

CPU-versus-Metal validation scripts:

| Script | Coverage |
|---|---|
| `validate_metal_fused_burton_miller.jl` | Fused exterior system against the four-operator system, multi-drive. `BLAB_VALIDATE_SYMMETRY` is a comma-separated arm list; the default runs `off,x,ground` and fails if any arm fails. |
| `validate_metal_singular_summation.jl` | The fused singular pair against the four-operator accumulation order, per pair, in Float32 against a Float64 reference: the two orders must agree in Float64, neither Float32 order may exceed the stated bound, and the fused order must not be systematically worse. |
| `validate_gmres_burton_miller.jl` | GMRES against the dense LU on a real assembled operator across the frequency band: true residual, three-way agreement between Krylov variants, restart independence, and that the failure mode the remedies cover is reachable. |
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
- The default `pair_gather` kernels are bitwise reproducible run to run, as
  are `pair_owned` and `entry_owned`. `pair_atomic` is not (atomic
  accumulation order); its differences are float32 summation noise.
- That holds for the singular stage as well, but only since the deterministic
  write-back landed. Before it, every mode routed its singular corrections
  through an atomic scatter, so nothing was actually reproducible.
  `BLAB_METAL_SINGULAR_WRITEBACK=scatter` restores the old behavior for
  comparison.
- Assembly being reproducible does not make a sweep reproducible: the CPU LU
  is multithreaded, and two runs of the same solve differ by about 3e-7
  relative in the exterior field. Golden-file comparisons belong on the CPU
  `reference` path, tolerance comparisons everywhere else.

## Resolved: the fused Burton-Miller gate at symmetry `xy`

`validate_metal_fused_burton_miller.jl` used to fail its `xy` arm, and `xy` was
kept out of the default arm list because of it. It passes now, on all four arms,
and the cause was the test fixture rather than any assembly code.

**A symmetry-reduced assembly needs a mesh that is one sector.** It folds mirror
images onto that sector. Hand it a mesh that already spans both sides of the
mirror plane and every contribution is counted twice, leaving an operator that is
close to singular. `sample.msh` spans x in [-0.198, 0.199] and y in
[-0.126, 0.127], so it is a valid fundamental domain for `off` and `ground` only.
The tell was that `snap_symmetry_planes` did nothing to it: identical faces,
vertices and triangle aspect ratios on all four arms, because there was nothing
to snap.

Condition number of the Burton-Miller left-hand side, 2 kHz, Apple M1 Pro:

| arm | mesh | kappa (cpu) | kappa (metal) | backends agree? |
|---|---|---|---|---|
| `off` | `sample.msh` | 4.44e+02 | 4.44e+02 | yes, to 4.2e-6 |
| `x` | `sample.msh` | 1.81e+07 | 2.43e+04 | **no, 100% apart** |
| `xy` | `sample.msh` | 1.33e+09 | 5.52e+04 | **no, 100% apart** |
| `x` | `sample_half.msh` | 5.29e+02 | 5.29e+02 | yes, to 4.2e-6 |
| `xy` | `sample_quarter.msh` | 5.23e+02 | 5.23e+02 | yes, to 4.1e-6 |

On the correct mesh every arm is well conditioned and the two backends agree.
The fold itself is sound. On the wrong mesh both backends produce garbage, and
different garbage, which is why the CPU and Metal columns diverge completely.

The script now picks the fundamental domain per arm — `sample.msh` for `off` and
`ground`, `sample_half.msh` for `x`, `sample_quarter.msh` for `xy` — and calls
`validate_symmetry_fundamental_domain!` after snapping, the same check the
drivers and every other symmetry script already ran. An invalid combination now
stops with a named vertex instead of returning a plausible-looking error:

```
ERROR: Mesh is not in the positive X fundamental domain for XY symmetry.
       Vertex 41 has x=-0.0098830005 m.
```

Results after the fix, M1 Pro, tolerance 5e-6:

| arm | mesh | lhs | rhs | pressure |
|---|---|---|---|---|
| `off` | `sample.msh` | 2.5e-7 | 5.4e-7 | 6.393e-7 |
| `x` | `sample_half.msh` | 2.532e-7 | 3.439e-7 | 6.170e-7 |
| `xy` | `sample_quarter.msh` | 2.559e-7 | 3.275e-7 | **5.397e-7** |
| `ground` | `sample.msh` | 1.6e-7 | 1.0e-6 | 2.839e-6 |

Against 1.6459548e-5 for `xy` on `sample.msh`. All four arms are in the default
list now.

### What this cost, and the note that misdirected it

An earlier version of this section blamed the atomic singular scatter and offered
two candidate fixes: a better-conditioned `xy` fixture, **or** a deterministic
scatter. The first was right. The second was not, and it was the one that got
built first.

The deterministic write-back was worth having on its own merits and is now the
default (see the singular write-back section above). It removed the run-to-run
spread completely. It did not move the arm:

| write-back | run 1 | run 2 | run 3 |
|---|---|---|---|
| `scatter` (atomics) | 2.3376e-5 | 1.7533e-5 | 2.1780e-5 |
| `gather` (deterministic) | 1.6459548e-5 | 1.6459548e-5 | 1.6459548e-5 |

The atomics were adding about plus or minus thirty percent of noise on top of a
real error near 1.6e-5, which was already three times the tolerance. The noise
was the visible symptom; the invalid fixture was the cause. A diagnosis that
explains only the variance and not the magnitude is not finished.
