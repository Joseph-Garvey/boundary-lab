# BEAT Engine Core

Boundary Lab's BEAT Engine, short for Boundary Element Acoustic Toolkit Engine,
is the Julia solver stack used for local exterior BEM and coupled FEM-BEM-LEM
solves. The Python side stages mesh assets and request JSON, while
`src/blab/solvers/julia_local/solver.jl` owns the numerical solve.
`BeatEngineCore.jl` provides shared BEM mesh, quadrature, formulation,
symmetry, Burton-Miller, and field-evaluation utilities used by BEAT Engine CPU,
Nvidia CUDA, and AMD ROCm.

This page describes the shared exterior-BEM core and its prescribed normal
velocity path. The FEM assembly, interface transfer, electrodynamic equations,
and coupled block systems are documented in [Coupled
Solver](../Coupled%20Solver.md). Local production solves use single precision
(`Float32/ComplexF32`).

Backend-specific details live in:

- [BEAT Engine CPU](beat-engine-CPU.md)
- [BEAT Engine AMD ROCm](beat-engine-rocm.md)
- [BEAT Engine CUDA](beat-engine-CUDA.md)

## Solve Pipeline

For each solve request, Julia:

1. Loads one or more Gmsh 2.2 ASCII triangle meshes.
2. Applies per-mesh scale and translation, then combines them into one `BoundaryMesh`.
3. Builds P1 pressure and DP0 velocity spaces.
4. Builds frequency-independent identity/mass matrices.
5. For each frequency, assembles Helmholtz boundary operators.
6. Solves the Burton-Miller Neumann system for boundary pressure.
7. Evaluates SPL at polar and optional spherical observation points.
8. Computes per-radiator acoustic impedance from pressure over driven elements.

Radiators are resolved by both physical tag and mesh id, so duplicate physical tags are allowed across meshes when the radiator specifies its mesh.

## Boundary Integral Formulation

The backend uses the outgoing Helmholtz Green function:

$$
G_k(x,y) = \frac{e^{i k \lVert y-x \rVert}}{4\pi \lVert y-x \rVert}.
$$

The assembled operator tuple contains:

- `single_layer`: maps DP0 Neumann data to P1 test functions.
- `double_layer`: maps P1 pressure data to P1 test functions.
- `adjoint_double_layer`: maps DP0 Neumann data to P1 test functions.
- `hypersingular`: maps P1 pressure data to P1 test functions.

The dense Burton-Miller system is formed as:

$$
\left(\frac{1}{2} I_{P1,P1} - D + \eta H\right)p
=
\left(-S - \eta\left(D^{*} + \frac{1}{2} I_{P1,DP0}\right)\right)q,
$$

where:

$$
\eta = \frac{i}{k}.
$$

Here \(p\) is the solved boundary pressure and \(q\) is the Neumann/radiator drive vector. Radiator normal velocity is converted to Neumann data with:

$$
q = i \rho \omega v_n.
$$

For a test triangle \(T_i\), trial triangle \(T_j\), test basis function \(\phi_a\), and trial basis function \(\psi_b\), a typical single-layer block entry is:

$$
S_{ab}^{ij}
=
\int_{T_i}\int_{T_j}
\phi_a(x) G_k(x,y) \psi_b(y)\,dS_y\,dS_x.
$$

The double-layer and adjoint double-layer use normal derivatives of \(G_k\):

$$
D_{ab}^{ij}
=
\int_{T_i}\int_{T_j}
\phi_a(x) \frac{\partial G_k(x,y)}{\partial n_y} \psi_b(y)\,dS_y\,dS_x,
$$

$$
(D^{*})_{ab}^{ij}
=
\int_{T_i}\int_{T_j}
\phi_a(x) \frac{\partial G_k(x,y)}{\partial n_x} \psi_b(y)\,dS_y\,dS_x.
$$

The hypersingular block is assembled with the surface-curl weak form:

$$
H_{ab}^{ij}
=
\int_{T_i}\int_{T_j}
G_k(x,y)
\left[
\operatorname{curl}_\Gamma \phi_a(x)\cdot\operatorname{curl}_\Gamma \psi_b(y)
- k^2 \phi_a(x)\psi_b(y)n_x\cdot n_y
\right]\,dS_y\,dS_x.
$$

BEAT Engine stores all four dense matrices explicitly. This makes direct dense solves straightforward, but memory usage scales quadratically with the P1/DP0 unknown counts.

## Singular And Regular Pairs

Operator assembly splits triangle pairs into regular pairs and adjacent/coincident singular pairs. Regular pairs are integrated directly with triangle quadrature. Adjacent, edge-sharing, vertex-sharing, and coincident pairs are handled with Duffy correction rules.

The singular path builds a frequency-independent correction cache for the mesh and singular quadrature order. The cache stores adjacent/coincident element pairs, orientation-remapped Duffy rules, surface curls, and pair geometry scalars once, then reuses them across the frequency loop. Per-frequency work still evaluates the Helmholtz kernels because they depend on \(k\).

The image-singular path for symmetry computes compact `singular - regular` correction blocks before adding them to the dense operators. CUDA scatters those blocks through GPU correction buffers with atomics, ROCm uses race-free device gather kernels, and CPU applies the same correction logic directly on host matrices.

## Symmetry Mode

BEAT Engine supports `off`, `x`, and `xy` symmetry modes in the CPU, Nvidia CUDA,
and AMD ROCm backends. The application disables the symmetry control for backends
that do not advertise symmetry support.

The application passes a reduced-domain mesh plus symmetry metadata. BEAT Engine does not infer or match mirrored element orbits. Instead, it assumes the global origin is the symmetry origin and validates that the provided mesh lies in the positive fundamental domain:

- `x`: all mesh vertices must be on or positive of the global X=0 plane.
- `xy`: all mesh vertices must be on or positive of both the global X=0 and Y=0 planes.

For each enabled mirror plane, the backend builds image transforms. Points and normals are reflected as ordinary vectors. Surface curls in the hypersingular weak form are reflected as pseudovectors, using `det(R) * R * curl` for each mirror transform. For `x` symmetry, the reduced operator includes the identity domain plus the X-reflected image. For `xy` symmetry, it includes the identity domain plus X, Y, and XY images.

The reduced Galerkin system remains defined on the reduced mesh's P1 pressure space and DP0 Neumann space. Regular operator assembly handles symmetry by adding reflected trial/source image contributions into the reduced matrices. This keeps the solved pressure vector on the reduced P1 dofs while accounting for the missing physical images.

P1 test rows receive symmetry orbit weights for vertices on symmetry planes. These weights are applied consistently to the Helmholtz operators and to the Burton-Miller identity/mass matrices. This is required for seam vertices on X=0 and Y=0 to match the corresponding full expanded system.

Singular handling has two parts:

- Identity-domain adjacent, edge-sharing, vertex-sharing, and coincident pairs use the normal Duffy correction cache.
- Reflected image pairs that become coincident, edge-adjacent, or vertex-adjacent across a symmetry plane use an image-singular correction cache.

Field evaluation is symmetry-aware by materializing mirrored quadrature sources in the field-evaluation cache. The existing field kernels are then reused unchanged: source points and normals are reflected for each symmetry transform, while source faces, source elements, quadrature weights, and P1 basis values continue to reference the reduced mesh.

Radiator impedance is computed from reduced-domain pressure and then scaled by the symmetry reduction factor: 2 for `x`, 4 for `xy`. This reports force over the full physical radiator image set while keeping the solve unknowns reduced.

## Field Evaluation

After boundary pressure is solved, the backend evaluates the potential at observation points:

$$
u(x)
=
\int_\Gamma
\frac{\partial G_k(x,y)}{\partial n_y}p(y)
- G_k(x,y)q(y)
\,dS_y.
$$

The implementation precomputes a field-evaluation cache containing quadrature source points, normals, weights, source faces, source elements, and P1 basis values.

The evaluated potential is:

$$
u(x)
=
\sum_m
\left[
\frac{\partial G_k(x,y_m)}{\partial n_m}p_m
- G_k(x,y_m)q_m
\right]w_m,
$$

where \(y_m\), \(n_m\), and \(w_m\) come from the cached source quadrature data.

SPL is reported as:

$$
\mathrm{SPL}(x)
=
20\log_{10}\left(\frac{|u(x)|}{20\ \mu\mathrm{Pa}}\right).
$$

Horizontal, vertical, and spherical observation points are concatenated into one field evaluation per frequency, then sliced back into result arrays. This avoids rebuilding source strengths for each observation set.

Physical-system result schema version 2 carries numeric arrays as typed,
little-endian, row-major base64 payloads. Observation coordinates remain in the
frequency-invariant result domains and are not repeated in each streamed
frequency result. The Python decoder continues to accept schema version 1
real/imaginary decimal arrays for compatibility with older workers.

## Adaptive Dense Solve

Exterior solves on the fused Burton-Miller path get one `N x N` system matrix
and an `N x drives` right-hand side, and there are two reasonable ways to solve
that. `BeatEngineDenseSolve.jl` chooses per solve.

- **Dense LU.** `(8/3)N^3` for the factorization, then one triangular solve per
  drive. One factorization serves every channel, which the CPU and Metal
  backends are deliberately built around.
- **GMRES, diagonal preconditioning.** One dense matvec per iteration, per
  drive, with no factorization to share.

The LU is `O(N^3)` and GMRES is `O(N^2)`, so they cross in the problem size;
the LU amortises across drives and GMRES does not, so they cross the other way
in the drive count. The router therefore compares the two costs directly rather
than testing a dof threshold:

    choose GMRES when   drives * t_gmres(N)  <  T_fact(N) + drives * t_tri(N)

A pair of independent thresholds — "above N dofs and below D drives" — would be
wrong, because the drive crossover moves with N. It would route a three-way
design to the slower path at exactly the size where the dof test says GMRES
wins.

### Calibration, and why it is not a constant

Four measured constants sit behind that comparison: the LU's effective GFLOP/s,
two terms of a matvec model `t = a N^2 + b N`, and the triangular solve's
bandwidth. The linear matvec term is not decoration — the effective bandwidth
of a dense complex matvec is still rising across the useful size range, so
fitting `iterations * 8N^2 / bandwidth` with a single bandwidth constant
misplaces the crossover at both ends.

The shipped values are measured on one machine, and every one is an environment
override. `scripts/calibrate_dense_solve.jl` re-measures them on any host and
prints the export lines. This matters most for the accelerator backends: the
CUDA and ROCm paths factor on the device (`cuda_dense_lu!`, `rocm_dense_lu!`),
so their arithmetic-to-bandwidth ratio is not the CPU's, and a threshold
calibrated here would be meaningless there. Those backends do not route through
this code yet.

The iteration constant is the exception: it is a property of the operator, not
the machine, and it is deliberately set toward the high end of the measured
range. Over-estimating iterations routes a marginal problem to the LU, which is
never catastrophically wrong; under-estimating routes it to a Krylov solve that
then costs several times the model's guess.

### Convergence, precision and the fallback

The tolerance is on the **true** relative residual `||b - Ax|| / ||b||`,
recomputed against the operator, not on the preconditioned residual the Givens
recursion tracks. The recursion only decides when to end an inner cycle.

The Krylov space is carried in Float64 while the operator stays Float32, with
DGKS reorthogonalisation. Both are load-bearing. An unreorthogonalised Float32
Arnoldi recurrence loses orthogonality on this operator and reports iteration
counts several times too large — which reads as bad conditioning rather than as
a bug. The Float64 basis costs 16 bytes per entry for a few dozen vectors,
kilobytes against a matrix measured in gigabytes.

Two signatures distinguish the failure modes, and they are opposites:

| symptom | cause |
|---|---|
| high count, and it changes with the restart length | orthogonality loss in the recurrence |
| high count, and the restart makes no difference | the cycle is ending early — breakdown test, budget, or stall check |
| low count, restart-invariant | healthy |

A related tell for the first: with a degraded basis, *more* restarting gives
*fewer* total iterations, because restarting discards the corrupted basis. That
is backwards for a healthy solver, where a larger Krylov space can only help.

GMRES falls back to the dense LU when it does not converge, reporting the
fallback in the run's diagnostics rather than raising. The operator does not
stagnate on the meshes tried, so this is a safety net rather than a load-bearing
part of the feature — but it earned its place during development, when an
unreachable tolerance turned a working solve into a failed one.

`scripts/validate_gmres_burton_miller.jl` gates all of this against a real
assembled operator across the frequency band. It asserts that the reference
variant needs at least ten iterations, and that plain single Gram-Schmidt fails
on the same system: three variants agreeing is evidence only because a fourth
demonstrably fails. A routine validated only on cases that converge in five
steps is not validated for the cases it exists for, and a mid-band-only gate
would pass straight through a low-frequency conditioning regression.

## Performance Shape

The expensive stages are:

- dense operator assembly, roughly \(O(N_e^2 q^2)\), where \(N_e\) is face count and \(q\) is quadrature point count
- dense direct solve, roughly \(O(N_p^3)\), where \(N_p\) is P1 dof count
- field evaluation, roughly \(O(N_{\mathrm{obs}} N_e q)\)

Symmetry can improve runtime by more than the simple physical-area reduction would suggest because it reduces the dense system dimension. Assembly has to include image contributions, so its scaling is closer to the expected half-domain or quarter-domain work plus image passes. The direct Burton-Miller solve, however, scales roughly as \(O(N_p^3)\). Halving the P1 unknown count can make the dense solve approach one eighth of the full cost, and quartering it can make that portion much smaller still.

That \(O(N_p^3)\) is the dense LU, and it is why reducing \(N_p\) has always
outranked accelerating the solve. Above the adaptive router's crossover the
exponent changes: with GMRES the solve becomes \(O(N_p^2)\) like assembly, the
whole pipeline is quadratic, and what sets the ceiling is memory rather than
time. Symmetry still helps, but by the square rather than the cube.

## Start-up And Precompilation

A worker process must load and compile the engine before it can solve
anything, and until the bundle packages existed it did all of that from
source. `solver.jl` `include`d 20,700 lines of engine and carried 1,300 lines
of driver itself, and Julia caches native code only for packages — so neither
the pkgimage cache nor the sysimage in `docker/` could see any of it. Against a
three-frequency A3 sweep that does about 1.8 s of real work, the start-up was
several times the compute.

Measured on an M1 Max, A1 (1,974 dofs), one frequency, arms interleaved so both
see the same machine, and only samples with no other Julia running counted:

| | before | after |
|---|---|---|
| bare Julia start-up | 0.09 s | 0.09 s |
| spawn to worker-ready | 10.0-10.6 s | 3.33-3.51 s |
| request to initialized | 2.9-3.2 s | 1.34-1.49 s |
| first solve, CPU | 5.8-6.3 s | 1.75-1.87 s |
| **time to first result, CPU** | **18.8-20.0 s** | **6.5-6.8 s** |
| **time to first result, Metal** | **28.9-30.0 s** | **16.3-17.3 s** |
| second solve, same worker | 1.46-1.61 s | 1.44-1.60 s |

The last row is the control: a warm solve is unchanged, which is what says this
is start-up and not the solver.

Subtracting the warm solve from the first one isolates what a first solve pays
on top. On the CPU backend that is 4.4 s before and 0.3 s after — Julia JIT,
which the precompile workload removes. The same subtraction on Metal leaves
about 11 s in *both* arms. That residual is GPU kernel compilation, and
GPUCompiler has no disk cache, so no sysimage or pkgimage can hold it. Keeping
a worker process alive between solves is the only remedy for that component,
which makes the persistent worker in the Python adapter load-bearing rather
than an optimisation.

The sources now live in packages, one per accelerator, under `julia_engine/`.
Each bundle `include`s the same engine and driver files and carries a
`PrecompileTools` workload that solves one frequency on a four-triangle
tetrahedron. The workload is what earns most of the win: without it a first
solve still JIT-compiles ~350 methods, with it ~90.

- **One package per backend** because a package gets one precompile cache and
  the engine picks its backend as it loads. `BEAT_ENGINE_BACKEND`, set by the
  bundle, is what `BeatEngineCore` reads instead of the environment — the
  environment it would read is the one the cache was *built* in.
- **The workload runs on the CPU backend even in a GPU bundle.** It compiles
  everything up to the backend branch, and it needs no device, which matters
  because precompilation runs in a sandboxed worker on a machine that may have
  no accelerator. What a GPU workload would add is kernel compilation, and that
  cannot be cached to disk at all — it is why the Python side keeps one worker
  alive across solves rather than starting one per job.
- **Invalidation is Julia's own.** Every `include`d file is recorded in the
  cache header and rechecked on load, so editing an engine source rebuilds the
  pkgimage on the next process. A sysimage has no such check, which is why the
  bundles are the mechanism and the sysimage in `docker/` is an optional extra
  built once inside an image from the sources shipped in that image.

Instantiating a backend environment is what installs its bundle
(`Pkg.develop` of the bundle path is already recorded in `julia_local`,
`julia_metal` and `julia_rocm`; the CUDA image does it in its Dockerfile). An
environment without one still works: `solver.jl` falls back to including the
sources directly, which is what the analysis scripts under `scripts/` and the
Julia test suite do in any case.

### Ahead-of-time codegen is not bit-identical to the JIT

Compiling the numerical kernels ahead of time does not produce the same
machine code as compiling them on first call, so a bundled worker's results
differ from the pre-bundle path in the last bits. Measured on the CPU backend
over the ATH ladder A1-A5 at 500, 2000 and 6000 Hz: SPL agrees to 2.0e-5 dB
and pressures to about one Float32 ulp (6.5e-9 of peak). For scale, the Metal
backend's atomics make it non-reproducible run-to-run at roughly 8e-7, two
orders of magnitude larger, and the solver emits Float32 — 1e-5 dB is the last
digit of the output format.

It is not zero, though, so `BLAB_BEAT_ENGINE_BUNDLE=0` forces the include
fallback, which is bit-identical to the pre-bundle path (verified over the
same cases). Use it when a number has to be reproduced exactly; it costs the
old start-up.

## Important Files

- `src/blab/solvers/beat_engine_backend.py`: Python adapter that stages assets and streams JSON events.
- `src/blab/solvers/julia_local/solver.jl`: the worker entry point — bundle resolution and dispatch only.
- `src/blab/solvers/julia_local/BeatEngineDriver.jl`: request handling, mesh/radiator setup, frequency loop, backend dispatch, drive calculation.
- `src/blab/solvers/julia_engine/BeatEngine{Cpu,Metal,Cuda,Rocm}Bundle`: the packages that include the engine and the driver so Julia can cache their native code.
- `src/blab/solvers/julia_local/src/BeatEngineCore.jl`: mesh representation, shared quadrature/formulation code, Burton-Miller solve, field evaluation interfaces.
- `src/blab/solvers/julia_local/src/BeatEngineDenseSolve.jl`: adaptive dense solve — the LU/GMRES cost model, the diagonally preconditioned GMRES, and the fallback.
- `src/blab/solvers/julia_local/src/BeatEngineCpu.jl`: include hub for the CPU implementation files.
- `src/blab/solvers/julia_local/src/BeatEngineCuda.jl`: include hub for the CUDA implementation files.
- `src/blab/solvers/julia_local/src/BeatEngineRocm.jl`: include hub for the ROCm implementation files.
