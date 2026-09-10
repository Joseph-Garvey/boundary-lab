# Options: speeding up FEM static condensation on Apple Metal

Written for review. This records where the time actually goes in the coupled
per-frequency budget on Apple Metal, the four options for reducing it, and the
tradeoffs of each. It is a decision document, not a description of shipped
behaviour — see [BEAT Engine Apple Metal](beat-engine-metal.md) for what the
backend does today.

## Why this came up

On BEM-light coupled models the Metal backend is barely faster than the CPU one.
The `F2B_FLH` example — 22,431 FEM vertices, 127k tetrahedra, an `xy`-symmetric
exterior mesh of about 2,500 triangles — runs a 30-point sweep in about the same
wall time on either backend. The reason is not the Metal kernels. It is that most
of each frequency is host code that both backends share.

Warm per-frequency mean on an M1 Pro, eight Julia threads:

| Stage | Runs on | BEAT CPU | BEAT Metal |
| --- | --- | ---: | ---: |
| `bem_operator_s` — BEM assembly | **GPU** on Metal | 0.95 s | 0.99 s |
| `fem_condensation_s` — UMFPACK + Schur | host on both | 1.23 s | **1.28 s** |
| `coupled_factorization_s` — dense LU | host on both | 0.24 s | 0.22 s |
| other block assembly | mixed | 0.10 s | 0.12 s |
| `field_s` — exterior field | **GPU** on Metal | 0.52 s | 0.05 s |
| **Total per frequency** | | **3.04 s** | **2.68 s** |

`fem_condensation_s` is the largest single cost and is entirely on the host. Even
a free GPU would only take the frequency from 2.68 s to about 1.64 s — a 1.85x
ceiling — and condensation is where most of that gap lives.

## Where the condensation time goes

`fem_condensation_s` splits into a partition, an interior UMFPACK factorization,
and the Schur complement. The Schur complement dominates, so
`_blocked_umfpack_schur_complement` now reports its three inner steps. Summed
across the eight worker tasks, `F2B_FLH` at 200 Hz:

| Step | Time | Share |
| --- | ---: | ---: |
| `ldiv!` — UMFPACK sparse triangular solves | **6.260 s** | **98.6%** |
| `mul!` — accumulate into the Schur block | 0.058 s | 0.9% |
| Densify sparse right-hand-side columns | 0.032 s | 0.5% |

(These are thread-summed, so they exceed the 1.07 s wall time of the stage by
roughly the thread count. Compare them against each other, not against
`fem_schur_extraction_s`.)

So the target is precisely one thing: **sparse triangular solves against a
21,329-unknown interior factorization, with about 1,100 right-hand sides, once
per frequency.** The partition is free and the accumulation is nearly free.

For context, the surrounding per-frequency costs on Metal are
`fem_condensation_partition_s` 0.006 s, `fem_condensation_factorization_s`
0.352 s (the interior LU itself), and `fem_schur_extraction_s` 1.074 s.

## A clarification about "LU on Apple Silicon"

Two different factorizations get conflated here, and only one of them is
GPU-constrained.

- **The dense `ComplexF32` LU** of the Burton-Miller or coupled system. Metal
  Performance Shaders factors `Float32` and `Float16` only, with no complex path,
  which is why `BLAB_METAL_DENSE_SOLVE` defaults to `cpu` and why the
  `equivalent_real` device alternative measures 5.7x slower. This is a real Metal
  limitation.
- **The sparse LU of the FEM interior**, which is what condensation needs. This
  has never run on a GPU on any backend except CUDA, where cuDSS provides it.
  It is SuiteSparse/UMFPACK running on the host, on Metal and ROCm and CPU alike.

Reusing a sparse symbolic factorization across frequencies (option C below) is
therefore not blocked by Apple Silicon at all — it is ordinary host code, and
`lu!` refactorization is available in Julia today. Verified on this machine: a
`ComplexF64` sparse `lu!` reusing an existing symbolic analysis returns a
correct factorization (residual 2.9e-13).

## Option A — port the triangular solve to Metal

Write a level-scheduled sparse triangular solver as Metal kernels and keep the
interior solve on the GPU.

**For.** It is the only option that removes host work rather than making it
faster, so it is the only one that raises the 1.85x ceiling itself.

**Against.**

- **Apple ships no sparse direct solver for Metal.** MPS covers dense `Float32`
  and `Float16`. Everything — factorization, permutation, triangular solve —
  would be ours to write and maintain.
- **UMFPACK's factors are not in a GPU-friendly form.** Extracting `L`, `U`, `P`,
  `Q`, and the scaling would be a prerequisite, and the result is a general
  unsymmetric multifrontal factorization, not something with clean level sets.
- **Sparse triangular solve is dependency-bound**, which is the workload GPUs are
  worst at. It frequently loses to a good multithreaded CPU implementation.
- **ROCm had the option and declined it.** `_build_rocm_hybrid_fem_condensation`
  has rocSPARSE available and still keeps UMFPACK on the host, uploading only the
  dense Schur block. That is a considered precedent from a more mature GPU stack
  than Metal's.

**Estimate.** Weeks of work, research-grade, genuinely uncertain payoff. Not
recommended as a first move.

## Option B — Apple Accelerate sparse LU

Replace the interior UMFPACK factorization and its triangular solves with
Accelerate's Sparse Solvers, which are CPU code heavily tuned for Apple Silicon.

**For.**

- **Complex is supported, including single precision.** The SDK exposes
  `SparseMatrix_Complex_Float` and `SparseMatrix_Complex_Double`, and the LU
  factorization is documented for "real or complex matrices". `ComplexF32`
  matches the rest of the FP32 Metal pipeline, where the current path forces
  `ComplexF64` and casts the result back down.
- **Multi-RHS solve is a first-class entry point.** `SparseSolve` takes a dense
  matrix of right-hand sides, which is exactly the 1,100-column block the Schur
  complement needs, instead of the current 64-column chunking.
- **A symbolic/numeric split is built in.** `_SparseSymbolicFactorLU` plus
  `_SparseNumericFactorLU`, and `_SparseRefactorLU` for refactoring in place,
  give this path its own route to option C. (Bound and verified, but not yet
  wired into the condensation — see option C.)
- Symbols verified present on this machine: `_SparseFactorLU_Complex_Float`,
  `_SparseNumericFactorLU_Complex_Float`, `_SparseRefactorLU_Complex_Float`,
  `_SparseSolveOpaque_Complex_Float`, `_SparseDestroyOpaqueNumeric_Complex_Float`.

**Against.**

- **Hand-written `ccall` bindings against C structs.** `SparseMatrixStructure`,
  the matrix and dense-matrix types, and both options structs must be mirrored in
  Julia exactly. A layout error is silent corruption, not a compile failure. This
  is the main risk and it argues for validating against UMFPACK on every solve
  during bring-up.
- **`_SparseNumericFactorLU` and `_SparseRefactorLU` require macOS 15.5.** The
  backend's stated floor is macOS 13, so the split-factorization path needs a
  version check with a fallback to the one-shot `SparseFactor`.
- **Apple-only, and it is a second sparse solver to maintain** beside UMFPACK,
  which the CPU, CUDA and ROCm backends keep using.
- **Dropping to `ComplexF32` for the interior solve is a numerical change.** The
  FEM interior can be poorly conditioned — the coupled validation fixture is
  already at `cond(A, 1) ≈ 3.8e9` — so this needs measuring against the existing
  5e-4 FP32 gate, and may have to stay at `ComplexF64`.

**Estimate.** Days, not weeks. Bounded, testable against the existing UMFPACK
path, and the only option that attacks the 98.6% directly without writing a
solver from scratch.

## Option C — reuse the UMFPACK symbolic factorization

`fem_system` is assembled from cached sparse blocks as `K - ω²M` plus loss and
wall terms, so **its sparsity pattern does not change with frequency**, but
`_build_host_fem_condensation` calls `lu()` afresh every frequency and redoes the
symbolic analysis each time. `lu!(F, A)` reuses it.

**For.** Small, safe, needs no new dependency, and — unlike option B — it is not
Apple-only. It applies to every backend that condenses:

- **`beat_cpu` on any platform.** Linux, Windows, and Intel macOS all run
  UMFPACK, and Accelerate reaches none of them.
- **`beat_rocm`**, which is Linux plus an AMD GPU, and keeps its interior solve
  on the host through `_build_rocm_hybrid_fem_condensation`.
- **`beat_metal` whenever the interior solver is `umfpack`** — which is the
  current default, so this is today's Metal path too.

On Apple Silicon itself Accelerate is effectively always present: it ships with
macOS, so `accelerate_sparse_available()` only returns false on an Intel Mac
(the binding requires `aarch64`) or if an SDK change breaks the struct-layout
check. The reason option C is not redundant is not that Accelerate might be
missing on a Mac — it is that **most backends do not run on a Mac at all.**

**Against.** It only touches `fem_condensation_factorization_s` (0.326 s), not the
1.070 s of triangular solves, so the ceiling is modest — perhaps 0.1-0.2 s per
frequency. It also requires caching the factorization across frequencies in the
coupled cache, and a guard for a refactorization that fails on a numerically
awkward frequency and has to fall back to a full `lu()`.

**Estimate.** Hours.

**Status: unimplemented, for both solvers.** Note that the Accelerate path does
*not* currently subsume this, contrary to what an earlier draft of this document
claimed. `_build_accelerate_fem_condensation` calls `accelerate_sparse_lu` fresh
on every frequency, exactly as the UMFPACK path calls `lu`. The binding does
expose `accelerate_sparse_refactor!` (`_SparseRefactorLU`, macOS 15.5+, verified
working at residual 1.8e-7), so the equivalent optimisation is available there —
it is simply not wired in. Doing option C properly means both halves: `lu!` for
UMFPACK and `accelerate_sparse_refactor!` for Accelerate.

## Option D — do nothing

**For.** The honest framing is that Metal is the wrong backend for a model shaped
like `F2B_FLH`, and the documentation now says so. A user with a large exterior
mesh already gets the full benefit; a user with a large FEM interior and a small
boundary should be on the CPU backend, where they lose almost nothing.

**Against.** The 1.85x ceiling stands, and coupled models with large interiors are
a normal loudspeaker workload, not an edge case.

## Measured result for option B

Option B was built and measured. `BLAB_METAL_FEM_CONDENSATION` selects the
interior solver on the Metal backend: `umfpack` (default), `accelerate`
(`ComplexF32`), or `accelerate_f64` (`ComplexF64`). Median of six warm
frequencies on `F2B_FLH`, M1 Pro, eight Julia threads:

| Stage | UMFPACK | Accelerate F32 | Accelerate F64 |
| --- | ---: | ---: | ---: |
| `fem_condensation_factorization_s` | 0.326 s | **0.212 s** | 0.306 s |
| `fem_schur_extraction_s` | 1.070 s | **0.507 s** | 1.101 s |
| **`fem_condensation_s`** | **1.416 s** | **0.755 s** | 1.426 s |
| `assembly_s` (whole frequency) | 2.683 s | **2.008 s** | 2.663 s |

**Accelerate in single precision is 1.87x faster on condensation and 1.34x on the
whole assembly stage. In double precision it is a wash.** The entire advantage
comes from the precision drop, not from a better solver — which also means the
speedup and the accuracy cost cannot be separated.

Two findings worth carrying forward:

- **`SparseSolve` cost grows superlinearly in right-hand sides per call.** Schur
  wall time by block size: 8 → 0.790 s, 16 → 0.639 s, 32 → 0.938 s, 64 → 1.502 s,
  256 → 7.433 s, 1200 → 29.104 s. The first implementation here used 256 and
  measured 6x *slower* than UMFPACK; the default is now 16, tunable with
  `BLAB_ACCELERATE_SCHUR_BLOCK`. Anyone re-measuring this path must sweep the
  block size before drawing a conclusion.
- **Accelerate does not parallelise a single solve**, and at large block sizes
  concurrent solves serialise against each other. The win at block 16 comes from
  Julia-level threading over narrow blocks, exactly as the UMFPACK path does.

### The accuracy cost

Against the UMFPACK result on `F2B_FLH` at 50/100/200 Hz:

| Metric | Worst observed |
| --- | ---: |
| Relative norm, any output quantity | 3.2e-3 |
| Magnitude error | 0.016 dB |
| Phase error | 0.18° |

The physical error is negligible — 0.016 dB is far below anything audible or
measurable. But **3.2e-3 exceeds the 5e-4 relative-norm gate** the coupled
validations use elsewhere in this project, so it is a real deviation from the
stated numerical standard even though it does not matter acoustically. The small
`validate_metal_coupled.jl` fixture does not show it: there both solvers agree to
the same 1e-6 figures, and that script passes under all three settings.

**The default therefore stays `umfpack`.** Flipping it is a judgement about
whether the 5e-4 gate is the right standard for a quantity that lands at 0.016 dB,
and that is a call for review rather than one to make silently in a performance
change.

## Outcome

**A fifth option turned out to dominate all four, and it is now shipped.** The
premise of this document was that the condensation had to get *faster*. It did
not: it had to stop being on the critical path. The condensation is independent
of the BEM operator assembly within a frequency, and on an accelerator backend
the two use different processors, so they now run concurrently — see [Stage
overlap](beat-engine-metal.md#stage-overlap). Together with the Schur block
balancing in [Schur block balance](beat-engine-metal.md#schur-block-balance),
`F2B_FLH` went from 2.68 s to 1.70 s per frequency, and Metal from a tie with
BEAT CPU to 1.62x ahead of it, with bit-identical outputs.

That reframes what follows. The table below is the pre-overlap budget, kept
because the measurements are still sound and still say where the host time goes.
But `fem_condensation_s` is now partly hidden behind device assembly, so the
value of shortening it further has dropped: only the part that exceeds
`bem_operator_s` is still on the critical path. On `F2B_FLH` that is about
0.14 s, not 1.37 s.

Option B in particular is worth much less than it was. It bought 1.34x on the
assembly stage when the condensation was fully exposed; overlapped, most of what
it saves lands in a window the GPU is busy anyway. Since it is also the only
option carrying an accuracy cost, and the 5e-4 coupled parity gate it misses is
the same gate `validate_rocm_coupled.jl` applies, the case for enabling it is now
weaker on both sides of the trade. **The default stays `umfpack`.**

Option C is unaffected: it touches `fem_condensation_factorization_s`, still
unimplemented, still cheap, still platform-independent. Option A stays closed.

## Recommendation (pre-overlap; superseded by the above)

**B is built and measured; the open question is whether to enable it.** It is
1.87x on the stage that dominates, and the struct-layout risk it carried is
retired — the ABI is asserted on load and both precisions are checked against a
reference solve. What it cannot do is deliver that speed without the precision
drop, as the `accelerate_f64` column shows. So the decision is the accuracy one
above, not a performance one.

**Option C is still worth doing on its own merits**, and independently of B. The
sparsity pattern of `K - ω²M` is frequency-invariant on every backend, and
`beat_cpu` and `beat_rocm` mostly run on hardware Accelerate cannot reach at all
— Linux, Windows, Intel Macs. It also helps `beat_metal` as currently
configured, since the default interior solver is UMFPACK.

**Option A should stay closed.** Nothing measured here argues for writing a
sparse triangular solver in Metal.
