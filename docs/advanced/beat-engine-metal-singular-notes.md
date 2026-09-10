# Notes: optimising the Metal singular kernel

Engineering notes from the 34x speed-up of the Metal singular correction stage.
Written to be re-readable later, so it records the reasoning and the dead ends,
not just the final patch.

Result: the singular stage went from 0.748 s to 0.022 s on `sample_detailed.msh`
(7,000 faces, 3,502 P1 unknowns, `q2/s2`), taking total Metal operator assembly
from 2.647 s to 1.915 s.

## The trap I nearly walked into

The performance table said:

> Singular kernel | 0.748 s | 28%

and the prose underneath said the singular kernel was the optimisation target
because it handled 0.2% of the pairs for 28% of the time. Every instinct said
"the Duffy quadrature is expensive, make the quadrature cheaper."

That reasoning was wrong, and it was wrong in a way that would have wasted a lot
of effort. A singular pair at `s2` needs 32 to 96 quadrature point-pairs against
a regular pair's 9, so the arithmetic-per-pair ratio *is* 4-11x. But there are
only 93,740 singular pairs against 48.9M regular ones. Multiply it out and the
singular blocks are about 2% of total assembly arithmetic — yet the stage was
taking 28% of the time. **The numbers did not reconcile, and that gap was the
actual signal.**

Lesson: when a stage's share of runtime is an order of magnitude off its share
of arithmetic, do not optimise the arithmetic. Find out where the time goes.

## Step one: split the stage before touching it

`_launch_metal_singular_block_gather_kernels!` fires four kernels behind one
timer. I wrote `scripts/benchmark_metal_singular.jl` to time each launch
separately, calling the internal kernels directly with `Metal.@metal`:

```
whole singular stage           0.7477 s
blocks: slp+adjoint            0.0036 s
blocks: dlp+hyp                0.0170 s
gather: slp+adjoint            0.1892 s
gather: dlp+hyp                0.5393 s

blocks total    0.0206 s  (3% of parts)
gather total    0.7284 s  (97% of parts)
```

The Duffy quadrature — the thing that *looks* expensive and that all the domain
complexity lives in — was 3% of the stage. The gather, which is bookkeeping, was
97%.

That benchmark took about twenty minutes to write and saved days. It is checked
in; re-run it before any future work on this stage.

## Why the gather was slow

The old design was **entry-owned**: one GPU thread per dense-matrix entry, each
thread searching for the singular pairs that contribute to it.

For the DLP/hypersingular gather that meant:

- launch `p1_dof_count²` threads = 3,502² = **12.26M threads**
- each thread loops over test-incident elements (~6) × trial-incident elements
  (~6) = ~36 combinations
- each combination runs `_metal_find_singular_pair`, a **linear scan** through
  that test element's pair list

So roughly 440M linear searches to place 93,740 pairs' worth of data. Worse,
almost every thread found nothing: only ~69k of the 12.26M entries receive a
correction at all. **99.4% of the launched threads did a pile of searching and
then wrote zero.**

This is a classic shape: the kernel was structured around the *output* (dense
matrix entries) when the work is defined by the *input* (a sparse pair list).

## The insight

Which dense-matrix entries receive a singular correction depends only on:

- mesh topology (which elements touch which)
- the singular pair list
- the P1/DP0 degree-of-freedom maps

None of that depends on frequency, wavenumber, or the quadrature values. It is
**frequency-independent**, and the caches in this codebase are already built
around that distinction — geometry, singular-pair, identity, and field caches
are all retained by the persistent worker and reused across a frequency sweep.

So the entire search can be hoisted out of the kernel and into cache
construction, where it runs once per mesh on the CPU instead of once per
frequency on 12.26M GPU threads.

## The data structure

A CSR-style contribution map, built once in
`_metal_singular_gather_maps`:

```
entry_indices    # Int32, one per corrected dense-matrix entry (linear index)
contrib_offsets  # Int32, length entry_count + 1, CSR row starts
contrib_values   # Int32, indices into the compact per-pair block value array
```

Construction is a flat push-then-`sortperm`-then-group over the pair list. For
each singular pair `(test_element, trial_element)`:

- SLP/adjoint: 3 contributions (3 test rows × 1 DP0 column)
- DLP/hypersingular: 9 contributions (3 test rows × 3 trial columns)

That is 93,740 × 12 ≈ 1.1M contributions, collapsing to ~86.7k SLP entries and
~69.2k DLP entries.

The kernel becomes trivial — one thread per *corrected* entry, no search:

```julia
index > entry_count && return nothing
position = contrib_offsets[index]
position_stop = contrib_offsets[index + 1] - 1
while position <= position_stop
    value_index = contrib_values[position]
    first_sum  += first_values[value_index]
    second_sum += second_values[value_index]
    position += 1
end
entry_index = entry_indices[index]
first_operator[entry_index]  += first_sum
second_operator[entry_index] += second_sum
```

The two old gather kernels collapsed into one generic
`_metal_singular_pair_gather_kernel!` taking a `(first, second)` operator pair,
because SLP+adjoint and DLP+hypersingular share a map and differ only in which
value arrays they read. `_metal_find_singular_pair` was deleted.

## Results

| | Before | After | Change |
| --- | ---: | ---: | ---: |
| Blocks (Duffy quadrature) | 0.0206 s | 0.0205 s | unchanged |
| Gather | 0.7284 s | 0.0009 s | 809x |
| Singular stage | 0.748 s | 0.022 s | **34x** |
| Total assembly | 2.647 s | 1.915 s | 1.38x |
| Assembly vs BEAT CPU (8 threads) | 1.17x | 1.81x | |

Threads launched for the DLP gather: 12,264,004 → 69,242.

The map costs ~53 ms to build on the host per mesh, and is reused across every
frequency in a sweep. On a single-frequency solve that is roughly a wash against
the old gather; on a sweep it disappears entirely.

Accuracy is unchanged. Operator relative errors moved in the 8th significant
figure (`hypersingular` 9.540e-7 → 9.580e-7), which is float summation-order
noise, not a behaviour change.

## Transferable lessons

1. **Reconcile time share against work share.** The mismatch is the diagnosis.
   If a stage takes 28% of the time for 2% of the arithmetic, the arithmetic is
   not the problem.

2. **Split a fused timer before optimising anything behind it.** Four kernels
   were hidden behind one number, and the number pointed at the wrong one.

3. **Entry-owned kernels are wrong for sparse work.** If most threads find
   nothing, the kernel is structured around the output when it should be
   structured around the input. Either invert the ownership or precompute the
   mapping.

4. **Search inside a kernel is a smell.** A linear scan per thread almost always
   means a lookup structure belongs in the cache instead. Ask what the search
   depends on: if the answer excludes the per-call inputs (here, frequency), it
   can be hoisted.

5. **Frequency-independence is the lever in this codebase.** Anything that
   depends only on mesh, spaces, and quadrature rules can move into cache
   construction. That boundary is already established here; use it.

6. **The expensive-looking code was not the expensive code.** Duffy quadrature
   has all the domain complexity — remapped rules, adjacency classification,
   Jacobian scaling. The bookkeeping around it was 30x more costly.

## What is still on the table

The regular kernel is now 97% of assembly (1.865 s). Per the existing notes it
runs well below device peak, is not transcendental-bound (fast-math
trigonometry moved it under 2%), and is memory or occupancy limited. It is the
next target, and it is a harder problem than this one was — there is no
99.4%-wasted-threads mistake to find, just genuine arithmetic that needs better
scheduling.

The same entry-owned/search-per-thread pattern exists in
`BeatEngineRocmSingular.jl`, which this file was ported from. The ROCm backend
would take the same 34x on its singular stage from the same change; it has not
been made there because there is no AMD hardware in this environment to
validate it on.

## Postscript: the same mistake in the coupled scope

Worth recording next to the kernel work, because it is the same failure mode one
level up.

The first pass at the coupled backend described Metal's host linear solve as a
fallback forced by Metal Performance Shaders having no sparse direct solver, and
then used that framing to exclude Metal from `CONDENSING_BACKEND_IDS`. Both
halves were wrong in the same way as "the singular kernel is slow because Duffy
quadrature is expensive": a plausible causal story accepted without checking what
the code actually does.

Checking it: `_build_rocm_hybrid_fem_condensation` is about fifty lines, and
exactly two touch AMDGPU — the `ROCArray` upload and the matching free. The
partition, the interior UMFPACK factorization, and
`_blocked_umfpack_schur_complement` are all host code. "Hybrid" in the name means
precisely that. So condensation never needed a device sparse solver on any
backend; only CUDA genuinely condenses on device, through cuDSS.

Measured on the curved-interface fixture (19,492 FEM vertices, 94,265 tetrahedra)
once the host variant existed:

| | Monolithic | Condensed | Ratio |
| --- | ---: | ---: | ---: |
| System order | 23,433 | 5,259 | 4.5x smaller |
| Build | 96.689 s | 6.305 s | 15.3x |
| Total | 99.489 s | 8.490 s | **11.7x** |

Lessons, and they rhyme with the ones above:

7. **"Platform X lacks feature Y" is a claim about the platform, not about the
   code.** Before letting it constrain a design, read what the existing backends
   actually do with Y. Here every backend but one already did the work on the
   host.

8. **Do not let a limitation framing propagate into a capability decision.** The
   host solve was the right call for its own reason — it is faster on unified
   memory. Describing it as a fallback made a second, unrelated exclusion look
   like it followed.

9. **Measure on a realistic model before setting a default.** The small coupled
   fixture would not have separated the two formulations; the production-scale
   interior separated them by 11.7x. Toy fixtures validate correctness, not
   policy.

## Incidental gotchas

Small things that cost time and are worth remembering:

- `Metal.@metal` resolves `Metal` at macro-expansion time in the calling module.
  A local binding (`Metal = BEC.METAL_MODULE`) does not satisfy it; the script
  needs `using Metal` at top level.
- Julia `do`-block syntax passes the closure as the **first** positional
  argument. `time_launch("label") do ... end` calls `time_launch(f, "label")`,
  so the signature must be `time_launch(launch!, label)`.
- Entry linear indices are `Int32`. That caps a single operator at ~46,340
  unknowns, but a 46,340² `ComplexF32` matrix is 17 GB per operator and there
  are four of them, so memory binds long before the index type does.
