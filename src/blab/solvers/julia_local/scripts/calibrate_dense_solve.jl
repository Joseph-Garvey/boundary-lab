# Refit the adaptive dense-solve cost model for this machine.
#
#   julia -t auto --project=<backend project> scripts/calibrate_dense_solve.jl [sizes...]
#
# The router in BeatEngineDenseSolve.jl chooses dense LU or GMRES by comparing
# two modelled costs, and the model carries four measured constants. They ship
# calibrated for an Apple M1 Max; any other machine should refit them, because
# what the model actually weighs is that machine's ratio of dense-GEMM
# throughput to memory bandwidth.
#
# This script measures all four on synthetic complex matrices of the same
# element type and shape the solver uses, and prints the export lines. It does
# not need a mesh: the primitives are BLAS, not BEM.
#
# Take at least three sizes. Two points cannot tell a wrong exponent from a
# wrong constant, and the linear term in the matvec model exists precisely
# because the effective bandwidth is still rising across the useful range.

using LinearAlgebra, Printf, Random

include(joinpath(get(ENV, "BLAB_CORE_DIR", joinpath(@__DIR__, "..", "src")), "BeatEngineCore.jl"))
using .BeatEngineCore

const SIZES = isempty(ARGS) ? [2048, 5107, 10230, 20422] : parse.(Int, ARGS)
const PASSES = parse(Int, get(ENV, "CALIBRATE_PASSES", "3"))
const ITERATION_SAMPLE = parse(Int, get(ENV, "CALIBRATE_MATVEC_ITERATIONS", "20"))

best(f) = minimum(begin f() end for _ in 1:PASSES)

function measure(n::Int)
    Random.seed!(20260902 + n)
    matrix = ComplexF32.(randn(Float32, n, n), randn(Float32, n, n))
    # Make it non-singular and diagonally sane, as the assembled operator is.
    @inbounds for index in 1:n
        matrix[index, index] += ComplexF32(2 * sqrt(n))
    end
    x = ComplexF32.(randn(Float32, n), randn(Float32, n))
    y = similar(x)

    matvec_seconds = best() do
        @elapsed for _ in 1:ITERATION_SAMPLE
            mul!(y, matrix, x)
        end
    end / ITERATION_SAMPLE

    factorization = nothing
    lu_seconds = best() do
        work = copy(matrix)
        elapsed = @elapsed(factorization = lu!(work))
        elapsed
    end

    triangular_seconds = best() do
        rhs = copy(x)
        @elapsed(ldiv!(factorization, rhs))
    end

    lu_gflops = (8 / 3) * float(n)^3 / lu_seconds / 1e9
    matvec_gbps = 8 * float(n)^2 / matvec_seconds / 1e9
    triangular_gbps = 8 * float(n)^2 / triangular_seconds / 1e9
    return (; n, matvec_seconds, lu_seconds, triangular_seconds,
            lu_gflops, matvec_gbps, triangular_gbps)
end

# Least squares for t = a N^2 + b N over the measured points.
function fit_matvec(samples)
    design = reduce(vcat, [float(s.n)^2 float(s.n)] for s in samples)
    observed = [s.matvec_seconds for s in samples]
    coefficients = design \ observed
    return coefficients[1], coefficients[2]
end

println("BLAS threads: $(BLAS.get_num_threads())   Julia threads: $(Threads.nthreads())   passes: $PASSES")
println()
@printf("%8s  %12s  %10s  %12s  %10s  %12s  %10s\n",
        "N", "matvec s", "GB/s", "LU s", "GFLOP/s", "trisolve s", "GB/s")
samples = map(SIZES) do n
    sample = measure(n)
    @printf("%8d  %12.6f  %10.1f  %12.4f  %10.1f  %12.6f  %10.1f\n",
            sample.n, sample.matvec_seconds, sample.matvec_gbps,
            sample.lu_seconds, sample.lu_gflops,
            sample.triangular_seconds, sample.triangular_gbps)
    sample
end

entry_seconds, dof_seconds = fit_matvec(samples)
lu_gflops = sum(s.lu_gflops for s in samples) / length(samples)
triangular_gbps = maximum(s.triangular_gbps for s in samples)

println()
println("Fitted matvec model: t = $(entry_seconds) N^2 + $(dof_seconds) N")
for sample in samples
    modelled = entry_seconds * float(sample.n)^2 + dof_seconds * float(sample.n)
    @printf("  N=%-8d measured %.6f s   model %.6f s   %.2fx\n",
            sample.n, sample.matvec_seconds, modelled, modelled / sample.matvec_seconds)
end

println()
println("Export these to calibrate this machine:")
println()
@printf("export %s=%.1f\n", BeatEngineCore.BEAT_DENSE_LU_GFLOPS_ENV, lu_gflops)
@printf("export %s=%.3e\n", BeatEngineCore.BEAT_DENSE_MATVEC_ENTRY_SECONDS_ENV, entry_seconds)
@printf("export %s=%.3e\n", BeatEngineCore.BEAT_DENSE_MATVEC_DOF_SECONDS_ENV, dof_seconds)
@printf("export %s=%.1f\n", BeatEngineCore.BEAT_DENSE_TRIANGULAR_GBPS_ENV, triangular_gbps)
println()
println("The iteration constant ($(BeatEngineCore.BEAT_GMRES_MODEL_ITERATIONS_ENV)) is a")
println("property of the operator rather than the machine and is not measured here.")
println("Read it off a real solve's reported iteration counts.")
println()
println("Crossover with the model as currently configured:")
for drives in 1:4
    @printf("  %d drive(s): %d dofs\n", drives, beat_dense_solve_crossover_dofs(drives))
end
