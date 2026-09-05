# Equivalence gate for the fused Burton-Miller assembly.
#
# The fused exterior path forms `0.5 I - D + (i/k) H` and `(-S - (i/k)(K' +
# 0.5 I)) q` inside the assembly kernels. This compares it against the
# four-operator path combined on the host by `burton_miller_neumann_matrices`
# on the same mesh, frequency and quadrature. Both run in Float32, so the two
# differ only by summation order; the tolerance below is float32 noise, not a
# physics tolerance, and this check is stronger than the exterior/symmetry
# validation scripts because it isolates the fusion from every other stage.
#
#   BLAB_VALIDATE_MESH_PATH   absolute mesh path (default: the bundled sample)
#   BLAB_VALIDATE_SCALE       mesh scale (default 0.001 for the sample)
#   BLAB_VALIDATE_SYMMETRY    comma-separated arms of off | x | xy | ground
#                             (default `off,x,ground`; every arm is run and the
#                             script fails if any of them fails)
#   BLAB_VALIDATE_DRIVES      number of independent drive columns (default 2)
#
# `xy` is deliberately not in the default arm list. On the bundled sample that
# combination fails `pressure_relative_error` on unmodified code, and two runs
# of the same tree disagree by ~2x: the operators agree to 1e-7 and the LU of
# the symmetry-reduced matrix amplifies the atomic-accumulation
# non-determinism of the singular scatter into 1e-5. It is a property of that
# fixture at that symmetry, not of the assembly path, and it is reproducible on
# a tree with no local changes. Run `BLAB_VALIDATE_SYMMETRY=xy` to reproduce it.
using LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

function relative_error(actual, reference)
    denominator = max(norm(reference), eps(real(eltype(reference))))
    return norm(actual - reference) / denominator
end

function validate_metal_fused_burton_miller(symmetry_mode::Symbol)
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
    metal.functional() || error("Metal.functional() is false.")

    mesh_path = get(ENV, "BLAB_VALIDATE_MESH_PATH", "")
    scale = Float32(parse(Float64, get(ENV, "BLAB_VALIDATE_SCALE", isempty(mesh_path) ? "0.001" : "1.0")))
    if isempty(mesh_path)
        mesh_path = joinpath(@__DIR__, "..", "test_meshes", get(ENV, "BLAB_VALIDATE_MESH", "sample.msh"))
    end
    drive_count = parse(Int, get(ENV, "BLAB_VALIDATE_DRIVES", "2"))
    regular_order = parse(Int, get(ENV, "BLAB_VALIDATE_REGULAR_ORDER", "4"))
    singular_order = parse(Int, get(ENV, "BLAB_VALIDATE_SINGULAR_ORDER", "4"))
    frequency_hz = Float32(parse(Float64, get(ENV, "BLAB_VALIDATE_FREQUENCY_HZ", "2000")))
    operator_tolerance = parse(Float64, get(ENV, "BLAB_VALIDATE_FUSED_TOLERANCE", "5e-6"))

    mesh = load_gmsh22_with_tags(mesh_path, scale)
    mesh = snap_symmetry_planes(mesh, symmetry_mode)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, regular_order)
    k = Float32(2pi) * frequency_hz / 343.0f0
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode)

    println("fixture=$(mesh_path) scale=$(scale) symmetry=$(symmetry_mode)")
    println("faces=$(length(mesh.faces)) p1_dofs=$(p1.global_dof_count) dp0_dofs=$(dp0.global_dof_count) q=$(regular_order) s=$(singular_order) f=$(frequency_hz) drives=$(drive_count)")
    println("device=$(metal.device().name)")
    flush(stdout)

    Random.seed!(20260902)
    q_neumann = ComplexF32.(randn(Float32, dp0.global_dof_count, drive_count),
                            randn(Float32, dp0.global_dof_count, drive_count))

    device_cache = build_metal_regular_assembly_cache(
        mesh, p1, dp0, rule; singular_order=singular_order, symmetry_mode=symmetry_mode,
    )
    device_singular_cache = build_metal_singular_correction_cache(singular_cache)

    reference_operators = nothing
    reference_seconds = @elapsed begin
        reference_operators = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=singular_order, backend=:metal,
            device_cache=device_cache, singular_cache=singular_cache,
            device_singular_cache=device_singular_cache, symmetry_mode=symmetry_mode,
        )
    end
    # metal_host_operators takes ownership of shared-storage buffers, so the
    # host tuple is what gets released, and only once the matrices built from
    # it are materialised.
    host_operators = metal_host_operators(reference_operators)
    reference_lhs, reference_rhs_operator = BeatEngineCore.burton_miller_neumann_matrices(
        host_operators, identity_p1_p1, identity_p1_dp0, k,
    )
    reference_rhs = reference_rhs_operator * q_neumann
    release_operator_storage!(host_operators)

    fused_timing = Dict{String,Float64}()
    fused = nothing
    fused_seconds = @elapsed begin
        fused = assemble_burton_miller_neumann_system_metal(
            mesh, p1, dp0, q_neumann, k, rule;
            device_cache=device_cache, singular_cache=singular_cache,
            device_singular_cache=device_singular_cache,
            identity_p1_p1=identity_p1_p1, identity_p1_dp0=identity_p1_dp0,
            singular_order=singular_order, symmetry_mode=symmetry_mode, timing=fused_timing,
        )
    end
    metal.synchronize()
    fused_lhs = Array(fused.matrix)
    fused_rhs = Array(fused.rhs)

    lhs_error = relative_error(fused_lhs, reference_lhs)
    rhs_error = relative_error(fused_rhs, reference_rhs)
    reference_pressure = lu(copy(reference_lhs)) \ reference_rhs
    fused_pressure = lu(copy(fused_lhs)) \ fused_rhs
    pressure_error = relative_error(fused_pressure, reference_pressure)

    reference_bytes = 6 * p1.global_dof_count * p1.global_dof_count * sizeof(ComplexF32)
    fused_bytes = (p1.global_dof_count * p1.global_dof_count + p1.global_dof_count * drive_count) *
        sizeof(ComplexF32)

    println("four_operator_assembly_s=$(round(reference_seconds, digits=4))")
    println("fused_assembly_s=$(round(fused_seconds, digits=4))")
    stage_text = join(["$(key)=$(round(value, digits=4))" for (key, value) in sort(collect(fused_timing))], "  ")
    println("fused_stages=$(stage_text)")
    println("operator_bytes four=$(reference_bytes) fused=$(fused_bytes) ratio=$(round(reference_bytes / fused_bytes, digits=2))x")
    println("lhs_relative_error=$(lhs_error)")
    println("rhs_relative_error=$(rhs_error)")
    println("pressure_relative_error=$(pressure_error)")
    flush(stdout)

    release_metal_burton_miller_system!(fused)
    release_metal_singular_correction_cache!(device_singular_cache)
    release_metal_regular_assembly_cache!(device_cache)

    failures = String[]
    lhs_error <= operator_tolerance || push!(failures, "lhs_relative_error $(lhs_error) > $(operator_tolerance)")
    rhs_error <= operator_tolerance || push!(failures, "rhs_relative_error $(rhs_error) > $(operator_tolerance)")
    pressure_error <= operator_tolerance || push!(failures, "pressure_relative_error $(pressure_error) > $(operator_tolerance)")
    if isempty(failures)
        println("ARM_RESULT symmetry=$(symmetry_mode) pass")
        return 0
    end
    for failure in failures
        println("FAILURE: symmetry=$(symmetry_mode) $(failure)")
    end
    println("ARM_RESULT symmetry=$(symmetry_mode) fail")
    return 1
end

function validate_metal_fused_burton_miller()
    arms = [Symbol(strip(arm)) for arm in split(get(ENV, "BLAB_VALIDATE_SYMMETRY", "off,x,ground"), ",")
            if !isempty(strip(arm))]
    isempty(arms) && error("BLAB_VALIDATE_SYMMETRY named no symmetry arm.")
    failed = Symbol[]
    for arm in arms
        println("=== symmetry arm: $(arm)")
        validate_metal_fused_burton_miller(arm) == 0 || push!(failed, arm)
        flush(stdout)
    end
    println("arms=$(join(arms, ","))")
    if isempty(failed)
        println("RESULT=pass")
        return 0
    end
    println("failed_arms=$(join(failed, ","))")
    println("RESULT=fail")
    return 1
end

exit(validate_metal_fused_burton_miller())
