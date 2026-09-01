using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

function relative_error(actual, reference)
    denominator = max(norm(reference), eps(real(eltype(reference))))
    return norm(actual - reference) / denominator
end

function validate_metal_exterior()
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
    metal.functional() || error("Metal.functional() is false.")

    mesh_name = get(ENV, "BLAB_VALIDATE_MESH", "sample.msh")
    mesh_path = joinpath(@__DIR__, "..", "test_meshes", mesh_name)
    mesh = load_gmsh22_with_tags(mesh_path, Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    regular_order = parse(Int, get(ENV, "BLAB_VALIDATE_REGULAR_ORDER", "1"))
    singular_order = parse(Int, get(ENV, "BLAB_VALIDATE_SINGULAR_ORDER", "2"))
    rule = triangle_rule(Float32, regular_order)
    frequency_hz = Float32(parse(Float64, get(ENV, "BLAB_VALIDATE_FREQUENCY_HZ", "500")))
    sound_speed = 343.0f0
    k = Float32(2pi) * frequency_hz / sound_speed
    singular_cache = build_singular_correction_cache(mesh, singular_order)

    println("fixture=$(mesh_path)")
    println("faces=$(length(mesh.faces)) p1_dofs=$(p1.global_dof_count) dp0_dofs=$(dp0.global_dof_count) q=$(regular_order) s=$(singular_order) f=$(frequency_hz)")
    println("device=$(metal.device().name)")
    flush(stdout)

    cpu_cache = build_beat_cpu_assembly_cache(
        mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=:off,
    )
    cpu_operators = nothing
    cpu_assembly_s = @elapsed begin
        cpu_operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=singular_order,
            backend=:cpu,
            singular_cache=singular_cache,
            cpu_cache=cpu_cache,
            symmetry_mode=:off,
        )
    end
    println("cpu_assembly_s=$(cpu_assembly_s)")
    flush(stdout)

    metal_cache = build_metal_regular_assembly_cache(
        mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=:off,
    )

    results = Dict{String,Any}()
    for singular_mode in ("host", "native")
        ENV["BLAB_METAL_SINGULAR_MODE"] = singular_mode
        metal_operators = nothing
        # First call includes kernel compilation; time the second.
        for pass in 1:2
            metal_operators === nothing || release_operator_storage!(metal_operators)
            metal_assembly_s = @elapsed begin
                metal_operators = assemble_regular_galerkin_operators(
                    mesh,
                    p1,
                    dp0,
                    k,
                    rule;
                    skip_singular=false,
                    singular_order=singular_order,
                    backend=:metal,
                    device_cache=metal_cache,
                    singular_cache=singular_cache,
                    metal_assembly_mode=:native,
                    symmetry_mode=:off,
                )
            end
            println("metal_assembly_s[$(singular_mode), pass $(pass)]=$(metal_assembly_s) mode=$(metal_operators.regular_assembly_mode)")
        end
        flush(stdout)
        operator_errors = Dict(
            "single_layer" => relative_error(Array(metal_operators.single_layer), cpu_operators.single_layer),
            "double_layer" => relative_error(Array(metal_operators.double_layer), cpu_operators.double_layer),
            "adjoint_double_layer" => relative_error(Array(metal_operators.adjoint_double_layer), cpu_operators.adjoint_double_layer),
            "hypersingular" => relative_error(Array(metal_operators.hypersingular), cpu_operators.hypersingular),
        )
        results[singular_mode] = (operators=metal_operators, errors=operator_errors)
        println("operator_relative_errors[$(singular_mode)]=$(operator_errors)")
        flush(stdout)
    end
    delete!(ENV, "BLAB_METAL_SINGULAR_MODE")
    metal_operators = results["native"].operators

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
    q_neumann = zeros(ComplexF32, length(mesh.faces))
    driven_tag = first(mesh.physical_tags)
    for face_index in eachindex(mesh.faces)
        if mesh.physical_tags[face_index] == driven_tag
            q_neumann[face_index] = ComplexF32(0, 1)
        end
    end

    cpu_pressure = nothing
    cpu_solve_s = @elapsed begin
        cpu_pressure = solve_burton_miller_neumann(
            cpu_operators,
            identity_p1_p1,
            identity_p1_dp0,
            q_neumann,
            k,
        )
    end
    identity_cache = build_metal_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)
    metal_pressure = nothing
    metal_solve_s = @elapsed begin
        metal_pressure = solve_burton_miller_neumann(metal_operators, identity_cache, q_neumann, k)
    end
    release_metal_burton_miller_identity_cache!(identity_cache)

    lhs, rhs_operator = burton_miller_neumann_matrices(
        cpu_operators,
        identity_p1_p1,
        identity_p1_dp0,
        k,
    )
    rhs = rhs_operator * q_neumann
    pressure_error = relative_error(metal_pressure, cpu_pressure)
    cpu_residual = relative_error(lhs * cpu_pressure, rhs)
    metal_residual = relative_error(lhs * metal_pressure, rhs)

    eval_points = fibonacci_sphere(64, Float32(2.0))
    cpu_field_cache = build_field_evaluation_cache(mesh, rule)
    cpu_field = evaluate_galerkin_field_cpu(eval_points, mesh, cpu_pressure, q_neumann, k, cpu_field_cache)
    metal_field_cache = build_metal_field_evaluation_cache(cpu_field_cache)
    metal_field = nothing
    metal_field_s = 0.0
    for pass in 1:2
        metal_field_s = @elapsed begin
            metal_field = evaluate_galerkin_field_metal(eval_points, mesh, metal_pressure, q_neumann, k, metal_field_cache)
        end
    end
    field_error = relative_error(metal_field, cpu_field)

    println("cpu_solve_s=$(cpu_solve_s) metal_solve_s=$(metal_solve_s) metal_field_s=$(metal_field_s)")
    println("pressure_relative_error=$(pressure_error)")
    println("cpu_residual=$(cpu_residual) metal_residual=$(metal_residual)")
    println("field_relative_error=$(field_error)")
    flush(stdout)

    operator_tolerance = regular_order == 1 && singular_order == 2 ? 1.0f-6 : 1.0f-5
    for singular_mode in ("host", "native")
        maximum(values(results[singular_mode].errors)) <= operator_tolerance ||
            error("Metal native operators ($(singular_mode) singular) differ from BEAT CPU.")
    end
    pressure_error <= 5.0f-3 || error("Metal pressure differs from BEAT CPU beyond tolerance.")
    metal_residual <= 5.0f-3 || error("Metal pressure residual exceeds tolerance.")
    field_error <= 5.0f-3 || error("Metal field differs from BEAT CPU beyond tolerance.")

    for singular_mode in ("host", "native")
        release_operator_storage!(results[singular_mode].operators)
    end
    release_metal_field_evaluation_cache!(metal_field_cache)
    release_metal_regular_assembly_cache!(metal_cache)
    println("METAL_EXTERIOR_VALIDATION_OK")
end

validate_metal_exterior()
