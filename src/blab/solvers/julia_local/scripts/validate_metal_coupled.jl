#!/usr/bin/env julia
# CPU-versus-Metal validation of the coupled FEM-BEM-LEM paths: the monolithic
# system (BEM operators from Metal, CPU coupled algebra) and the CPU condensed
# solver with Metal-assembled operators, against the pure CPU build.

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupledCondensed

using LinearAlgebra, Printf, SparseArrays, StaticArrays

const FIXTURE_ROOT = normpath(
    joinpath(@__DIR__, "..", "..", "..", "..", "..", "tests", "fixtures"),
)

function relative_error(reference, candidate)
    scale = norm(reference)
    return norm(candidate - reference) / max(scale, eps(Float64))
end

function timed_build(f)
    value = nothing
    elapsed = @elapsed value = f()
    return value, elapsed
end

function main()
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
    metal.functional() || error("Metal.functional() is false.")

    quadrature_order = parse(Int, get(ENV, "BLAB_COUPLED_QUADRATURE_ORDER", "1"))
    singular_order = parse(Int, get(ENV, "BLAB_COUPLED_SINGULAR_ORDER", "1"))
    fem_mesh = load_gmsh41_volume(joinpath(FIXTURE_ROOT, "femvolume.msh"), 0.001f0)
    bem_mesh = load_gmsh22_with_tags(
        joinpath(FIXTURE_ROOT, "exterior_conforming.msh"),
        0.001f0,
    )
    interface_map = build_conforming_interface_map(
        fem_mesh,
        bem_mesh,
        physical_tag(fem_mesh, 2, "Interface"),
        2,
    )
    radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
    transducer = ElectrodynamicTransducer{Float32}(
        "component:metal-validation",
        [radiator_tag],
        Float32[1],
        [1],
        Float32[-1],
        SVector(0f0, 0f0, 1f0),
        2f0,
        1,
        6f0,
        0.0005f0,
        7f0,
        0.015f0,
        0.0005f0,
        1f0,
    )
    common_options = (
        quadrature_order=quadrature_order,
        singular_order=singular_order,
        bulk_loss_factor=0.01f0,
        transducers=[transducer],
    )
    velocity_excitation = (
        kind=:normal_velocity,
        fem_boundary_tags=[radiator_tag],
        fem_boundary_weights=Float32[1],
        bem_source_index=0,
        transducer_index=0,
        amplitude=ComplexF32(1, 0),
    )
    voltage_excitation = (
        kind=:voltage,
        fem_boundary_tags=Int[],
        fem_boundary_weights=Float32[],
        bem_source_index=0,
        transducer_index=1,
        amplitude=ComplexF32(2.83, 0),
    )

    cpu_system = metal_system = cpu_condensed = metal_condensed = nothing
    try
        cpu_system, cpu_build_s = timed_build() do
            build_coupled_system(
                fem_mesh, bem_mesh, interface_map, 500f0, 343f0, 1.21f0;
                common_options..., bem_backend=:cpu,
                validation_diagnostics=false, static_condensation=false,
            )
        end
        # Warm-up pass compiles the Metal kernels; the timed build follows.
        warm = build_coupled_system(
            fem_mesh, bem_mesh, interface_map, 500f0, 343f0, 1.21f0;
            common_options..., bem_backend=:metal,
            validation_diagnostics=false, static_condensation=false,
        )
        release_coupled_system!(warm)
        metal_system, metal_build_s = timed_build() do
            build_coupled_system(
                fem_mesh, bem_mesh, interface_map, 500f0, 343f0, 1.21f0;
                common_options..., bem_backend=:metal,
                validation_diagnostics=false, static_condensation=false,
            )
        end
        cpu_condensed, cpu_condensed_build_s = timed_build() do
            build_condensed_coupled_system(
                fem_mesh, bem_mesh, interface_map, 500f0, 343f0, 1.21f0;
                common_options..., validation_diagnostics=false,
            )
        end
        metal_cache = prepare_condensed_coupled_cache(
            fem_mesh, bem_mesh, interface_map;
            quadrature_order=quadrature_order, singular_order=singular_order,
            retained_fem_vertices=cpu_condensed.retained_fem_vertices,
            bulk_loss_factor_by_vertex=fill(0.01f0, length(fem_mesh.vertices)),
            bem_backend=:metal,
        )
        metal_condensed, metal_condensed_build_s = timed_build() do
            build_condensed_coupled_system(
                fem_mesh, bem_mesh, interface_map, 500f0, 343f0, 1.21f0;
                common_options..., cache=metal_cache, validation_diagnostics=false,
            )
        end

        cpu_v = only(solve_coupled_excitations(cpu_system, [velocity_excitation]))
        metal_v = only(solve_coupled_excitations(metal_system, [velocity_excitation]))
        cpu_u = only(solve_coupled_excitations(cpu_system, [voltage_excitation]))
        metal_u = only(solve_coupled_excitations(metal_system, [voltage_excitation]))
        cpu_cv = only(solve_condensed_coupled_excitations(cpu_condensed, [velocity_excitation]))
        metal_cv = only(solve_condensed_coupled_excitations(metal_condensed, [velocity_excitation]))
        cpu_cu = only(solve_condensed_coupled_excitations(cpu_condensed, [voltage_excitation]))
        metal_cu = only(solve_condensed_coupled_excitations(metal_condensed, [voltage_excitation]))

        checks = Dict(
            "monolithic_fem" => relative_error(cpu_v.fem_pressure, metal_v.fem_pressure),
            "monolithic_bem" => relative_error(cpu_v.bem_pressure, metal_v.bem_pressure),
            "monolithic_flux" => relative_error(cpu_v.interface_flux, metal_v.interface_flux),
            "monolithic_voltage_fem" => relative_error(cpu_u.fem_pressure, metal_u.fem_pressure),
            "monolithic_voltage_bem" => relative_error(cpu_u.bem_pressure, metal_u.bem_pressure),
            "monolithic_voltage_flux" => relative_error(cpu_u.interface_flux, metal_u.interface_flux),
            "monolithic_voltage_velocity" => relative_error(cpu_u.diaphragm_velocity, metal_u.diaphragm_velocity),
            "monolithic_voltage_current" => relative_error(cpu_u.voice_coil_current, metal_u.voice_coil_current),
            "condensed_fem" => relative_error(cpu_cv.fem_pressure, metal_cv.fem_pressure),
            "condensed_bem" => relative_error(cpu_cv.bem_pressure, metal_cv.bem_pressure),
            "condensed_flux" => relative_error(cpu_cv.interface_flux, metal_cv.interface_flux),
            "condensed_voltage_fem" => relative_error(cpu_cu.fem_pressure, metal_cu.fem_pressure),
            "condensed_voltage_bem" => relative_error(cpu_cu.bem_pressure, metal_cu.bem_pressure),
            "condensed_voltage_flux" => relative_error(cpu_cu.interface_flux, metal_cu.interface_flux),
            "condensed_voltage_velocity" => relative_error(cpu_cu.diaphragm_velocity, metal_cu.diaphragm_velocity),
            "condensed_voltage_current" => relative_error(cpu_cu.voice_coil_current, metal_cu.voice_coil_current),
            "condensed_vs_monolithic_bem" => relative_error(metal_v.bem_pressure, metal_cv.bem_pressure),
        )

        points = SVector{3,Float32}[
            SVector(0.20f0, 0.13f0, 0.30f0),
            SVector(-0.27f0, 0.08f0, 0.24f0),
        ]
        cpu_field = evaluate_galerkin_field_cpu(
            points, bem_mesh, cpu_v.bem_pressure, cpu_v.bem_neumann,
            cpu_system.wavenumber, cpu_system.field_cache,
        )
        metal_field = evaluate_galerkin_field_metal(
            points, bem_mesh, metal_v.bem_pressure, metal_v.bem_neumann,
            metal_system.wavenumber, metal_system.field_cache,
        )
        condensed_field = evaluate_galerkin_field_metal(
            points, bem_mesh, metal_cv.bem_pressure, metal_cv.bem_neumann,
            metal_condensed.wavenumber, metal_condensed.field_cache,
        )
        checks["monolithic_field"] = relative_error(cpu_field, metal_field)
        checks["condensed_field"] = relative_error(cpu_field, condensed_field)

        println("BEAT coupled Metal validation at 500 Hz (q$quadrature_order/s$singular_order)")
        println("device=$(metal.device().name)")
        @printf "  %-16s build=%8.3f s linear=%s order=%d\n" "CPU" cpu_build_s cpu_system.linear_backend cpu_system.solved_system_order
        @printf "  %-16s build=%8.3f s linear=%s bem=%s\n" "Metal" metal_build_s metal_system.linear_backend metal_system.bem_backend
        @printf "  %-16s build=%8.3f s order=%d/%d\n" "CPU condensed" cpu_condensed_build_s cpu_condensed.solved_system_order cpu_condensed.full_system_order
        @printf "  %-16s build=%8.3f s order=%d/%d bem=%s\n" "Metal condensed" metal_condensed_build_s metal_condensed.solved_system_order metal_condensed.full_system_order metal_condensed.bem_backend
        for name in sort!(collect(keys(checks)))
            @printf "  %-28s %.6e\n" name checks[name]
        end
        @printf "  pressure continuity     %.6e\n" metal_v.pressure_continuity_error
        @printf "  flux conservation       %.6e\n" metal_v.flux_conservation_error

        # Same gates as the ROCm script: physical outputs at 5e-4 / 1e-3, and the
        # ill-conditioned voltage interface flux at 2e-3 (cond(A) ~ 4e9 in FP32).
        maximum(checks[name] for name in (
            "monolithic_fem", "monolithic_bem", "monolithic_flux", "monolithic_field",
            "monolithic_voltage_fem", "monolithic_voltage_bem",
            "monolithic_voltage_velocity", "monolithic_voltage_current",
        )) < 5e-4 || error("Metal monolithic path exceeds the 5e-4 parity gate.")
        checks["monolithic_voltage_flux"] < 2e-3 ||
            error("Metal monolithic voltage interface-flux parity exceeds 2e-3.")
        maximum(checks[name] for name in (
            "condensed_fem", "condensed_bem", "condensed_flux", "condensed_field",
            "condensed_voltage_fem", "condensed_voltage_bem",
            "condensed_voltage_velocity", "condensed_voltage_current",
            "condensed_vs_monolithic_bem",
        )) < 1e-3 || error("Metal condensed path exceeds the 1e-3 parity gate.")
        checks["condensed_voltage_flux"] < 2e-3 ||
            error("Metal condensed voltage interface-flux parity exceeds 2e-3.")
        metal_condensed.formulation == :fem_interface_condensed ||
            error("Metal condensed build did not select the condensed formulation.")
        metal_condensed.solved_system_order < metal_condensed.full_system_order ||
            error("Metal condensed build did not reduce the system order.")
        metal_v.pressure_continuity_error < 1e-4 || error("Pressure continuity exceeds 1e-4.")
        metal_v.flux_conservation_error < 1e-4 || error("Flux conservation exceeds 1e-4.")
        println("METAL_COUPLED_VALIDATION_OK")
    finally
        cpu_system === nothing || release_coupled_system!(cpu_system)
        metal_system === nothing || release_coupled_system!(metal_system)
        cpu_condensed === nothing || release_condensed_coupled_system!(cpu_condensed)
        metal_condensed === nothing || release_condensed_coupled_system!(metal_condensed)
    end
end

main()
