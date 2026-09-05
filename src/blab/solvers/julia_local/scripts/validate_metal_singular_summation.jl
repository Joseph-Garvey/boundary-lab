# Summation-order gate for the fused singular Burton-Miller pair.
#
# `_metal_singular_pair_fused_bm_blocks` forms the Burton-Miller combination
# inside the Duffy quadrature loop; `_metal_singular_pair_blocks`, which the
# four-operator path still uses, accumulates the four operator blocks and
# combines afterwards. The two are algebraically identical, so in Float32 they
# differ only by summation order — and a singular pair is where cancellation is
# worst, so "only summation order" is a claim that has to be measured rather
# than asserted.
#
# This evaluates the same pairs three ways on the host, from the exact arrays
# the device kernel reads (face vertices, normals, curls, Jacobian scales,
# normal products and the Duffy rules), replicating the kernel's `part_count`
# split so the summation trees match:
#
#   blocks_f32  accumulate slp/adj/dlp/hb + g_total, combine after
#   quad_f32    combine per quadrature point, curl term hoisted after the loop
#   ref_f64     the same two, in Float64
#
# and gates three things:
#
#   1. the two orders are the same algebra   — they must agree in Float64
#   2. neither Float32 order loses accuracy  — both within the stated bound of
#                                              the Float64 reference
#   3. the fused order is not systematically worse than the reference order —
#      it must not be worse on a majority of pairs, nor by more than a small
#      factor at the maximum
#
# It runs on the host and is bit-deterministic: no atomics, no device
# arithmetic, a fixed pair selection and a fixed frequency set. On the bundled
# `sample.msh` at symmetry `off` the worst per-pair Float32 error of either
# order is 8.0e-5 (20 kHz), the two Float64 orders agree to 2e-15, the fused
# order is worse than the reference order on 47.7-48.4% of pairs, and the
# maximum-error ratio is 1.001-1.003. A Metal device is needed only to build
# the caches the pair data is read back from.
#
#   BLAB_VALIDATE_MESH        bundled fixture name (default sample.msh)
#   BLAB_VALIDATE_MESH_PATH   absolute mesh path, overrides the fixture
#   BLAB_VALIDATE_SCALE       mesh scale (default 0.001 for the bundled sample)
#   BLAB_VALIDATE_SYMMETRY    off | x | xy | ground (default off)
#   BLAB_VALIDATE_SINGULAR_ORDER          Duffy order (default 4)
#   BLAB_VALIDATE_SUMMATION_MAX_PAIRS     stratified sample cap (default 4000)
#   BLAB_VALIDATE_SUMMATION_TOLERANCE     Float32-vs-Float64 bound (default 2e-4)
#   BLAB_VALIDATE_SUMMATION_MAX_RATIO     max(quad error)/max(blocks error) bound (default 1.25)
#   BLAB_VALIDATE_SUMMATION_WORSE_SHARE   share of pairs where fused may be worse (default 0.6)
#   BLAB_METAL_SINGULAR_PARTS             rule split, as the kernel reads it (default 4)
using LinearAlgebra, Printf, Random, StaticArrays, Statistics

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const SUMMATION_SEED = 20260905
const SUMMATION_FREQUENCIES_HZ = (100.0, 2000.0, 20000.0)

function validate_metal_singular_summation()
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
    metal.functional() || error("Metal.functional() is false.")

    mesh_path = get(ENV, "BLAB_VALIDATE_MESH_PATH", "")
    scale = Float32(parse(Float64, get(ENV, "BLAB_VALIDATE_SCALE", isempty(mesh_path) ? "0.001" : "1.0")))
    if isempty(mesh_path)
        mesh_path = joinpath(@__DIR__, "..", "test_meshes", get(ENV, "BLAB_VALIDATE_MESH", "sample.msh"))
    end
    symmetry_mode = Symbol(get(ENV, "BLAB_VALIDATE_SYMMETRY", "off"))
    order = parse(Int, get(ENV, "BLAB_VALIDATE_SINGULAR_ORDER", "4"))
    parts = parse(Int, get(ENV, "BLAB_METAL_SINGULAR_PARTS", "4"))
    max_pairs = parse(Int, get(ENV, "BLAB_VALIDATE_SUMMATION_MAX_PAIRS", "4000"))
    tolerance = parse(Float64, get(ENV, "BLAB_VALIDATE_SUMMATION_TOLERANCE", "2e-4"))
    max_ratio = parse(Float64, get(ENV, "BLAB_VALIDATE_SUMMATION_MAX_RATIO", "1.25"))
    worse_share = parse(Float64, get(ENV, "BLAB_VALIDATE_SUMMATION_WORSE_SHARE", "0.6"))
    f64_tolerance = 1e-12

    mesh = snap_symmetry_planes(load_gmsh22_with_tags(mesh_path, scale), symmetry_mode)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, order)
    device_cache = build_metal_regular_assembly_cache(
        mesh, p1, dp0, rule; singular_order=order, symmetry_mode=symmetry_mode,
    )
    singular_cache = build_singular_correction_cache(mesh, order)
    device_singular_cache = build_metal_singular_correction_cache(singular_cache)

    # Host copies of the exact arrays the kernel indexes.
    face_vertices = Array(device_cache.face_vertices)
    normals = Array(device_cache.normals)
    curls = Array(device_cache.curls)
    test_indices = Array(device_singular_cache.test_indices)
    trial_indices = Array(device_singular_cache.trial_indices)
    rule_indices = Array(device_singular_cache.rule_indices)
    jac_scales = Array(device_singular_cache.jac_scales)
    normal_products = Array(device_singular_cache.normal_products)
    rule_offsets = Array(device_singular_cache.rule_offsets)
    rule_test_points = Array(device_singular_cache.rule_test_points)
    rule_trial_points = Array(device_singular_cache.rule_trial_points)
    rule_weights = Array(device_singular_cache.rule_weights)
    face_count = Int(device_cache.face_count)
    pair_count = Int(device_singular_cache.pair_count)
    rule_point_count = length(rule_weights)

    println("fixture=$(mesh_path) scale=$(scale) symmetry=$(symmetry_mode)")
    @printf("faces=%d p1_dofs=%d singular_pairs=%d rule_points=%d s=%d parts=%d\n",
            length(mesh.faces), p1.global_dof_count, pair_count, rule_point_count, order, parts)
    println("device=$(metal.device().name)")
    flush(stdout)

    face_point(vertices, index, b1, b2, b3, ::Type{T}) where {T} = (
        T(vertices[index]) * b1 + T(vertices[index + face_count]) * b2 +
            T(vertices[index + 2face_count]) * b3,
        T(vertices[index + 3face_count]) * b1 + T(vertices[index + 4face_count]) * b2 +
            T(vertices[index + 5face_count]) * b3,
        T(vertices[index + 6face_count]) * b1 + T(vertices[index + 7face_count]) * b2 +
            T(vertices[index + 8face_count]) * b3,
    )

    function curl_products(::Type{T}, test_index, trial_index) where {T}
        t = ntuple(i -> T(curls[test_index + (i - 1) * face_count]), 9)
        r = ntuple(i -> T(curls[trial_index + (i - 1) * face_count]), 9)
        return SVector{9,T}(
            t[1]*r[1]+t[2]*r[2]+t[3]*r[3], t[4]*r[1]+t[5]*r[2]+t[6]*r[3], t[7]*r[1]+t[8]*r[2]+t[9]*r[3],
            t[1]*r[4]+t[2]*r[5]+t[3]*r[6], t[4]*r[4]+t[5]*r[5]+t[6]*r[6], t[7]*r[4]+t[8]*r[5]+t[9]*r[6],
            t[1]*r[7]+t[2]*r[8]+t[3]*r[9], t[4]*r[7]+t[5]*r[8]+t[6]*r[9], t[7]*r[7]+t[8]*r[8]+t[9]*r[9],
        )
    end

    # One (pair, part) in precision T, accumulated in `mode` order.
    function pair_part(::Type{T}, pair::Int, part::Int, k::T, mode::Symbol) where {T}
        test_index = Int(test_indices[pair])
        trial_index = Int(trial_indices[pair])
        rule_index = Int(rule_indices[pair])
        q_first = Int(rule_offsets[rule_index])
        q_last = Int(rule_offsets[rule_index + 1]) - 1
        per_part = cld(q_last - q_first + 1, parts)
        q = q_first + (part - 1) * per_part
        q_stop = min(q + per_part - 1, q_last)
        jac_scale = T(jac_scales[pair])
        normal_product = T(normal_products[pair])
        test_normal = SVector{3,T}(T(normals[test_index]), T(normals[test_index + face_count]),
                                   T(normals[test_index + 2face_count]))
        trial_normal = SVector{3,T}(T(normals[trial_index]), T(normals[trial_index + face_count]),
                                    T(normals[trial_index + 2face_count]))
        inv_four_pi = T(0.07957747154594767)
        inverse_k = one(T) / k
        curl_scale = inverse_k * k * k * normal_product
        slp_re = zero(SVector{3,T}); slp_im = zero(SVector{3,T})
        adj_re = zero(SVector{3,T}); adj_im = zero(SVector{3,T})
        dlp_re = zero(SVector{9,T}); dlp_im = zero(SVector{9,T})
        hb_re = zero(SVector{9,T}); hb_im = zero(SVector{9,T})
        lhs_re = zero(SVector{9,T}); lhs_im = zero(SVector{9,T})
        rhs_re = zero(SVector{3,T}); rhs_im = zero(SVector{3,T})
        g_total_re = zero(T); g_total_im = zero(T)
        while q <= q_stop
            test_xi = T(rule_test_points[q]); test_eta = T(rule_test_points[q + rule_point_count])
            trial_xi = T(rule_trial_points[q]); trial_eta = T(rule_trial_points[q + rule_point_count])
            weight = T(rule_weights[q]) * jac_scale
            tb1 = one(T) - test_xi - test_eta
            rb1 = one(T) - trial_xi - trial_eta
            x, y, z = face_point(face_vertices, test_index, tb1, test_xi, test_eta, T)
            sx, sy, sz = face_point(face_vertices, trial_index, rb1, trial_xi, trial_eta, T)
            dx = sx - x; dy = sy - y; dz = sz - z
            radius2 = dx * dx + dy * dy + dz * dz
            if radius2 > zero(T)
                inv_radius = one(T) / sqrt(radius2)
                radius = radius2 * inv_radius
                phase = k * radius
                green_scale = inv_radius * inv_four_pi * weight
                green_re = cos(phase) * green_scale
                green_im = sin(phase) * green_scale
                grad_re = -green_re * inv_radius - green_im * k
                grad_im = green_re * k - green_im * inv_radius
                test_dot = -(dx * test_normal[1] + dy * test_normal[2] + dz * test_normal[3]) * inv_radius
                trial_dot = (dx * trial_normal[1] + dy * trial_normal[2] + dz * trial_normal[3]) * inv_radius
                tb = SVector{3,T}(tb1, test_xi, test_eta)
                outer = SVector{9,T}(
                    tb1 * rb1, test_xi * rb1, test_eta * rb1,
                    tb1 * trial_xi, test_xi * trial_xi, test_eta * trial_xi,
                    tb1 * trial_eta, test_xi * trial_eta, test_eta * trial_eta,
                )
                if mode === :blocks
                    slp_re += tb * green_re; slp_im += tb * green_im
                    adj_re += tb * (grad_re * test_dot); adj_im += tb * (grad_im * test_dot)
                    dlp_re += outer * (grad_re * trial_dot); dlp_im += outer * (grad_im * trial_dot)
                    hb_re += outer * green_re; hb_im += outer * green_im
                else
                    rhs_re += tb * (-green_re + inverse_k * (grad_im * test_dot))
                    rhs_im += tb * (-green_im - inverse_k * (grad_re * test_dot))
                    u_re = -(grad_re * trial_dot) + curl_scale * green_im
                    u_im = -(grad_im * trial_dot) - curl_scale * green_re
                    lhs_re += outer * u_re; lhs_im += outer * u_im
                end
                g_total_re += green_re; g_total_im += green_im
            end
            q += 1
        end
        products = curl_products(T, test_index, trial_index)
        if mode === :blocks
            k2n = k * k * normal_product
            hyp_re = products * g_total_re - hb_re * k2n
            hyp_im = products * g_total_im - hb_im * k2n
            lhs_re = -dlp_re - inverse_k * hyp_im
            lhs_im = -dlp_im + inverse_k * hyp_re
            rhs_re = -slp_re + inverse_k * adj_im
            rhs_im = -slp_im - inverse_k * adj_re
        else
            lhs_re -= products * (inverse_k * g_total_im)
            lhs_im += products * (inverse_k * g_total_re)
        end
        return lhs_re, lhs_im, rhs_re, rhs_im
    end

    # The parts summed as the scatter kernel sums them, ascending.
    function pair_value(::Type{T}, pair::Int, k::T, mode::Symbol) where {T}
        lhs = zeros(ComplexF64, 9)
        rhs = zeros(ComplexF64, 3)
        for part in 1:parts
            lhs_re, lhs_im, rhs_re, rhs_im = pair_part(T, pair, part, k, mode)
            for i in 1:9
                lhs[i] += ComplexF64(Float64(lhs_re[i]), Float64(lhs_im[i]))
            end
            for i in 1:3
                rhs[i] += ComplexF64(Float64(rhs_re[i]), Float64(rhs_im[i]))
            end
        end
        return vcat(lhs, rhs)
    end

    # Longest edge / (2 * inradius); 1.0 is equilateral.
    function aspect_ratio(face)
        v = mesh.vertices
        a = norm(v[face[2]] - v[face[1]])
        b = norm(v[face[3]] - v[face[2]])
        c = norm(v[face[1]] - v[face[3]])
        s = (a + b + c) / 2
        area = sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        area <= 0 && return Inf
        return max(a, b, c) * s / (2 * area)
    end
    aspects = [aspect_ratio(face) for face in mesh.faces]
    @printf("aspect_ratio min=%.3f median=%.3f p99=%.3f max=%.3f\n",
            minimum(aspects), median(aspects), quantile(aspects, 0.99), maximum(aspects))

    # Deterministic stratification: every pair when the mesh is small, otherwise
    # all of the worst-aspect half plus a seeded random half.
    pair_aspects = [max(aspects[Int(test_indices[p])], aspects[Int(trial_indices[p])]) for p in 1:pair_count]
    selected = if pair_count <= max_pairs
        collect(1:pair_count)
    else
        ordered = sortperm(pair_aspects; rev=true)
        rng = MersenneTwister(SUMMATION_SEED)
        unique(vcat(ordered[1:div(max_pairs, 2)], randperm(rng, pair_count)[1:div(max_pairs, 2)]))
    end
    @printf("evaluating %d of %d pairs (seed=%d)\n", length(selected), pair_count, SUMMATION_SEED)
    flush(stdout)

    failures = String[]
    for frequency_hz in SUMMATION_FREQUENCIES_HZ
        k32 = Float32(2pi) * Float32(frequency_hz) / 343.0f0
        k64 = Float64(k32)
        blocks_error = Float64[]
        quad_error = Float64[]
        selected_aspects = Float64[]
        f64_gap = 0.0
        for pair in selected
            reference = pair_value(Float64, pair, k64, :quad)
            reference_blocks = pair_value(Float64, pair, k64, :blocks)
            magnitude = norm(reference)
            magnitude == 0 && continue
            f64_gap = max(f64_gap, norm(reference - reference_blocks) / magnitude)
            push!(blocks_error, norm(pair_value(Float32, pair, k32, :blocks) - reference) / magnitude)
            push!(quad_error, norm(pair_value(Float32, pair, k32, :quad) - reference) / magnitude)
            push!(selected_aspects, pair_aspects[pair])
        end
        isempty(blocks_error) && error("no singular pair carried a non-zero contribution at $(frequency_hz) Hz.")
        worst = sortperm(selected_aspects; rev=true)[1:min(200, length(selected_aspects))]
        worse = count(i -> quad_error[i] > blocks_error[i], eachindex(blocks_error))
        worse_fraction = worse / length(blocks_error)
        ratio = maximum(quad_error) / max(maximum(blocks_error), floatmin(Float64))

        @printf("\nf=%.1f Hz k=%.4f pairs=%d\n", frequency_hz, k32, length(blocks_error))
        @printf("  f64_order_gap=%.3e\n", f64_gap)
        @printf("  blocks_f32 med=%.3e p99=%.3e max=%.3e\n",
                median(blocks_error), quantile(blocks_error, 0.99), maximum(blocks_error))
        @printf("  fused_f32  med=%.3e p99=%.3e max=%.3e\n",
                median(quad_error), quantile(quad_error, 0.99), maximum(quad_error))
        @printf("  worst_aspect blocks_max=%.3e fused_max=%.3e (aspect >= %.2f)\n",
                maximum(blocks_error[worst]), maximum(quad_error[worst]), minimum(selected_aspects[worst]))
        @printf("  fused_worse_share=%.3f max_error_ratio=%.3f\n", worse_fraction, ratio)
        flush(stdout)

        label = @sprintf("%.0f Hz", frequency_hz)
        f64_gap <= f64_tolerance ||
            push!(failures, "$(label): the two accumulation orders differ in Float64 by $(f64_gap) > $(f64_tolerance)")
        maximum(blocks_error) <= tolerance ||
            push!(failures, "$(label): blocks_f32 max error $(maximum(blocks_error)) > $(tolerance)")
        maximum(quad_error) <= tolerance ||
            push!(failures, "$(label): fused_f32 max error $(maximum(quad_error)) > $(tolerance)")
        worse_fraction <= worse_share ||
            push!(failures, "$(label): fused worse than blocks on $(worse_fraction) of pairs > $(worse_share)")
        ratio <= max_ratio ||
            push!(failures, "$(label): max error ratio fused/blocks $(ratio) > $(max_ratio)")
    end

    release_metal_singular_correction_cache!(device_singular_cache)
    release_metal_regular_assembly_cache!(device_cache)

    println()
    if isempty(failures)
        println("RESULT=pass")
        return 0
    end
    for failure in failures
        println("FAILURE: $(failure)")
    end
    println("RESULT=fail")
    return 1
end

exit(validate_metal_singular_summation())
