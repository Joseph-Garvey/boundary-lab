# Traced to collect precompile statements for the CUDA sysimage.
#
# It drives the bundle rather than `include`ing `solver.jl`, because the
# statements PackageCompiler collects are only kept if they resolve when the
# image is built -- and anything defined in `Main` by an `include` does not.
# That is why the previous version of this file, which included the script and
# called its parsing helpers, contributed nothing to the image.

using BeatEngineCudaBundle

const B = BeatEngineCudaBundle

request = Dict{String,Any}(
    "beat_engine_backend" => "cuda",
    "frequencies_hz" => [100.0],
    "config" => Dict{String,Any}(
        "mesh_file" => "warmup.msh",
        "scale_factor" => 1.0,
        "distance" => 1.0,
        "step_size" => 90.0,
        "min_angle" => 0.0,
        "max_angle" => 0.0,
        "tag_throat" => 2,
        "radiators" => [Dict{String,Any}(
            "name" => "warmup",
            "tag" => 2,
            "channel" => "main",
            "hpf" => Dict{String,Any}("type" => "none"),
            "lpf" => Dict{String,Any}("type" => "none"),
        )],
        "channels" => [Dict{String,Any}(
            "name" => "main",
            "hpf" => Dict{String,Any}("type" => "none"),
            "lpf" => Dict{String,Any}("type" => "none"),
        )],
    ),
)

B.beat_backend_from_request(request)
mesh_inputs = B.mesh_inputs_from_config(request["config"])
B.radiator_inputs_from_config(request["config"], mesh_inputs)
B.channel_inputs_from_config(request["config"])
B.polar_observation_points(request["config"], Float32)
B.spherical_observation(request["config"], Float32)
