from __future__ import annotations

import subprocess
from pathlib import Path

from blab.metal import (
    MINIMUM_MACOS_MAJOR,
    default_metal_project,
    discover_metal,
    metal_unavailable_reason,
)
from blab.solvers.beat_engine_backend import (
    BEAT_ENGINE_METAL_BACKEND,
    DEFAULT_BEAT_ENGINE_METAL_PROJECT,
    _default_beat_engine_project,
    _julia_project_backend_label,
    _normalize_beat_engine_backend,
)
from blab.solvers.coupled_backend import (
    COUPLED_BEM_BACKENDS,
    PhysicalSystemProductionBackend,
)
from blab.solvers.registry import (
    backend_condenses_fem_interior,
    backend_info,
    create_backend,
    normalize_backend_id,
    supports_physical_system_solves,
)
from repo_paths import REPO_ROOT


def _runner(version: str = "15.7.7", chip: str = "Apple M1 Pro"):
    def run(command, **_kwargs):
        text = version if command[0] == "sw_vers" else chip
        return subprocess.CompletedProcess(command, 0, stdout=f"{text}\n", stderr="")

    return run


def test_discovery_accepts_supported_apple_silicon_host() -> None:
    installation = discover_metal(system="Darwin", machine="arm64", runner=_runner())
    assert installation is not None
    assert installation.is_apple_silicon
    assert installation.macos_version == "15.7.7"
    assert installation.chip == "Apple M1 Pro"


def test_discovery_rejects_non_macos_host() -> None:
    assert discover_metal(system="Linux", machine="x86_64", runner=_runner()) is None
    assert "requires macOS" in metal_unavailable_reason(system="Linux", machine="x86_64")


def test_discovery_rejects_intel_mac() -> None:
    assert discover_metal(system="Darwin", machine="x86_64", runner=_runner()) is None
    reason = metal_unavailable_reason(system="Darwin", machine="x86_64", runner=_runner())
    assert "Apple Silicon" in reason


def test_discovery_rejects_macos_older_than_minimum() -> None:
    old = _runner(version=f"{MINIMUM_MACOS_MAJOR - 1}.4.0")
    assert discover_metal(system="Darwin", machine="arm64", runner=old) is None
    reason = metal_unavailable_reason(system="Darwin", machine="arm64", runner=old)
    assert f"macOS {MINIMUM_MACOS_MAJOR}" in reason


def test_backend_aliases_normalize_to_metal() -> None:
    for alias in ("metal", "beat_metal", "apple", "mps", "Metal"):
        assert _normalize_beat_engine_backend(alias) == BEAT_ENGINE_METAL_BACKEND


def test_metal_backend_resolves_its_own_julia_project() -> None:
    project = _default_beat_engine_project(BEAT_ENGINE_METAL_BACKEND)
    assert project == DEFAULT_BEAT_ENGINE_METAL_PROJECT
    assert project.name == "julia_metal"
    assert default_metal_project().name == "julia_metal"


def test_metal_julia_project_ships_a_metal_environment() -> None:
    project_file = DEFAULT_BEAT_ENGINE_METAL_PROJECT / "Project.toml"
    assert project_file.is_file()
    assert "Metal" in project_file.read_text()


def test_registry_exposes_metal_backend() -> None:
    assert normalize_backend_id("beat_metal") == "beat_metal"
    info = backend_info("beat_metal")
    assert info.label == "BEAT Engine (Apple Metal)"
    assert info.capabilities.supports_symmetry
    assert info.capabilities.supports_channel_resynthesis
    assert not info.capabilities.is_remote


def test_registry_factory_builds_a_metal_backend_not_a_cuda_one() -> None:
    backend = create_backend("beat_metal")
    assert backend.backend_id == "beat_metal"
    assert backend.beat_engine_backend == BEAT_ENGINE_METAL_BACKEND
    assert Path(backend.julia_project).name == "julia_metal"


def test_metal_is_offered_for_coupled_physical_system_solves() -> None:
    assert supports_physical_system_solves("beat_metal")
    assert supports_physical_system_solves("beat_rocm")


def test_metal_condenses_the_fem_interior_on_the_host() -> None:
    # The Schur complement is CPU work on every backend, so Metal condenses too.
    # Measured 11.7x faster than monolithic on a 19,492-vertex FEM interior.
    assert backend_condenses_fem_interior("beat_metal")
    assert backend_condenses_fem_interior("beat_cuda")
    assert backend_condenses_fem_interior("beat_rocm")


def test_metal_coupled_backend_constructs_and_uses_the_metal_julia_project() -> None:
    # Regression. The Julia coupled path accepted :metal, but the Python adapter
    # in coupled_backend.py still rejected it, so a coupled solve failed with
    # "Coupled BEM backend must be cpu, cuda, or rocm." Once that was fixed, the
    # project map had no "metal" entry either, so the solver would have launched
    # under the CPU Julia project, where Metal.jl is not a dependency and the
    # BLAB_BEAT_ENGINE_GPU_BACKEND hint is never set.
    assert "metal" in COUPLED_BEM_BACKENDS
    backend = PhysicalSystemProductionBackend(bem_backend="metal", persistent_worker=False)
    assert backend.bem_backend == "metal"
    assert Path(backend.julia_project) == DEFAULT_BEAT_ENGINE_METAL_PROJECT


def test_metal_condensation_runs_on_the_host() -> None:
    coupled = (REPO_ROOT / "src/blab/solvers/julia_local/src/BeatEngineCoupled.jl").read_text(encoding="utf-8")
    assert "_build_host_fem_condensation" in coupled
    assert "_solve_host_condensed_excitations" in coupled
    assert ":cpu_hybrid" in coupled
    assert "static_condensation && !(bem_backend in (:cuda, :rocm, :metal))" in coupled


def test_solver_jl_dispatches_the_metal_backend() -> None:
    # The exterior solve driver rejected "metal" until the backend was wired,
    # so the Metal kernels were only reachable from the validation scripts.
    #
    # solver.jl is now a loader: it picks a precompiled bundle package and
    # falls back to including the driver from source. The dispatch itself
    # lives in BeatEngineDriver.jl, which is what both routes end up running.
    solver = (REPO_ROOT / "src/blab/solvers/julia_local/solver.jl").read_text(encoding="utf-8")
    assert "BeatEngineDriver.jl" in solver
    assert "BeatEngineMetalBundle" in solver

    driver = (REPO_ROOT / "src/blab/solvers/julia_local/BeatEngineDriver.jl").read_text(encoding="utf-8")
    assert '"cuda", "cpu", "rocm", "metal"' in driver
    assert "evaluate_galerkin_field_metal" in driver
    assert "build_metal_regular_assembly_cache" in driver
    assert "build_metal_singular_correction_cache" in driver
    assert "release_metal_regular_assembly_cache!" in driver


def test_coupled_solver_dispatches_the_metal_backend() -> None:
    coupled = (REPO_ROOT / "src/blab/solvers/julia_local/src/BeatEngineCoupled.jl").read_text(encoding="utf-8")
    assert "bem_backend in (:cpu, :cuda, :rocm, :metal)" in coupled
    # Metal assembles the BEM operators on the GPU and then hands them to the
    # host, because Metal.jl has no GPU LU and unified memory makes the handoff
    # free. So it must NOT be routed into the device block builder that CUDA
    # and ROCm use -- that path frees device buffers and would be handed a host
    # Matrix. The guards below stay two-way on purpose.
    assert "metal_host_operators(operators)" in coupled
    assert "bem_blocks = if bem_backend in (:cuda, :rocm)" in coupled
    assert "_metal_coupled_bem_blocks" not in coupled


def test_metal_validation_scripts_cover_symmetry_and_coupled() -> None:
    scripts = REPO_ROOT / "src/blab/solvers/julia_local/scripts"
    for name, marker in (
        ("validate_metal_exterior.jl", "METAL_EXTERIOR_VALIDATION_OK"),
        ("validate_metal_symmetry.jl", "METAL_SYMMETRY_VALIDATION_OK"),
        ("validate_metal_coupled.jl", "METAL_COUPLED_VALIDATION_OK"),
    ):
        script = scripts / name
        assert script.is_file(), f"missing {name}"
        assert marker in script.read_text(encoding="utf-8")


def test_metal_coupled_julia_test_block_matches_the_rocm_gate() -> None:
    tests = (REPO_ROOT / "src/blab/solvers/julia_local/tests/coupled_solver_tests.jl").read_text(encoding="utf-8")
    assert "BLAB_RUN_COUPLED_METAL" in tests
    assert "metal_available()" in tests
    runtests = (REPO_ROOT / "src/blab/solvers/julia_local/tests/runtests.jl").read_text(encoding="utf-8")
    assert "metal_available()" in runtests


def test_metal_assembly_mode_reads_its_own_environment_variable() -> None:
    assembly = (REPO_ROOT / "src/blab/solvers/julia_local/src/BeatEngineMetalAssembly.jl").read_text(encoding="utf-8")
    assert "BLAB_METAL_ASSEMBLY_MODE" in assembly
    assert "BLAB_ROCM_ASSEMBLY_MODE" not in assembly


def test_backend_label_lookup_reports_metal() -> None:
    label = _julia_project_backend_label(DEFAULT_BEAT_ENGINE_METAL_PROJECT, BEAT_ENGINE_METAL_BACKEND)
    assert label == "BEAT Engine (Apple Metal)"


def _accelerate_source() -> str:
    return (REPO_ROOT / "src/blab/solvers/julia_local/src/BeatEngineAccelerateSparse.jl").read_text(encoding="utf-8")


def test_accelerate_binding_asserts_its_struct_layouts() -> None:
    source = _accelerate_source()
    # A ccall layout mistake is silent corruption, so the binding must check the
    # ABI it was written against rather than trusting the SDK to be unchanged.
    assert "accelerate_sparse_abi_matches_reference" in source
    assert "sizeof(type) == size" in source
    assert "fieldoffset(type, index) == offset" in source
    assert "accelerate_sparse_available()" in source


def test_accelerate_binding_covers_both_precisions() -> None:
    source = _accelerate_source()
    for symbol in (
        "_SparseFactorLU_",
        "_SparseRefactorLU_",
        "_SparseSolveOpaque_",
        "_SparseDestroyOpaqueNumeric_",
    ):
        assert symbol in source, symbol
    assert '"Complex_Float"' in source
    assert '"Complex_Double"' in source


def test_condensation_backend_switch_is_documented_and_wired() -> None:
    coupled = (REPO_ROOT / "src/blab/solvers/julia_local/src/BeatEngineCoupled.jl").read_text(encoding="utf-8")
    assert "BLAB_METAL_FEM_CONDENSATION" in coupled
    assert "BLAB_ACCELERATE_SCHUR_BLOCK" in coupled
    # The default must stay UMFPACK: the Accelerate path is faster but exceeds
    # the 5e-4 relative-norm gate the coupled validations use.
    assert 'get(ENV, "BLAB_METAL_FEM_CONDENSATION", "umfpack")' in coupled
    doc = (REPO_ROOT / "docs/advanced/beat-engine-metal-condensation-options.md").read_text(encoding="utf-8")
    assert "BLAB_METAL_FEM_CONDENSATION" in doc
    assert "accelerate_f64" in doc


def test_accelerate_abi_probe_is_checked_in() -> None:
    probe = REPO_ROOT / "scripts/probe-accelerate-sparse-abi.c"
    assert probe.is_file()
    text = probe.read_text(encoding="utf-8")
    assert "SparseOpaqueFactorization_Complex_Float" in text
    assert "offsetof" in text


def test_metal_doc_records_the_condensation_default_and_its_accuracy_cost() -> None:
    doc = (REPO_ROOT / "docs/advanced/beat-engine-metal.md").read_text(encoding="utf-8")
    section = doc[doc.index("### Interior solver: UMFPACK by default") :]
    assert "0.016 dB" in section
    assert "3.2e-3" in section
    assert "5e-4" in section
    # The block-size trap inverts the measurement, so it has to stay documented.
    assert "BLAB_ACCELERATE_SCHUR_BLOCK" in section
    assert "superlinearly" in section
