"""Apple Metal backend discovery for Boundary Lab.

Unlike ROCm, Metal needs no separate SDK install and no search for a toolchain
root: the driver ships with macOS and Metal.jl talks to it directly. Discovery
is therefore a capability check rather than a filesystem hunt.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

MINIMUM_MACOS_MAJOR = 13
"""Metal.jl requires macOS 13 (Ventura) or newer."""


@dataclass(frozen=True)
class MetalInstallation:
    """A usable Metal target."""

    system: str
    machine: str
    macos_version: str
    chip: str
    source: str

    @property
    def is_apple_silicon(self) -> bool:
        return self.machine == "arm64"


def _macos_version(runner=subprocess.run) -> str:
    try:
        completed = runner(
            ["sw_vers", "-productVersion"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "").strip()


def _chip_name(runner=subprocess.run) -> str:
    if not shutil.which("sysctl"):
        return ""
    try:
        completed = runner(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "").strip()


def _version_major(version: str) -> int:
    head = version.split(".", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return 0


def discover_metal(
    *,
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
    machine: str | None = None,
    runner=subprocess.run,
) -> MetalInstallation | None:
    """Return a :class:`MetalInstallation` when this host can run Metal solves.

    Returns ``None`` on any non-macOS host, on Intel Macs, and on macOS releases
    older than Metal.jl's minimum. Callers treat ``None`` as "backend
    unavailable" rather than as an error.
    """

    resolved_system = system if system is not None else platform.system()
    if resolved_system != "Darwin":
        return None

    resolved_machine = machine if machine is not None else platform.machine()
    if resolved_machine != "arm64":
        # Metal.jl supports Apple Silicon only; Intel Macs have no supported path.
        return None

    version = _macos_version(runner=runner)
    if version and _version_major(version) < MINIMUM_MACOS_MAJOR:
        return None

    return MetalInstallation(
        system=resolved_system,
        machine=resolved_machine,
        macos_version=version,
        chip=_chip_name(runner=runner),
        source="platform",
    )


def metal_unavailable_reason(
    *,
    system: str | None = None,
    machine: str | None = None,
    runner=subprocess.run,
) -> str:
    """Human-readable explanation for why Metal is not usable on this host."""

    resolved_system = system if system is not None else platform.system()
    if resolved_system != "Darwin":
        return f"BEAT Engine Metal requires macOS. This host reports {resolved_system}."

    resolved_machine = machine if machine is not None else platform.machine()
    if resolved_machine != "arm64":
        return f"BEAT Engine Metal requires Apple Silicon. This host reports {resolved_machine}."

    version = _macos_version(runner=runner)
    if version and _version_major(version) < MINIMUM_MACOS_MAJOR:
        return f"BEAT Engine Metal requires macOS {MINIMUM_MACOS_MAJOR} or newer. This host reports macOS {version}."
    return "BEAT Engine Metal is unavailable on this host."


def default_metal_project(solvers_root: Path | None = None) -> Path:
    """Path to the dedicated Julia environment for the Metal backend."""

    root = solvers_root or Path(__file__).with_name("solvers")
    return root / "julia_metal"
