# SPDX-License-Identifier: Apache-2.0
"""Build and register the plugin's AscendC kernels during worker preparation.

Importing this module does not import Torch-NPU, initialize a device, or build
anything. Call ``ensure_loaded`` before graph capture, never from serving. The
cache is local to the user and protected across TP workers by a file lock.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

_RECIPE_VERSION = 1
_ARCH = "dav-2201"
_LIBRARY = "libgdn_recoverssm_ac.so"
_LOADED = None
_PROCESS_LOCK = threading.Lock()


class AscendCBuildError(RuntimeError):
    """An unavailable compiler, failed build, or registration blocks readiness."""


@dataclass(frozen=True)
class BuildArtifact:
    library_path: Path
    fingerprint: str
    cache_hit: bool
    log_path: Path


@dataclass(frozen=True)
class Toolchain:
    cann_root: str
    compiler: str
    compiler_version: str
    cmake: str
    cmake_version: str
    torch_prefix: str
    torch_version: str
    torch_abi: int
    torch_npu_root: str
    torch_npu_version: str
    machine: str
    cann_versions: tuple[tuple[str, str], ...]


def _version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AscendCBuildError(
            f"cannot query AscendC toolchain {command[0]}: {exc}"
        ) from exc
    return (result.stdout + result.stderr).strip()


def _cann_root() -> Path:
    candidates = [
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/latest",
    ]
    for item in candidates:
        if item and Path(item).is_dir():
            return Path(item).resolve()
    raise AscendCBuildError(
        "AscendC development toolkit not found in the existing CANN environment; "
        "the op-dev plugin requires BiSheng and headers before worker READY"
    )


def _discover_toolchain() -> Toolchain:
    if platform.system() != "Linux":
        raise AscendCBuildError(
            "AscendC NPU build requires the Linux deployment environment"
        )
    root = _cann_root()
    machine = platform.machine()
    candidates = [
        root / "compiler/ccec_compiler/bin/bisheng",
        root / f"{machine}-linux/ccec_compiler/bin/bisheng",
        root / "ccec_compiler/bin/bisheng",
    ]
    compiler = next((p for p in candidates if p.is_file()), None)
    if compiler is None:
        found = shutil.which("bisheng")
        compiler = Path(found).resolve() if found else None
    cmake = shutil.which("cmake")
    if compiler is None or cmake is None:
        raise AscendCBuildError(
            "AscendC build requires BiSheng and CMake in the deployment toolkit"
        )
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    # Importing torch does not guarantee that torch.utils.cmake_prefix_path has
    # been materialized on every supported wheel; the package prefix is stable.
    prefix = getattr(
        getattr(torch, "utils", None),
        "cmake_prefix_path",
        str(Path(torch.__file__).resolve().parent / "share/cmake"),
    )
    npu_root = Path(torch_npu.__file__).resolve().parent
    versions = []
    for relative in (
        "version.cfg",
        "version.info",
        f"{machine}-linux/ascend_toolkit_install.info",
    ):
        path = root / relative
        if path.is_file():
            versions.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    return Toolchain(
        cann_root=str(root),
        compiler=str(compiler),
        compiler_version=_version([str(compiler), "--version"]),
        cmake=str(Path(cmake).resolve()),
        cmake_version=_version([cmake, "--version"]),
        torch_prefix=str(prefix),
        torch_version=str(torch.__version__),
        torch_abi=int(torch._C._GLIBCXX_USE_CXX11_ABI),
        torch_npu_root=str(npu_root),
        torch_npu_version=str(torch_npu.__version__),
        machine=machine,
        cann_versions=tuple(versions),
    )


def _fingerprint(source: Path, toolchain: Toolchain) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"recipe": _RECIPE_VERSION, "arch": _ARCH, "toolchain": asdict(toolchain)},
            sort_keys=True,
        ).encode()
    )
    files = sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and (p.suffix in (".h", ".cpp", ".cmake") or p.name == "CMakeLists.txt")
    )
    if not files or not (source / "ops.cpp").is_file():
        raise AscendCBuildError(
            "packaged AscendC sources are incomplete: missing csrc/ops.cpp"
        )
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_root() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "gdn_recoverssm_ascend" / "ascendc"


def _valid_cache(directory: Path, fingerprint: str) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        library = directory / _LIBRARY
        return (
            manifest["fingerprint"] == fingerprint
            and library.is_file()
            and hashlib.sha256(library.read_bytes()).hexdigest()
            == manifest["library_sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _run(command: list[str], log) -> None:
    log.write("COMMAND " + json.dumps(command) + "\n")
    log.flush()
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def _build(
    source: Path, directory: Path, fingerprint: str, toolchain: Toolchain
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "build.log"
    try:
        with log_path.open("w") as log:
            _run(
                [
                    toolchain.cmake,
                    "-S",
                    str(source),
                    "-B",
                    str(directory / "build"),
                    f"-DCMAKE_CXX_COMPILER={toolchain.compiler}",
                    f"-DCMAKE_PREFIX_PATH={toolchain.torch_prefix}",
                    f"-DGDN_CANN_ROOT={toolchain.cann_root}",
                    f"-DGDN_TORCH_NPU_ROOT={toolchain.torch_npu_root}",
                    f"-DGDN_OUTPUT_DIR={directory}",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                log,
            )
            # Serial compilation bounds memory when every TP worker starts at
            # once. Only one process per cache fingerprint reaches this command.
            _run(
                [
                    toolchain.cmake,
                    "--build",
                    str(directory / "build"),
                    "--parallel",
                    "1",
                ],
                log,
            )
        library = directory / _LIBRARY
        if not library.is_file() or library.stat().st_size == 0:
            raise AscendCBuildError("CMake did not produce a nonempty AscendC library")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "arch": _ARCH,
                    "recipe": _RECIPE_VERSION,
                    "toolchain": asdict(toolchain),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (OSError, subprocess.SubprocessError, AscendCBuildError) as exc:
        raise AscendCBuildError(
            f"AscendC build failed before worker READY; full compiler log: {log_path}; {exc}"
        ) from exc


def _build_cached(source: Path, cache: Path, toolchain: Toolchain) -> BuildArtifact:
    # Linux-only and imported lazily: CPU-only imports remain portable.
    import fcntl

    fingerprint = _fingerprint(source, toolchain)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / fingerprint
    with (cache / f"{fingerprint}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = _valid_cache(target, fingerprint)
        if not hit:
            # Build entirely outside the published directory. Failed attempts
            # retain their log, and another rank cannot load a partial library.
            staging = Path(tempfile.mkdtemp(prefix=f".{fingerprint[:16]}-", dir=cache))
            snapshot = staging / "source"
            shutil.copytree(source, snapshot)
            if _fingerprint(snapshot, toolchain) != fingerprint:
                raise AscendCBuildError(
                    "AscendC sources changed during preparation; refusing an unkeyed binary"
                )
            _build(snapshot, staging, fingerprint, toolchain)
            if target.exists():
                # A corrupt cache is quarantined rather than loaded or deleted
                # while another process could still hold its shared library.
                target.rename(staging.with_name(staging.name + "-invalid-cache"))
            staging.rename(target)
        return BuildArtifact(target / _LIBRARY, fingerprint, hit, target / "build.log")


def ensure_loaded() -> BuildArtifact:
    """Compile/cache/register the AC library, once per process, before capture.

    No fallback occurs here: callers must not publish READY on build/load error.
    A cached binary is keyed by source bytes and the actual compiler/Torch/NPU
    toolchain, not just a plugin version string.
    """
    global _LOADED
    with _PROCESS_LOCK:
        if _LOADED is not None:
            return _LOADED
        toolchain = _discover_toolchain()
        artifact = _build_cached(
            Path(__file__).parent / "csrc", _cache_root(), toolchain
        )
        try:
            importlib.import_module("torch").ops.load_library(
                str(artifact.library_path)
            )
        except Exception as exc:
            raise AscendCBuildError(
                f"AscendC registration failed for {artifact.library_path}; "
                f"compiler log: {artifact.log_path}; {exc}"
            ) from exc
        _LOADED = artifact
        return artifact
