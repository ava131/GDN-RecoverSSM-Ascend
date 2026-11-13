# SPDX-License-Identifier: Apache-2.0
"""CPU checks for startup build ownership, cache integrity and failure handling."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture
def build(monkeypatch):
    path = Path(__file__).parents[1] / "gdn_recoverssm_ascend/ascendc_build.py"
    spec = importlib.util.spec_from_file_location("_test_ac_build", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toolchain(build):
    return build.Toolchain(
        cann_root="/toolkit",
        compiler="/toolkit/bisheng",
        compiler_version="bisheng 1",
        cmake="/bin/cmake",
        cmake_version="cmake 3.28",
        torch_prefix="/torch/share/cmake",
        torch_version="2.10.0",
        torch_abi=1,
        torch_npu_root="/torch_npu",
        torch_npu_version="2.10.0.post4",
        machine="aarch64",
        cann_versions=(("version.cfg", "9.1"),),
    )


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "csrc"
    path.mkdir()
    (path / "ops.cpp").write_text("// host and device launch\n")
    (path / "common.h").write_text("// geometry\n")
    (path / "CMakeLists.txt").write_text("project(gdn)\n")
    return path


def _publish_fake(build, directory, fingerprint):
    library = directory / build._LIBRARY
    library.write_bytes(b"compiled operators")
    (directory / "build.log").write_text("successful build\n")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
        )
    )


def test_import_does_not_load_device_packages(build, monkeypatch):
    # Running tool discovery is deliberately separate from importing this file.
    monkeypatch.setattr(
        build.importlib, "import_module", lambda _: pytest.fail("device import")
    )
    assert build._LOADED is None
    assert build.BuildArtifact is not None


def test_fingerprint_tracks_source_and_toolchain(build, source, toolchain):
    first = build._fingerprint(source, toolchain)
    assert first == build._fingerprint(source, toolchain)
    assert first != build._fingerprint(
        source, replace(toolchain, torch_npu_version="next")
    )
    assert first != build._fingerprint(
        source, replace(toolchain, compiler_version="bisheng 2")
    )
    assert first != build._fingerprint(source, replace(toolchain, torch_abi=0))
    (source / "common.h").write_text("// different geometry\n")
    assert first != build._fingerprint(source, toolchain)


def test_toolchain_discovery_uses_existing_install_and_wheel_metadata(
    build, tmp_path, monkeypatch
):
    toolkit = tmp_path / "toolkit"
    compiler = toolkit / "compiler/ccec_compiler/bin/bisheng"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("compiler")
    (toolkit / "version.cfg").write_text("9.1")
    fake_torch = types.SimpleNamespace(
        __file__=str(tmp_path / "torch/__init__.py"),
        __version__="2.10.0",
        _C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
        utils=types.SimpleNamespace(cmake_prefix_path="/actual/torch/cmake"),
    )
    fake_npu = types.SimpleNamespace(
        __file__=str(tmp_path / "torch_npu/__init__.py"), __version__="2.10.0.post4"
    )
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(build, "_cann_root", lambda: toolkit)
    monkeypatch.setattr(build.shutil, "which", lambda _: "/bin/cmake")
    monkeypatch.setattr(build, "_version", lambda command: command[0] + " version")
    monkeypatch.setattr(
        build.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_npu,
    )
    got = build._discover_toolchain()
    assert got.compiler == str(compiler)
    assert got.torch_prefix == "/actual/torch/cmake"
    assert got.torch_abi == 1
    assert got.torch_npu_version == "2.10.0.post4"
    assert got.cann_versions == (("version.cfg", hashlib.sha256(b"9.1").hexdigest()),)


def test_non_linux_discovery_refuses_before_importing_device_packages(
    build, monkeypatch
):
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        build.importlib, "import_module", lambda _: pytest.fail("device import")
    )
    with pytest.raises(build.AscendCBuildError, match="Linux deployment"):
        build._discover_toolchain()


def test_distribution_declares_kernel_sources_as_package_data(build):
    import tomllib

    project = Path(build.__file__).parents[1] / "pyproject.toml"
    config = tomllib.loads(project.read_text())
    data = config["tool"]["setuptools"]["package-data"]["gdn_recoverssm_ascend"]
    assert "csrc/CMakeLists.txt" in data
    assert "csrc/*.cpp" in data
    assert "csrc/*.h" in data


def test_two_workers_build_once_and_use_verified_cache(
    build, source, toolchain, tmp_path, monkeypatch
):
    calls = []
    lock = threading.Lock()

    def compile_fake(snapshot, directory, fingerprint, _):
        assert snapshot != source
        assert (snapshot / "ops.cpp").read_bytes() == (source / "ops.cpp").read_bytes()
        with lock:
            calls.append(directory)
        time.sleep(0.02)
        _publish_fake(build, directory, fingerprint)

    monkeypatch.setattr(build, "_build", compile_fake)
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(
                lambda _: build._build_cached(source, cache, toolchain), range(2)
            )
        )
    assert len(calls) == 1
    assert results[0].library_path == results[1].library_path
    assert sorted(x.cache_hit for x in results) == [False, True]
    assert build._valid_cache(results[0].library_path.parent, results[0].fingerprint)


def test_failed_build_is_not_published_and_retry_can_succeed(
    build, source, toolchain, tmp_path, monkeypatch
):
    def fail(_, directory, *args):
        (directory / build._LIBRARY).write_bytes(b"incomplete")
        raise build.AscendCBuildError("compiler failed")

    cache = tmp_path / "cache"
    monkeypatch.setattr(build, "_build", fail)
    with pytest.raises(build.AscendCBuildError, match="compiler failed"):
        build._build_cached(source, cache, toolchain)
    assert not (cache / build._fingerprint(source, toolchain)).exists()
    monkeypatch.setattr(build, "_build", lambda _, d, f, t: _publish_fake(build, d, f))
    result = build._build_cached(source, cache, toolchain)
    assert not result.cache_hit
    assert result.library_path.read_bytes() == b"compiled operators"


def test_corrupt_cached_library_is_quarantined_and_rebuilt(
    build, source, toolchain, tmp_path, monkeypatch
):
    monkeypatch.setattr(build, "_build", lambda _, d, f, t: _publish_fake(build, d, f))
    cache = tmp_path / "cache"
    first = build._build_cached(source, cache, toolchain)
    first.library_path.write_bytes(b"corruption")
    second = build._build_cached(source, cache, toolchain)
    assert not second.cache_hit
    assert second.library_path.read_bytes() == b"compiled operators"
    assert len(list(cache.glob("*-invalid-cache"))) == 1


def test_build_command_has_exact_fast_launch_arch_and_preserves_log(
    build, source, toolchain, tmp_path, monkeypatch
):
    calls = []
    output = tmp_path / "attempt"

    def run(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("compiler diagnostic\n")
        if "--build" in command:
            (output / build._LIBRARY).write_bytes(b"operators")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build.subprocess, "run", run)
    build._build(source, output, "key", toolchain)
    assert "-DCMAKE_CXX_COMPILER=/toolkit/bisheng" in calls[0]
    assert "-DCMAKE_PREFIX_PATH=/torch/share/cmake" in calls[0]
    assert calls[1][-2:] == ["--parallel", "1"]
    assert "compiler diagnostic" in (output / "build.log").read_text()
    assert build._valid_cache(output, "key")
    cmake = Path(build.__file__).parent / "csrc/CMakeLists.txt"
    assert "--npu-arch=dav-2201;-xasc" in cmake.read_text()
    assert "-ffast-math" not in cmake.read_text()


def test_compile_failure_exposes_original_diagnostic_log(
    build, source, toolchain, tmp_path, monkeypatch
):
    def run(command, **kwargs):
        kwargs["stdout"].write("bad AscendC instruction\n")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(build.subprocess, "run", run)
    output = tmp_path / "failed"
    with pytest.raises(build.AscendCBuildError, match="before worker READY") as error:
        build._build(source, output, "key", toolchain)
    assert str(output / "build.log") in str(error.value)
    assert "bad AscendC instruction" in (output / "build.log").read_text()
    assert not (output / "manifest.json").exists()


def test_load_once_and_registration_failure_never_marks_loaded(
    build, toolchain, tmp_path, monkeypatch
):
    artifact = build.BuildArtifact(
        tmp_path / "kernel.so", "key", False, tmp_path / "build.log"
    )
    monkeypatch.setattr(build, "_discover_toolchain", lambda: toolchain)
    monkeypatch.setattr(build, "_build_cached", lambda *args: artifact)
    calls = []
    fake_torch = types.SimpleNamespace(
        ops=types.SimpleNamespace(load_library=calls.append)
    )
    monkeypatch.setattr(build.importlib, "import_module", lambda _: fake_torch)
    assert build.ensure_loaded() is artifact
    assert build.ensure_loaded() is artifact
    assert calls == [str(artifact.library_path)]
    build._LOADED = None

    def fail(_):
        raise RuntimeError("undefined symbol")

    fake_torch.ops.load_library = fail
    with pytest.raises(build.AscendCBuildError, match="registration failed"):
        build.ensure_loaded()
    assert build._LOADED is None
