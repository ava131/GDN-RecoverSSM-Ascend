# SPDX-License-Identifier: Apache-2.0
"""Configure the real CMake recipe and inspect the generated link command.

The fake SDK files are never linked/executed. This checks dependency selection,
not BiSheng or NPU compatibility.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "gdn_recoverssm_ascend/csrc"


def configure(tmp_path, *, missing=None, architecture_only=False):
    cmake = shutil.which("cmake")
    if cmake is None:
        candidate = Path(sys.executable).parent / "cmake"
        cmake = str(candidate) if candidate.is_file() else None
    compiler = shutil.which("clang++") or shutil.which("c++")
    if cmake is None or compiler is None or shutil.which("make") is None:
        pytest.skip("CMake, C++ and make needed for the real configure regression")
    toolkit = tmp_path / "sdk"
    include = toolkit / "include"
    include.mkdir(parents=True)
    (include / "kernel_operator.h").write_text("// fake configure-only header\n")
    lib = toolkit / (
        f"{platform.machine()}-linux/lib64" if architecture_only else "lib64"
    )
    lib.mkdir(parents=True)
    for name in ("tiling_api", "platform"):
        if name != missing:
            (lib / f"lib{name}.so").write_bytes(b"configure-only; not a binary")
    torch = tmp_path / "torch"
    torch.mkdir()
    (torch / "TorchConfig.cmake").write_text(
        'set(Torch_FOUND TRUE)\nset(TORCH_LIBRARIES "")\n'
        'set(TORCH_INCLUDE_DIRS "")\nset(TORCH_CXX_FLAGS "")\n'
    )
    out = tmp_path / "build"
    # Use the real recipe; only external packages/SDK files are simulated.
    result = subprocess.run(
        [
            cmake,
            "-G",
            "Unix Makefiles",
            "-S",
            str(SOURCE),
            "-B",
            str(out),
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DTorch_DIR={torch}",
            f"-DGDN_CANN_ROOT={toolkit}",
            f"-DGDN_TORCH_NPU_ROOT={tmp_path / 'npu'}",
            f"-DGDN_OUTPUT_DIR={tmp_path / 'output'}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, out, lib


@pytest.mark.parametrize("architecture_only", [False, True])
def test_real_recipe_links_both_matched_sdk_dependencies(tmp_path, architecture_only):
    result, out, lib = configure(tmp_path, architecture_only=architecture_only)
    assert result.returncode == 0, result.stdout + result.stderr
    link = (out / "CMakeFiles/gdn_recoverssm_ac.dir/link.txt").read_text()
    assert str(lib / "libtiling_api.so") in link
    assert str(lib / "libplatform.so") in link
    assert "-Wl,--no-undefined" in link
    assert "--no-allow-shlib-undefined" not in link


@pytest.mark.parametrize("missing", ["tiling_api", "platform"])
def test_missing_direct_sdk_dependency_fails_before_compilation(tmp_path, missing):
    result, _, _ = configure(tmp_path, missing=missing)
    assert result.returncode != 0
    assert f"gdn_{missing}" in result.stderr
