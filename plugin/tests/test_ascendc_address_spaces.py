# SPDX-License-Identifier: Apache-2.0
"""Compile the real Host pointer adapter with an address-qualified GM type.

An empty __gm__ stub cannot detect the BiSheng error reported in issue #4.
Clang's address_space attribute preserves that distinction without NPU hardware.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "gdn_recoverssm_ascend/csrc/ops.cpp"


def compile_probe(tmp_path, macro):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang needed for the address-space compiler regression")
    source = tmp_path / "gm_cast.cpp"
    source.write_text(
        "#include <cstdint>\n"
        "#define __gm__ __attribute__((address_space(1)))\n"
        "using GM_ADDR = __gm__ unsigned char *;\n"
        "struct Tensor { void *data_ptr() const; };\n" + macro + "\n"
        "void launch(GM_ADDR);\n"
        "void host_launch(const Tensor& tensor) { launch(PTR(tensor)); }\n"
        "__gm__ float *same_space(GM_ADDR p) {\n"
        "  return reinterpret_cast<__gm__ float *>(p);\n}\n"
        "__gm__ float *address_table(int64_t p) {\n"
        "  return reinterpret_cast<__gm__ float *>(p);\n}\n"
    )
    return subprocess.run(
        [compiler, "-std=c++17", "-Werror", "-fsyntax-only", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_real_host_adapter_preserves_qualified_gm_pointer(tmp_path):
    macro = re.search(r"^#define PTR\(tensor\).*$", SOURCE.read_text(), re.M)
    assert macro is not None
    result = compile_probe(tmp_path, macro.group())
    assert result.returncode == 0, result.stderr


def test_previous_reinterpret_cast_fails_like_bisheng(tmp_path):
    result = compile_probe(
        tmp_path, "#define PTR(tensor) reinterpret_cast<GM_ADDR>((tensor).data_ptr())"
    )
    assert result.returncode != 0
    assert "reinterpret_cast" in result.stderr
    assert "not allowed" in result.stderr
