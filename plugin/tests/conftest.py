"""Keep the older dependency-stub tests from leaking into real CPU checks."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolate_dependency_stubs(request):
    roots = ("vllm", "vllm_ascend", "gdn_recoverssm_ascend")

    def selected(name):
        return name == "torch" or any(
            name == r or name.startswith(r + ".") for r in roots
        )

    saved = {k: v for k, v in sys.modules.items() if selected(k)}
    if request.module.__name__.endswith("test_layer_patch_mock"):
        # That module deliberately supplies a FakeTensor, including on hosts
        # which now have real torch installed for the numerical filter tests.
        for name in saved:
            sys.modules.pop(name, None)
        if request.node.name == "test_loggers_live_under_vllm":
            request.module._install_torch_stub()
            request.module._install_vllm_stubs()
    yield
    for name in list(sys.modules):
        if selected(name) and name not in saved:
            sys.modules.pop(name, None)
    sys.modules.update(saved)
