# SPDX-License-Identifier: Apache-2.0
"""Readiness checks the model's bound methods, independent of donor status."""

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def startup(monkeypatch):
    root = Path(__file__).parents[1] / "gdn_recoverssm_ascend"
    package = types.ModuleType("_gdn_mixed_ready")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.__name__ + ".selftest", types.ModuleType("selftest"))
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    spec = importlib.util.spec_from_file_location(
        package.__name__ + ".startup_patch", root / "startup_patch.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def bound_layer():
    def adapted(*args):
        pass

    adapted._gdn_mixed_source_digest = "a" * 64
    adapted._gdn_mixed_patch_counts = dict.fromkeys(
        ("preflight", "spec_qkv", "gates", "spec_verify", "merge"), 1
    )

    class Layer:
        def _forward_core(self, *args):
            pass

    Layer._forward_core._gdn_mixed_core = adapted
    return Layer()


def test_readiness_checks_all_actual_layers_and_describes_source(startup):
    registry = {"layer.0": bound_layer(), "layer.1": bound_layer()}
    config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(static_forward_context=registry)
    )
    builders = [types.SimpleNamespace(layer_names=list(registry))]
    evidence = startup._check_mixed_adapters(
        types.SimpleNamespace(vllm_config=config), builders
    )
    assert evidence == {
        "layers": 2,
        "native_source_digests": {"a" * 64: 2},
        "prefill_observation_layers": 0,
        "scope": "BOUND_METHOD_CHECK_NOT_DEVICE_TIMING",
    }


@pytest.mark.parametrize("change", ["missing_layer", "unpatched", "incomplete", "no_source"])
def test_readiness_rejects_an_unprepared_real_layer(startup, change):
    first, second = bound_layer(), bound_layer()
    registry = {"layer.0": first, "layer.1": second}
    if change == "missing_layer":
        registry.pop("layer.1")
    elif change == "unpatched":
        second._forward_core.__func__._gdn_mixed_core = None
    elif change == "incomplete":
        second._forward_core._gdn_mixed_core._gdn_mixed_patch_counts["merge"] = 0
    else:
        second._forward_core._gdn_mixed_core._gdn_mixed_source_digest = None
    config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(static_forward_context=registry)
    )
    with pytest.raises(RuntimeError, match="mixed adapter"):
        startup._check_mixed_adapters(
            types.SimpleNamespace(vllm_config=config),
            [types.SimpleNamespace(layer_names=["layer.0", "layer.1"])],
        )
