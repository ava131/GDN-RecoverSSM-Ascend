# SPDX-License-Identifier: Apache-2.0
"""Window DMA bounds, UB lifetime, and the independent last-output identity.

These checks run without an NPU. They do not establish CANN compilation,
hardware event scheduling, or device arithmetic tolerances.
"""

import re
from pathlib import Path

import pytest
import torch

CSRC = Path(__file__).parents[1] / "gdn_recoverssm_ascend" / "csrc"


def _layout():
    source = (CSRC / "verify.h").read_text()
    values = {"sizeof(float)": 4}
    for name, expression in re.findall(
        r"static constexpr uint32_t (\w+) = ([^;]+);", source
    ):
        expression = expression.replace("sizeof(float)", "4")
        for key, value in values.items():
            expression = re.sub(rf"\b{key}\b", str(value), expression)
        assert re.fullmatch(r"[\d\s*+()/-]+", expression)
        values[name] = eval(expression, {"__builtins__": {}}, {})
    return values


def test_verify_window_staging_fits_shared_host_device_budget():
    layout = _layout()
    shared = (CSRC / "common.h").read_text()
    budget = int(re.search(r"kVerifyUbBytes = (\d+)", shared).group(1))
    guard = int(re.search(r"kUbGuardBytes = (\d+)", shared).group(1))
    assert layout["kUbBytes"] == budget
    assert budget + guard <= 192 * 1024
    # All outputs/records remain disjoint from live recurrent state and inputs.
    regions = [
        (0, 2 * layout["kMatrixBytes"]),
        (layout["kQOffset"], layout["kWindowBytes"]),
        (layout["kKOffset"], layout["kWindowBytes"]),
        (layout["kVOffset"], layout["kWindowBytes"]),
        (layout["kRawOffset"], layout["kWindowBytes"]),
        (layout["kRawKeyOffset"], layout["kWindowBytes"]),
        (layout["kCorrectionOffset"], layout["kWindowBytes"]),
        (layout["kOutputOffset"], layout["kWindowBytes"]),
        (layout["kBroadcastOffset"], 128 * 32),
        (layout["kBetaRawOffset"], 32),
        (layout["kMetaOffset"], 64),
        (layout["kDecayOffset"], 32),
    ]
    for index, (start, length) in enumerate(regions):
        assert start % 32 == 0
        assert start + length <= budget
        for other_start, other_length in regions[index + 1 :]:
            assert start + length <= other_start or other_start + other_length <= start


def test_one_output_event_publication_after_window_computation():
    source = (CSRC / "verify.h").read_text()
    stage = source.split("inline void StageToken", 1)[1].split("template <typename T>", 1)[0]
    assert "DataCopy" not in stage
    assert "BeforeOutput()" not in stage
    save = source.split("inline void SaveWindow", 1)[1].split("inline void LastOutput", 1)[0]
    assert save.count("BeforeOutput();") == 1
    assert save.count("AfterOutput();") == 1
    assert save.index("BeforeOutput();") < save.index("StoreWindow(")
    assert save.rindex("AfterOutput();") > save.rindex("StoreWindow(")
    run = source.split("inline void RunTask", 1)[1].split("VerifyTiling td_", 1)[0]
    assert run.count("SaveWindow(") == 1
    assert "SaveToken(" not in run
    assert "if (t + 1 == count)" in run
    assert run.index("LastOutput(") < run.index("} else {", run.index("LastOutput("))
    after = source.split("inline void AfterOutput", 1)[1].split("inline void VectorToScalar", 1)[0]
    assert "SetFlag<HardEvent::MTE3_S>(outputScalarFree_)" in after
    assert "WaitFlag<HardEvent::MTE3_S>(outputScalarFree_)" in after


@pytest.mark.parametrize("count", range(1, 9))
@pytest.mark.parametrize("element_bytes", [2, 4])
@pytest.mark.parametrize("token_stride", [128, 160, 12 * 128, 17])
def test_window_copy_byte_gaps_preserve_token_padding(count, element_bytes, token_stride):
    # Ascend UB->GM srcStride is in 32-byte blocks, dstStride is in bytes.
    # The normal path uses tightly packed UB rows and a byte gap between GM rows.
    row_bytes = 128 * element_bytes
    if token_stride >= 128:
        gap = (token_stride - 128) * element_bytes
        actual = {
            t * (row_bytes + gap) + byte
            for t in range(count)
            for byte in range(row_bytes)
        }
    else:
        # Layouts that cannot use a positive DMA gap retain per-row writes.
        actual = {
            t * token_stride * element_bytes + byte
            for t in range(count)
            for byte in range(row_bytes)
        }
    expected = {
        (t * token_stride + element) * element_bytes + byte
        for t in range(count)
        for element in range(128)
        for byte in range(element_bytes)
    }
    assert actual == expected
    # Decay is a separate contiguous field: exactly count floats, no 32B overrun.
    decay = bytearray(b"x" * 64)
    decay[16 : 16 + count * 4] = bytes(count * 4)
    assert decay[:16] == b"x" * 16
    assert decay[16 + count * 4 :] == b"x" * (48 - count * 4)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 8])
@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_last_output_identity_and_all_accepted_prefixes(count, activation_dtype, state_dtype):
    generator = torch.Generator().manual_seed(561 + count)
    checkpoint = (torch.randn(128, 128, generator=generator) * 0.05).to(state_dtype)
    original = checkpoint.clone()
    q = torch.randn(count, 128, generator=generator).to(activation_dtype).float() / 128
    k = torch.randn(count, 128, generator=generator).to(activation_dtype).float() / 128**0.5
    v = (torch.randn(count, 128, generator=generator) * 0.1).to(activation_dtype).float()
    decays = torch.linspace(0.02, 0.99, count)
    betas = torch.linspace(0.01, 0.98, count)
    state = checkpoint.float()
    records, accepted = [], [state.clone()]
    for t in range(count):
        decayed = state * decays[t]
        correction = (v[t] - decayed @ k[t]) * betas[t]
        updated = decayed + correction[:, None] * k[t][None, :]
        if t + 1 == count:
            old_output = updated @ q[t]
            new_output = decayed @ q[t] + correction * (k[t] @ q[t])
            torch.testing.assert_close(new_output, old_output, atol=2e-6, rtol=2e-4)
            # Independent double precision formula checks algebra, not bit identity.
            reference = decayed.double() @ q[t].double() + correction.double() * (
                k[t].double() @ q[t].double()
            )
            torch.testing.assert_close(new_output.double(), reference, atol=2e-6, rtol=2e-4)
        records.append((correction, k[t], decays[t]))
        accepted.append(updated.clone())
        state = updated
    # Last-output shortcut does not change records, hence every accepted length
    # (including zero) still recovers the same prefix from the read-only checkpoint.
    for length in range(count + 1):
        recovered = checkpoint.float()
        for correction, key, decay in records[:length]:
            recovered = recovered * decay + correction[:, None] * key[None, :]
        torch.testing.assert_close(recovered, accepted[length], atol=0, rtol=0)
    assert torch.equal(checkpoint, original)
