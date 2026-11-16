"""CPU proofs for the UB resources used by deferred output DMA.

These tests check a resource/lifetime contract, not device execution. In
particular they cannot establish Ascend instruction timing or performance.
The arithmetic oracle and deployment startup self-test remain independent.
"""

import re
from pathlib import Path

import pytest

CSRC = Path(__file__).parents[1] / "gdn_recoverssm_ascend" / "csrc"


def _constants(filename):
    """Evaluate the header's actual constant UB layout, with no CANN imports."""
    values = {}
    for name, expression in re.findall(
        r"static constexpr uint32_t (\w+) = ([^;]+);",
        (CSRC / filename).read_text(),
    ):
        expression = expression.replace("sizeof(float)", "4")
        expression = expression.replace("sizeof(half)", "2")
        for key in sorted(values, key=len, reverse=True):
            expression = re.sub(rf"\b{key}\b", str(values[key]), expression)
        assert re.fullmatch(r"[0-9\s+*()/-]+", expression), expression
        values[name] = eval(expression, {"__builtins__": {}}, {})
    return values


def _region(start, length):
    return (start, start + length)


def _overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _assert_disjoint(regions):
    for index, left in enumerate(regions):
        for right in regions[index + 1 :]:
            assert not _overlaps(left, right), (left, right)


def test_verify_combined_loads_have_disjoint_dma_destinations():
    c = _constants("verify.h")
    for state_bytes in (2, 4):
        state_start = c["kMatrixBytes"] if state_bytes == 2 else 0
        for activation_bytes in (2, 4):
            regions = [_region(state_start, c["kElements"] * state_bytes)]
            regions += [
                _region(c[name], c["kWindowElements"] * activation_bytes)
                for name in ("kRawOffset", "kRawKeyOffset", "kRawValueOffset")
            ]
            _assert_disjoint(regions)
            assert all(end <= c["kUbBytes"] for _, end in regions)


def test_host_launch_guard_covers_the_actual_device_layouts():
    common = (CSRC / "common.h").read_text()
    verify = _constants("verify.h")["kUbBytes"]
    fold = _constants("fold.h")["kUbBytes"]
    for name, actual in (("kVerifyUbBytes", verify), ("kFoldUbBytes", fold)):
        declared = int(re.search(rf"constexpr uint32_t {name} = (\d+);", common)[1])
        assert declared == actual
    assert verify >= fold
    assert "kRequiredUbBytes = kVerifyUbBytes + kUbGuardBytes" in common
    assert "ubBytes>=kRequiredUbBytes" in (CSRC / "ops.cpp").read_text()


@pytest.mark.parametrize("count", range(1, 9))
@pytest.mark.parametrize("activation_bytes,key_bytes", ((2, 2), (2, 4), (4, 4)))
def test_pending_token_dma_cannot_alias_later_vector_writes(
    count, activation_bytes, key_bytes
):
    """Worst case: MTE3 reads none of its sources until the final drain.

    Recurrence and output conversion may advance all Q tokens while every
    previous output remains in flight. Every future write must still be safe.
    """
    c = _constants("verify.h")
    width = c["kDim"]
    pending = []
    key_base = c["kRawKeyOffset"] if key_bytes == activation_bytes else c["kKOffset"]
    for token in range(count):
        writes = [
            _region(0, 2 * c["kMatrixBytes"]),
            _region(c["kBroadcastOffset"], width * 32),
            _region(c["kCorrectionOffset"] + token * width * 4, width * 4),
            _region(c["kOutputOffset"] + token * width * 4, width * 4),
            _region(
                c["kRawOffset"] + token * width * activation_bytes,
                width * activation_bytes,
            ),
            _region(c["kDecayOffset"] + token * 32, 32),
        ]
        for write in writes:
            assert write[1] <= c["kUbBytes"]
            assert not any(_overlaps(write, source) for source in pending), (
                token,
                write,
                pending,
            )
        pending.extend(
            [
                writes[2],
                writes[3] if activation_bytes == 4 else writes[4],
                _region(writes[5][0], 4),
                _region(key_base + token * width * key_bytes, width * key_bytes),
            ]
        )
    # A new task will overwrite the raw-key input allocation. The final MTE3
    # drain is therefore essential even though there is no wait between tokens.
    next_input = _region(c["kRawKeyOffset"], c["kWindowBytes"])
    if key_bytes == activation_bytes:
        assert any(_overlaps(next_input, source) for source in pending)


def test_headers_keep_real_waits_at_the_resource_reuse_boundaries():
    text = (CSRC / "verify.h").read_text()
    save = text.split("void SaveToken(", 1)[1].split("void RunTask(", 1)[0]
    task = text.split("void RunTask(", 1)[1].split("VerifyTiling td_;", 1)[0]
    assert "AfterOutput();" not in save
    assert "[t * kDim]" in save and "[t * 8]" in save
    assert task.count("InputReady();") == 1
    assert task.count("AfterOutput();") == 1
    assert task.index("InputReady();") < task.index("ConvertMatrix();")
    assert task.rindex("AfterOutput();") > task.rindex("SaveToken(")

    fold = (CSRC / "fold.h").read_text()
    store = fold.split("void StoreState(", 1)[1].split("void RunTask(", 1)[0]
    assert "HardEvent::MTE3_V" in store
    fold_task = fold.split("void RunTask(", 1)[1].split("FoldTiling td_;", 1)[0]
    assert fold_task.count("AwaitInput();") == 1
    assert fold_task.index("AwaitInput();") < fold_task.index("ConvertState();")
    assert fold_task.index("AwaitInput();") < fold_task.index("ConvertKeys(accepted);")


def test_both_ub_layouts_fit_a2_with_alignment_and_guard():
    for filename in ("verify.h", "fold.h"):
        constants = _constants(filename)
        assert constants["kUbBytes"] + 2048 <= 192 * 1024
        for name, value in constants.items():
            if name.endswith("Offset"):
                assert value % 32 == 0, (filename, name, value)


def test_fold_combined_loads_do_not_overlap_for_any_supported_dtype():
    c = _constants("fold.h")
    for state_bytes in (2, 4):
        for key_bytes in (2, 4):
            state_start = c["kIoOffset"] if state_bytes == 2 else 0
            key_start = c["kRawKeysOffset"] if key_bytes == 2 else c["kKeysOffset"]
            regions = [
                _region(state_start, c["kElements"] * state_bytes),
                _region(key_start, c["kWindow"] * c["kDim"] * key_bytes),
                _region(c["kCorrectionsOffset"], c["kWindowBytes"]),
            ]
            _assert_disjoint(regions)
            assert all(end <= c["kUbBytes"] for _, end in regions)
