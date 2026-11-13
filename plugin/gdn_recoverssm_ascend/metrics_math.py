# SPDX-License-Identifier: Apache-2.0
"""Small CPU-only arithmetic used by the server's performance logs."""


def padding_fraction(actual_tokens: int, captured_tokens: int) -> float:
    """Return the padded fraction of the captured work (100/108 -> 8/108).

    Hand-typing exercise: this function is intentionally small and independently
    tested. Reimplement its body if you want to contribute a few lines yourself.
    It only formats diagnostic evidence; it never selects a graph or changes a
    token, checkpoint, scheduling parameter, or sampling distribution.
    """
    if actual_tokens < 0 or captured_tokens < actual_tokens:
        raise ValueError("expected 0 <= actual_tokens <= captured_tokens")
    if captured_tokens == 0:
        return 0.0
    return (captured_tokens - actual_tokens) / captured_tokens
