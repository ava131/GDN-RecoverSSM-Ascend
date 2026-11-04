# SPDX-License-Identifier: Apache-2.0
"""Static contract check for the Triton kernels in ``kernels.py``.

A missing ``tl.constexpr`` argument is not a Python error: it surfaces as a
``CompilationError`` inside ``kernel[grid](...)`` at the first launch, i.e. on the
machine, in the middle of a request. This test reads ``kernels.py`` as text --
no Triton, no NPU, no import of the plugin -- and asserts, for every kernel and
every launch site:

* every declared ``tl.constexpr`` parameter is supplied by the call (or is one of
  the names ``triton.heuristics`` injects),
* the call does not pass keywords the kernel does not declare,
* the number of positional arguments matches the non-constexpr parameters,
  unless the launch forwards ``*args`` (in which case the tuple is checked).

It exists because a missing ``NUM_BLOCKS`` constexpr in the commit-plan kernel
crashed a real run exactly this way.
"""

from __future__ import annotations

import ast
import pathlib

KERNELS = pathlib.Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend" / "kernels.py"

# Supplied by @triton.heuristics at launch time, so they never appear as keywords.
HEURISTIC_NAMES = {"HAS_REQUEST_INDICES", "ALIGN_MODE", "ALIGN"}
LAUNCH_OPTIONS = {"num_warps", "num_stages", "num_ctas"}


def _is_constexpr(annotation) -> bool:
    return isinstance(annotation, ast.Attribute) and annotation.attr == "constexpr"


def _collect(tree: ast.AST):
    kernels: dict[str, tuple[list[str], list[str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_kernel"):
            params = [a.arg for a in node.args.args]
            constexpr = [
                a.arg for a in node.args.args if _is_constexpr(a.annotation)
            ]
            kernels[node.name] = (params, constexpr)

    launches: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in kernels
        ):
            launches.append((node.func.value.id, node))
    return kernels, launches


def check() -> list[str]:
    tree = ast.parse(KERNELS.read_text())
    kernels, launches = _collect(tree)
    problems: list[str] = []

    assert kernels, "no kernels found — did the file move?"
    assert launches, "no kernel launch sites found"

    for name, call in launches:
        params, constexpr = kernels[name]
        keywords = {k.arg for k in call.keywords if k.arg}
        starred = any(isinstance(a, ast.Starred) for a in call.args)

        required = [c for c in constexpr if c not in HEURISTIC_NAMES]
        missing = [c for c in required if c not in keywords]
        if missing:
            problems.append(f"{name}: missing constexpr argument(s) {missing}")

        undeclared = keywords - set(constexpr) - LAUNCH_OPTIONS
        if undeclared:
            problems.append(f"{name}: undeclared keyword(s) {sorted(undeclared)}")

        if not starred:
            positional = len([a for a in call.args if not isinstance(a, ast.Starred)])
            expected = len(params) - len(constexpr)
            if positional != expected:
                problems.append(
                    f"{name}: {positional} positional arguments, kernel declares "
                    f"{expected} non-constexpr parameters"
                )
    return problems


def check_local_use_before_assignment() -> list[str]:
    """Catch ``UnboundLocalError`` patterns statically.

    A name that is assigned somewhere in a function is local for the whole
    function, so reading it before the first assignment raises at runtime. That
    is exactly how ``num_work`` crashed the commit path. Only names that are
    assigned *inside* the same function are considered, so outer-scope and
    imported names are ignored.
    """
    tree = ast.parse(KERNELS.read_text())
    problems: list[str] = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        assigned: dict[str, int] = {}
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name) and leaf.id not in assigned:
                        assigned[leaf.id] = node.lineno
            # ``for`` / comprehension targets bind in their own scope, so they
            # are not function-local in the sense that matters here.
        params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        loads: dict[str, int] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loads.setdefault(node.id, node.lineno)
        for name, load_line in loads.items():
            if name in params or name not in assigned:
                continue
            if load_line < assigned[name]:
                problems.append(
                    f"{fn.name}: '{name}' read at line {load_line} but first "
                    f"assigned at line {assigned[name]} (UnboundLocalError)"
                )
    return problems


def test_kernel_launch_contract() -> None:
    problems = check() + check_local_use_before_assignment()
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    issues = check() + check_local_use_before_assignment()
    for issue in issues:
        print("FAIL", issue)
    assert not issues, "kernel launch contract violated"
    print("kernel launch contract OK")
