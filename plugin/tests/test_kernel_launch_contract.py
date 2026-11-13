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
import sys

KERNELS = (
    pathlib.Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend" / "kernels.py"
)
SOURCES = (KERNELS, KERNELS.with_name("grouped_commit.py"))

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
            constexpr = [a.arg for a in node.args.args if _is_constexpr(a.annotation)]
            kernels[node.name] = (params, constexpr)

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def owner(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return tree

    def resolve(value, call):
        seen = set()
        while isinstance(value, ast.Name) and value.id not in seen:
            seen.add(value.id)
            matches = [
                n
                for n in ast.walk(owner(call))
                if isinstance(n, ast.Assign)
                and owner(n) is owner(call)
                and n.lineno < call.lineno
                and any(isinstance(t, ast.Name) and t.id == value.id for t in n.targets)
            ]
            if not matches:
                break
            value = max(matches, key=lambda n: n.lineno).value
        return value

    launches: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in kernels
        ):
            launches.append((node.func.value.id, node))
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_LaunchSpec"
        ):
            continue
        fields = {kw.arg: kw.value for kw in node.keywords}
        kernel = fields.get("kernel")
        if not isinstance(kernel, ast.Name) or kernel.id not in kernels:
            continue
        args = resolve(fields["args"], node)
        options = resolve(fields["kwargs"], node)
        assert isinstance(args, (ast.Tuple, ast.List)), (
            f"{kernel.id}: cannot resolve LaunchSpec args at line {node.lineno}"
        )
        if isinstance(options, ast.Dict):
            assert all(
                isinstance(k, ast.Constant) and isinstance(k.value, str)
                for k in options.keys
            ), "LaunchSpec kwargs need explicit names"
            keywords = [
                ast.keyword(arg=k.value, value=v)
                for k, v in zip(options.keys, options.values)
            ]
        else:
            assert (
                isinstance(options, ast.Call)
                and isinstance(options.func, ast.Name)
                and options.func.id == "dict"
            ), "cannot resolve LaunchSpec kwargs"
            keywords = options.keywords
        call = ast.Call(func=kernel, args=args.elts, keywords=keywords)
        call.lineno = node.lineno
        launches.append((kernel.id, call))
    return kernels, launches


def check() -> list[str]:
    kernels, launches = {}, []
    for path in SOURCES:
        found, calls = _collect(ast.parse(path.read_text()))
        kernels.update(found)
        launches.extend(calls)
    problems: list[str] = []

    assert kernels, "no kernels found — did the file move?"
    assert launches, "no kernel launch sites found"
    assert set(kernels) == {name for name, _ in launches}, (
        "every declared kernel must have a checked direct or LaunchSpec call"
    )

    for name, call in launches:
        params, constexpr = kernels[name]
        keywords = {k.arg for k in call.keywords if k.arg}
        starred = any(isinstance(a, ast.Starred) for a in call.args)
        if any(
            isinstance(a, (ast.Tuple, ast.List))
            or (
                isinstance(a, ast.Call)
                and isinstance(a.func, ast.Name)
                and a.func.id == "tuple"
            )
            for a in call.args
        ):
            problems.append(
                f"{name}: runtime tuple arguments are unsupported by Triton-Ascend 3.2.2"
            )

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


def check_no_tuple_values_in_kernels() -> list[str]:
    """A triton kernel may not hold a tuple in a variable on this toolchain.

    Triton-Ascend 3.2.2 fails on both shapes we tried for a staged window:

        staged = (load(t0), load(t1), ...)   # TypeError: cannot convert tuple
        staged = (q0, q1, ...); staged[t]    # ValueError: unsupported tensor index

    So, inside a kernel: a tuple literal may not be assigned to a name (only
    unpacking into names is allowed), and a name bound to a tuple may not be
    subscripted. Both are checked statically; only the NPU compiler would catch
    them otherwise.
    """
    tree = ast.parse(KERNELS.read_text())
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_kernel = any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "jit"
            for d in node.decorator_list
        ) or node.name.endswith("_kernel")
        if not is_kernel:
            continue
        tuple_names: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Tuple):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Name):
                        tuple_names.add(tgt.id)
                        problems.append(
                            f"{node.name}: binds a tuple to {tgt.id!r} "
                            f"(line {sub.lineno})"
                        )
            if isinstance(sub, ast.Tuple):
                for elt in sub.elts:
                    if isinstance(elt, ast.Tuple):
                        problems.append(
                            f"{node.name}: nested tuple literal at line {elt.lineno}"
                        )
            if (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in tuple_names
            ):
                problems.append(
                    f"{node.name}: indexes tuple {sub.value.id!r} "
                    f"(line {sub.lineno})"
                )
    return problems


def check_dataclass_field_order() -> list[str]:
    """No dataclass field with a default may precede one without.

    Adding ``record_rows: int = 0`` in the middle of GDNRecoverSSMCommitContext
    made every worker die at import time with
    "TypeError: non-default argument 'num_blocks' follows default argument".
    """
    tree = ast.parse(KERNELS.read_text())
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(d, ast.Name)
            and d.id == "dataclass"
            or isinstance(d, ast.Call)
            and getattr(d.func, "id", "") == "dataclass"
            for d in node.decorator_list
        ):
            continue
        seen_default = None
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(
                item.target, ast.Name
            ):
                continue
            has_default = item.value is not None
            if has_default and seen_default is None:
                seen_default = item.target.id
            elif not has_default and seen_default is not None:
                problems.append(
                    f"{node.name}: {item.target.id} (no default) follows "
                    f"{seen_default} (default)"
                )
    return problems


def test_kernel_launch_contract() -> None:
    problems = (
        check()
        + check_local_use_before_assignment()
        + check_dataclass_field_order()
        + check_no_tuple_values_in_kernels()
    )
    assert not problems, "\n".join(problems)


def test_launchspec_missing_constexpr_is_detected(tmp_path, monkeypatch):
    tree = ast.parse(KERNELS.read_text())
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_LaunchSpec"
        ):
            continue
        fields = {kw.arg: kw.value for kw in node.keywords}
        if getattr(fields.get("kernel"), "id", None) != "_gdn_recoverssm_verify_kernel":
            continue
        options = fields["kwargs"]
        index = next(
            i for i, key in enumerate(options.keys) if key.value == "NUM_BLOCKS"
        )
        options.keys.pop(index)
        options.values.pop(index)
        break
    else:
        raise AssertionError("verify LaunchSpec was not found")
    modified = tmp_path / "kernels.py"
    modified.write_text(ast.unparse(tree))
    monkeypatch.setattr(sys.modules[__name__], "SOURCES", (modified, SOURCES[1]))
    assert any("missing constexpr" in p and "NUM_BLOCKS" in p for p in check())


def test_dataclass_guard_is_wired_into_main_test(monkeypatch):
    import pytest

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "check", lambda: [])
    monkeypatch.setattr(module, "check_local_use_before_assignment", lambda: [])
    monkeypatch.setattr(
        module, "check_dataclass_field_order", lambda: ["bad field order"]
    )
    with pytest.raises(AssertionError, match="bad field order"):
        test_kernel_launch_contract()


if __name__ == "__main__":
    issues = (
        check() + check_local_use_before_assignment() + check_dataclass_field_order()
    )
    for issue in issues:
        print("FAIL", issue)
    assert not issues, "kernel launch contract violated"
    print("kernel launch contract OK")
