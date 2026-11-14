#!/usr/bin/env python3
"""Reproduce/check the reviewed HMS engine extraction from a read-only checkout.

This is a maintainer tool, never a runtime dependency or an automatic updater.
--write overwrites the explicitly selected port files; review changes first.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REVISION = "ba1eac18146f017f8cc3cae6acdb7963b8f4149e"
URL = "https://github.com/aaiauea/hidden-markov-singer"
ROOTS = ("hms.core.synthesizer", "hms.core.trainer")
EXCLUDED_TEST_FILES = {"test_cli.py", "test_evaluate.py", "test_minimal_example.py"}
EXCLUDED_TEST_IMPORTS = {"hms.cli.main", "hms.config", "hms.core.evaluate"}
PROJECT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(source, *args):
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, timeout=15,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()


def imports(text):
    """Adapt import statements only; on-disk 'hms.source.*' format IDs stay intact."""
    text = re.sub(r"(?m)^([ \t]*from[ \t]+)hms(?=[. \t])", r"\1hmvc.engine", text)
    return re.sub(r"(?m)^([ \t]*import[ \t]+)hms(?=[. \t]|$)", r"\1hmvc.engine", text)


def header(path, marker="#"):
    return (f"{marker} Derived from {URL}, revision {REVISION}, {path}.\n"
            f"{marker} MIT License; Copyright (c) 2026 aaiauea. See LICENSE.hms / THIRD_PARTY_NOTICES.md.\n")


def module_graph(source):
    modules = {}
    for path in sorted((source / "hms").rglob("*.py")):
        if "tests" in path.relative_to(source).parts:
            continue
        parts = path.relative_to(source).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules[name] = path
    graph = {}
    for name, path in modules.items():
        pieces = name.split(".")
        dependencies = {".".join(pieces[:i]) for i in range(1, len(pieces))}
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in modules:
                    dependencies.add(node.module)
                for alias in node.names:
                    candidate = node.module + "." + alias.name
                    if candidate in modules:
                        dependencies.add(candidate)
            elif isinstance(node, ast.Import):
                dependencies.update(a.name for a in node.names if a.name in modules)
        graph[name] = sorted(dependencies - {name})
    closure = set(ROOTS)
    while True:
        expanded = closure | {dep for name in closure for dep in graph[name]}
        if expanded == closure:
            return modules, graph, closure
        closure = expanded


def prune_tests(text):
    """Omit only tests/helpers requiring non-engine CLI/config/evaluation APIs."""
    tree = ast.parse(text)
    removed_imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                       and node.module in EXCLUDED_TEST_IMPORTS]
    bad = {alias.asname or alias.name for node in removed_imports for alias in node.names}
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    while True:
        expanded = bad | {name for name, node in functions.items()
                          if any(isinstance(child, ast.Name) and child.id in bad
                                 for child in ast.walk(node))}
        if expanded == bad:
            break
        bad = expanded
    removed = removed_imports + [node for name, node in functions.items() if name in bad]
    lines = text.splitlines(keepends=True)
    for node in removed:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        for index in range(start, node.end_lineno):
            lines[index] = ""
    return "".join(lines), sorted(set(functions) & bad)


def build_port(source):
    modules, graph, closure = module_graph(source)
    if len(closure) != 35:
        raise ValueError(f"Reviewed closure has 35 Python files; found {len(closure)}")
    files, records, omitted = {}, [], {}

    def add(upstream, destination, text=None, category="engine", dependencies=None):
        original = (source / upstream).read_bytes()
        content = original if text is None else text.encode("utf-8")
        files[destination] = content
        record = {"upstream": upstream, "local": destination, "category": category,
                  "upstream_sha256": digest(original), "local_sha256": digest(content)}
        if dependencies is not None:
            record["internal_imports_including_deferred"] = dependencies
        records.append(record)

    for name in sorted(closure):
        path = modules[name]
        relative = path.relative_to(source).as_posix()
        text = imports(path.read_text(encoding="utf-8"))
        if relative == "hms/vocoder/world_native.py":
            old = ('    root = Path(__file__).resolve().parents[2]\n'
                   '    script = root / "tools" / "build_world.sh"')
            if text.count(old) != 1:
                raise ValueError("Native helper path changed upstream; review required")
            text = text.replace(old, '    script = _NATIVE_DIR / "build_world.sh"')
            text = text.replace("tools/build_world.sh", "vocoder/_native/build_world.sh")
        add(relative, "src/hmvc/engine/" + relative.removeprefix("hms/"),
            header(relative) + text, dependencies=graph[name])

    asset = "hms/config/phonemes.yaml"
    add(asset, "src/hmvc/engine/config/phonemes.yaml", category="required default resource")
    script = "tools/build_world.sh"
    text = (source / script).read_text(encoding="utf-8")
    text = text.replace('root="$(cd "$here/.." && pwd)"\n', "")
    text = text.replace('out_dir="${1:-$root/hms/vocoder/_native}"', 'out_dir="${1:-$here}"')
    text = text.replace('$here/world_native/World', '$here/World')
    text = text.replace('$here/world_native/hms_world_capi.cpp', '$here/hms_world_capi.cpp')
    text = text.replace('tools/world_native/hms_world_capi.cpp', 'hms_world_capi.cpp')
    text = text.replace('tools/world_native/World', 'World (beside this script)')
    text = text.replace('tools/build_world.sh [output_dir]', 'bash build_world.sh [output_dir]')
    first, rest = text.split("\n", 1)
    add(script, "src/hmvc/engine/vocoder/_native/build_world.sh",
        first + "\n" + header(script) + rest, category="optional native build support")
    capi = "tools/world_native/hms_world_capi.cpp"
    add(capi, "src/hmvc/engine/vocoder/_native/hms_world_capi.cpp",
        header(capi, "//") + (source / capi).read_text(encoding="utf-8"),
        category="optional native build support")
    add("LICENSE", "src/hmvc/engine/LICENSE.hms", category="MIT notice")

    # DemoSinger is synthetic test input generation ONLY, never shipped in hmvc.
    demo = "hms/data/demo_singer.py"
    add(demo, "tests/_support/demo_singer.py", header(demo) + imports((source / demo).read_text(encoding="utf-8")),
        category="test-only synthetic fixture generator")
    for path in sorted((source / "hms/tests").glob("*.py")):
        relative = path.relative_to(source).as_posix()
        if path.name in EXCLUDED_TEST_FILES:
            omitted[relative] = [node.name for node in ast.parse(path.read_text()).body
                                 if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
            continue
        text = path.read_text(encoding="utf-8")
        text, names = prune_tests(text)
        if names:
            omitted[relative] = names
        text = text.replace("from hms.data.demo_singer import", "from tests._support.demo_singer import")
        text = imports(text)
        # These are monkeypatch import paths, NOT model-format identifiers.
        for old in ("hms.core.synthesizer.match_global_variance",
                    "hms.core.synthesizer.optimize_global_variance",
                    "hms.core.trainer.wavio.read_wav"):
            text = text.replace('"' + old + '"', '"hmvc.engine.' + old.removeprefix("hms.") + '"')
        if path.name == "conftest.py":
            shim = ('ROOT = Path(__file__).resolve().parents[2]\n'
                    'if str(ROOT) not in sys.path:                     # `pytest` from a checkout\n'
                    '    sys.path.insert(0, str(ROOT))\n')
            if shim not in text:
                raise ValueError("Upstream test path shim changed; review required")
            text = text.replace(shim, "")
        add(relative, "tests/upstream/" + path.name, header(relative) + text, category="ported upstream tests")

    manifest = {
        "repository": URL, "revision": REVISION, "upstream_engine_version": "0.1.0",
        "roots": list(ROOTS), "python_engine_files": len(closure),
        "namespace": "hmvc.engine", "runtime_depends_on_hms": False,
        "adaptations": [
            "Import statements: hms -> hmvc.engine; original class/function names unchanged",
            "world_native._try_build resolves package-local _native/build_world.sh",
            "Native shell helper: package-local output, WORLD source, and C ABI shim paths",
            "MIT/provenance headers; non-import strings including model format IDs unchanged",
            "Tests: namespace and three monkeypatch paths; synthetic generator test-only; remove checkout sys.path shim",
            "Omit CLI/config-loader/evaluation/example tests; keep applicable engine assertions unchanged",
        ],
        "excluded_runtime_modules": sorted(set(modules) - closure),
        "omitted_tests_and_helpers": omitted,
        "files": sorted(records, key=lambda row: row["local"]),
    }
    files["src/hmvc/engine/UPSTREAM.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return files, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pristine, separate HMS checkout")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Write the reviewed extraction into HMVC")
    mode.add_argument("--check", action="store_true", help="Compare local port with the deterministic adaptations")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if git(source, "rev-parse", "HEAD") != REVISION:
        parser.error("HMS revision differs from the reviewed pin")
    status_args = ("status", "--porcelain", "--untracked-files=all", "--ignored=matching")
    if git(source, *status_args):
        parser.error("HMS must be pristine, including no caches/build artifacts")
    before = {p.relative_to(source).as_posix(): digest(p.read_bytes())
              for p in source.rglob("*") if p.is_file() and ".git" not in p.relative_to(source).parts}
    files, manifest = build_port(source)
    different = []
    for relative, data in files.items():
        target = PROJECT / relative
        if args.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if target.suffix == ".sh":
                target.chmod(0o755)
        elif not target.is_file() or target.read_bytes() != data:
            different.append(relative)
    after = {p.relative_to(source).as_posix(): digest(p.read_bytes())
             for p in source.rglob("*") if p.is_file() and ".git" not in p.relative_to(source).parts}
    if before != after or git(source, *status_args):
        raise RuntimeError("The original HMS checkout changed")
    if different:
        print("Port differs from reviewed extraction:\n" + "\n".join(different))
        return 1
    print(f"{'Wrote' if args.write else 'Verified'} {len(files)} selected engine/notice/test files; "
          f"{manifest['python_engine_files']} runtime Python files; original HMS unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
