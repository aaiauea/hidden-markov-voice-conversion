"""Local engine imports, provenance, resource packaging, and dependency boundaries."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import hmvc.engine as engine

ENGINE = Path(engine.__file__).resolve().parent
MANIFEST = json.loads((ENGINE / "UPSTREAM.json").read_text(encoding="utf-8"))
MODULES = []
for row in MANIFEST["files"]:
    if row["category"] == "engine" and row["local"].endswith(".py"):
        parts = Path(row["local"]).relative_to("src").with_suffix("").parts
        MODULES.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))


@pytest.mark.parametrize("name", MODULES)
def test_module_import_in_fresh_interpreter_outside_checkout(name, tmp_path):
    # -I ignores PYTHONPATH and the current directory. This works both for an
    # editable install and for the wheel installed in a completely clean venv.
    code = '''
import importlib, json, sys
from pathlib import Path
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'hms','torch','librosa','scipy','soundfile','pyworld','pysptk','numba','sklearn'}:
            raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
sys.meta_path.insert(0, Guard())
module = importlib.import_module(sys.argv[1])
import hmvc.engine
root = Path(hmvc.engine.__file__).resolve().parent
assert Path(module.__file__).resolve().is_relative_to(root)
assert not any(n == 'hms' or n.startswith('hms.') for n in sys.modules)
print(json.dumps({'module': module.__name__}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, name],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30,
                            env=dict(os.environ, HMS_NO_AUTO_BUILD="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["module"] == name


def test_complete_engine_closure_and_no_original_runtime_imports():
    assert len(MODULES) == MANIFEST["python_engine_files"] == 35
    assert MANIFEST["runtime_depends_on_hms"] is False
    assert {"hmvc.engine.core.synthesizer", "hmvc.engine.core.trainer"} <= set(MODULES)
    forbidden = {"hms", "torch", "librosa", "scipy", "soundfile", "numba", "pysptk", "sklearn"}
    for path in ENGINE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, path
            if isinstance(node, ast.Import):
                assert not {a.name.split(".")[0] for a in node.names} & forbidden, path
    for name in ("cli", "core.evaluate", "data.demo_singer"):
        assert importlib.util.find_spec("hmvc.engine." + name) is None
    assert not (ENGINE / "config/__init__.py").exists()
    requirements = importlib.metadata.requires("hidden-markov-voice-conversion")
    runtime = {r.split(">=")[0] for r in requirements if "extra ==" not in r}
    assert runtime == {"numpy", "PyYAML"}


def test_reviewed_engine_files_and_mit_notice_are_intact():
    for record in MANIFEST["files"]:
        if not record["local"].startswith("src/hmvc/engine/"):
            continue
        path = ENGINE / record["local"].removeprefix("src/hmvc/engine/")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["local_sha256"], path
    notice = (ENGINE / "LICENSE.hms").read_text(encoding="utf-8")
    assert "MIT License" in notice and "Copyright (c) 2026 aaiauea" in notice
    assert engine.__version__ == "0.1.0"  # upstream version retained in voice metadata


def test_default_resource_and_original_model_identifiers(tmp_path):
    from hmvc.engine.core.model import MODEL_FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS
    from hmvc.engine.core.phonemes import PhonemeSet
    from hmvc.engine.source.hmm import SOURCE_MODEL_FORMAT, SOURCE_MODEL_FORMAT_VERSION
    from hmvc.engine.source.pca import SourcePCA
    assert PhonemeSet.default().resolve("a").type == "vowel"
    assert MODEL_FORMAT_VERSION == 4 and SUPPORTED_FORMAT_VERSIONS == (2, 3, 4)
    assert (SOURCE_MODEL_FORMAT, SOURCE_MODEL_FORMAT_VERSION) == ("hms.source.hmm", 1)
    pca = SourcePCA.fit(np.arange(32.0).reshape(4, 8), n_components=2)
    pca.save(tmp_path / "pca.npz")
    with np.load(tmp_path / "pca.npz", allow_pickle=False) as data:
        assert json.loads(str(data["meta"]))["format"] == "hms.source.pca"


def test_native_support_paths_are_package_local_without_building(monkeypatch):
    from hmvc.engine.vocoder import world_native
    native = ENGINE / "vocoder/_native"
    assert (native / "build_world.sh").is_file()
    assert (native / "hms_world_capi.cpp").is_file()
    monkeypatch.delenv("HMS_NO_AUTO_BUILD", raising=False)
    monkeypatch.setattr(world_native, "_find_library", lambda: None)
    calls = []
    monkeypatch.setattr(world_native.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    assert world_native._try_build(timeout=1) is None
    assert calls == [["bash", str(native / "build_world.sh")]]


def test_named_pcm16_io_and_inherited_wav_limitations(tmp_path):
    from hmvc.engine.data import wavio
    samples = np.array([0.0, 0.25, -0.25, 0.5, -0.5])
    wavio.write_wav(tmp_path / "pcm16.wav", samples, 16000)
    actual, fs = wavio.read_wav(tmp_path / "pcm16.wav")
    assert fs == 16000
    np.testing.assert_allclose(actual, samples, atol=1e-4)
    # Preserve and disclose existing behavior rather than slipping in a DSP fix.
    with pytest.raises(AttributeError, match="struct"):
        exec("from hmvc.engine.data.wavio import *", {})
    wavio.write_wav(tmp_path / "float32.wav", samples, 16000, bit_depth=32)
    actual, _ = wavio.read_wav(tmp_path / "float32.wav")
    assert not np.allclose(actual, samples, atol=1e-4)
