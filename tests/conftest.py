"""Keep all regression checks independent of HMS and optional heavy backends."""
from __future__ import annotations

import os
import sys

# Preserve production backend behavior; disable downloads/builds in tests only.
os.environ["HMS_NO_AUTO_BUILD"] = "1"

FORBIDDEN = {"hms", "torch", "librosa", "scipy", "soundfile", "numba", "pysptk", "sklearn", "pyworld"}


class EngineDependencyGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise ModuleNotFoundError(
                f"{fullname} is excluded: the extracted engine must run independently",
                name=fullname.split(".")[0])


sys.meta_path.insert(0, EngineDependencyGuard())


def pytest_sessionstart(session):
    loaded = {name.split(".")[0] for name in sys.modules} & FORBIDDEN
    if loaded:
        raise RuntimeError(f"Start a clean test interpreter; excluded modules already loaded: {sorted(loaded)}")
