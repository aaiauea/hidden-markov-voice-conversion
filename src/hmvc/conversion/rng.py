"""Seeded 64-bit LCG — the reproducible engine used for the vocoder's noise.

The original HTML calls ``Math.random()`` twice per spectral bin (once for
the magnitude, once for the phase).  That is intentionally nondeterministic.
For reproducibility (tests, reference comparison) this package accepts an
optional integer seed, which drives a 64-bit linear congruential generator
(``state = state * 6364136223846793005 + 1442695040888963407 (mod 2^64)``)
and returns ``float(state >> 11) / 2**53``.  The exact same constants are used
by ``tools/generate_conversion_reference.js`` so the Node reference output is
bit-compatible with the Python port when seeded.
"""

from __future__ import annotations

_MASK64 = (1 << 64) - 1
_MULTIPLIER = 6364136223846793005
_INCREMENT = 1442695040888963407
_SCALE = 1.0 / 9007199254740992.0  # 1 / (2 ** 53)


class LcgRandom:
    """Deterministic replacement for ``Math.random()`` (returns [0, 1))."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        state = int(seed) & _MASK64
        if state == 0:
            state = 1
        self._state = state

    def random(self) -> float:
        self._state = (self._state * _MULTIPLIER + _INCREMENT) & _MASK64
        return float(self._state >> 11) * _SCALE


def make_rng(seed):
    """Return an object mimicking ``Math.random()``.

    With ``seed`` set, a deterministic :class:`LcgRandom` is returned; with
    ``seed=None`` a process-local random generator is returned, exactly like
    the original browser (nondeterministic, unseeded) behavior.
    """
    if seed is None:
        import random

        return random.Random()
    return LcgRandom(seed)
