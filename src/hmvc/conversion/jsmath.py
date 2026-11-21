"""JavaScript-exact scalar math primitives (``Math.log``/``Math.exp`` semantics).

The original code relies on JavaScript quirks: ``Math.log(0)`` is ``-Infinity``
(not an exception), ``Math.log(x < 0)`` is ``NaN``, and ``Math.exp`` overflows
to ``Infinity``.  These helpers replicate those semantics.
"""

from __future__ import annotations

import math


def js_log(x: float) -> float:
    """JavaScript ``Math.log`` — ``-inf`` for 0, ``NaN`` for negatives."""
    if x == 0.0:
        return -math.inf
    if x < 0.0:
        return math.nan
    return math.log(x)


def js_exp(x: float) -> float:
    """JavaScript ``Math.exp`` — ``Infinity`` on overflow."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def js_round(x: float) -> int:
    """JavaScript ``Math.round`` — half toward +inf."""
    return math.floor(x + 0.5)
