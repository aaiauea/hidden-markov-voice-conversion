"""In-place radix-2 iterative FFT — a line-for-line port of the HTML original.

The original JavaScript FFT:

* uses a float32 lookup table for cos/sin (stored as ``Float32Array``),
* swaps by bit-reversal,
* iterates over decimation lengths 2, 4, ..., N,
* computes every butterfly in float64 (JavaScript reads ``Float32Array``
  elements as float64) and rounds every stored value to float32,
* divides by ``N`` on inverse (in place).

The :func:`fft` implementation below reproduces the same algorithm.  Because
the butterflies in one stage read and write disjoint indices, the per-stage
float32 store-back is arithmetic-equivalent to the original's per-butterfly
rounding (every original butterfly stores every result to a Float32Array,
which is the same rounding applied to the same stage-complete buffer).
Everything else — bit-reversal, table construction, per-stage arithmetic,
inverse scaling — is identical.
"""

from __future__ import annotations

import numpy as np

from hmvc.conversion.constants import N

#: float32 tables, as in the original.
cosT = np.array([np.cos(2.0 * np.pi * i / N) for i in range(N // 2)], dtype=np.float32)
sinT = np.array([-np.sin(2.0 * np.pi * i / N) for i in range(N // 2)], dtype=np.float32)
def _bit_reversals():
    out = np.zeros(N, dtype=np.int32)
    for i in range(N):
        r = 0
        for j in range(10):
            r = (r << 1) | ((i >> j) & 1)
        out[i] = r
    return out


_REV = _bit_reversals()

#: Hann window (.5 - .5 * cos(2 pi i / N)), stored as float32, as in the original.
WIN = np.array([0.5 - 0.5 * np.cos(2.0 * np.pi * i / N) for i in range(N)], dtype=np.float32)


def fft(re: np.ndarray, im: np.ndarray, inv: int) -> None:
    """In-place radix-2 Cooley-Tukey FFT of the interleaved buffers.

    Replicates the original implementation exactly (see module docstring).

    Parameters
    ----------
    re, im : np.ndarray
        ``(N,)`` float32 arrays, mutated in place.
    inv : int
        ``0`` for the forward transform, any nonzero value for the inverse
        (which additionally divides by ``N``).
    """
    # Bit-reversal swap (only j > i, in place).
    for i in range(N):
        j = _REV[i]
        if j > i:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]

    length = 2
    while length <= N:
        h = length >> 1
        st = N // length
        idx = np.arange(h)
        c = cosT[idx * st].astype(np.float64)
        s = (-sinT[idx * st] if inv else sinT[idx * st]).astype(np.float64)
        re_blk = re.reshape(-1, length)
        im_blk = im.reshape(-1, length)
        a = re_blk[:, :h].astype(np.float64)
        b = re_blk[:, h:].astype(np.float64)
        ai = im_blk[:, :h].astype(np.float64)
        bi = im_blk[:, h:].astype(np.float64)
        tr = b * c - bi * s
        ti = b * s + bi * c
        re_blk[:, :h] = a + tr
        re_blk[:, h:] = a - tr
        im_blk[:, :h] = ai + ti
        im_blk[:, h:] = ai - ti
        # Store back (float32 rounding, as the original Float32Array did).
        re_blk[:] = np.float32(re_blk)
        im_blk[:] = np.float32(im_blk)
        length <<= 1

    if inv:
        re[:] = np.float32(re / N)
        im[:] = np.float32(im / N)
