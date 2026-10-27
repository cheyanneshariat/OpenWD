"""Humlicek (1982) W4 approximation to the Faddeeva function."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def humlicek_w4(x: ArrayLike, y: ArrayLike) -> NDArray[np.complex128]:
    """Return ``w(x + i y) = exp(-z^2) erfc(-i z)`` for ``y >= 0``.

    Humlicek (1982, JQSRT 27, 437) region algorithm W4, accurate to about
    1e-4 relative. The real part is the Voigt function; the imaginary part is
    its causal (magneto-optical) partner. The compiled line kernel uses the
    same four regions and coefficients.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.broadcast_to(np.asarray(y, dtype=np.float64), x.shape)
    t = y - 1j * x
    s = np.abs(x) + y
    w = np.empty(x.shape, dtype=np.complex128)
    region1 = s >= 15.0
    region2 = (s >= 5.5) & ~region1
    region3 = (s < 5.5) & (y >= 0.195 * np.abs(x) - 0.176)
    region4 = ~(region1 | region2 | region3)
    if np.any(region1):
        tt = t[region1]
        w[region1] = tt * 0.5641896 / (0.5 + tt * tt)
    if np.any(region2):
        tt = t[region2]
        u = tt * tt
        w[region2] = tt * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))
    if np.any(region3):
        tt = t[region3]
        w[region3] = (
            16.4955 + tt * (20.20933 + tt * (11.96482 + tt * (3.778987 + tt * 0.5642236)))
        ) / (
            16.4955 + tt * (38.82363 + tt * (39.27121 + tt * (21.69274 + tt * (6.699398 + tt))))
        )
    if np.any(region4):
        tt = t[region4]
        u = tt * tt
        w[region4] = np.exp(u) - tt * (
            36183.31 - u * (3321.9905 - u * (1540.787 - u * (219.0313 - u * (
                35.76683 - u * (1.320522 - u * 0.56419)))))
        ) / (
            32066.6 - u * (24322.84 - u * (9022.228 - u * (2186.181 - u * (
                364.2191 - u * (61.57037 - u * (1.841439 - u))))))
        )
    return w
