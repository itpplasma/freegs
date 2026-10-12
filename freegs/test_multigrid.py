"""
Tests of the multigrid restriction operator
"""

import numpy as np

from . import multigrid


def _grid(n):
    """Index coordinates i, j on an n x n grid (unit spacing)"""
    i, j = np.meshgrid(
        np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij"
    )
    return i, j


def _interior(a):
    return a[1:-1, 1:-1]


def test_restrict_constant():
    """Full weighting weights sum to one, so a constant is preserved"""
    fine = np.full((17, 17), 3.5)

    coarse = multigrid.restrict(fine)

    assert coarse.shape == (9, 9)
    assert np.allclose(_interior(coarse), 3.5, rtol=0.0, atol=1e-12)


def test_restrict_linear_exact():
    """Full weighting reproduces linear functions exactly"""
    i, j = _grid(17)
    fine = 2.0 + 0.3 * i - 0.7 * j

    coarse = multigrid.restrict(fine)

    ci, cj = _grid(9)
    expected = 2.0 + 0.3 * (2 * ci) - 0.7 * (2 * cj)
    assert np.allclose(_interior(coarse), _interior(expected), rtol=0.0, atol=1e-12)


def test_restrict_quadratic_error():
    """
    The 1D marginal of the 9-point stencil has weights (1/4, 1/2, 1/4) at
    offsets (-1, 0, 1), so for f = i**2 the restriction error is exactly 1/2
    in fine-grid index units squared, independent of position
    """
    i, _ = _grid(17)
    fine = i**2

    coarse = multigrid.restrict(fine)

    ci, _ = _grid(9)
    assert np.allclose(_interior(coarse), _interior((2 * ci) ** 2 + 0.5), rtol=1e-13)


def test_restrict_smooth_second_order():
    """Restriction of a smooth function converges at second order"""

    def error(n):
        i, j = _grid(n)
        h = 1.0 / (n - 1)
        x, y = i * h, j * h
        fine = np.sin(2.0 * x) * np.cos(3.0 * y)
        coarse = multigrid.restrict(fine)
        ci, cj = _grid((n - 1) // 2 + 1)
        hc = 2.0 * h
        xc, yc = ci * hc, cj * hc
        exact = np.sin(2.0 * xc) * np.cos(3.0 * yc)
        return np.max(np.abs(_interior(coarse) - _interior(exact)))

    e_coarse = error(17)
    e_fine = error(33)

    assert e_fine < e_coarse
    assert 3.5 < e_coarse / e_fine < 4.5
