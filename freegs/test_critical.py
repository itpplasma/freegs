import numpy as np
import pytest
from scipy import interpolate

from . import critical


def test_one_opoint():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one O-point at (r0,z0) and no x-points
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / 0.3**2)

    opoints, xpoints = critical.find_critical(r2d, z2d, psi_func(r2d, z2d))

    assert len(xpoints) == 0
    assert len(opoints) == 1
    assert np.isclose(opoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(opoints[0][1], z0, atol=1.0 / ny)


def test_one_xpoint():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one X-point at (r0,z0) and no O-points
    def psi_func(R, Z):
        return (R - r0) ** 2 - (Z - z0) ** 2

    opoints, xpoints = critical.find_critical(r2d, z2d, psi_func(r2d, z2d))

    assert len(xpoints) == 1
    assert len(opoints) == 0
    assert np.isclose(xpoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(xpoints[0][1], z0, atol=1.0 / ny)


def test_doublet():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.1

    # This has two O-points, and one x-point at (r0, z0)
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0 - 0.3) ** 2) / 0.3**2) + np.exp(
            -((R - r0) ** 2 + (Z - z0 + 0.3) ** 2) / 0.3**2
        )

    opoints, xpoints = critical.find_critical(r2d, z2d, psi_func(r2d, z2d))

    assert len(xpoints) == 1
    assert len(opoints) == 2
    assert np.isclose(xpoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(xpoints[0][1], z0, atol=1.0 / ny)


def test_mask_zero_psi_bndry():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one O-point at (r0,z0) and no x-points
    # Range from around -0.5 to +0.5
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / 0.3**2) - 0.5

    psi = psi_func(r2d, z2d)

    opoints, xpoints = critical.find_critical(r2d, z2d, psi)

    assert len(xpoints) == 0
    assert len(opoints) == 1
    assert np.isclose(opoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(opoints[0][1], z0, atol=1.0 / ny)

    mask = critical.core_mask(r2d, z2d, psi, opoints, xpoints, psi_bndry=0.0)

    # Some of the mask must equal 1, some 0
    assert np.any(np.equal(mask, 1))
    assert np.any(np.equal(mask, 0))


def _grid(rmin, rmax, zmin, zmax, nx):
    r1d = np.linspace(rmin, rmax, nx)
    z1d = np.linspace(zmin, zmax, nx)
    return np.meshgrid(r1d, z1d, indexing="ij")


def _two_o_one_x(R, Z, r0, z0, a, w):
    """Two Gaussian wells at (r0, z0 +- a) with a saddle at (r0, z0) by symmetry"""
    return np.exp(-((R - r0) ** 2 + (Z - z0 - a) ** 2) / w**2) + np.exp(
        -((R - r0) ** 2 + (Z - z0 + a) ** 2) / w**2
    )


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, -1e-3, -1.0])
def test_critical_invariant_to_psi_scaling(scale):
    # Critical points depend on the shape of psi, not its units
    R, Z = _grid(1.0, 2.0, -1.0, 1.0, 65)
    psi = _two_o_one_x(R, Z, 1.5, 0.0, 0.3, 0.3)
    ref_o, ref_x = critical.find_critical(R, Z, psi)
    opoints, xpoints = critical.find_critical(R, Z, scale * psi)

    assert len(ref_o) == 2 and len(ref_x) == 1
    assert len(opoints) == 2
    assert len(xpoints) == 1
    # Saddle is on the symmetry point, independent of the scale
    assert xpoints[0][0] == pytest.approx(1.5, abs=1e-6)
    assert xpoints[0][1] == pytest.approx(0.0, abs=1e-6)
    for p, q in zip(sorted(opoints, key=lambda o: o[1]), sorted(ref_o, key=lambda o: o[1])):
        assert p[0] == pytest.approx(q[0], abs=1e-6)
        assert p[1] == pytest.approx(q[1], abs=1e-6)
        assert p[2] == pytest.approx(scale * q[2] / 1.0, rel=1e-6)


def test_critical_points_are_converged():
    # The gradient must vanish to a small fraction of a cell, not only to
    # an absolute tolerance on B.
    R, Z = _grid(1.0, 2.0, -1.0, 1.0, 65)
    psi = _two_o_one_x(R, Z, 1.5, 0.0, 0.3, 0.3)
    opoints, xpoints = critical.find_critical(R, Z, 1e-3 * psi)
    f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], 1e-3 * psi)
    for r, z, _ in opoints + xpoints:
        grad = np.hypot(f(r, z, dx=1)[0, 0], f(r, z, dy=1)[0, 0])
        hess = abs(f(r, z, dx=2)[0, 0]) + abs(f(r, z, dy=2)[0, 0])
        assert grad < 1e-6 * (2.0 / 64) * hess


def test_critical_distinct_points_closer_than_old_dedup_radius():
    # Small domain: distinct critical points are 2 mm apart, closer than the
    # former fixed 3 mm duplicate radius, but many cells apart.
    R, Z = _grid(1.0, 1.01, -0.005, 0.005, 129)
    psi = _two_o_one_x(R, Z, 1.005, 0.0, 0.001, 0.001)
    opoints, xpoints = critical.find_critical(R, Z, psi)
    assert len(opoints) == 2
    assert len(xpoints) == 1


def test_primary_opoint_not_nearest_to_centre():
    # Deep well near the edge of the domain and a shallow vacuum O-point at the
    # domain centre. The primary O-point is the well, not the nearest to centre.
    R, Z = _grid(1.0, 3.0, -1.0, 1.0, 129)
    k = 3.0
    psi = np.exp(-((R - 1.3) ** 2 + Z**2) / 0.06) + 0.05 * np.cos(k * (R - 2.0)) * np.cos(
        k * Z
    )
    opoints, xpoints = critical.find_critical(R, Z, psi)
    assert len(opoints) >= 2
    i, j = np.unravel_index(np.argmax(psi), psi.shape)
    assert opoints[0][2] == pytest.approx(psi.max(), rel=1e-3)
    assert opoints[0][0] == pytest.approx(R[i, j], abs=0.02)
    assert opoints[0][1] == pytest.approx(Z[i, j], abs=0.02)
    # The saddle between the two O-points is connected to the primary one
    assert any(abs(x[0] - 1.9) < 0.1 and abs(x[1]) < 0.01 for x in xpoints)


def test_degenerate_point_is_neither_opoint_nor_xpoint():
    # Monkey saddle psi = Re[(z)^3]: grad = 0 and Hessian = 0 at the centre
    R, Z = _grid(1.0, 2.0, -0.5, 0.5, 65)
    x = R - 1.5
    psi = x**3 - 3 * x * Z**2
    opoints, xpoints = critical.find_critical(R, Z, psi)
    assert len(opoints) == 0
    assert len(xpoints) == 0


def test_find_psisurface_not_bracketed_raises():
    R, Z = _grid(1.0, 2.0, -1.0, 1.0, 65)
    psin = np.exp(-((R - 1.5) ** 2 + Z**2) / 0.1)  # peak 1 at the axis
    psin = 1.0 - psin  # 0 on axis, rising to < 1 everywhere in the domain
    f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psin)

    class Eq:
        Rmin, Rmax, Zmin, Zmax = 1.0, 2.0, -1.0, 1.0

    with pytest.raises(ValueError, match="not bracketed"):
        critical.find_psisurface(Eq, f, 1.5, 0.0, 1.9, 0.0, psival=1.0)

    # A bracketed surface is found where the analytic profile crosses it
    level = 0.5
    r, z = critical.find_psisurface(Eq, f, 1.5, 0.0, 1.9, 0.0, psival=level)
    assert r == pytest.approx(1.5 + np.sqrt(-0.1 * np.log(1 - level)), abs=2e-3)
    assert z == pytest.approx(0.0)


def test_find_separatrix_without_xpoint_raises():
    R, Z = _grid(1.0, 2.0, -1.0, 1.0, 65)
    psi = np.exp(-((R - 1.5) ** 2 + Z**2) / 0.1)

    class Eq:
        Rmin, Rmax, Zmin, Zmax = 1.0, 2.0, -1.0, 1.0
        psi_bndry = 0.5

    Eq.R, Eq.Z = R, Z
    with pytest.raises(ValueError, match="X-point"):
        critical.find_separatrix(Eq, psi=psi)


def _disc_mask_area(n, smooth, radius, centre=(0.013, -0.021)):
    # psi = squared distance from the centre: one O-point, boundary psi = radius^2
    R, Z = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    psi = (R - centre[0]) ** 2 + (Z - centre[1]) ** 2
    opoint = [(centre[0], centre[1], 0.0)]
    mask = critical.core_mask(R, Z, psi, opoint, [], psi_bndry=radius**2, smooth=smooth)
    h = 2.0 / (n - 1)
    return mask.sum() * h * h, np.pi * radius**2


def test_smooth_core_mask_area_converges_second_order():
    # The area of a disc is known analytically; rms error over a set of radii
    # must converge at second order and beat the hard mask.
    radii = np.linspace(0.45, 0.65, 5)
    rms = {True: [], False: []}
    for smooth in (False, True):
        for n in (33, 65, 129):
            err = [a / exact - 1 for a, exact in (_disc_mask_area(n, smooth, r) for r in radii)]
            rms[smooth].append(np.sqrt(np.mean(np.square(err))))
    assert all(s < h for s, h in zip(rms[True], rms[False]))
    # second order: the error drops by ~16 over two halvings of the cell size
    assert rms[True][0] / rms[True][2] > 2 ** (2 * 1.75)


def test_smooth_core_mask_is_continuous_in_psi():
    # Moving the boundary by a small amount changes the mask area by a small
    # amount (d area / d psi_bndry = pi for psi = r^2); the hard mask jumps by
    # whole cells.
    n = 65
    R, Z = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    psi = R**2 + Z**2
    opoint = [(0.0, 0.0, 0.0)]
    h = 2.0 / (n - 1)
    levels = 0.36 + np.linspace(-2e-3, 2e-3, 201)
    jumps = {}
    for smooth in (True, False):
        area = np.array(
            [
                critical.core_mask(R, Z, psi, opoint, [], psi_bndry=lev, smooth=smooth).sum()
                for lev in levels
            ]
        ) * h * h
        jumps[smooth] = np.max(np.abs(np.diff(area)))
    assert jumps[True] < 2 * np.pi * (levels[1] - levels[0])
    assert jumps[False] >= 0.9 * h * h


def test_primary_opoint_ignores_grid_scale_spikes():
    """
    A resolved well at the centre of the domain and a lattice of grid-scale
    spikes, like the filaments of a coil inside the grid. The spikes are
    O-points connected to many saddles, but are not the magnetic axis.
    """
    n = 257
    r1d = np.linspace(1.0, 3.0, n)
    z1d = np.linspace(-1.0, 1.0, n)
    R, Z = np.meshgrid(r1d, z1d, indexing="ij")
    h = r1d[1] - r1d[0]
    psi = 1.0 * np.exp(-((R - 2.0) ** 2 + Z**2) / 0.08)
    for rc in (2.6, 2.7, 2.8):
        for zc in (0.5, 0.6, 0.7):
            psi += 1.2 * np.exp(-((R - rc) ** 2 + (Z - zc) ** 2) / (2.0 * h) ** 2)

    opoints, xpoints = critical.find_critical(R, Z, psi)

    assert len(opoints) > 2
    assert opoints[0][0] == pytest.approx(2.0, abs=0.01)
    assert opoints[0][1] == pytest.approx(0.0, abs=0.01)
