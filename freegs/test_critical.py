import numpy as np

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
