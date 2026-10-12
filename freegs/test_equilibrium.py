import pytest

from . import equilibrium
from . import boundary
from . import jtor
from . import picard
from . import control
from .machine import TestTokamak, TestTokamakLimited

import numpy as np

def test_inoutseparatrix():
    eq = equilibrium.Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=65, ny=65)

    # Two O-points, one X-point half way between them
    psi = np.exp((-((eq.R - 1.0) ** 2) - eq.Z**2) * 3) + np.exp(
        (-((eq.R - 1.0) ** 2) - (eq.Z + 1) ** 2) * 3
    )

    eq._updatePlasmaPsi(psi)

    Rin, Rout = eq.innerOuterSeparatrix()

    assert Rin >= eq.Rmin
    assert Rout >= eq.Rmin
    assert Rin <= eq.Rmax
    assert Rout <= eq.Rmax


def test_fixed_boundary_psi():
    # This is adapted from example 5

    eq = equilibrium.Equilibrium(
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=boundary.fixedBoundary,
    )

    # Plasma pressure on axis [Pascals]  # Plasma current [Amps]
    profiles = jtor.ConstrainPaxisIp(eq, 1e3, 1e5, 1.0)  # fvac = R*Bt

    # Nonlinear solve
    picard.solve(eq, profiles)

    psi = eq.psi()
    assert psi[0, 0] == 0.0  # Boundary is fixed
    assert psi[32, 32] != 0.0  # Solution is not all zero

    assert eq.psi_bndry == 0.0
    assert eq.poloidalBeta() > 0.0


def test_fixed_boundary_psi_check_limited():
    # This is adapted from example 5

    tokamak = TestTokamak()

    eq = equilibrium.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=boundary.fixedBoundary,
        check_limited=True,
    )

    profiles = jtor.ConstrainPaxisIp(
        eq, 1e3, 1e5, 1.0  # Plasma pressure on axis [Pascals]  # Plasma current [Amps]
    )  # fvac = R*Bt

    # Nonlinear solve
    picard.solve(eq, profiles)

    psi = eq.psi()
    assert psi[0, 0] == 0.0  # Boundary is fixed
    assert psi[32, 32] != 0.0  # Solution is not all zero

    assert eq.psi_bndry == 0.0
    assert eq.poloidalBeta() > 0.0


def test_setSolverVcycle():
    eq = equilibrium.Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=65, ny=65)

    oldsolver = eq._solver
    eq.setSolverVcycle(nlevels=2, ncycle=1, niter=5)
    assert eq._solver != oldsolver


def _solve_diverted(nx, ny, Ip=2e5):
    # Adapted from example 1: free boundary, lower single null, diverted
    # (the separatrix does not touch the TestTokamak wall)
    tokamak = TestTokamak()
    eq = equilibrium.Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=nx, ny=ny
    )
    profiles = jtor.ConstrainPaxisIp(eq, 1e3, Ip, 2.0)
    constrain = control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    picard.solve(eq, profiles, constrain, check_limited=True)
    return eq


def test_check_limited_diverted_plasma():
    eq = _solve_diverted(65, 65)

    # The limiter flux must be the flux at the reported contact point
    assert np.isclose(eq.psi_limit, eq.psiRZ(eq.Rlim, eq.Zlim))

    # Every wall point is outside the separatrix, so the plasma is diverted
    psi_wall = eq.psiRZ(eq.tokamak.limit_points_R, eq.tokamak.limit_points_Z)
    assert np.all(psi_wall < eq.psi_xpt)
    assert not eq.is_limited
    assert eq.psi_bndry == eq.psi_xpt


def test_check_limited_non_square_grid():
    # psi[nx, ny] must be passed to the limiter interpolation untransposed;
    # with nx != ny a transposed array raises in RectBivariateSpline
    eq = _solve_diverted(65, 33)
    assert not eq.is_limited


def test_check_limited_diverted_negative_ip():
    # With Ip < 0 psi increases outward, so the separatrix flux is the
    # minimum, not the maximum, of the X-point and limiter fluxes
    eq = _solve_diverted(65, 65, Ip=-2e5)
    assert not eq.is_limited
    assert eq.psi_bndry == eq.psi_xpt


def _solve_limited(Ip):
    # Limited plasma: the separatrix touches the TestTokamakLimited wall
    tokamak = TestTokamakLimited()
    eq = equilibrium.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=boundary.freeBoundaryHagenow,
    )
    profiles = jtor.ConstrainPaxisIp(eq, 1e3, Ip, 2.0)
    constrain = control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    picard.solve(eq, profiles, constrain, check_limited=True)
    return eq


def test_check_limited_sign_of_ip():
    # The limiter contact point must be the separatrix for either sign of Ip:
    # every wall point lies outside it (psiN >= 1) and the +/-Ip solutions
    # are mirror images (psi flips sign with the current)
    eq_pos = _solve_limited(2e5)
    eq_neg = _solve_limited(-2e5)

    for eq in (eq_pos, eq_neg):
        assert eq.is_limited
        wall = eq.tokamak.wall
        psi_wall = np.array([eq.psiRZ(r, z) for r, z in zip(wall.R, wall.Z)])
        psi_n = (psi_wall - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)
        assert np.min(psi_n) >= 1.0

    assert (eq_pos.Rlim, eq_pos.Zlim) == (eq_neg.Rlim, eq_neg.Zlim)
    assert np.isclose(eq_pos.psi_bndry, -eq_neg.psi_bndry, rtol=1e-4)
    assert np.isclose(eq_pos.psi_axis, -eq_neg.psi_axis, rtol=1e-4)
    scale = np.max(np.abs(eq_pos.psi()))
    assert np.max(np.abs(np.abs(eq_pos.psi()) - np.abs(eq_neg.psi()))) < 1e-4 * scale


def test_psiN_without_boundary_flux_raises():
    """
    With no X-point (and no limiter contact) psi_bndry is None, so psiN is
    undefined and must raise a clear error
    """
    eq = equilibrium.Equilibrium(
        tokamak=TestTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=boundary.freeBoundaryHagenow,
    )
    # A single maximum of psi: an O-point, but no X-point
    eq._updatePlasmaPsi(-((eq.R - 1.0) ** 2 + eq.Z**2) * 10.0)
    assert eq.psi_bndry is None

    with pytest.raises(RuntimeError, match="psi_bndry"):
        eq.psiN()


@pytest.mark.parametrize("n", [33, 65, 129])
def test_grid_size_power_of_two_accepted(n):
    equilibrium.Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=n, ny=n)


@pytest.mark.parametrize(("nx", "ny"), [(64, 65), (65, 100), (1, 65)])
def test_grid_size_not_power_of_two_rejected(nx, ny):
    with pytest.raises(ValueError, match="2\\^k \\+ 1"):
        equilibrium.Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=nx, ny=ny)
