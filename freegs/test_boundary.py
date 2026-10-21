"""On-boundary Hagenow quadrature and the public TestTokamak regression."""

from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root

from . import Equilibrium, boundary, control, critical, jtor, machine, solve
from .gradshafranov import Greens, mu0


@pytest.mark.parametrize("aspect", [1, 2])
def test_hagenow_filament_vacuum_convergence(aspect):
    # A filament inside the domain has an exactly known vacuum field near
    # every boundary point. Avoid a discretized delta source: solve only its
    # smooth harmonic boundary lift, then supply the resulting zero-boundary
    # field to Hagenow's normal derivative and singular integral calculation.
    errors = []
    for n in [17, 33, 65]:
        eq = Equilibrium(
            Rmin=0.4,
            Rmax=1.8,
            Zmin=-0.8,
            Zmax=0.7,
            nx=n,
            ny=aspect * (n - 1) + 1,
            order=4,
        )
        exact = Greens(1.03, 0.12, eq.R, eq.Z)
        rhs = np.zeros_like(exact)
        rhs[[0, -1], :] = exact[[0, -1], :] / mu0
        rhs[:, [0, -1]] = exact[:, [0, -1]] / mu0
        lift = eq.callSolver(np.zeros_like(rhs), rhs)
        fixed = lift - exact / mu0
        result = np.zeros_like(rhs)
        with patch.object(eq, "callSolver", return_value=fixed):
            boundary.freeBoundaryHagenow(eq, np.zeros_like(rhs), result)
        errors.append(
            max(
                np.max(np.abs(result[[0, -1], :] - exact[[0, -1], :])),
                np.max(np.abs(result[:, [0, -1]] - exact[:, [0, -1]])),
            )
        )
        assert np.all(result[1:-1, 1:-1] == 0)

    # Fourth-order derivatives, PDE and cubic product integration: allow
    # pre-asymptotic variation, but require at least order log2(10) per step.
    assert np.all(np.array(errors[:-1]) / errors[1:] > 10), errors
    assert errors[-1] < 1e-12, errors


def _tokamak_observables(method):
    # 01-freeboundary.py physical case, with the E2.3 solve/readback tolerances.
    eq = Equilibrium(
        tokamak=machine.TestTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        order=4,
        boundary=method,
    )
    profiles = jtor.ConstrainPaxisIp(
        eq,
        1e3,
        2e5,
        2.0,
        alpha_m=1.0,
        alpha_n=2.0,
        Raxis=1.0,
    )
    constraints = control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)],
        isoflux=[(1.1, -0.6, 1.1, 0.6)],
        gamma=1e-12,
    )
    solve(eq, profiles, constraints, rtol=1e-9, atol=1e-12, maxits=500, show=False)
    ray = partial(critical.find_psisurface, n=800)
    with patch.object(critical, "find_psisurface", ray):
        q95 = critical.find_safety(eq, psinorm=[0.95], ntheta=1024)[0]

    # Remove find_critical's fixed 1 mT stopping floor when comparing positions.
    _, xpoints = critical.find_critical(eq.R, eq.Z, eq.psi())
    spline = RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], eq.psi())
    positions = []
    for point in sorted(xpoints, key=lambda p: p[1]):

        def gradient(v):
            return [spline.ev(*v, dx=1), spline.ev(*v, dy=1)]

        polished = root(gradient, point[:2], tol=1e-11)
        assert np.linalg.norm(gradient(polished.x)) < 1e-10
        positions.append(polished.x)
    assert len(positions) == 2
    return np.array([eq.psi_axis, eq.psi_bndry, q95]), np.array(positions)


def test_hagenow_testtokamak_against_direct_green():
    # Direct-Green 129 -> 257 convergence indicators from the published case:
    # https://github.com/itpplasma/kin6d/tree/3bc4b5b68181d6a15c6c6ce1bb09deb1dfabc355/benchmarks/freegs
    # reference.dat (fluxes/q95), critical-refinement.dat (polished X-points).
    # Use their differences as budgets, not a fitted tolerance for Hagenow.
    direct_ladder = np.array(
        [
            [0.09052254644576879, 0.037167930797614344, 7.805833464007265],
            [0.0905225838990162, 0.037168441708540494, 7.805362969547644],
        ]
    )
    xpoint_ladder = np.array(
        [
            [
                [1.0942765080370582, -0.6038555619229298],
                [1.1077553261449438, 0.7982280491279263],
            ],
            [
                [1.0942765330580697, -0.6038556664517302],
                [1.1077546969979575, 0.7982279255861822],
            ],
        ]
    )
    flux_q_budget = np.abs(direct_ladder[1] - direct_ladder[0])
    position_budget = np.linalg.norm(xpoint_ladder[1] - xpoint_ladder[0], axis=1)
    direct, direct_x = _tokamak_observables(boundary.freeBoundary)
    hagenow, hagenow_x = _tokamak_observables(boundary.freeBoundaryHagenow)
    assert np.all(np.abs(hagenow - direct) < flux_q_budget), (
        hagenow - direct,
        flux_q_budget,
    )
    assert np.all(np.linalg.norm(hagenow_x - direct_x, axis=1) < position_budget)
