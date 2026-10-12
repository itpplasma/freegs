import numpy as np

import freegs


def test_q_accepts_list():
    """Equilibrium.q documents a scalar, list or array of normalised flux"""
    eq = freegs.Equilibrium(
        tokamak=freegs.machine.TestTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=33,
        ny=33,
        boundary=freegs.boundary.freeBoundaryHagenow,
    )
    profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    constrain = freegs.control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    freegs.solve(eq, profiles, constrain, rtol=1e-4)

    psinorm = [0.2, 0.5, 0.9]
    q_list = eq.q(psinorm)

    assert np.allclose(q_list, eq.q(np.array(psinorm)))
    # Interior and extrapolated (near the axis and separatrix) values
    assert np.allclose(eq.q([0.5, 0.995]), eq.q(np.array([0.5, 0.995])))
