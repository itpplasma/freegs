"""
Test GEQDSK write/read round trip, and that reading logs rather than prints
"""

import io
import logging

import pytest

import freegs
from freegs import geqdsk


@pytest.fixture(scope="module")
def solved_equilibrium():
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
    freegs.solve(eq, profiles, constrain, rtol=1e-4, maxits=100)
    return eq


def test_geqdsk_read_logs_instead_of_printing(solved_equilibrium, capsys, caplog):
    buf = io.StringIO()
    geqdsk.write(solved_equilibrium, buf)
    buf.seek(0)

    with caplog.at_level(logging.INFO, logger="freegs.geqdsk"):
        read_eq = geqdsk.read(buf, freegs.machine.TestTokamak())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Plasma current" in caplog.text

    # Round trip: the plasma current is reproduced to a few per cent on this
    # coarse grid, where the read equilibrium is re-solved from the file
    ip_in = solved_equilibrium.plasmaCurrent()
    ip_out = read_eq.plasmaCurrent()
    assert abs(ip_out - ip_in) / abs(ip_in) < 0.05
