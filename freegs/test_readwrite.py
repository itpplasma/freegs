import contextlib
import dataclasses
import io

import numpy as np
import pytest
from freeqdsk import geqdsk
from numpy import allclose
from numpy.testing import assert_allclose
from scipy import interpolate
from shapely.geometry import Point, Polygon

import freegs
import freegs.geqdsk


def test_readwrite():
    """Test reading/writing to a file round-trip"""

    for tokamak in [freegs.machine.TestTokamak(), freegs.machine.MAST_sym()]:
        eq = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=0.1,
            Rmax=2.0,
            Zmin=-1.0,
            Zmax=1.0,
            nx=17,
            ny=17,
            boundary=freegs.boundary.freeBoundaryHagenow,
        )
        profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e4, 1e6, 2.0)

        # Note here the X-point locations and isoflux locations are not the same.
        # The result will be an unbalanced double null configuration, where the
        # X-points are on different flux surfaces.
        xpoints = [(1.1, -0.6), (1.1, 0.8)]
        isoflux = [(1.1, -0.6, 1.1, 0.6)]
        constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

        freegs.solve(eq, profiles, constrain, maxits=25, atol=1e-3, rtol=1e-1)

        memory_file = io.BytesIO()

        with freegs.OutputFile(memory_file, "w") as f:
            f.write_equilibrium(eq)

        with freegs.OutputFile(memory_file, "r") as f:
            read_eq = f.read_equilibrium()

        assert tokamak == read_eq.tokamak
        assert allclose(eq.psi(), read_eq.psi())


# GEQDSK export/import. The format definition used for the grid conventions is
# the FreeQDSK one (https://freeqdsk.readthedocs.io, freeqdsk.geqdsk): the 1D
# profiles are on numpy.linspace(simagx, sibdry, nx), i.e. include both the
# axis and the boundary.


@pytest.fixture(scope="module")
def diverted_eq():
    tokamak = freegs.machine.TestTokamak()
    eq = freegs.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=freegs.boundary.freeBoundaryHagenow,
    )
    profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    constrain = freegs.control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    freegs.solve(eq, profiles, constrain, rtol=1e-5, maxits=200)
    return eq


def written_text(eq):
    buf = io.StringIO()
    freegs.geqdsk.write(eq, buf)
    return buf.getvalue()


def read_back(text, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return freegs.geqdsk.read(
            io.StringIO(text), freegs.machine.TestTokamak(), rtol=1e-3, **kwargs
        )


def test_geqdsk_profile_grid_includes_axis_and_boundary(diverted_eq):
    """1D profiles are on linspace(0, 1, nx) in psiN, with both end points"""
    eq = diverted_eq
    data = geqdsk.read(io.StringIO(written_text(eq)))
    nx = data["nx"]
    psinorm = np.linspace(0.0, 1.0, nx)

    # Boundary values: F = fvac and p' = 0 on the LCFS of this profile family,
    # p' = 0 and the pressure vanishes
    assert data["fpol"][-1] == pytest.approx(eq.fvac(), rel=1e-8)
    assert data["pprime"][-1] == pytest.approx(0.0, abs=1e-6)
    assert data["pres"][-1] == pytest.approx(0.0, abs=1e-6)
    assert data["pprime"][0] == pytest.approx(float(eq.pprime(0.0)))
    for name, func in [
        ("fpol", eq.fpol),
        ("pres", eq.pressure),
        ("pprime", eq.pprime),
        ("ffprime", eq.ffprime),
    ]:
        assert_allclose(data[name], func(psinorm), rtol=1e-7, atol=1e-6, err_msg=name)

    # Flux is relative to the axis; psi spans simagx..sibdry on the 2D grid
    assert data["simagx"] == 0.0
    assert data["sibdry"] == pytest.approx(eq.psi_bndry - eq.psi_axis)
    assert data["cpasma"] == pytest.approx(eq.plasmaCurrent())


def test_geqdsk_qpsi_cocos3_and_axis_value(diverted_eq):
    """qpsi is signed for COCOS 3, and q(axis) is the analytic axis value"""
    eq = diverted_eq
    data = geqdsk.read(io.StringIO(written_text(eq)))
    qpsi = np.asarray(data["qpsi"])

    # COCOS 3: sign(q) = -sign(Ip * F)
    assert np.all(np.sign(qpsi) == -np.sign(data["cpasma"] * data["fpol"]))
    assert_allclose(qpsi[1:-1], eq.q(np.linspace(0, 1, len(qpsi))[1:-1]), rtol=2e-2)

    # Near the axis psi = psi0 + (x, z) H (x, z) / 2, whose flux surfaces have
    # |q0| = |F| / (R0 sqrt(det H)) (psi per radian, B_p = |grad psi| / R)
    R0, Z0 = data["rmagx"], data["zmagx"]
    h = 2e-3

    def psi(r, z):
        return float(eq.psiRZ(r, z))

    p0 = psi(R0, Z0)
    hrr = (psi(R0 + h, Z0) - 2 * p0 + psi(R0 - h, Z0)) / h**2
    hzz = (psi(R0, Z0 + h) - 2 * p0 + psi(R0, Z0 - h)) / h**2
    hrz = (
        psi(R0 + h, Z0 + h) - psi(R0 + h, Z0 - h) - psi(R0 - h, Z0 + h)
        + psi(R0 - h, Z0 - h)
    ) / (4 * h**2)  # fmt: skip
    q0 = abs(data["fpol"][0]) / (R0 * np.sqrt(hrr * hzz - hrz**2))
    assert abs(qpsi[0]) == pytest.approx(q0, rel=2e-2)
    # Not a copy of the first interior point
    assert qpsi[0] != qpsi[1]


def test_geqdsk_boundary_contains_xpoint(diverted_eq):
    """The closed boundary polygon has the X-point as a vertex"""
    eq = diverted_eq
    data = geqdsk.read(io.StringIO(written_text(eq)))
    rb, zb = np.asarray(data["rbdry"]), np.asarray(data["zbdry"])
    _, xpoint = freegs.critical.find_critical(eq.R, eq.Z, eq.psi())
    rx, zx = xpoint[0][0:2]

    assert (rb[0], zb[0]) == (rb[-1], zb[-1])  # closed
    assert np.min(np.hypot(rb - rx, zb - zx)) < 1e-6  # file precision
    assert Polygon(np.c_[rb, zb]).buffer(1e-6).contains(Point(rx, zx))
    # Every vertex is on the separatrix psiN = 1
    assert_allclose(eq.psiNRZ(rb, zb), 1.0, atol=2e-3)


def test_geqdsk_roundtrip(diverted_eq):
    eq = diverted_eq
    eq2 = read_back(written_text(eq))
    s = np.linspace(0.0, 1.0, 11)

    assert eq2.plasmaCurrent() == pytest.approx(eq.plasmaCurrent(), rel=1e-3)
    assert eq2.fvac() == pytest.approx(eq.fvac())
    assert (eq2.psi_bndry - eq2.psi_axis) == pytest.approx(
        eq.psi_bndry - eq.psi_axis, rel=1e-3
    )
    assert_allclose(eq2.fpol(s), eq.fpol(s), rtol=1e-6)
    assert_allclose(eq2.pressure(s), eq.pressure(s), rtol=1e-6, atol=1e-3)
    # p' and ff' are the (spline) derivatives of the file's p and f^2 / 2
    for name in ("pprime", "ffprime"):
        ref = np.asarray(getattr(eq, name)(s))
        assert_allclose(
            getattr(eq2, name)(s), ref, atol=2e-3 * np.max(np.abs(ref)), err_msg=name
        )


def test_geqdsk_read_dict_style_data(diverted_eq, monkeypatch):
    """freeqdsk < 0.5 returns a dict, with rlim only if the file has a limiter"""
    text = written_text(diverted_eq)
    original = geqdsk.read

    def read_dict(fh, **kwargs):
        data = original(fh, **kwargs)
        data = dataclasses.asdict(data) if dataclasses.is_dataclass(data) else data
        for key in ("rlim", "zlim"):
            data.pop(key, None)
        return data

    monkeypatch.setattr(freegs.geqdsk.geqdsk, "read", read_dict)
    machine = freegs.machine.TestTokamak()
    machine.wall = None
    with contextlib.redirect_stdout(io.StringIO()):
        eq = freegs.geqdsk.read(io.StringIO(text), machine, rtol=1e-3)
    assert machine.wall is None
    assert eq.plasmaCurrent() == pytest.approx(2e5, rel=1e-3)


def test_geqdsk_read_wall(diverted_eq):
    """The limiter in the file becomes the machine wall"""
    text = written_text(diverted_eq)
    assert diverted_eq.tokamak.wall is not None
    machine = freegs.machine.TestTokamak()
    machine.wall = None
    with contextlib.redirect_stdout(io.StringIO()):
        freegs.geqdsk.read(io.StringIO(text), machine, rtol=1e-3)
    assert_allclose(machine.wall.R, diverted_eq.tokamak.wall.R)
    assert_allclose(machine.wall.Z, diverted_eq.tokamak.wall.Z)


@pytest.mark.parametrize(
    "domain", [(0.2, 1.9, -0.9, 0.9), (0.05, 2.1, -1.1, 1.1)]
)  # smaller, larger than the file's 0.1..2.0 x -1..1
def test_geqdsk_read_domain(diverted_eq, domain):
    """read(domain=...) preserves psi in the plasma on the overlapping region"""
    eq = diverted_eq
    eq2 = read_back(written_text(eq), domain=list(domain))
    assert domain == (eq2.Rmin, eq2.Rmax, eq2.Zmin, eq2.Zmax)
    assert eq2.nx >= 65
    assert eq2.ny >= 65

    psirange = eq.psi_bndry - eq.psi_axis
    reference = interpolate.RectBivariateSpline(
        eq.R[:, 0], eq.Z[0, :], eq.psi() - eq.psi_axis
    )
    overlap = (
        (eq.Rmin <= eq2.R) & (eq.Rmax >= eq2.R)
        & (eq.Zmin <= eq2.Z) & (eq.Zmax >= eq2.Z)
        & (eq2.mask > 0)
    )  # fmt: skip
    assert np.any(overlap)
    error = (eq2.psi() - eq2.psi_axis) - reference(eq2.R, eq2.Z, grid=False)
    assert np.max(np.abs(error[overlap])) < 2e-3 * abs(psirange)
    assert eq2.plasmaCurrent() == pytest.approx(eq.plasmaCurrent(), rel=1e-2)
