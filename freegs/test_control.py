import numpy as np
import pytest
from scipy.optimize import lsq_linear

from . import control
from .coil import Coil
from .equilibrium import Equilibrium
from .machine import Circuit, Machine
from .multi_coil import MultiCoil

PTS = [(1.0, 0.0), (1.2, 0.3), (1.0, -0.3), (1.4, 0.1), (0.9, 0.2), (1.3, -0.2)]
VALS = [0.02, 0.015, 0.02, 0.01, 0.02, 0.012]


def make_eq(coils):
    return Equilibrium(
        tokamak=Machine(coils), Rmin=0.1, Rmax=2.0, Zmin=-1.2, Zmax=1.2, nx=33, ny=33
    )


def six_coils(turns=1):
    pos = [(0.5, 0.8), (1.8, 0.8), (0.5, -0.8), (1.8, -0.8), (1.2, 1.1), (1.2, -1.1)]
    return [(f"C{i}", Coil(r, z, turns=turns)) for i, (r, z) in enumerate(pos)]


def linear_system(eq, pts=PTS, vals=VALS):
    tok = eq.getMachine()
    A = np.array([tok.controlPsi(r, z) for r, z in pts])
    b = np.array(vals) - np.array([eq.psiRZ(r, z) for r, z in pts])
    return A, b


def test_psi_invariant_under_turns():
    """Same physical coils with different turns give the same achieved psi."""
    achieved = []
    for turns in (1, 1000):
        eq = make_eq(six_coils(turns))
        con = control.constrain(
            psivals=[(r, z, v) for (r, z), v in zip(PTS, VALS)], gamma=1e-7
        )
        con(eq)
        achieved.append([eq.psiRZ(r, z) for r, z in PTS])
    np.testing.assert_allclose(achieved[0], VALS, atol=1e-10)
    np.testing.assert_allclose(achieved[1], VALS, atol=1e-10)


def test_unbounded_matches_exact_solution():
    eq = make_eq(six_coils())
    A, b = linear_system(eq)
    con = control.constrain(psivals=[(r, z, v) for (r, z), v in zip(PTS, VALS)])
    con(eq)
    np.testing.assert_allclose(con.current_change, np.linalg.solve(A, b), rtol=1e-6)


def test_bounded_matches_lsq_linear():
    """With active bounds the result is the bounded least squares optimum."""
    rng = np.random.default_rng(3)
    for _ in range(3):
        pts = [(rng.uniform(0.8, 1.5), rng.uniform(-0.4, 0.4)) for _ in range(8)]
        vals = rng.uniform(0.01, 0.03, 8)
        eq = make_eq(six_coils())
        A, b = linear_system(eq, pts, vals)
        free = np.linalg.lstsq(A, b, rcond=None)[0]
        lim = 0.4 * np.abs(free)
        lims = [(-l, l) for l in lim]
        ref = lsq_linear(A, b, bounds=(-lim, lim), tol=1e-14)
        con = control.constrain(
            psivals=[(r, z, v) for (r, z), v in zip(pts, vals)], current_lims=lims
        )
        con(eq)
        cur = np.array(eq.getMachine().controlCurrents())
        assert np.all(np.abs(cur) <= lim * (1 + 1e-9))
        assert np.isclose(np.sum(np.abs(cur) >= lim * (1 - 1e-6)), np.sum(ref.active_mask != 0))
        assert np.linalg.norm(A @ cur - b) <= np.linalg.norm(A @ ref.x - b) * (1 + 1e-6)


def test_limits_are_per_control():
    """A non-controlled coil does not shift the indexing of current_lims."""
    coils = six_coils()
    coils[0][1].control = False
    eq = make_eq(coils)
    A, b = linear_system(eq)
    free = np.linalg.lstsq(A, b, rcond=None)[0]
    lims = [(-0.5 * abs(x), 0.5 * abs(x)) for x in free]
    con = control.constrain(
        psivals=[(r, z, v) for (r, z), v in zip(PTS, VALS)], current_lims=lims
    )
    con(eq)
    cur = np.array(eq.getMachine().controlCurrents())
    assert np.all(np.abs(cur) <= 0.5 * np.abs(free) * (1 + 1e-9))
    with pytest.raises(ValueError):
        control.constrain(psivals=[(1.0, 0.0, 0.0)], current_lims=lims[:-1])(eq)


def test_max_total_current():
    eq = make_eq(six_coils())
    A, b = linear_system(eq)
    free = np.linalg.solve(A, b)
    limit = 0.5 * np.sum(np.abs(free))
    con = control.constrain(
        psivals=[(r, z, v) for (r, z), v in zip(PTS, VALS)], max_total_current=limit
    )
    con(eq)
    assert np.sum(np.abs(eq.getMachine().controlCurrents())) <= limit * (1 + 1e-6)


def test_circuit_ties_controls():
    """A Circuit ties CS1U = CS1L; the fit equals the reduced linear problem."""
    cs = Circuit([("U", Coil(0.5, 0.4), 1.0), ("L", Coil(0.5, -0.4), 1.0)])
    coils = [("CS", cs), ("P1", Coil(1.8, 0.8)), ("P2", Coil(1.8, -0.8))]
    eq = make_eq(coils)
    A, b = linear_system(eq)
    con = control.constrain(psivals=[(r, z, v) for (r, z), v in zip(PTS, VALS)])
    con(eq)
    assert A.shape[1] == 3
    assert cs["U"].current == cs["L"].current
    ref = np.linalg.lstsq(A, b, rcond=None)[0]
    np.testing.assert_allclose(con.current_change, ref, rtol=1e-6)


def test_xpoint_and_isoflux_weights():
    """Row tolerances change the weighting but not an exactly solvable fit."""
    eq = make_eq(six_coils())
    con = control.constrain(
        xpoints=[(1.0, -0.9)],
        isoflux=[(1.0, 0.0, 1.4, 0.1), (1.0, 0.0, 0.9, 0.2)],
        psi_tol=1e-3,
        B_tol=1e-2,
    )
    con(eq)
    assert abs(eq.Br(1.0, -0.9)) < 1e-6
    assert abs(eq.Bz(1.0, -0.9)) < 1e-6
    assert abs(eq.psiRZ(1.0, 0.0) - eq.psiRZ(1.4, 0.1)) < 1e-9


def test_ampere_turn_limits():
    mc = MultiCoil([1.0, 1.1, 1.2, 1.3], [0.5] * 4)
    coils = [("A", Coil(0.5, 0.8, turns=10)), ("M", mc), ("P", Coil(1.8, 0.8))]
    lims = control.ampere_turn_limits(
        Machine(coils), [(-1000.0, 2000.0), (-400.0, 400.0), (-5.0, 5.0)]
    )
    assert lims == [(-100.0, 200.0), (-100.0, 100.0), (-5.0, 5.0)]


def _target_case():
    eq = make_eq(six_coils())
    control.constrain(psivals=[(r, z, v) for (r, z), v in zip(PTS, VALS)])(eq)
    return eq, eq.psi().copy(), np.array(eq.getMachine().controlCurrents())


def _misfit(eq, target):
    psi = eq.psi()
    return np.linalg.norm((psi - psi.mean()) - (target - target.mean()))


def test_psi2d_recovers_currents():
    """ConstrainPsi2D recovers the currents that produced the target psi."""
    eq, target, currents = _target_case()
    eq.getMachine().setControlCurrents(0.8 * currents)
    control.ConstrainPsi2D(target)(eq)
    np.testing.assert_allclose(eq.getMachine().controlCurrents(), currents, rtol=1e-3)


@pytest.mark.parametrize("cls", ["psi", "psinorm"])
def test_advanced_controls_default_limits(cls):
    """The Advanced controls run with current_lims=None and reduce the misfit."""
    eq, target, currents = _target_case()
    eq.getMachine().setControlCurrents(0.8 * currents)
    if cls == "psi":
        before = _misfit(eq, target)
        control.ConstrainPsi2DAdvanced(target)(eq)
        assert _misfit(eq, target) < before
    else:
        # A vacuum field has no plasma boundary: use the psi range instead
        def update(psi=None):
            psi = eq.psi() if psi is None else psi
            eq.psi_axis, eq.psi_bndry = psi.max(), psi.min()

        eq._updateBoundaryPsi = update
        update()
        target_norm = (eq.psi() - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)
        control.ConstrainPsiNorm2DAdvanced(target_norm)(eq)
        assert np.all(np.isfinite(eq.getMachine().controlCurrents()))


def test_psi2d_advanced_limits_per_control():
    eq, target, currents = _target_case()
    lims = [(-abs(c), abs(c)) for c in 0.9 * currents]
    eq.getMachine().setControlCurrents(0.5 * currents)
    control.ConstrainPsi2DAdvanced(target, current_lims=lims)(eq)
    cur = np.array(eq.getMachine().controlCurrents())
    assert np.all(np.abs(cur) <= 0.9 * np.abs(currents) * (1 + 1e-9))
