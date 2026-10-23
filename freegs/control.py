"""
Plasma control system

Use constraints to adjust coil currents
"""

import numpy as np
from numpy import array, eye, inf
from numpy.linalg import norm
from scipy import optimize


def ampere_turn_limits(tokamak, limits):
    """
    Convert limits in ampere-turns to limits on the control currents,
    for use as current_lims.

    tokamak - A Machine
    limits  - List of (lower, upper) in ampere-turns, one per control, in
              the order of tokamak.controlCurrents()

    The control current of a Coil or FilamentCoil is the current per turn,
    so the limit is divided by coil.turns. For a MultiCoil the control
    current is the current in each filament, so the limit is divided by the
    number of filaments. Other controls (ShapedCoil, Circuit, ...) are
    left unchanged: a Circuit limit is on the circuit current, not on the
    ampere-turns of its member coils.
    """
    from .coil import Coil
    from .filament_coil import FilamentCoil
    from .multi_coil import MultiCoil

    controls = [coil for _, coil in tokamak.coils if coil.control]
    if len(limits) != len(controls):
        raise ValueError(
            f"Provided {len(limits)} limits for {len(controls)} controls."
        )
    result = []
    for coil, (lo, hi) in zip(controls, limits):
        if isinstance(coil, MultiCoil):
            turns = len(coil.Rfil)
        elif isinstance(coil, (Coil, FilamentCoil)):
            turns = abs(coil.turns)
        else:
            turns = 1
        result.append((lo / turns, hi / turns))
    return result


def _control_bounds(current_lims, ncontrols):
    """
    Array (ncontrols, 2) of lower and upper bounds, one pair per control.
    No limits (None) means unbounded.
    """
    bnds = np.empty((ncontrols, 2))
    bnds[:, 0] = -inf
    bnds[:, 1] = inf
    if current_lims is not None:
        if len(current_lims) != ncontrols:
            raise ValueError(
                "Should provide current limits for all controls. "
                f"Provided {len(current_lims)} limits for {ncontrols} controls."
            )
        for i, lim in enumerate(current_lims):
            bnds[i] = (min(lim), max(lim))
    return bnds


class constrain:
    """
    Adjust coil currents using constraints. To use this class,
    first create an instance by specifying the constraints

    >>> controlsystem = constrain(xpoints = [(1.0, 1.1), (1.0,-1.0)])

    controlsystem will now attempt to create x-points at
    (R,Z) = (1.0, 1.1) and (1.0, -1.1) in any Equilibrium

    >>> controlsystem(eq)

    where eq is an Equilibrium object which is modified by
    the call.

    The constraints which can be set are:

    xpoints - A list of X-point (R,Z) locations

    isoflux - A list of tuples (R1,Z1, R2,Z2)

    psivals - A list of (R,Z,psi) values

    At least one of the above constraints must be included.

    gamma - A scalar, minimises the magnitude of the coil currents. It is
    dimensionless: the Tikhonov parameter relative to the largest singular
    value of the constraint matrix after the controls have been scaled, so
    the result does not depend on the units (e.g. turns) of the controls.

    The following constraints are entirely optional:

    current_lims - A list of tuples [(l1,u1),(l2,u2)...(lN,uN)] for the upper
    and lower bounds on the current of each *control* (not each coil), in
    the order of Machine.controlCurrents(). The current of a control is the
    quantity set by coil.current: the current per turn for a Coil, per
    filament for a MultiCoil, and the circuit current for a Circuit. Limits
    given in ampere-turns can be converted with ampere_turn_limits().

    max_total_current - The maximum total current through the coilset
    (sum of the absolute control currents).

    current_scale - Optional list with a typical size of each control current
    (e.g. its limit). By default each control is scaled by the norm of its
    response column.

    psi_tol, B_tol - Optional tolerances for flux [Wb] and field [T]
    constraints. Constraint rows are divided by these, so that psi and
    Br/Bz rows are weighted against each other. Default is no weighting.

    Controls can be tied together (e.g. CS1U = CS1L) by putting the coils
    in a freegs.machine.Circuit; the circuit is then one control.

    Without bounds the problem is solved by an SVD based least squares;
    with bounds by scipy.optimize.lsq_linear. SLSQP is only used if
    max_total_current is violated by that solution.
    """

    def __init__(
        self,
        xpoints=None,
        gamma=1e-12,
        isoflux=None,
        psivals=None,
        current_lims=None,
        max_total_current=None,
        current_scale=None,
        psi_tol=None,
        B_tol=None,
    ):
        """
        Create an instance, specifying the constraints to apply
        """
        self.current_scale = current_scale
        self.psi_tol = psi_tol
        self.B_tol = B_tol
        if psivals is None:
            psivals = []
        if isoflux is None:
            isoflux = []
        if xpoints is None:
            xpoints = []
        self.xpoints = xpoints
        self.gamma = gamma
        self.isoflux = isoflux
        self.psivals = psivals
        self.current_lims = current_lims
        self.max_total_current = max_total_current

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()

        constraint_matrix = []
        constraint_rhs = []
        # Row weights (1 / tolerance), so that rows with different units
        # (Br, Bz in Tesla; psi in Weber) are comparable
        wB = 1.0 / self.B_tol if self.B_tol is not None else 1.0
        wpsi = 1.0 / self.psi_tol if self.psi_tol is not None else 1.0
        for xpt in self.xpoints:
            # Each x-point introduces two constraints
            # 1) Br = 0

            Br = eq.Br(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Br * wB)
            constraint_matrix.append(np.multiply(tokamak.controlBr(xpt[0], xpt[1]), wB))

            # 2) Bz = 0

            Bz = eq.Bz(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Bz * wB)
            constraint_matrix.append(np.multiply(tokamak.controlBz(xpt[0], xpt[1]), wB))

        # Constrain points to have the same flux
        for r1, z1, r2, z2 in self.isoflux:
            # Get Psi at (r1,z1) and (r2,z2)
            p1 = eq.psiRZ(r1, z1)
            p2 = eq.psiRZ(r2, z2)
            constraint_rhs.append((p2 - p1) * wpsi)

            # Coil responses
            c1 = tokamak.controlPsi(r1, z1)
            c2 = tokamak.controlPsi(r2, z2)
            # Control for the difference between p1 and p2
            c = [(c1val - c2val) * wpsi for c1val, c2val in zip(c1, c2)]
            constraint_matrix.append(c)

        # Constrain the value of psi
        for r, z, psi in self.psivals:
            p1 = eq.psiRZ(r, z)
            constraint_rhs.append((psi - p1) * wpsi)

            # Coil responses
            c = np.multiply(tokamak.controlPsi(r, z), wpsi)
            constraint_matrix.append(c)

        if not constraint_rhs:
            raise ValueError("No constraints given")

        # Constraint matrix
        A = array(constraint_matrix, dtype=float)
        b = np.reshape(array(constraint_rhs, dtype=float), (-1,))

        # Number of controls (length of x)
        ncontrols = A.shape[1]
        if ncontrols == 0:
            self.current_change = np.zeros(0)
            eq._constraints = self
            return

        # Scale the controls so that the problem is independent of the units
        # (e.g. number of turns) of each control: x = scale * z, with z
        # dimensionless. By default scale = 1 / (norm of the column).
        if self.current_scale is None:
            colnorm = norm(A, axis=0)
            scale = np.where(colnorm > 0.0, 1.0 / np.where(colnorm > 0.0, colnorm, 1.0), 1.0)
        else:
            scale = np.abs(np.asarray(self.current_scale, dtype=float))
            if scale.shape != (ncontrols,) or np.any(scale <= 0.0):
                raise ValueError(
                    f"current_scale must be {ncontrols} positive values, one per control"
                )
        As = A * scale

        # Tikhonov regularisation, with gamma relative to the largest singular
        # value of the scaled matrix:
        # minimise || As z - b ||^2 + || gamma sigma_max z ||^2
        sigma_max = np.linalg.svd(As, compute_uv=False)[0]
        reg = self.gamma * sigma_max
        A2 = np.concatenate([As, reg * eye(ncontrols)])
        b2 = np.concatenate([b, np.zeros(ncontrols)])

        # Establish bounds on the changes in control currents from the present
        # current and the max/min limits (one pair per control)
        current_change_bnds = self._change_bounds(tokamak, ncontrols)
        lower = current_change_bnds[:, 0] / scale
        upper = current_change_bnds[:, 1] / scale

        # Least squares solve. Without bounds this is the SVD-based
        # solution of the regularised problem.
        if np.all(np.isinf(lower)) and np.all(np.isinf(upper)):
            z = np.linalg.lstsq(A2, b2, rcond=None)[0]
        else:
            z = optimize.lsq_linear(
                A2, b2, bounds=(lower, upper), method="bvls", tol=1e-14
            ).x

        # The total current limit is the only nonlinear constraint. Only
        # refine the solution if it is violated.
        if self.max_total_current is not None:
            currents = np.asarray(tokamak.controlCurrents(), dtype=float)

            def max_total_currents(zz):
                return self.max_total_current - np.sum(np.abs(currents + scale * zz))

            if max_total_currents(z) < 0.0:
                bnds = list(zip(lower, upper))
                bnorm2 = max(np.dot(b2, b2), np.finfo(float).tiny)
                sol = optimize.minimize(
                    lambda zz: np.sum((A2 @ zz - b2) ** 2) / bnorm2,
                    z,
                    method="SLSQP",
                    bounds=bnds,
                    constraints=[{"type": "ineq", "fun": max_total_currents}],
                )
                z = sol.x

        self.current_change = scale * z
        tokamak.controlAdjust(self.current_change)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def _change_bounds(self, tokamak, ncontrols):
        """
        Bounds on the change of each control current, shape (ncontrols, 2).
        current_lims has one (lower, upper) pair per control.
        """
        bnds = _control_bounds(self.current_lims, ncontrols)
        currents = np.asarray(tokamak.controlCurrents(), dtype=float)
        bnds -= currents[:, None]
        return bnds

    def plot(self, axis=None, show=True):
        """
        Plots constraints used for coil current control

        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning

        """
        from .plotting import plotConstraints

        return plotConstraints(self, axis=axis, show=show)


class ConstrainPsi2D:
    """
    Adjusts coil currents to minimise the square differences
    between psi[R,Z] and a target psi.

    Ignores constant offset differences between psi array
    """

    def __init__(self, target_psi, weights=None):
        """
        target_psi : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psi
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        """
        if weights is None:
            weights = np.full(target_psi.shape, 1.0)

        # Remove the average so constant offsets are ignored
        self.target_psi = target_psi - np.average(target_psi, weights=weights)

        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psi_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psi_difference(self, currents, eq):
        """
        Difference between psi from equilibrium with the given currents
        and the target psi
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()
        psi_av = np.average(psi, weights=self.weights)
        return (
            (psi - psi_av - self.target_psi) * self.weights
        ).ravel()  # flatten array


class ConstrainPsiNorm2D:
    """
    Adjusts coil currents to minimise the square differences
    between normalised psi[R,Z] and a target normalised psi.
    """

    def __init__(self, target_psinorm, weights=1.0):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        """
        self.target_psinorm = target_psinorm
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psinorm_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psinorm_difference(self, currents, eq):
        """
        Difference between normalised psi from equilibrium with the given currents
        and the target psinorm
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()

        eq._updateBoundaryPsi(psi)
        psi_bndry = eq.psi_bndry
        psi_axis = eq.psi_axis

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        return (
            (psi_norm - self.target_psinorm) * self.weights
        ).ravel()  # flatten array


class ConstrainPsi2DAdvanced:
    """
    Adjusts coil currents to minimise the square differences
    between psi[R,Z] and a target psi.

    Attempts to also constrain the coil currents as in the 'constrain' class.
    """

    def __init__(
        self, target_psi, weights=1.0, current_lims=None, max_total_current=None
    ):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        current_lims: List of tuples
            Optional list of tuples representing constraints on the control
            currents, one per control (see constrain), to be used
            when reconstructing the equilibrium from the geqdsk file.
            [(l1,u1),(l2,u2)...(lN,uN)]. Default is no limits.

        Create an instance, specifying the constraints to apply
        """

        self.current_lims = current_lims
        self.target_psi = target_psi
        # A scalar weight (the default) must be expanded for np.average
        self.weights = np.broadcast_to(weights, np.shape(target_psi))

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = np.asarray(tokamak.controlCurrents())
        ncontrols = len(start_currents)

        # In order for the optimisation to work, the initial guess must be within the
        # bounds supplied. Hence, check start_currents and adjust accordingly to be within bounds
        current_bnds = _control_bounds(self.current_lims, ncontrols)
        start_currents = np.clip(start_currents, current_bnds[:, 0], current_bnds[:, 1])

        # Least squares optimisation of difference in target v achieved normalised psi
        # applied with bounds on coil currents
        end_currents = optimize.minimize(
            self.psi_difference,
            start_currents,
            method="L-BFGS-B",
            bounds=current_bnds,
            args=(eq,),
        ).x

        # Set the latest coil currents
        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psi_difference(self, currents, eq):
        """
        Sum of the squares of the differences between the achieved
        psi and the target psi.
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()
        eq._updateBoundaryPsi(psi)

        psi_av = np.average(psi, weights=self.weights)
        diff = (psi - psi_av - self.target_psi) * self.weights
        return np.sum(diff * diff)


class ConstrainPsiNorm2DAdvanced:
    """
    Adjusts coil currents to minimise the square differences
    between normalised psi[R,Z] and a target normalised psi.

    Attempts to also constrain the coil currents as in the 'constrain' class.
    """

    def __init__(self, target_psinorm, weights=1.0, current_lims=None):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        current_lims: List of tuples
            Optional list of tuples representing constraints on the control
            currents, one per control (see constrain), to be used
            when reconstructing the equilibrium from the geqdsk file.
            [(l1,u1),(l2,u2)...(lN,uN)]. Default is no limits.

        Create an instance, specifying the constraints to apply
        """

        self.current_lims = current_lims
        self.target_psinorm = target_psinorm
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = np.asarray(tokamak.controlCurrents())
        ncontrols = len(start_currents)

        # In order for the optimisation to work, the initial guess must be within the
        # bounds supplied. Hence, check start_currents and adjust accordingly to be within bounds
        current_bnds = _control_bounds(self.current_lims, ncontrols)
        start_currents = np.clip(start_currents, current_bnds[:, 0], current_bnds[:, 1])

        # Least squares optimisation of difference in target v achieved normalised psi
        # applied with bounds on coil currents
        end_currents = optimize.minimize(
            self.psinorm_difference,
            start_currents,
            method="L-BFGS-B",
            bounds=current_bnds,
            args=(eq,),
        ).x

        # Set the latest coil currents
        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psinorm_difference(self, currents, eq):
        """
        Sum of the squares of the differences between the achieved normalised
        psi and the target normalised psi.
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()

        eq._updateBoundaryPsi(psi)
        psi_bndry = eq.psi_bndry
        psi_axis = eq.psi_axis

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
        diff = (psi_norm - self.target_psinorm) * self.weights
        return np.sum(diff * diff)
