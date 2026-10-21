"""
Functions to impose boundary conditions on psi(plasma)

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.

"""

from functools import lru_cache

import numpy as np
from numpy import concatenate, sqrt
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.integrate import romb  # Romberg integration

from .gradshafranov import Greens, mu0


def fixedBoundary(eq, Jtor, psi):
    """
    Set psi=0 on all boundaries

    Inputs
    ------

    eq    Equilibrium object (not used)
    Jtor  2D array of toroidal current (not used)
    psi   2D array of psi values (modified by call)

    Returns
    -------
    None
    """
    psi[0, :] = 0.0
    psi[:, 0] = 0.0
    psi[-1, :] = 0.0
    psi[:, -1] = 0.0


def freeBoundary(eq, Jtor, psi):
    """
    Apply a free boundary condition using Green's functions

    Note: This method is inefficient because it requires
    an integral over the area of the domain for each point
    on the boundary.

    Inputs
    ------

    eq    Equilibrium object (not used)
    Jtor  2D array of toroidal current (not used)
    psi   2D array of psi values (modified by call)

    Returns
    -------

    None
    """

    # Get the (R,Z) coordinates from the Equilibrium object
    R = eq.R
    Z = eq.Z

    nx, ny = psi.shape

    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    # List of indices on the boundary
    bndry_indices = concatenate(
        [
            [(x, 0) for x in range(nx)],
            [(x, ny - 1) for x in range(nx)],
            [(0, y) for y in range(ny)],
            [(nx - 1, y) for y in range(ny)],
        ]
    )

    for x, y in bndry_indices:
        # Calculate the response of the boundary point
        # to each cell in the plasma domain
        greenfunc = Greens(R, Z, R[x, y], Z[x, y])

        # Prevent infinity/nan by removing (x,y) point
        greenfunc[x, y] = 0.0

        # Integrate over the domain
        psi[x, y] = romb(romb(greenfunc * Jtor)) * dR * dZ


def freeBoundaryHagenow(eq, Jtor, psi):
    """
    Apply a free boundary using von Hagenow's method

    Inputs
    ------

    eq    Equilibrium object (not used)
    Jtor  2D array of toroidal current (not used)
    psi   2D array of psi values (modified by call)

    Returns
    -------

    None
    """

    # Get the (R,Z) coordinates from the Equilibrium object
    R = eq.R
    Z = eq.Z

    nx, ny = psi.shape

    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    # solver RHS
    rhs = eq.R * Jtor

    # Apply a zero-value boundary condition
    rhs[0, :] = 0.0
    rhs[:, 0] = 0.0
    rhs[-1, :] = 0.0
    rhs[:, -1] = 0.0

    # Solve with a fixed boundary
    psi_fixed = eq.callSolver(psi, rhs)

    # Differentiate at the boundary
    # Note: normal is out of the domain, so this is -dU/dx

    # Fourth-order one-sided differences
    coeffs = [(0, 25.0 / 12), (1, -4.0), (2, 3.0), (3, -16.0 / 12), (4, 1.0 / 4)]

    dUdn_L = (
        sum([weight * psi_fixed[index, :] for index, weight in coeffs]) / dR
    )  # left boundary

    dUdn_R = (
        sum([weight * psi_fixed[-(1 + index), :] for index, weight in coeffs]) / dR
    )  # Right boundary

    dUdn_D = (
        sum([weight * psi_fixed[:, index] for index, weight in coeffs]) / dZ
    )  # Down boundary

    dUdn_U = (
        sum([weight * psi_fixed[:, -(1 + index)] for index, weight in coeffs]) / dZ
    )  # Upper boundary

    dd = sqrt(dR**2 + dZ**2)  # Diagonal spacing

    # Left down corner
    dUdn_L[0] = dUdn_D[0] = (
        sum([weight * psi_fixed[index, index] for index, weight in coeffs]) / dd
    )

    # Left upper corner
    dUdn_L[-1] = dUdn_U[0] = (
        sum([weight * psi_fixed[index, -(1 + index)] for index, weight in coeffs]) / dd
    )

    # Right down corner
    dUdn_R[0] = dUdn_D[-1] = (
        sum([weight * psi_fixed[-(1 + index), index] for index, weight in coeffs]) / dd
    )

    # Right upper corner
    dUdn_R[-1] = dUdn_U[-1] = (
        sum(
            [weight * psi_fixed[-(1 + index), -(1 + index)] for index, weight in coeffs]
        )
        / dd
    )

    # Integrate at the boundary itself. Each side uses cubic product
    # integration, including analytic logarithmic moments at coincident nodes.
    weights = _hagenow_weights(nx, ny, R[0, 0], R[-1, 0], Z[0, 0], Z[0, -1])
    values = weights @ concatenate([dUdn_L, dUdn_R, dUdn_D, dUdn_U])
    psi[0, :] = values[:ny]
    psi[-1, :] = values[ny : 2 * ny]
    psi[:, 0] = values[2 * ny : 2 * ny + nx]
    psi[:, -1] = values[2 * ny + nx :]


def _hagenow_side(R, Z, Robs, Zobs):
    """Weights for integral G * dUdn / R on one straight boundary side.

    Interpolate dUdn/sqrt(R) cubically on each panel. On panels touching the
    observation point, split the kernel into a smooth part and a smooth
    coefficient times log(distance/panel_length). Integrate the latter with
    logarithmic product weights; eight-point Gauss handles the smooth part
    and other panels. Work per observation is linear in the side length.
    """
    n = len(R)
    h = np.hypot(R[1] - R[0], Z[1] - Z[0])
    panels = np.arange(n - 1)
    indices = np.minimum(np.maximum(panels - 1, 0), n - 4)[:, None] + np.arange(4)
    nodes = indices - panels[:, None]
    t, gauss, log_gauss = _hagenow_quadrature()
    basis = np.ones((n - 1, 8, 4))
    for j in range(4):
        for k in range(4):
            if j != k:
                basis[:, :, j] *= (t[None, :] - nodes[:, k, None]) / (
                    nodes[:, j, None] - nodes[:, k, None]
                )

    Rq = R[:-1, None] + t * np.diff(R)[:, None]
    Zq = Z[:-1, None] + t * np.diff(Z)[:, None]
    distance2 = (Rq - Robs) ** 2 + (Zq - Zobs) ** 2
    # Compute the complementary elliptic parameter directly: unlike Greens'
    # clipping, ellipkm1 retains the logarithm arbitrarily close to a source.
    complement = distance2 / ((Rq + Robs) ** 2 + (Zq - Zobs) ** 2)
    k2 = 1 - complement
    coefficient = mu0 * np.sqrt(Robs) / (2 * np.pi)
    kernel = (
        coefficient
        * ((1 + complement) * ellipkm1(complement) - 2 * ellipe(k2))
        / np.sqrt(k2)
    )

    correction = np.zeros_like(kernel)
    for endpoint, parameter, log_weights in [
        (0, t, log_gauss),
        (1, 1 - t, log_gauss[::-1]),
    ]:
        coincident = np.flatnonzero(
            (R[panels + endpoint] == Robs) & (Z[panels + endpoint] == Zobs)
        )
        for panel in coincident:
            c = complement[panel]
            # K(1-c) has log(4/sqrt(c)) coefficient 2*K(c)/pi;
            # E(1-c) has coefficient 2*(K(c)-E(c))/pi. Thus this is
            # the complete log coefficient, including higher powers of c.
            log_coefficient = (
                coefficient
                * (2 / np.pi)
                * (2 * ellipe(c) - (1 - c) * ellipk(c))
                / np.sqrt(1 - c)
            )
            kernel[panel] += log_coefficient * np.log(parameter)
            correction[panel] = -log_coefficient * log_weights

    panel_weights = h * np.einsum("pq,pqj->pj", kernel * gauss + correction, basis)
    weights = np.zeros(n)
    np.add.at(weights, indices.ravel(), panel_weights.ravel())
    return weights / np.sqrt(R)


@lru_cache(maxsize=1)
def _hagenow_quadrature():
    """Gauss nodes and product weights for integral_0^1 f(t)*log(t) dt.

    Integrate each degree-seven cardinal polynomial analytically using
    integral_0^1 t^k log(t) dt = -1/(k+1)^2. No length cutoff is involved.
    """
    nodes, weights = leggauss(8)
    nodes = (nodes + 1) / 2
    log_weights = []
    for j, node in enumerate(nodes):
        others = np.delete(nodes, j)
        cardinal = Polynomial.fromroots(others) / np.prod(node - others)
        log_weights.append(sum(-c / (k + 1) ** 2 for k, c in enumerate(cardinal.coef)))
    return nodes, weights / 2, np.array(log_weights)


@lru_cache(maxsize=4)
def _hagenow_weights(nx, ny, Rmin, Rmax, Zmin, Zmax):
    """Cache geometry only; no equilibrium or current-dependent state."""
    r = np.linspace(Rmin, Rmax, nx)
    z = np.linspace(Zmin, Zmax, ny)
    sides = [
        (np.full(ny, Rmin), z),
        (np.full(ny, Rmax), z),
        (r, np.full(nx, Zmin)),
        (r, np.full(nx, Zmax)),
    ]
    observations = np.concatenate([np.column_stack(side) for side in sides])
    return np.array(
        [
            np.concatenate([_hagenow_side(R, Z, Robs, Zobs) for R, Z in sides])
            for Robs, Zobs in observations
        ]
    )
