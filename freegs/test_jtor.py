import numpy as np

from . import equilibrium, jtor


def test_psinorm_range():
    """Test that the profiles produce finite values outside core"""

    eq = equilibrium.Equilibrium(Rmin=0.5, Rmax=1.5, Zmin=-1.0, Zmax=1.0, nx=33, ny=33)

    for profiles in [
        jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0),
        jtor.ConstrainBetapIp(eq, 1.0, 2e5, 2.0),
    ]:
        current_density = profiles.Jtor(eq.R, eq.Z, eq.psi())
        assert np.all(np.isfinite(current_density))

        assert profiles.pprime(1.0) == 0.0
        assert profiles.pprime(1.1) == 0.0
        assert np.isfinite(profiles.pprime(-0.32))

        assert profiles.ffprime(1.0) == 0.0
        assert profiles.ffprime(1.1) == 0.0
        assert np.isfinite(profiles.ffprime(-0.32))


def _two_wells(R, Z, shift):
    # Two wells (O-points) joined by a saddle at (1.5, 0): one X-point. The
    # primary O-point is the deeper one; ``shift`` moves the saddle flux.
    well = np.exp(-((R - 1.5) ** 2 + (Z - 0.3) ** 2) / 0.09)
    well2 = 0.9 * np.exp(-((R - 1.5) ** 2 + (Z + 0.3) ** 2) / 0.09)
    return well + well2 + shift * np.exp(-((R - 1.5) ** 2 + Z**2) / 0.5)


def test_cut_profile_current_is_continuous_in_psi():
    """With p' and ff' finite at the boundary the current must vary
    continuously with psi: a tiny change of psi changes Ip by a tiny amount.
    A hard cell cutoff jumps by whole cells, so a Picard iteration on it has
    no exact fixed point."""
    n = 65
    r1d = np.linspace(1.0, 2.0, n)
    z1d = np.linspace(-1.0, 1.0, n)
    R, Z = np.meshgrid(r1d, z1d, indexing="ij")
    dA = (r1d[1] - r1d[0]) * (z1d[1] - z1d[0])

    shifts = np.linspace(0.0, 0.03, 301)
    jump = {}
    for smooth in (True, False):
        prof = jtor.ProfilesPprimeFfprime(
            lambda x: -1e3 * np.ones_like(x), lambda x: -0.1 * np.ones_like(x), 2.0
        )
        prof.smooth_mask = smooth
        ip = np.array(
            [np.sum(prof.Jtor(R, Z, _two_wells(R, Z, s))) * dA for s in shifts]
        )
        jump[smooth] = np.max(np.abs(np.diff(ip)))
    # Current carried by one boundary cell
    cell = np.max(np.abs(prof.Jtor(R, Z, _two_wells(R, Z, 0.0)))) * dA
    # Smooth: 1e-4 in the saddle flux moves a small fraction of a cell's current
    assert jump[True] < 0.05 * cell
    # Hard: whole cells switch on and off
    assert jump[False] > 0.3 * cell


def test_equilibrium_smooth_mask_reaches_profiles():
    eq = equilibrium.Equilibrium(
        Rmin=0.5, Rmax=1.5, Zmin=-1.0, Zmax=1.0, nx=33, ny=33, smooth_mask=True
    )
    prof = jtor.ProfilesPprimeFfprime(lambda x: 0.0 * x, lambda x: 0.0 * x, 2.0)
    assert not prof.smooth_mask
    eq.solve(prof, Jtor=np.zeros_like(eq.R))
    assert prof.smooth_mask
