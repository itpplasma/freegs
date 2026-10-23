"""Safety-factor oracle for nested surfaces without an X-point."""
from types import SimpleNamespace
import numpy as np
import pytest
from .equilibrium import Equilibrium
from .boundary import fixedBoundary


def circular_equilibrium():
    R0, a, C, F = 6.2, .62, 1.7, 32.86
    r = np.linspace(R0-1.25*a,R0+1.25*a,65)
    z = np.linspace(-1.25*a,1.25*a,65)
    R,Z = np.meshgrid(r,z,indexing="ij")
    psi = C*((R-R0)**2+Z**2)
    eq = Equilibrium(Rmin=r[0],Rmax=r[-1],Zmin=z[0],Zmax=z[-1],
                     nx=65,ny=65,psi=psi,boundary=fixedBoundary)
    eq.psi_bndry = C*a*a
    eq._profiles = SimpleNamespace(fpol=lambda s:np.full_like(s,F))
    return eq,R0,a,C,F


def test_q_without_xpoint_matches_circular_line_integral():
    eq,R0,a,C,F = circular_equilibrium()
    s = np.array([.2,.5,.8])
    # q = (F/(2 C)) <1/R> around each circular surface.
    expected = F/(2*C*np.sqrt(R0*R0-s*a*a))
    np.testing.assert_allclose(eq.q(s),expected,rtol=1e-3)


def test_q_without_xpoint_requires_known_boundary_flux():
    eq,*_ = circular_equilibrium()
    eq.psi_bndry = None
    with pytest.raises(ValueError,match="boundary"):
        eq.q(np.array([.5]))


def signed_circular_equilibrium(sign_psi, sign_f):
    # psi = sign_psi*C*r^2: Ip along -phi when sign_psi > 0, B0 sign from F.
    eq,R0,a,C,F = circular_equilibrium()
    eq.plasma_psi = sign_psi*eq.plasma_psi
    eq.psi_bndry = sign_psi*C*a*a
    eq.psi_axis = 0.0
    eq._profiles = SimpleNamespace(fpol=lambda s:np.full_like(s,sign_f*F))
    return eq,R0,a,C,F


@pytest.mark.parametrize("sign_psi",[1,-1])
@pytest.mark.parametrize("sign_f",[1,-1])
def test_q_sign_follows_cocos_and_toroidal_flux_is_exact(sign_psi,sign_f):
    eq,R0,a,C,F = signed_circular_equilibrium(sign_psi,sign_f)
    # psi increasing outward (sign_psi>0) is Ip<0: Ip = -sign_psi.
    ip_b0 = -sign_psi*sign_f
    s = np.array([.2,.5,.8])
    magnitude = F/(2*C*np.sqrt(R0*R0-s*a*a))
    # Sauter COCOS tables: sign(q) = sigma_rho_theta_phi*sign(Ip*B0)
    assert eq.cocos == 3
    np.testing.assert_allclose(eq.q(s),-ip_b0*magnitude,rtol=1e-3)
    np.testing.assert_allclose(eq.q(s,cocos=1),ip_b0*magnitude,rtol=1e-3)
    np.testing.assert_allclose(eq.q(s,cocos=13),-ip_b0*magnitude,rtol=1e-3)
    # Flux through a circle: F*int dA/R = 2 pi F (R0 - sqrt(R0^2 - rho^2))
    psi = eq.psi_axis+s*(eq.psi_bndry-eq.psi_axis)
    expected = sign_f*2*np.pi*F*(R0-np.sqrt(R0*R0-s*a*a))
    np.testing.assert_allclose(eq.tor_flux(psi),expected,rtol=2e-3)
    sn = np.linspace(0,1,21)
    rho = eq.rhotor(eq.psi_axis+sn*(eq.psi_bndry-eq.psi_axis))
    assert rho[0] == pytest.approx(0,abs=1e-6) and rho[-1] == pytest.approx(1)
    assert np.all(np.diff(rho) > 0)
    expected = (R0-np.sqrt(R0*R0-sn*a*a))/(R0-np.sqrt(R0*R0-a*a))
    np.testing.assert_allclose(rho**2,expected,rtol=2e-3,atol=1e-4)


def test_q_scalar_list_array_and_default_tor_flux():
    eq,*_ = circular_equilibrium()
    values = [.02,.5,.995]
    from_array = eq.q(np.array(values))
    np.testing.assert_allclose(eq.q(values),from_array)
    for v,expected in zip(values,from_array):
        assert isinstance(eq.q(v),float)
        assert eq.q(v) == pytest.approx(expected)
    assert eq.q([.5]) == pytest.approx(eq.q(.5))
    flux = eq.tor_flux()
    assert flux.shape == (101,) and flux[0] == 0
    assert np.all(np.diff(flux) > 0)
    assert eq.tor_flux(eq.psi_axis+.5*(eq.psi_bndry-eq.psi_axis)) == pytest.approx(
        flux[50],rel=1e-12)
    with pytest.raises(ValueError,match="COCOS"):
        eq.q(.5,cocos=9)
