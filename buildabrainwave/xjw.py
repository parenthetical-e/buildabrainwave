from __future__ import division

from sdeint import itoint
from functools import partial

from numpy import exp, allclose, asarray, linspace, argmax, zeros, diag
from numpy import abs as npabs
from numpy import mean as npmean
from numpy.random import normal

from buildabrainwave.util import ornstein_uhlenbeck, create_times
from buildabrainwave.util import threshold_linear as phi


def xjw(ys,
        t,
        J_ee=2.1,
        J_ie=1.9,
        J_ei=1.5,
        J_ii=1.1,
        tau_e=10e-3,
        tau_i=40e-3,
        I_e=120,
        I_i=150):
    """A version of XJW's classic two-population rate model.
    
    Citation
    --------
    Dayan P & Abbott LF, Theoretical Neuroscience, MIT Press, 2005, p266.
    """
    re, ri, s_ee, s_ie, s_ei, s_ii = ys

    # Internal synaptic dynamics
    I_syn_e = (re * J_ee) - (ri * J_ie) + I_e
    I_syn_i = (re * J_ei) - (ri * J_ii) + I_i

    # Update rates, passing synaptic currents (I_syn_*) through
    # the output nonlinearity, phi.
    re = (-re + phi(I_syn_e)) / tau_e
    ri = (-ri + phi(I_syn_i)) / tau_i

    # Repackage
    ys = [re, ri, s_ee, s_ie, s_ei, s_ii]

    return asarray(ys)


def run(t,
        re_0=8.0,
        ri_0=12.0,
        J_ee=2.1,
        J_ie=1.9,
        J_ei=1.5,
        J_ii=1.1,
        tau_e=10e-3,
        tau_i=30e-3,
        I_e=12,
        I_i=8,
        sigma=1,
        dt=1e-4):

    rs_0 = asarray([re_0, ri_0, re_0, ri_0, re_0, ri_0])

    # !
    times = create_times((0, t), dt)

    # If sigma > 0: we re-define xjw as a stochastic ODE.
    g = partial(ornstein_uhlenbeck, sigma=sigma, loc=[0, 1])  # re/i locs
    f = partial(
        xjw,
        J_ee=J_ee,
        J_ei=J_ei,
        J_ie=J_ie,
        J_ii=J_ii,
        tau_e=tau_e,
        tau_i=tau_i,
        I_e=I_e,
        I_i=I_i)

    rs = itoint(f, g, rs_0, times)

    return times, rs
