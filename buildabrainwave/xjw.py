from __future__ import division

import fire

from sdeint import itoint
from functools import partial

from numpy import exp, allclose, asarray, linspace, argmax, zeros, diag
from numpy import abs as npabs
from numpy import mean as npmean
from numpy.random import normal

from buildabrainwave.util import phi, ornstein_uhlenbeck, create_times


def xjw(ys,
        t,
        J_ee=2.1,
        J_ie=1.9,
        J_ei=1.5,
        J_ii=1.1,
        tau_e=40e-3,
        tau_i=20e-3,
        tau_n=20e-4,
        I_e=120,
        I_i=150):
    """A version of XJW's classic two-population rate model.
    
    Citation
    -------
    TODO
    """
    re, ri, s_ee, s_ie, s_ei, s_ii = ys

    # Output nonlinear params
    c = 1.1  # lit?
    g = 1 / 10  # lit?

    # Internal synaptic dynamics
    s_ee = (-s_ee / tau_e) + re
    s_ie = (-s_ie / tau_i) + ri
    s_ei = (-s_ei / tau_e) + re
    s_ii = (-s_ii / tau_i) + ri

    I_syn_e = (s_ee * J_ee) - (s_ie * J_ie) + I_e
    I_syn_i = (s_ei * J_ei) - (s_ii * J_ii) + I_i

    # Update rates, passing synaptic currents (I_syn_*) through
    # the output nonlinearity, phi.
    re = (-re + phi(I_syn_e, I_e, c, g)) / tau_n
    ri = (-ri + phi(I_syn_i, I_i, c, g)) / tau_n

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
        tau_e=40e-3,
        tau_i=20e-3,
        tau_n=20e-4,
        I_e=120,
        I_i=85,
        sigma=1,
        dt=1e-4):
    rs_0 = asarray([re_0, ri_0, 0, 0, 0, 0])

    # !
    times = create_times((0, t), dt)
    g = partial(ornstein_uhlenbeck, sigma=sigma, loc=[0, 1])  # re/i locs
    f = partial(
        xjw,
        J_ee=J_ee,
        J_ei=J_ei,
        J_ie=J_ie,
        J_ii=J_ii,
        tau_e=tau_e,
        tau_i=tau_i,
        tau_n=tau_n,
        I_e=I_e,
        I_i=I_i)

    rs = itoint(f, g, rs_0, times)

    return times, rs


# demo
if __name__ == "__main__":
    fire.Fire(run)