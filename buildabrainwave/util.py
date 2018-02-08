import numpy as np
from numpy import exp, allclose, asarray, linspace, argmax, zeros, diag
from numpy import abs as npabs
from numpy import mean as npmean
from numpy.random import normal
from fakespikes.rates import stim, constant
import sys


def create_times(tspan, dt):
    """Define time
    
    Params
    ------
    tspan : tuple (float, float)
        Start and stop times (seconds)
    dt : numeric
        Time step length
    """
    t0 = tspan[0]
    t1 = tspan[1]
    return linspace(t0, t1, np.int(np.round((t1 - t0) / dt)))


def threshold_linear(x):
    """Output non-linearity."""

    return max(0, x)


def sigmoid(Isyn, I, c, g):
    """Output non-linearity."""
    return ((c * Isyn) - I) / (1 - exp(-g * ((c * Isyn) - I)))


def ornstein_uhlenbeck(rs, t, sigma=0.5, loc=None):
    """A (independent) Brownian noise process."""
    if loc is None:
        loc = range(rs.size)

    sigmas = zeros(rs.size)
    sigmas[loc] = sigma  # Locations of re1,ri1,re2,ri2

    return diag(sigmas)
