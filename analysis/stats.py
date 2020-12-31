"""
Statistical distributions and related functions
"""

# import numba
import numpy as np
import scipy.stats


# Distributions

def exp_fit(x, slope, offset):
    """Equation 3 of the article"""
    return np.exp(slope*x + offset)


# @numba.njit
def gaussian(x, mu, sigma):
    return scipy.stats.norm.pdf(x, mu, sigma)


# @numba.njit
def gaussian_double(x, a1, m1, s1, a2, m2, s2):
    """Sum of two Gaussians"""
    return a1*gaussian(x, m1, s1) + a2*gaussian(x, m2, s2)


# @numba.njit
def gaussian_scaled(x, a, mu, sigma):
    return a*scipy.stats.norm.pdf(x, mu, sigma)


# Utilities

def gaussian_fwhm(sigma):
    return 2*np.sqrt(2*np.log(2))*sigma
