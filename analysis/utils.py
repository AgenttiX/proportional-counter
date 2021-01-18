import typing as tp

import numpy as np
import sympy as sp

import const


def diethorn(V, a, b, p, delta_V, K, std_V, std_a, std_b, std_p, std_delta_V, std_K):
    """
    Get the natural logarithm of the gas multiplication factor.
    The equation 5 of the article is wrong!
    The correct form is the equation 6.10 of Knoll's book (and equation 1 of Wolff, 1973)

    :param V: bias voltage
    :param a: anode radius
    :param b: cathode radius
    :param p: gas pressure
    :param delta_V: dependent on the gas mixture
    :param K: dependent on the gas mixture
    """
    # ln_ba = np.log(b/a)
    # val = V / ln_ba * np.log(2) / delta_V * (np.log(V / (p*a*ln_ba)) - np.log(K))

    V_sym, a_sym, b_sym, p_sym, delta_V_sym, K_sym = sp.symbols("V a b p ΔV K", real=True, positive=True)
    ln_ba_sym = sp.log(b/a)
    func = V_sym / ln_ba_sym * sp.log(2) / delta_V_sym * (sp.log(V_sym / (p_sym*a_sym*ln_ba_sym)) - sp.log(K))
    syms = [V_sym, a_sym, b_sym, p_sym, delta_V_sym, K_sym]
    vals = np.array([V, a, b, p, delta_V, K])
    stds = np.array([std_V, std_a, std_b, std_p, std_delta_V, std_K])

    val, variance = error_propagation(func, syms, vals, stds=stds)
    return float(val), np.sqrt(float(variance))


def error_propagation(
        func: sp.Function,
        syms: tp.List[sp.Symbol],
        vals: np.ndarray,
        covar: np.ndarray = None,
        stds: np.ndarray = None):
    """Error propagation for an arbitrary function"""
    if (covar is None) == (stds is None):
        raise ValueError("Give either covariances or stds")
    if covar is None:
        covar = np.diag(stds)**2

    # Variables and their means in a form that SymPy understands
    subs = list(zip(syms, vals))
    # Value of the function at the means
    val = func.subs(subs)

    # Partial derivatives for each variable
    A_sym = [func.diff(sym) for sym in syms]
    # Numerical values for the partial derivatives (equation 1.56 of Cowan, 1998)
    A_vec = np.array([A.subs(subs) for A in A_sym])
    # Error propagation according to equation 1.55 of Cowan, 1998
    U = A_vec @ covar @ A_vec
    return val, U


def gain(gain_pre, gain_spec, gain_std_pre, gain_std_spec):
    """Total gain for a system consisting of a pre-amplifier and a spectral amplifier"""
    gain = gain_pre * gain_spec
    gain_std = np.sqrt((gain_spec*gain_std_pre)**2 + (gain_pre*gain_std_spec)**2)
    return gain, gain_std


def gas_mult_factor(Q, E_rad, E_ion_pair, e: float = const.ELEMENTARY_CHARGE):
    """
    Get the gas multiplication factor M.
    Equation 4 of the article.

    :param Q: collected charge
    :param E_rad: energy of the incident radiation
    :param E_ion_pair: energy required to create an electron-ion pair
    :param e: electron charge
    """
    return Q / ((E_rad / E_ion_pair) * e)


def log_gas_mult_factor_p10(V, a, b, p, std_V, std_a, std_b, std_p):
    """
    Theoretical as multiplication factor for P-10 (90 % argon, 10 % CH4) gas.
    Values are from the table 1 of
    "Measurement of the gas constants for various
    proportional-counter gas mixtures (Wolff, 1973).
    """
    K_Wolff = 4.8e-4  # V/(cm atm)
    std_K_Wolff = 0.3e-4  # V/(cm atm)
    K = K_Wolff / (const.ATM_TO_PA * 0.01)
    std_K = std_K_Wolff / (const.ATM_TO_PA * 0.01)
    print(f"K = {K:.3e} +/- {std_K:.3e} V/m Pa")

    # TODO: this is a manual fix
    K = 10
    std_K = 10

    delta_V = 23.6  # eV
    std_delta_V = 21.8  # eV
    return diethorn(
        V=V, a=a, b=b, p=p, delta_V=delta_V, K=K,
        std_V=std_V, std_a=std_a, std_b=std_b, std_p=std_p, std_delta_V=std_delta_V, std_K=std_K
    )


def print_title(title: str):
    print("###")
    print(title)
    print("###")
