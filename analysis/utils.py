import numpy as np


def gas_mult_factor(Q, E_rad, E_ion_pair, e):
    """
    Get the gas multiplication factor M.
    Equation 4 of the article.

    :param Q: collected charge
    :param E_rad: energy of the incident radiation
    :param E_ion_pair: energy required to create an electron-ion pair
    :param e: electron charge
    """
    return Q / ((E_rad / E_ion_pair) * e)


def diethorn(V, a, b, p, delta_V, K):
    """
    Get the natural logarithm of the gas multiplication factor.
    Equation 5 of the article.

    :param V: bias voltage
    :param a: anode radius
    :param b: cathode radius
    :param p: gas pressure
    :param delta_V: dependent on the gas mixture
    :param K: dependent on the gas mixture
    """
    ln_ba = np.log(b/a)
    return V / ln_ba * np.log(2) / delta_V * np.log(V / (p*a*ln_ba) - np.log(K))


def gas_mult_factor_p10(V, a, b, p):
    """
    Gas multiplication factor for P-10 (90 % argon, 10 % CH4) gas.
    Values are from the table 1 of
    "Measurement of the gas constants for various
    proportional-counter gas mixtures (Wolff, 1973).
    """
    K = 4.8e-4
    K_std = 0.3e-4
    delta_V = 23.6  # eV
    delta_V_std = 21.8  # eV
    # TODO


def print_title(title: str):
    print("###")
    print(title)
    print("###")
