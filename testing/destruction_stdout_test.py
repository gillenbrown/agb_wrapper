import pytest

import numpy as np
from scipy import special
from pytest import approx

def true_f_bound(eps_int):
    alpha_star = 0.48
    f_sat = 0.94
    term_a = special.erf(np.sqrt(3 * eps_int / alpha_star))
    term_b = np.sqrt(12 * eps_int / (np.pi * alpha_star))
    term_c = np.exp(-3 * eps_int / alpha_star)
    return (term_a - term_b * term_c) * f_sat

def test_destruction_stdout():
    num_completed = 0
    with open("./f_bound_stdout.txt", "r") as stdout:
        for idx, line in enumerate(stdout):
            if "eps_int" not in line:
                continue

            split_line = line.split(" ")

            initial_mass = float(split_line[-7].replace(",", ""))
            star_ibound = float(split_line[-5].replace(",", ""))
            eps_int = float(split_line[-3].replace(",", ""))
            f_bound = float(split_line[-1].replace(",", ""))

            true_eps_int = initial_mass / star_ibound
            assert true_eps_int == approx(eps_int, rel=1E-6, abs=1E-15)
            assert true_f_bound(true_eps_int) == approx(f_bound, rel=1E-6, abs=1E-15)

            num_completed += 1

    assert num_completed > 0
