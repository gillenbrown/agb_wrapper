import pytest
import random

import numpy as np
from scipy import special
from pytest import approx

from pathlib import Path
this_dir = Path(__file__).absolute().parent

def true_f_bound(eps_int):
    alpha_star = 0.48
    f_sat = 0.94
    term_a = special.erf(np.sqrt(3 * eps_int / alpha_star))
    term_b = np.sqrt(12 * eps_int / (np.pi * alpha_star))
    term_c = np.exp(-3 * eps_int / alpha_star)
    return (term_a - term_b * term_c) * f_sat

# go through the file and parse it
f_bound_points = []
stdout_file = this_dir/"f_bound_stdout.txt"
with stdout_file.open("r") as stdout:
    for idx, line in enumerate(stdout):
        if "eps_int" not in line:
            continue

        split_line = line.split(" ")

        initial_mass = float(split_line[-7].replace(",", ""))
        star_ibound = float(split_line[-5].replace(",", ""))
        eps_int = float(split_line[-3].replace(",", ""))
        f_bound = float(split_line[-1].replace(",", ""))

        f_bound_points.append({"initial_mass": initial_mass,
                               "star_ibound": star_ibound,
                               "eps_int": eps_int,
                               "f_bound": f_bound})

assert len(f_bound_points) > 10
# we only need some of these
n_tests = 100000000
if len(f_bound_points) > n_tests:
    f_bound_points = random.sample(f_bound_points, n_tests)

@pytest.mark.parametrize("point", f_bound_points)
def test_eps_int(point):
    true_eps_int = point["initial_mass"] / point["star_ibound"]
    assert true_eps_int == approx(point["eps_int"])

@pytest.mark.parametrize("point", f_bound_points)
def test_eps_int(point):
    assert true_f_bound(point["eps_int"]) == approx(point["f_bound"])

