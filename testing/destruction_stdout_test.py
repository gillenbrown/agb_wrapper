import pytest

from scipy import special
from pytest import approx

def true_f_bound(ept_int):
    return 1

def test_destruction_stdout():
    num_completed = 0
    with open("./stdout", "r") as stdout:
        for idx, line in enumerate(stdout):
            if "eps_int" not in line:
                continue

            split_line = line.split(",")

            initial_mass = float(split_line[-7])
            star_ibound = float(split_line[-5])
            eps_int = float(split_line[-3])
            f_bound = float(split_line[-1])

            assert initial_mass / star_ibound == approx(eps_int)
            assert true_f_bound(eps_int) == approx(f_bound)

            num_completed += 1

    assert num_completed > 0
