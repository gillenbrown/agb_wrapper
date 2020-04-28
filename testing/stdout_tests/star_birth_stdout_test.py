import pytest
from pytest import approx

import random

from parse_stdout import parse_file

n_tests = 4
rel = 1E-6

timesteps_all = parse_file("./stdout_star_birth.txt", "star_birth")

# we need some of each
def test_good_amount_of_tests():
    assert len(timesteps_all) >= n_tests

# draw a random sample of these to compare against, to make tests faster
# this sampling is done without replacement
# draw a random sample of these to compare against, to make tests faster
def random_sample(full_sample):
    if len(full_sample) < n_tests:
        return full_sample
    return random.sample(full_sample, n_tests)


timesteps_all = random_sample(timesteps_all)



# ==============================================================================
#
# Test the initial metallicities
#
# ==============================================================================
elts = ["SNII", "SNIa", "AGB", "C", "N", "O", "Mg", "S", "Ca", "Fe"]

@pytest.mark.parametrize("elt", elts)
@pytest.mark.parametrize("step", timesteps_all)
def test_formation_metallicities(elt, step):
    test_z = step["star metallicity {}".format(elt)]
    true_z = step["{} current".format(elt)] / step["total current"]
    assert approx(true_z, abs=0, rel=rel) == test_z

# ==============================================================================
#
# Test the removal from gas
#
# ==============================================================================
fields = elts + ["HI", "HII", "H2", "HeI", "HeII", "HeIII"]

@pytest.mark.parametrize("step", timesteps_all)
def test_remove_total_density(step):
    removed_density = step["particle_mass"] * step["1/vol"]
    old_density = step["total current"]
    new_density_test = step["total new"]
    new_density_true = old_density - removed_density
    assert approx(new_density_true, abs=0, rel=rel) == new_density_test

@pytest.mark.parametrize("field", fields)
@pytest.mark.parametrize("step", timesteps_all)
def test_remove_total_density(field, step):
    # test that the gas metallicities don't change
    old_fraction = step["{} current".format(field)] / step["total current"]
    new_fraction = step["{} new".format(field)] / step["total new"]

    assert approx(old_fraction, abs=0, rel=rel) == new_fraction
