import pytest
from pytest import approx

import random

from parse_stdout import parse_file

n_tests = 10
rel = 1E-6

timesteps_all = parse_file("./stdout_star_growth.txt", "star_growth")

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
# Test the mass growth
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_growth_masses(step):
    test_mass = step["star mass new"]
    true_mass = step["star mass current"] + step["add mass"]

    assert approx(true_mass, abs=0, rel=rel) == test_mass


# ==============================================================================
#
# Test the changed metallicities
#
# ==============================================================================
elts = ["SNII", "SNIa", "AGB", "C", "N", "O", "Mg", "S", "Ca", "Fe"]

@pytest.mark.parametrize("elt", elts)
@pytest.mark.parametrize("step", timesteps_all)
def test_growth_metallicities(elt, step):
    old_z = step["star metallicity {} current".format(elt)]
    old_metals = step["star mass current"] * old_z

    gas_z = step["{} cell current".format(elt)] / step["total cell current"]
    added_metals = step["add mass"] * gas_z

    new_metals = old_metals + added_metals

    new_z_true = new_metals / step["star mass new"]
    new_z_test = step["star metallicity {} new".format(elt)]

    assert pytest.approx(new_z_true, abs=0, rel=rel) == new_z_test

# ==============================================================================
#
# Test the removal from gas
#
# ==============================================================================
fields = elts + ["HI", "HII", "H2", "HeI", "HeII", "HeIII"]

@pytest.mark.parametrize("step", timesteps_all)
def test_remove_total_density(step):
    removed_density = step["add_mass"] * step["1/vol"]
    old_density = step["total cell current"]
    new_density_test = step["total cell new"]
    new_density_true = old_density - removed_density
    assert approx(new_density_true, abs=0, rel=rel) == new_density_test

@pytest.mark.parametrize("field", fields)
@pytest.mark.parametrize("step", timesteps_all)
def test_remove_total_density(field, step):
    # test that the gas metallicities don't change
    old_fraction = step["{} cell current".format(field)] / step["total cell current"]
    new_fraction = step["{} cell new".format(field)] / step["total cell new"]

    assert approx(old_fraction, abs=0, rel=rel) == new_fraction
