import pytest
from pytest import approx
import sys
from pathlib import Path
import random

from astropy import units as u
from astropy import constants as c
import numpy as np
from scipy import integrate

import tabulation
from parse_stdout import parse_file

# add directory of compiled C code to my path so it can be imported
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent))

from wind_enrich_ia_elts_cluster_discrete import lib as wind_c_code
from core_enrich_ia_elts_cluster_discrete import lib as core_c_code

core_c_code.detailed_enrichment_init()
wind_c_code.detailed_enrichment_init()

lt = tabulation.Lifetimes("Raiteri_96")

n_tests = 10
rel = 1E-6

timesteps_all = parse_file(str(this_dir/"stdout_wind.txt"), "wind")

# then go through them and put them in a few categories to parse them more
# carefully later
timesteps_after_sn = []
timesteps_before_sn = []

for timestep in timesteps_all:
    if timestep["m_turnoff_next"] > 50:
        timesteps_before_sn.append(timestep)
    elif timestep["m_turnoff_next"] <= 50:
        timesteps_after_sn.append(timestep)
    else:  # should not happen!
        raise ValueError("Bad SN checks")


# we need some of each
def test_good_amount_of_tests():
    assert len(timesteps_before_sn) >= n_tests
    assert len(timesteps_after_sn) >= n_tests


# draw a random sample of these to compare against, to make tests faster
# this sampling is done without replacement
# draw a random sample of these to compare against, to make tests faster
def random_sample(full_sample):
    if len(full_sample) < n_tests:
        return full_sample
    return random.sample(full_sample, n_tests)


timesteps_before_sn = random_sample(timesteps_before_sn)
timesteps_after_sn = random_sample(timesteps_after_sn)

# then make some supersets that will be useful later
# timesteps_all = timesteps_before_sn + timesteps_after_sn
timesteps_all = timesteps_after_sn

# ==============================================================================
#
# Convenience functions
#
# ==============================================================================
# first define some functions to get code units based on my understanding of
# them. This will help me make sure I'm using units correctly
# the cosmological parameters are from the old IC
base_code_length = u.def_unit("base_code_length", 4 * u.Mpc / 128)
h = 0.6814000010490417
H_0 = 100 * h * u.km / (u.second * u.Mpc)
omega_m = 0.3035999834537506
code_mass = u.def_unit("code_mass", 3 * H_0**2 * omega_m / (8 * np.pi * c.G) *
                       base_code_length**3)

# ==============================================================================
#
# Timesteps and age
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_ave_age_range(step):
    """Check that ave_birth is past particle birth, but that it's not past
    15 Myr, the end of particle formation"""
    diff = step["ave_birth"] - step["birth"]
    assert 0 < diff < 15E6

@pytest.mark.parametrize("step", timesteps_all)
def test_age_calculation(step):
    test_age = step["time"] - step["ave_birth"]
    assert step["age"] == test_age

@pytest.mark.parametrize("step", timesteps_all)
def test_age_value(step):
    # The first timestep a star forms, it's ave_birth will be set to half of
    # the current timestep after the current time, so in this first timestep
    # the star can have negative ages.
    assert step["age"] > 0 or \
           step["age"] == -0.5 * step["dt"]

@pytest.mark.parametrize("step", timesteps_all)
def test_dt(step):
    test_dt = step["next"] - step["time"]
    assert step["dt"] == test_dt

# ==============================================================================
#
# Turnoff masses
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_turnoff_now_exact_values(step):
    true_turnoff_mass = lt.turnoff_mass(step["age"], step["metallicity"])
    # early times are a bit sketchier, since the lifetime function in ART
    # isn't quite as good as it could be.
    if true_turnoff_mass > 70:
        assert step["m_turnoff_now"] > 70
    else:
        # require exact values
        assert step["m_turnoff_now"] == approx(true_turnoff_mass,
                                               abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
def test_turnoff_next_exact_values(step):
    true_turnoff_mass = lt.turnoff_mass(step["age"] + step["dt"],
                                        step["metallicity"])
    # early times are a bit sketchier, since the lifetime function in ART
    # isn't quite as good as it could be.
    if true_turnoff_mass > 70:
        assert step["m_turnoff_next"] > 70
    else:
        # require exact values
        assert step["m_turnoff_next"] == approx(true_turnoff_mass,
                                               abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
def test_age_50(step):
    true_age_50 = lt.lifetime(50, step["metallicity"])
    assert step["age_50"] == approx(true_age_50, abs=0, rel=rel)

# ==============================================================================
#
# code details
#
# ==============================================================================
# Then we can use these to check the values in the code
@pytest.mark.parametrize("step", timesteps_all)
def test_vol(step):
    assert step["1/vol"] == step["true 1/vol"]

@pytest.mark.parametrize("step", timesteps_all)
def test_cluster_mass_range(step):
    assert 1 < step["stellar mass Msun"] < 1E8

@pytest.mark.parametrize("step", timesteps_all)
def test_mass_conversion(step):
    msun = step["stellar mass Msun"]
    code = step["stellar mass code"]

    assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=rel)
    assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=rel)

# ==============================================================================
#
# Metallicity
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_metallicity_range(step):
    # just a check that we're getting the correct thing
    assert 0 < step["metallicity SNII"] <= step["metallicity"]
    assert 0 < step["metallicity SNIa"] < step["metallicity"]
    assert 0 < step["metallicity AGB"] < step["metallicity"]
    assert 0 < step["metallicity C"] < step["metallicity"]
    assert 0 < step["metallicity N"] < step["metallicity"]
    assert 0 < step["metallicity O"] < step["metallicity"]
    assert 0 < step["metallicity S"] < step["metallicity"]
    assert 0 < step["metallicity Ca"] < step["metallicity"]
    assert 0 < step["metallicity Fe"] < step["metallicity"]
    assert 0 < step["metallicity"] < 1

@pytest.mark.parametrize("step", timesteps_all)
def test_sum_of_elements(step):
    # the sum of the elements should be less than the total metallicity
    elt_sum = step["metallicity C"] + \
              step["metallicity N"] + \
              step["metallicity O"] + \
              step["metallicity Mg"] + \
              step["metallicity S"] + \
              step["metallicity Ca"] + \
              step["metallicity Fe"]
    # at early times sometimes this can fail, since all these quantities are
    # initially set to the same value, so the first stars will fail this
    assert 0 < elt_sum < step["metallicity"] or step["metallicity"] < 1E-25


@pytest.mark.parametrize("step", timesteps_all)
def test_total_metallicity(step):
    test_total_z = step["metallicity SNII"] + \
                   step["metallicity SNIa"] + \
                   step["metallicity AGB"]
    assert step["metallicity"] == test_total_z

# ==============================================================================
#
# yield checking - simple checks
#
# ==============================================================================
elts = ["C", "N", "O", "Mg", "S", "Ca", "Fe", "AGB", "SNII", "SNIa"]
fields = elts + ["total"]

# This test is commented out because oftentimes the additions are too small to
# budge the floating point representation
# @pytest.mark.parametrize("step", timesteps_all)
# @pytest.mark.parametrize("elt", fields)
# def test_increase_elements(step, elt):
#     current = step["{} current".format(elt)]
#     new = step["{} new".format(elt)]
#     assert current < new

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", fields)
def test_have_added(step, elt):
    assert step["{} added".format(elt)] > 0

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", fields)
def test_actual_density_addition(step, elt):
    current = step["{} current".format(elt)]
    added = step["{} added".format(elt)]
    new_expected = current + added
    new = step["{} new".format(elt)]
    assert new == approx(new_expected, abs=0, rel=rel)


# ==============================================================================
#
# Yields - actual checks
#
# ==============================================================================
# This also checks the rates, since we have to use the cumulative yield tables
# to get these quantities
@pytest.mark.parametrize("step", timesteps_all)
def test_total_ejecta(step):
    ejecta_0 = core_c_code.get_cumulative_mass_winds_py(step["age"],
                                                        step["m_turnoff_now"],
                                                        step["metallicity"],
                                                        step["age_50"])
    ejecta_1 = core_c_code.get_cumulative_mass_winds_py(step["age"] + step["dt"],
                                                        step["m_turnoff_next"],
                                                        step["metallicity"],
                                                        step["age_50"])
    ejecta_true = (ejecta_1 - ejecta_0) * step["stellar mass Msun"]
    # this is in stellar masses, have to convert to code masses
    ejecta_true = (ejecta_true * u.Msun).to(code_mass).value
    added_density_true = ejecta_true * step["1/vol"]

    assert step["total added"] == approx(added_density_true, abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_before_sn)
def test_total_ejecta_early(step):
    age_50 = lt.lifetime(50, step["metallicity"])
    ejecta_50 = core_c_code.get_cumulative_mass_winds_py(age_50, 50,
                                                         step["metallicity"],
                                                         age_50)
    ejecta_true = step["stellar mass Msun"] * ejecta_50 * step["dt"] / age_50
    # this is in stellar masses, have to convert to code masses
    ejecta_true = (ejecta_true * u.Msun).to(code_mass).value
    added_density_true = ejecta_true * step["1/vol"]

    assert step["total added"] == approx(added_density_true, abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", elts)
def test_ejected_elements(step, elt):
    true_elt_ejecta = step["total added"] * step["metallicity {}".format(elt)]
    assert step["{} added".format(elt)] == approx(true_elt_ejecta, abs=0, rel=rel)

# ==============================================================================
#
# Particle mass loss
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_mass_loss(step):
    old_mass = step["particle_mass current"]
    lost_density = step["total added"]
    lost_mass = lost_density / step["1/vol"]

    expected_new_mass = old_mass - lost_mass
    assert step["particle_mass new"] == pytest.approx(expected_new_mass, abs=0, rel=rel)