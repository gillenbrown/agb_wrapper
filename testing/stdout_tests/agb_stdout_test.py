import pytest
from pytest import approx
import sys
from pathlib import Path
import random

p = sys.path

from astropy import units as u
from astropy import constants as c
import numpy as np
from scipy import integrate
import yt
yt.funcs.mylog.setLevel(50)

import tabulation
from parse_stdout import parse_file

# add directory of compiled C code to my path so it can be imported
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent/"build"))

from agb_enrich_ia_elts_cluster_discrete import lib as agb_c_code
from core_enrich_ia_elts_cluster_discrete import lib as core_c_code

core_c_code.detailed_enrichment_init()
agb_c_code.detailed_enrichment_init()

lt = tabulation.Lifetimes("Raiteri_96")
imf = tabulation.IMF("Kroupa", 0.08, 50, total_mass=1.0)

ds = yt.load(str(this_dir/"art_dataset/continuous_a0.1044.art"))

n_tests = 10
rel = 1E-6

timesteps_all = parse_file(str(this_dir/"stdout_agb.txt"), "AGB")

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

def get_stars_in_mass_range(step):
    m_low = min(step["m_turnoff_next"], 8.0)
    m_high = min(step["m_turnoff_now"], 8.0)

    imf.normalize(step["stellar mass Msun"])
    n_stars = integrate.quad(imf.normalized_dn_dm, m_low, m_high)[0]
    return n_stars

def get_mean_stellar_mass(step):
    return 0.5 * (step["m_turnoff_now"] + step["m_turnoff_next"])

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
    assert step["age"] > lt.lifetime(8.0, step["metallicity"])

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
    assert step["m_turnoff_now"] == approx(true_turnoff_mass, abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
def test_turnoff_next_exact_values(step):
    true_turnoff_mass = lt.turnoff_mass(step["age"] + step["dt"],
                                        step["metallicity"])
    assert step["m_turnoff_next"] == approx(true_turnoff_mass, abs=0, rel=rel)

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

@pytest.mark.parametrize("step", timesteps_all)
def test_mass_conversion_with_yt(step):
    msun_raw = step["stellar mass Msun"]
    code_raw = step["stellar mass code"]

    msun_yt = ds.quan(msun_raw, "Msun")
    code_yt = ds.quan(code_raw, "code_mass")

    assert code_yt.to("code_mass").value == approx(code_raw, abs=0, rel=rel)
    assert msun_yt.to("code_mass").value == approx(code_raw, abs=0, rel=rel)

    assert code_yt.to("Msun").value == approx(msun_raw, abs=0, rel=rel)
    assert msun_yt.to("Msun").value == approx(msun_raw, abs=0, rel=rel)


# ==============================================================================
#
# Metallicity
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_metallicity_range(step):
    # just a check that we're getting the correct thing
    assert 0 < step["metallicity II"] <= step["metallicity"]
    assert 0 < step["metallicity Ia"] < step["metallicity"]
    assert 0 < step["metallicity AGB"] < step["metallicity"]
    assert 0 < step["metallicity S"] < step["metallicity"]
    assert 0 < step["metallicity Ca"] < step["metallicity"]
    assert 0 < step["metallicity Fe"] < step["metallicity"]
    assert 0 < step["metallicity"] < 1

@pytest.mark.parametrize("step", timesteps_all)
def test_sum_of_elements(step):
    # the sum of the elements should be less than the total metallicity, e
    # especially since for AGB we only get a few metallicities
    elt_sum = step["metallicity S"] + \
              step["metallicity Ca"] + \
              step["metallicity Fe"]
    assert 0 < elt_sum < step["metallicity"]

@pytest.mark.parametrize("step", timesteps_all)
def test_total_metallicity(step):
    test_total_z = step["metallicity II"] + \
                   step["metallicity Ia"] + \
                   step["metallicity AGB"]
    assert step["metallicity"] == test_total_z

# ==============================================================================
#
# yield checking - simple checks
#
# ==============================================================================
modified_elts = ["C", "N", "O", "Mg", "S", "Ca", "Fe", "AGB", "total"]
not_modified_elts = ["SNII", "SNIa"]
directly_returned_elts = ["C", "N", "O", "Mg", "total"]
scaled_elts = ["S", "Ca", "Fe"]
all_elts = modified_elts + not_modified_elts
# set up indices for accessing the individual yields
idxs = {"C": 0, "N": 1, "O":2, "Mg":3, "some_metals":4, "total": 5}

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", not_modified_elts)
def test_snii_and_snia_never_change(step, elt):
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    assert current == new

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", directly_returned_elts)
def test_increase_directly_returned_elements(step, elt):
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    assert current < new

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", directly_returned_elts)
def test_increase_scaled_elements(step, elt):
    # sometimes these additions are tiny, so aren't big enough to move the
    # floating point representation
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    assert current <= new

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", modified_elts)
def test_sn_have_added(step, elt):
    assert step["{} added".format(elt)] > 0

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", modified_elts)
def test_actual_density_addition(step, elt):
    current = step["{} current".format(elt)]
    added = step["{} added".format(elt)]
    new_expected = current + added
    new = step["{} new".format(elt)]
    assert new_expected == approx(new, abs=0, rel=rel)


# ==============================================================================
#
# Yields - actual checks
#
# ==============================================================================
# This also checks the IMF integration, since we need to know how many start
# left the main sequence to correctly get the yields


@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", directly_returned_elts)
def test_ejected_yields_directly_ejected(step, elt):
    z = step["metallicity"]
    m = get_mean_stellar_mass(step)
    # get the number of AGB based on the IMF
    n_agb = get_stars_in_mass_range(step)

    yields_agb = core_c_code.get_yields_raw_agb_py(z, m)[idxs[elt]]

    mass_ejected = yields_agb * n_agb
    # this is in stellar masses, have to convert to code masses
    mass_ejected_code = (mass_ejected * u.Msun).to(code_mass).value

    density_ejected_code = mass_ejected_code * step["1/vol"]
    assert step["{} added".format(elt)] == approx(density_ejected_code,
                                                  abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", scaled_elts)
def test_ejected_yields_scaled(step, elt):
    z = step["metallicity"]
    m = get_mean_stellar_mass(step)
    # get the number of AGB based on the IMF
    n_agb = get_stars_in_mass_range(step)

    # scale the ejecta by the metallicity of this element
    all_ejecta_agb = core_c_code.get_yields_raw_agb_py(z, m)[idxs["total"]]
    mass_ejected = all_ejecta_agb * n_agb * step["metallicity {}".format(elt)]

    # this is in stellar masses, have to convert to code masses
    mass_ejected_code = (mass_ejected * u.Msun).to(code_mass).value

    density_ejected_code = mass_ejected_code * step["1/vol"]
    assert step["{} added".format(elt)] == approx(density_ejected_code,
                                                  abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
def test_ejected_yields_metals(step):
    z = step["metallicity"]
    m = get_mean_stellar_mass(step)
    # get the number of AGB based on the IMF
    n_agb = get_stars_in_mass_range(step)

    # scale the ejecta by the metallicity of this element
    metals = core_c_code.get_yields_raw_agb_py(z, m)[idxs["some_metals"]]
    metals *= n_agb
    # then add each of the elements
    all_ejecta_agb = core_c_code.get_yields_raw_agb_py(z, m)[idxs["total"]]
    for elt in scaled_elts:
        metals += all_ejecta_agb * n_agb * step["metallicity {}".format(elt)]

    # this is in stellar masses, have to convert to code masses
    mass_ejected_code = (metals * u.Msun).to(code_mass).value

    density_ejected_code = mass_ejected_code * step["1/vol"]
    assert step["AGB added".format(elt)] == approx(density_ejected_code,
                                                   abs=0, rel=rel)


# ==============================================================================
#
# H and He
#
# ==============================================================================
he_fraction = 0.24
h_fraction  = 1.0 - he_fraction
# then correct for the 4 nucleons, which matters to ART
he_fraction *= 0.25

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("field", ["HI", "H2", "HeI", "HeII"])
def test_no_change_unionized_h_he(step, field):
    current = step["{} current".format(field)]
    new = step["{} new".format(field)]
    assert current == new

@pytest.mark.parametrize("step", timesteps_all)
def test_HII_change(step):
    current = step["HII current"]
    new = step["HII new"]
    # actual injected H is the total minus the metals added
    non_metals = step["total added"] - step["AGB added"]
    h_added = non_metals * h_fraction
    correct_new = current + h_added

    assert new == approx(correct_new, abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
def test_HeIII_change(step):
    current = step["HeIII current"]
    new = step["HeIII new"]
    # actual injected H is the total minus the metals added
    non_metals = step["total added"] - step["AGB added"]
    he_added = non_metals * he_fraction
    correct_new = current + he_added

    assert new == approx(correct_new, abs=0, rel=rel)

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
    assert step["particle_mass new"] == approx(expected_new_mass,
                                               abs=0, rel=rel)

# ==============================================================================
#
# compare to output of C code
#
# ==============================================================================
# set up indices for accessing the results of the AGB calculations.
idxs_c = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6, "AGB": 7,
          "total": 8}

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", modified_elts)
def test_comp_elts_to_c_code(step, elt):
    code_mass_added = step["{} added".format(elt)] / step["1/vol"]

    ejecta = agb_c_code.get_ejecta_agb_py(step["m_turnoff_now"],
                                          step["m_turnoff_next"],
                                          step["stellar mass Msun"],
                                          step["metallicity"],
                                          step["metallicity S"],
                                          step["metallicity Ca"],
                                          step["metallicity Fe"])
    true_mass_added = ejecta[idxs_c[elt]]
    true_mass_added = (true_mass_added * u.Msun).to(code_mass).value

    assert pytest.approx(true_mass_added, abs=0, rel=rel) == code_mass_added



