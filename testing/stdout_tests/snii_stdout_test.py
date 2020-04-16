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

from snii_enrich_ia_elts_cluster_discrete import lib as snii_c_code
from core_enrich_ia_elts_cluster_discrete import lib as core_c_code

core_c_code.detailed_enrichment_init()
snii_c_code.detailed_enrichment_init()
snii_c_code.init_rand()

lt = tabulation.Lifetimes("Raiteri_96")
imf = tabulation.IMF("Kroupa", 0.08, 50)

n_tests = 10

timesteps_all = parse_file(str(this_dir/"snii_stdout.txt"), "SNII")

# then go through them and put them in a few categories to parse them more
# carefully later
timesteps_good_with_hn_and_sn = []
timesteps_good_with_sn_only = []
timesteps_good_without_sn = []
timesteps_early = []

for timestep in timesteps_all:
    if timestep["energy added"] > 0 and timestep["m_turnoff_now"] > 20:
        timesteps_good_with_hn_and_sn.append(timestep)
    elif timestep["energy added"] > 0 and timestep["m_turnoff_now"] <= 20:
        timesteps_good_with_sn_only.append(timestep)
    elif timestep["energy added"] == 0 and timestep["m_turnoff_next"] > 50:
        timesteps_early.append(timestep)
    elif timestep["energy added"] == 0 and timestep["m_turnoff_next"] < 50:
        timesteps_good_without_sn.append(timestep)
    else:  # should not happen!
        raise ValueError("Bad SN checks")

# we need some of each
def test_good_amount_of_tests():
    assert len(timesteps_good_with_hn_and_sn) >= n_tests
    assert len(timesteps_good_with_sn_only) >= n_tests
    assert len(timesteps_good_without_sn) >= n_tests
    assert len(timesteps_early) >= n_tests

# draw a random sample of these to compare against, to make tests faster
# this sampling is done without replacement
# draw a random sample of these to compare against, to make tests faster
def random_sample(full_sample):
    if len(full_sample) < n_tests:
        return full_sample
    return random.sample(full_sample, n_tests)

timesteps_good_with_hn_and_sn = random_sample(timesteps_good_with_hn_and_sn)
timesteps_good_with_sn_only = random_sample(timesteps_good_with_sn_only)
timesteps_good_without_sn = random_sample(timesteps_good_without_sn)
timesteps_early = random_sample(timesteps_early)

# then make some supersets that will be useful later
timesteps_with_sn = timesteps_good_with_hn_and_sn + timesteps_good_with_sn_only
timesteps_without_sn = timesteps_good_without_sn + timesteps_early
timesteps_all = timesteps_with_sn + timesteps_without_sn

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
def test_age(step):
    test_age = step["time"] - step["ave_birth"]
    assert step["age"] == test_age
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
        assert step["m_turnoff_now"] == approx(true_turnoff_mass)

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
        assert step["m_turnoff_next"] == approx(true_turnoff_mass)

# ==============================================================================
#
# code details
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
base_code_time = u.def_unit("base_code_time", 2.0 / (H_0 * np.sqrt(omega_m)))
def code_time_func(a_box):
    return u.def_unit("code_time", base_code_time * a_box**2)

def code_length_func(a_box):
    return u.def_unit("code_length", base_code_length * a_box)

def code_energy_func(a_box):
    e_val = code_mass * (code_length_func(a_box) / code_time_func(a_box))**2
    return u.def_unit("code_energy", e_val)

# Then we can use these to check the values in the code
@pytest.mark.parametrize("step", timesteps_all)
def test_time_factor_value(step):
    factor = step["code time"]

    code_time = code_time_func(step["abox[level]"])
    assert (1.0 * code_time).to(u.second).value == approx(factor, rel=1E-10, abs=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_length_factor_value(step):
    factor = step["code length"]
    code_length = code_length_func(step["abox[level]"])
    assert (1.0 * code_length).to(u.cm).value == approx(factor, rel=1E-10, abs=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_energy_factor_value(step):
    factor = step["code energy"]
    code_energy = code_energy_func(step["abox[level]"])
    # broader tolerance, a^2 leaves it more vulnerable
    assert (1.0 * code_energy).to(u.erg).value == approx(factor, rel=1E-4, abs=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_energy_1E51_value(step):
    ergs_1E51 = step["1E51 ergs"]

    code_energy = code_energy_func(step["abox[level]"])

    # broader tolerance, a^2 leaves it more vulnerable
    assert (1E51 * u.erg).to(code_energy).value == approx(ergs_1E51, rel=1E-4, abs=0)


@pytest.mark.parametrize("step", timesteps_all)
def test_snii_vol(step):
    assert step["1/vol"] == step["true 1/vol"]

@pytest.mark.parametrize("step", timesteps_all)
def test_cluster_mass_range(step):
    assert 1 < step["stellar mass Msun"] < 1E8

@pytest.mark.parametrize("step", timesteps_all)
def test_mass_conversion(step):
    msun = step["stellar mass Msun"]
    code = step["stellar mass code"]

    assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
    assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)

# ==============================================================================
#
# Metallicity
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_metallicity_range(step):
    # just a check that we're getting the correct thing
    assert 0 < step["metallicity II"] < 1
    assert 0 < step["metallicity Ia"] < 1
    assert 0 < step["metallicity AGB"] < 1
    assert 0 < step["metallicity"] < 1


@pytest.mark.parametrize("step", timesteps_all)
def test_total_metallicity(step):
    test_total_z = step["metallicity II"] + \
                   step["metallicity Ia"] + \
                   step["metallicity AGB"]
    assert step["metallicity"] == test_total_z

# ==============================================================================
#
# unexploded SN
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_unexploded_sn_current_reasonable(step):
    assert 0 <= step["unexploded_sn current"] < 1.0

@pytest.mark.parametrize("step", timesteps_all)
def test_unexploded_sn_new_reasonable(step):
    assert 0 <= step["unexploded_sn new"] < 1.0

@pytest.mark.parametrize("step", timesteps_without_sn)
def test_unexploded_sn_new_exact_no_sn(step):
    m_low = max(min(step["m_turnoff_next"], 50.0), 8.0)
    m_high = max(min(step["m_turnoff_now"], 50.0), 8.0)

    imf.normalize(step["stellar mass Msun"])
    n_stars = integrate.quad(imf.normalized_dn_dm, m_low, m_high)[0]

    old_counter = step["unexploded_sn current"]
    new_counter = step["unexploded_sn new"]
    assert new_counter == approx(old_counter + n_stars, abs=1E-5, rel=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_unexploded_sn_new_exact(step):
    m_low  = max(min(step["m_turnoff_next"], 50.0), 8.0)
    m_high = max(min(step["m_turnoff_now"],  50.0), 8.0)

    imf.normalize(step["stellar mass Msun"])
    n_stars = integrate.quad(imf.normalized_dn_dm, m_low, m_high)[0]

    old_counter = step["unexploded_sn current"]
    new_counter = step["unexploded_sn new"]
    correct_counter = (old_counter + n_stars) % 1.0
    assert new_counter == approx(correct_counter, abs=1E-5, rel=0)

# ==============================================================================
#
# SN energy
#
# ==============================================================================
def code_energy_to_erg(energy_code, a):
    code_energy = code_energy_func(a)
    return (energy_code * code_energy).to(u.erg).value

# These are a bit tricky, since we have to figure out whether we have a
# supernovae or hypernova. I will first write a function to determine how many
# supernovae or hypernovae happened based on the energy injected
def sn_and_hn(code_energy_added, sn_mass, a):
    energy_ergs = code_energy_to_erg(code_energy_added, a)
    # boost it every so slightly to account for floating point errors, since
    # in the upcoming code I"ll subtract from this initial value
    energy_ergs *= 1.000001
    if sn_mass > 20.0:  # we can have HN
        hn_energy_this_mass = snii_c_code.hn_energy_py(sn_mass)
        # hn are always dominant over SN, so the dominant fraction of energy
        # will be from HN, if they exist
        n_hn = int(energy_ergs // hn_energy_this_mass)
        energy_ergs -= n_hn * hn_energy_this_mass
    else:
        n_hn = 0
    # then everything that's left over is SN
    n_sn = int(round(energy_ergs / 1E51, 0))  # each SN is 1E51 ergs
    return n_sn, n_hn

@pytest.mark.parametrize("step", timesteps_with_sn)
def test_number(step):
    # get the number of supernovae based on the energy
    mass = 0.5 * (step["m_turnoff_now"] + step["m_turnoff_next"])

    n_sn, n_hn = sn_and_hn(step["energy added"], mass, step["abox[level]"])
    # then get the number expected by the IMF integration
    m_low = max(min(step["m_turnoff_next"], 50.0), 8.0)
    m_high = max(min(step["m_turnoff_now"], 50.0), 8.0)

    imf.normalize(step["stellar mass Msun"])
    n_stars = integrate.quad(imf.normalized_dn_dm, m_low, m_high)[0]

    expected_n_sn = (step["unexploded_sn current"] + n_stars) // 1.0

    assert n_sn + n_hn == expected_n_sn


# ==============================================================================
#
# yield checking - simple checks
#
# ==============================================================================
modified_elts = ["C", "N", "O", "Mg", "S", "Ca", "Fe", "SNII"]
not_modified_elts = ["AGB", "SNIa"]
all_elts = modified_elts + not_modified_elts

@pytest.mark.parametrize("step", timesteps_without_sn)
@pytest.mark.parametrize("elt", all_elts)
def test_no_change_when_no_sn(step, elt):
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    assert current == new

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", not_modified_elts)
def test_agb_and_snia_never_change(step, elt):
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    assert current == new

@pytest.mark.parametrize("step", timesteps_with_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_sn_increase_elements(step, elt):
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    assert current < new

@pytest.mark.parametrize("step", timesteps_with_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_actual_density_addition(step, elt):
    current = step["{} current".format(elt)]
    added = step["{}_added".format(elt)]
    new_expected = current + added
    new = step["{} new".format(elt)]
    assert new_expected == approx(new, abs=1E-5, rel=1E-5)


# ==============================================================================
#
# Yields - actual checks
#
# ==============================================================================
# set up indices for accessing the individual yields
idxs = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6, "SNII": 7,
        "total": 8}

@pytest.mark.parametrize("step", timesteps_with_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_ejected_yields(step, elt):
    m = 0.5 * (step["m_turnoff_next"] + step["m_turnoff_now"])
    z = step["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(step["energy added"], m, step["abox[level]"])

    yields_snii = core_c_code.get_yields_raw_sn_ii_py(z, m)[idxs[elt]]
    yields_hnii = core_c_code.get_yields_raw_hn_ii_py(z, m)[idxs[elt]]

    total_ejected = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code density
    total_ejected_code = (total_ejected * u.Msun).to(code_mass).value

    ejected_density = total_ejected_code * step["1/vol"]
    assert ejected_density == approx(step["{}_added".format(elt)])

#TODO: test that the mass loss is done correctly