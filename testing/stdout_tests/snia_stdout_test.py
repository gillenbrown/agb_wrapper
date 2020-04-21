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

from snia_enrich_ia_elts_cluster_discrete import lib as snia_c_code
from core_enrich_ia_elts_cluster_discrete import lib as core_c_code

core_c_code.detailed_enrichment_init()
snia_c_code.detailed_enrichment_init()

lt = tabulation.Lifetimes("Raiteri_96")

n_tests = 10
rel = 1E-6

timesteps_all = parse_file(str(this_dir/"stdout_snia.txt"), "SNIa")

# then go through them and put them in a few categories to parse them more
# carefully later
timesteps_with_sn = []
timesteps_without_sn = []

for timestep in timesteps_all:
    if timestep["energy added"] > 0:
        timesteps_with_sn.append(timestep)
    elif timestep["energy added"] == 0:
        timesteps_without_sn.append(timestep)
    else:  # should not happen!
        raise ValueError("Bad SN checks")


# we need some of each
def test_good_amount_of_tests():
    assert len(timesteps_with_sn) >= n_tests
    assert len(timesteps_without_sn) >= n_tests


# draw a random sample of these to compare against, to make tests faster
# this sampling is done without replacement
# draw a random sample of these to compare against, to make tests faster
def random_sample(full_sample):
    if len(full_sample) < n_tests:
        return full_sample
    return random.sample(full_sample, n_tests)


timesteps_with_sn = random_sample(timesteps_with_sn)
timesteps_without_sn = random_sample(timesteps_without_sn)

# then make some supersets that will be useful later
timesteps_all = timesteps_with_sn + timesteps_without_sn

# ==============================================================================
#
# Convenience functions
#
# ==============================================================================
def true_n_sn(step):
    # see https://nbviewer.jupyter.org/github/gillenbrown/agb_wrapper/blob/master/testing/informal_tests/snia.ipynb
    # and
    # https://nbviewer.jupyter.org/github/gillenbrown/Tabulation/blob/master/notebooks/sn_Ia.ipynb
    norm = step["stellar mass Msun"] * 1.6E-3 * 2.3480851917 / 0.13

    age_1 = step["age"]
    age_2 = step["age"] + step["dt"]
    return norm * (age_1**(-0.13) - age_2**(-0.13))

def step_n_sn(step):
    return step["energy added"] / (step["energy per SN"] * step["1/vol"])

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

def code_energy_to_erg(energy_in_code_units, a):
    code_energy = code_energy_func(a)
    return (energy_in_code_units * code_energy).to(u.erg).value

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
    assert step["age"] > step["t_start"] > 0

@pytest.mark.parametrize("step", timesteps_all)
def test_dt(step):
    test_dt = step["next"] - step["time"]
    assert step["dt"] == test_dt

@pytest.mark.parametrize("step", timesteps_all)
def test_t_start(step):
    true_t_start = lt.lifetime(8.0, step["metallicity"])
    assert step["t_start"] == approx(true_t_start, abs=0, rel=rel)

# ==============================================================================
#
# code details
#
# ==============================================================================
# Then we can use these to check the values in the code
@pytest.mark.parametrize("step", timesteps_all)
def test_time_factor_value(step):
    # in the code this is the number of seconds in a code time unit
    factor = step["code time"]
    code_time = code_time_func(step["abox[level]"])
    assert (1.0 * code_time).to(u.second).value == approx(factor, rel=rel, abs=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_length_factor_value(step):
    # in the code this is the number of cm in a code length unit
    factor = step["code length"]
    code_length = code_length_func(step["abox[level]"])
    assert (1.0 * code_length).to(u.cm).value == approx(factor, rel=rel, abs=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_energy_factor_value(step):
    # in the code this is the number of ergs in a code energy unit
    factor = step["code energy"]
    code_energy = code_energy_func(step["abox[level]"])
    # broader tolerance, a^2 leaves it more vulnerable
    assert (1.0 * code_energy).to(u.erg).value == approx(factor, rel=rel, abs=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_energy_2E51_value(step):
    ergs = code_energy_to_erg(step["energy per SN"],
                              step["abox[level]"])
    # broader tolerance, a^2 leaves it more vulnerable
    assert ergs == pytest.approx(2E51, rel=1E-4, abs=0)

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
    assert 0 < step["metallicity II"] <= step["metallicity"]
    assert 0 < step["metallicity Ia"] < step["metallicity"]
    assert 0 < step["metallicity AGB"] < step["metallicity"]
    assert 0 < step["metallicity"] < 1


@pytest.mark.parametrize("step", timesteps_all)
def test_total_metallicity(step):
    test_total_z = step["metallicity II"] + \
                   step["metallicity Ia"] + \
                   step["metallicity AGB"]
    assert step["metallicity"] == test_total_z

# ==============================================================================
#
# unexploded SN and numbers of SN
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
    true_sn = true_n_sn(step)

    old_counter = step["unexploded_sn current"]
    new_counter = step["unexploded_sn new"]
    assert new_counter == approx(old_counter + true_sn, abs=1E-5, rel=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_unexploded_sn_new_exact(step):
    true_sn = true_n_sn(step)

    old_counter = step["unexploded_sn current"]
    new_counter = step["unexploded_sn new"]

    correct_counter = (old_counter + true_sn) % 1.0
    assert new_counter == approx(correct_counter, abs=1E-5, rel=0)

# ==============================================================================
#
# energy and number of SN
#
# ==============================================================================
# I don't directly test energy, since we use that to calculate the number of SN
# and then check that number
@pytest.mark.parametrize("step", timesteps_with_sn)
def test_integer_number_sn(step):
    n_sn = step_n_sn(step)
    assert int(n_sn) == approx(n_sn, abs=1E-5, rel=0)

@pytest.mark.parametrize("step", timesteps_with_sn)
def test_number(step):
    sn_in_step = true_n_sn(step)
    old_counter = step["unexploded_sn current"]
    expected_sn = (old_counter + sn_in_step) // 1

    assert step_n_sn(step) == expected_sn

@pytest.mark.parametrize("step", timesteps_with_sn)
def test_cell_gas_energy_increases_with_sn(step):
    assert step["cell_gas_energy new"] > step["cell_gas_energy current"]

@pytest.mark.parametrize("step", timesteps_without_sn)
def test_cell_gas_energy_unchanged_with_no_sn(step):
    assert step["cell_gas_energy new"] == step["cell_gas_energy current"]

@pytest.mark.parametrize("step", timesteps_with_sn)
def test_cell_gas_internal_energy_increases_with_sn(step):
    assert step["cell_gas_internal_energy new"] > step["cell_gas_internal_energy current"]

@pytest.mark.parametrize("step", timesteps_without_sn)
def test_cell_gas_internal_energy_unchanged_with_no_sn(step):
    assert step["cell_gas_internal_energy new"] == step["cell_gas_internal_energy current"]

@pytest.mark.parametrize("step", timesteps_with_sn)
def test_cell_gas_turbulent_energy_increases_with_sn(step):
    assert step["cell_gas_turbulent_energy new"] > step["cell_gas_turbulent_energy current"]

@pytest.mark.parametrize("step", timesteps_without_sn)
def test_cell_gas_turbulent_energy_unchanged_with_no_sn(step):
    assert step["cell_gas_turbulent_energy new"] == step["cell_gas_turbulent_energy current"]


@pytest.mark.parametrize("step", timesteps_with_sn)
def test_cell_gas_pressure_increases_with_sn(step):
    assert step["cell_gas_pressure new"] > step["cell_gas_pressure current"]

@pytest.mark.parametrize("step", timesteps_without_sn)
def test_cell_gas_internal_energy_unchanged_with_no_sn(step):
    assert step["cell_gas_pressure new"] == step["cell_gas_pressure current"]

# ==============================================================================
#
# yield checking - simple checks
#
# ==============================================================================
modified_elts = ["C", "N", "O", "Mg", "S", "Ca", "Fe", "SNIa", "total"]
not_modified_elts = ["AGB", "SNII"]
all_elts = modified_elts + not_modified_elts

@pytest.mark.parametrize("step", timesteps_without_sn)
@pytest.mark.parametrize("elt", all_elts)
def test_no_change_when_no_sn(step, elt):
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    assert current == new

@pytest.mark.parametrize("step", timesteps_without_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_no_added_when_no_sn(step, elt):
    assert step["{} added".format(elt)] == 0

@pytest.mark.parametrize("step", timesteps_all)
@pytest.mark.parametrize("elt", not_modified_elts)
def test_agb_and_snii_never_change(step, elt):
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    assert current == new

@pytest.mark.parametrize("step", timesteps_with_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_sn_increase_elements(step, elt):
    current = step["{} current".format(elt)]
    new = step["{} new".format(elt)]
    # sometimes N addition is too small to notice
    if elt == "N":
        assert current <= new
    else:
        assert current < new

@pytest.mark.parametrize("step", timesteps_with_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_sn_have_added(step, elt):
    assert step["{} added".format(elt)] > 0

@pytest.mark.parametrize("step", timesteps_with_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_actual_density_addition(step, elt):
    current = step["{} current".format(elt)]
    added = step["{} added".format(elt)]
    new_expected = current + added
    new = step["{} new".format(elt)]
    assert new == approx(new_expected, abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
def test_metals_are_total(step):
    assert step["total added"] == step["SNIa added"]


# ==============================================================================
#
# Yields - actual checks
#
# ==============================================================================
# set up indices for accessing the individual yields
idxs = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6, "SNIa": 7,
        "total": 8}

@pytest.mark.parametrize("step", timesteps_with_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_ejected_yields(step, elt):
    z = step["metallicity"]
    # get the number of supernovae based on yields
    n_sn = step_n_sn(step)

    yields_snia = core_c_code.get_yields_sn_ia_py(z)[idxs[elt]]

    mass_ejected = yields_snia * n_sn
    # this is in stellar masses, have to convert to code masses
    mass_ejected_code = (mass_ejected * u.Msun).to(code_mass).value

    density_ejected_code = mass_ejected_code * step["1/vol"]
    assert step["{} added".format(elt)] == approx(density_ejected_code,
                                                  abs=0, rel=rel)

# ==============================================================================
#
# Particle mass loss
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_with_sn)
def test_sn_mass_loss(step):
    old_mass = step["particle_mass current"]
    lost_density = step["total added"]
    lost_mass = lost_density / step["1/vol"]

    expected_new_mass = old_mass - lost_mass
    assert step["particle_mass new"] == pytest.approx(expected_new_mass,
                                                      abs=0, rel=0.1*lost_mass)

@pytest.mark.parametrize("step", timesteps_without_sn)
def test_no_sn_mass_loss(step,):
    assert step["particle_mass new"] == step["particle_mass current"]

