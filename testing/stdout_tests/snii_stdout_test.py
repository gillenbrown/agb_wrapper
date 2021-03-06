import pytest
from pytest import approx
import sys
from pathlib import Path
import random

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

from snii_enrich_ia_elts_cluster_discrete import lib as snii_c_code
from core_enrich_ia_elts_cluster_discrete import lib as core_c_code

core_c_code.detailed_enrichment_init()
snii_c_code.detailed_enrichment_init()
snii_c_code.init_rand()

lt = tabulation.Lifetimes("Raiteri_96")
imf = tabulation.IMF("Kroupa", 0.08, 50, total_mass=1.0)

ds = yt.load(str(this_dir/"art_dataset/continuous_a0.1507.art"))

n_tests = 10
rel = 1E-6

timesteps_all = parse_file(str(this_dir/"stdout_snii.txt"), "SNII")

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
timesteps_without_hn = timesteps_good_with_sn_only + timesteps_without_sn
timesteps_all = timesteps_with_sn + timesteps_without_sn


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

def get_stars_in_mass_range(step):
    m_low = max(min(step["m_turnoff_next"], 50.0), 8.0)
    m_high = max(min(step["m_turnoff_now"], 50.0), 8.0)

    imf.normalize(step["stellar mass Msun"])
    return integrate.quad(imf.normalized_dn_dm, m_low, m_high)[0]

def get_sn_mass(step):
    # get the number of supernovae based on the energy
    return 0.5 * (step["m_turnoff_now"] + step["m_turnoff_next"])

def sn_and_hn(step):
    # Determine how many SN and HN are happening in a given timestep. First
    # we'll use the IMF limits to estimate how many SN are possible in this
    # timestep. then we'll test all possible combinations of SN and HN in this
    # timestep to see if they match the energy injected
    energy_ergs = code_energy_to_erg(step["energy added"], step["abox[level]"])
    sn_mass = get_sn_mass(step)
    total_sn = int(step["number SN"])

    if sn_mass < 20.0:  # no HN
        n_sn = int(round(energy_ergs / 1E51, 0))
        return n_sn, 0
    else:
        hn_energy = snii_c_code.hn_energy_py(sn_mass)
        for n_hn in range(total_sn + 1): # have iteration with all HN
            n_sn = total_sn - n_hn
            this_E = n_sn * 1E51 + n_hn * hn_energy
            if this_E == pytest.approx(energy_ergs, abs=0, rel=1E-4):
                return n_sn, n_hn
        # if we got here we didn't find an answer
        assert False

def get_true_hn_fraction(z):
    return max(0.5 * np.exp(-z/0.001), 0.001)

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
    assert (1.0 * code_energy).to(u.erg).value == approx(factor, rel=1E-4, abs=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_energy_1E51_value(step):
    ergs = code_energy_to_erg(step["1E51 ergs in code energy"],
                              step["abox[level]"])
    # broader tolerance, a^2 leaves it more vulnerable
    assert ergs == pytest.approx(1E51, rel=1E-4, abs=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_vol(step):
    assert step["1/vol"] == step["true 1/vol"]

@pytest.mark.parametrize("step", timesteps_all)
def test_cluster_mass_range(step):
    assert 1 < step["stellar mass Msun"] < 1E8

# When testing the units with yt, we need to be a bit loose on the tolerances,
# since the energy factors in the output are not at the same time as the ones
# in the stdout file, since the size of the box has changed somewhat
@pytest.mark.parametrize("step", timesteps_all)
def test_time_factor_with_yt(step):
    code_time_test = step["code time"]
    code_time_true = ds.time_unit.to("s").value
    assert code_time_test == approx(code_time_true, abs=0, rel=1E-2)

@pytest.mark.parametrize("step", timesteps_all)
def test_length_factor_with_yt(step):
    code_length_test = step["code length"]
    code_length_true = ds.length_unit.to("cm").value
    assert code_length_test == approx(code_length_true, abs=0, rel=1E-2)

@pytest.mark.parametrize("step", timesteps_all)
def test_energy_factor_with_yt(step):
    code_energy_test = step["code energy"]
    code_energy_true = (ds.mass_unit * ds.velocity_unit**2).to("erg").value
    assert code_energy_test == approx(code_energy_true, abs=0, rel=1E-2)


@pytest.mark.parametrize("step", timesteps_all)
def test_sn_energy_with_yt(step):
    code_energy_in_sn = step["1E51 ergs in code energy"]

    erg_per_code_energy = (ds.mass_unit * ds.velocity_unit ** 2).to("erg").value

    erg_in_sn = code_energy_in_sn * erg_per_code_energy
    assert erg_in_sn == approx(1E51, abs=0, rel=1E-2)


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
    n_stars = get_stars_in_mass_range(step)

    old_counter = step["unexploded_sn current"]
    new_counter = step["unexploded_sn new"]
    assert new_counter == approx(old_counter + n_stars, abs=1E-5, rel=0)

@pytest.mark.parametrize("step", timesteps_all)
def test_unexploded_sn_new_exact(step):
    n_stars = get_stars_in_mass_range(step)

    old_counter = step["unexploded_sn current"]
    new_counter = step["unexploded_sn new"]
    correct_counter = (old_counter + n_stars) % 1.0
    assert new_counter == approx(correct_counter, abs=1E-5, rel=0)

# ==============================================================================
#
# HN fraction
#
# ==============================================================================
@pytest.mark.parametrize("step", timesteps_all)
def test_hn_fraction(step):
    test_f_hn = step["hn_fraction"]
    true_f_hn = get_true_hn_fraction(step["metallicity"])
    assert test_f_hn == approx(true_f_hn, abs=1E-5, rel=0)

# ==============================================================================
#
# energy and number of SN
#
# ==============================================================================
# these are tied together since we determine the number of SN from the energy
@pytest.mark.parametrize("step", timesteps_with_sn)
def test_number_is_integer(step):
    n_sn = step["number SN"]
    assert n_sn == approx(int(n_sn), abs=1E-10, rel=0)

@pytest.mark.parametrize("step", timesteps_with_sn)
def test_number(step):
    n_sn = step["number SN"]
    # then get the number expected by the IMF integration
    n_stars = get_stars_in_mass_range(step)
    expected_n_sn = (step["unexploded_sn current"] + n_stars) // 1.0
    assert n_sn == expected_n_sn

@pytest.mark.parametrize("step", timesteps_with_sn)
def test_energy_is_possible(step):
    n_sn_code = step["number SN"]
    # getting the number of SN and HN will check if the energy combo is possible
    n_sn, n_hn = sn_and_hn(step)
    # this check should always pass since this is assumed in the sn_and_hn
    # function, the real checking is done in the function above
    assert n_sn + n_hn == n_sn_code

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
def test_cell_gas_pressure_unchanged_with_no_sn(step):
    assert step["cell_gas_pressure new"] == step["cell_gas_pressure current"]

# ==============================================================================
#
# yield checking - simple checks
#
# ==============================================================================
modified_elts = ["C", "N", "O", "Mg", "S", "Ca", "Fe", "SNII", "total"]
not_modified_elts = ["AGB", "SNIa"]
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
def test_sn_have_added(step, elt):
    assert step["{} added".format(elt)] > 0

@pytest.mark.parametrize("step", timesteps_with_sn)
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
# set up indices for accessing the individual yields
idxs = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6, "SNII": 7,
        "total": 8}

@pytest.mark.parametrize("step", timesteps_with_sn)
@pytest.mark.parametrize("elt", modified_elts)
def test_ejected_yields(step, elt):
    m = get_sn_mass(step)
    z = step["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(step)

    yields_snii = core_c_code.get_yields_raw_sn_ii_py(z, m)[idxs[elt]]
    yields_hnii = core_c_code.get_yields_raw_hn_ii_py(z, m)[idxs[elt]]

    mass_ejected = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code masses
    mass_ejected_code = (mass_ejected * u.Msun).to(code_mass).value

    density_ejected_code = mass_ejected_code * step["1/vol"]
    assert density_ejected_code == approx(step["{} added".format(elt)],
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
    non_metals = step["total added"] - step["SNII added"]
    h_added = non_metals * h_fraction
    correct_new = current + h_added

    assert new == approx(correct_new, abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_all)
def test_HeIII_change(step):
    current = step["HeIII current"]
    new = step["HeIII new"]
    # actual injected H is the total minus the metals added
    non_metals = step["total added"] - step["SNII added"]
    he_added = non_metals * he_fraction
    correct_new = current + he_added

    assert new == approx(correct_new, abs=0, rel=rel)


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
                                                      abs=0, rel=rel)

@pytest.mark.parametrize("step", timesteps_without_sn)
def test_no_sn_mass_loss(step):
    assert step["particle_mass new"] == step["particle_mass current"]

# ==============================================================================
#
# compare to output of C code
#
# ==============================================================================
# set up indices for accessing the results of the SNII calculations. I cannot
# do this with hypernovae, since there is some randomness.
idxs_c = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6, "SNII": 7,
          "total": 8, "energy": 9, "N_SN": 10, "N_SN_left": 11}

@pytest.mark.parametrize("step", timesteps_without_hn)
@pytest.mark.parametrize("elt", modified_elts)
def test_comp_elts_to_c_code(step, elt):
    code_mass_added = step["{} added".format(elt)] / step["1/vol"]

    ejecta = snii_c_code.get_ejecta_sn_ii_py(step["unexploded_sn current"],
                                             step["m_turnoff_now"],
                                             step["m_turnoff_next"],
                                             step["stellar mass Msun"],
                                             step["metallicity"])
    true_mass_added = ejecta[idxs_c[elt]]
    true_mass_added = (true_mass_added * u.Msun).to(code_mass).value

    assert pytest.approx(true_mass_added, abs=0, rel=rel) == code_mass_added

@pytest.mark.parametrize("step", timesteps_without_hn)
def test_comp_energy_to_c_code(step):
    code_energy_added = step["energy added"]

    energy_erg_added = code_energy_to_erg(code_energy_added, step["abox[level]"])

    ejecta = snii_c_code.get_ejecta_sn_ii_py(step["unexploded_sn current"],
                                             step["m_turnoff_now"],
                                             step["m_turnoff_next"],
                                             step["stellar mass Msun"],
                                             step["metallicity"])
    true_ergs_added = ejecta[idxs_c["energy"]]

    # energy tolerance needs to be a bit bigger
    assert pytest.approx(true_ergs_added, abs=0, rel=1E-4) == energy_erg_added

@pytest.mark.parametrize("step", timesteps_without_hn)
def test_comp_n_sn_to_c_code(step):
    test_n_sn = step["number SN"]

    ejecta = snii_c_code.get_ejecta_sn_ii_py(step["unexploded_sn current"],
                                             step["m_turnoff_now"],
                                             step["m_turnoff_next"],
                                             step["stellar mass Msun"],
                                             step["metallicity"])
    true_n_sn = ejecta[idxs_c["N_SN"]]

    assert true_n_sn == test_n_sn

@pytest.mark.parametrize("step", timesteps_without_hn)
def test_comp_n_sn_leftover_to_c_code(step):
    test_leftover = step["unexploded_sn new"]

    ejecta = snii_c_code.get_ejecta_sn_ii_py(step["unexploded_sn current"],
                                             step["m_turnoff_now"],
                                             step["m_turnoff_next"],
                                             step["stellar mass Msun"],
                                             step["metallicity"])
    true_leftover = ejecta[idxs_c["N_SN_left"]]

    assert pytest.approx(true_leftover, abs=1E-5, rel=0) == test_leftover
