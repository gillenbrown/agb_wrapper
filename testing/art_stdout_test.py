import sys, os

import pytest
from pytest import approx
import random
from scipy import interpolate
from astropy import units as u
from astropy import constants as c
import numpy as np

import tabulation

import parse_stdout
from art_enrich import lib as c_code
c_code.detailed_enrichment_init()

"""
This file tests the implementation of the code directly in ART by parsing the
stdout file generated as it runs and checking that what's written is correct.
"""

imf = tabulation.IMF("Kroupa", 0.1, 50)
lifetimes = tabulation.Lifetimes("Raiteri_96")
number_sn_ia = 1.6E-3
ssp_check = tabulation.SSPYields("Kroupa", 0.1, 50, 1,
                                 "Raiteri_96",
                                 "Kobayashi_06", 0.5, 8, 50,
                                 "art power law", "Nomoto_18",
                                 {"number_sn_ia":1.6E-3},
                                 "NuGrid", 0.1, 8)
sn_ia_check = ssp_check.sn_ia_model
agb_check = ssp_check.agb_model

# Create yields information to check against
sn_ia_yields_interp = dict()
elts = ["C", "N", "O", "Fe", "total_metals"]
sn_ia_z = sn_ia_check.metallicities
for elt in elts:
    yield_z_low = sn_ia_check.ejected_mass(elt, sn_ia_z[0])
    yield_z_high = sn_ia_check.ejected_mass(elt, sn_ia_z[1])
    interp = interpolate.interp1d(x=sn_ia_z, y=[yield_z_low, yield_z_high],
                                  kind="linear", bounds_error=False,
                                  fill_value=(yield_z_low, yield_z_high))
    sn_ia_yields_interp[elt] = interp

def mass_loss_agb(time, dt, z, elt):
    # pick the indices
    agb_z = ssp_check.agb_model.metallicities
    agb_z_idx = c_code.find_z_bound_idxs_agb_py(z)
    z0 = agb_z[agb_z_idx[0]]
    z1 = agb_z[agb_z_idx[1]]

    true_c_0 = ssp_check.mass_lost_end_ms(elt, time, time + dt, z0, "AGB")
    true_c_1 = ssp_check.mass_lost_end_ms(elt, time, time + dt, z1, "AGB")

    interp = interpolate.interp1d(x=[z0, z1], y=[true_c_0, true_c_1],
                                  kind="linear", bounds_error=False,
                                  fill_value=(true_c_0, true_c_1))
    return interp(z)

def mass_loss_sn_ii(time, dt, z, elt):
    # pick the indices
    sn_z = ssp_check.sn_ii_model.sn.metallicities
    sn_z_idx = c_code.find_z_bound_idxs_sn_ii_py(z)
    z0 = sn_z[sn_z_idx[0]]
    z1 = sn_z[sn_z_idx[1]]

    true_c_0 = ssp_check.mass_lost_end_ms(elt, time, time + dt, z0, "SNII")
    true_c_1 = ssp_check.mass_lost_end_ms(elt, time, time + dt, z1, "SNII")

    interp = interpolate.interp1d(x=[z0, z1], y=[true_c_0, true_c_1],
                                  kind="linear", bounds_error=False,
                                  fill_value=(true_c_0, true_c_1))
    return interp(z)


# create the code units to check against
code_length = u.def_unit("code_length", 4 * u.Mpc / 128)
h = 0.6814000010490417
H_0 = 100 * h * u.km / (u.second * u.Mpc)
omega_m = 0.3035999834537506
code_mass = u.def_unit("code_mass", 3 * H_0**2 * omega_m / (8 * np.pi * c.G) *
                       code_length**3)
code_time = u.def_unit("code_time", 2.0 / (H_0 * np.sqrt(omega_m)))

# ==============================================================================
#
# pick the stars to test
#
# ==============================================================================
number_to_check = 1000
# We want some that have winds, some that have SN, and some that have AGB
good = False
all_timesteps = parse_stdout.parse_file(10**4)
while not good:
    print("Redrawing random sample")
    timesteps_to_check = random.sample(all_timesteps, number_to_check)

    agb_flag = False
    snii_flag = False
    wind_flag = False
    snia_off_flag = False
    snia_on_flag = False
    for star in timesteps_to_check:
        # first throw away negative ages. These only happen when star particles
        # are just forming
        if star.snia["age"] < 0 or star.detail["age"] < 0:
            good = False
            break

        # check the ages when various ejecta are sure to be active
        if star.snia["age"] < 30E6:
            snia_off_flag = True
        elif star.snia["age"] > 50E6:
            snia_on_flag = True
        if star.detail["age"] < 3E6:
            wind_flag = True
        elif 5E6 < star.detail["age"] < 30E6:
            snii_flag = True
        elif star.detail["age"] > 50E6:
            agb_flag = True

        if agb_flag and snii_flag and wind_flag and snia_off_flag and snia_on_flag:
            good = True
            break

# Make a wrapper to test functions on all stars, but without using pytests
# parameterize. I don't want to do this because it counts each iteration as an
# extra test, making the number of tests go crazy. This doesn't do that.
def all_stars(func):
    def internal():
        for star in timesteps_to_check:
            func(star)
    return internal


# ==============================================================================
#
# SN Ia
#
# ==============================================================================
@all_stars
def test_snia_ave_age_range(star):
    """Check that ave_birth is past particle birth, but that it's not past
    15 Myr, the end of particle formation"""
    diff = star.snia["ave_birth"] - star.snia["birth"]
    assert 0 < diff < 15E6


@all_stars
def test_snia_age(star):
    test_age = star.snia["time"] - star.snia["ave_birth"]
    assert star.snia["age"] == test_age
    assert star.snia["age"] > 0


@all_stars
def test_snia_dt(star):
    test_dt = star.snia["next"] - star.snia["time"]
    assert star.snia["dt"] == test_dt


@all_stars
def test_snia_vol(star):
    assert star.snia["1/vol"] == star.snia["true 1/vol"]


@all_stars
def test_snia_mass_factor_consistency(star):
    factor = star.snia["Msol_to_code_mass"]
    inv_factor = star.snia["1/Msol_to_code_mass"]

    assert 1.0 / factor == approx(inv_factor, abs=0, rel=1E-10)


@all_stars
def test_snia_mass_factor_value(star):
    factor = star.snia["Msol_to_code_mass"]
    inv_factor = star.snia["1/Msol_to_code_mass"]

    assert (1.0 * u.Msun).to(code_mass).value == approx(factor, abs=0, rel=1E-7)
    assert (1.0 * code_mass).to(u.Msun).value == approx(inv_factor, abs=0, rel=1E-7)


@all_stars
def test_snia_mass_conversion(star):
    msun = star.snia["stellar mass Msun"]
    code = star.snia["stellar mass code"]

    assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
    assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)

@all_stars
def test_snia_mass_range(star):
    assert 1 < star.snia["stellar mass Msun"] < 1E8


@all_stars
def test_snia_metallicity_range(star):
    # just a check that we're getting the correct thing
    assert 0 < star.snia["metallicity II"] < 1
    assert 0 < star.snia["metallicity Ia"] < 1
    assert 0 < star.snia["metallicity AGB"] < 1
    assert 0 < star.snia["metallicity"] < 1


@all_stars
def test_snia_total_metallicity(star):
    test_total_z = star.snia["metallicity II"] + \
                   star.snia["metallicity Ia"] + \
                   star.snia["metallicity AGB"]
    assert star.snia["metallicity"] == test_total_z


@all_stars
def test_snia_early_rate(star):
    if star.snia["age"] < 20E6:
        assert star.snia["Ia rate"] == 0


@all_stars
def test_snia_late_rate(star):
    if star.snia["age"] > 50E6:
        true_rate = sn_ia_check.sn_dtd(star.snia["age"],
                                       star.snia["metallicity"])
        assert star.snia["Ia rate"] == approx(true_rate, abs=0, rel=1E-7)


@all_stars
def test_snia_all_rate(star):
    true_rate = sn_ia_check.sn_dtd(star.snia["age"],
                                   star.snia["metallicity"])
    assert star.snia["Ia rate"] == approx(true_rate, abs=0, rel=1E-7)


@all_stars
def test_snia_num_sn(star):
    true_rate = sn_ia_check.sn_dtd(star.snia["age"],
                                   star.snia["metallicity"])
    true_num = true_rate * star.snia["dt"] * star.snia["stellar mass Msun"]
    assert star.snia["num Ia"] == approx(true_num, abs=0, rel=1E-7)


@all_stars
def test_snia_ejecta_c_c_code_msun(star):
    true_c = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[0]
    assert star.snia["C ejecta Msun"] == approx(true_c, abs=0, rel=1E-7)


@all_stars
def test_snia_ejecta_n_c_code_msun(star):
    true_n = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[1]
    assert star.snia["N ejecta Msun"] == approx(true_n, abs=0, rel=1E-7)


@all_stars
def test_snia_ejecta_o_c_code_msun(star):
    true_o = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[2]
    assert star.snia["O ejecta Msun"] == approx(true_o, abs=0, rel=1E-7)


@all_stars
def test_snia_ejecta_fe_c_code_msun(star):
    true_fe = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[3]
    assert star.snia["Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-7)


@all_stars
def test_snia_ejecta_metals_c_code_msun(star):
    true_met = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[4]
    assert star.snia["metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-7)


@all_stars
def test_snia_ejecta_c_py_code_msun(star):
    true_c = sn_ia_yields_interp["C"](star.snia["metallicity"])
    assert star.snia["C ejecta Msun"] == approx(true_c, abs=0, rel=1E-6)


@all_stars
def test_snia_ejecta_n_py_code_msun(star):
    true_n = sn_ia_yields_interp["N"](star.snia["metallicity"])
    assert star.snia["N ejecta Msun"] == approx(true_n, abs=0, rel=1E-6)


@all_stars
def test_snia_ejecta_o_py_code_msun(star):
    true_o = sn_ia_yields_interp["O"](star.snia["metallicity"])
    assert star.snia["O ejecta Msun"] == approx(true_o, abs=0, rel=1E-6)


@all_stars
def test_snia_ejecta_fe_py_code_msun(star):
    true_fe = sn_ia_yields_interp["Fe"](star.snia["metallicity"])
    assert star.snia["Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-6)


@all_stars
def test_snia_ejecta_metals_py_code_msun(star):
    true_met = sn_ia_yields_interp["total_metals"](star.snia["metallicity"])
    assert star.snia["metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-6)

@all_stars
def test_snia_code_conversion_constant(star):
    c_ratio = star.snia["C ejecta Msun"] / star.snia["C ejecta code"]
    n_ratio = star.snia["N ejecta Msun"] / star.snia["N ejecta code"]
    o_ratio = star.snia["O ejecta Msun"] / star.snia["O ejecta code"]
    fe_ratio = star.snia["Fe ejecta Msun"] / star.snia["Fe ejecta code"]
    metals_ratio = star.snia["metals ejecta Msun"] / star.snia["metals ejecta code"]
    assert n_ratio == approx(c_ratio, abs=0, rel=1E-7)
    assert o_ratio == approx(c_ratio, abs=0, rel=1E-7)
    assert fe_ratio == approx(c_ratio, abs=0, rel=1E-7)
    assert metals_ratio == approx(c_ratio, abs=0, rel=1E-7)

@all_stars
def test_snia_code_mass_conversion_ejecta(star):
    for element in ["C", "N", "O", "Fe", "metals"]:
        msun = star.snia["{} ejecta Msun".format(element)]
        code = star.snia["{} ejecta code".format(element)]
        assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
        assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)


@all_stars
def test_snia_initial_densities_nonzero(star):
    for elt in ["C", "N", "O", "Fe", "Ia"]:
        assert star.snia["{} current".format(elt)] > 0


@all_stars
def test_end_ms_initial_densities_sun(star):
    elts_tot = sum([star.snia["{} current".format(elt)]
                    for elt in ["C", "N", "O", "Fe"]])
    sources_tot = sum([star.snia["{} current".format(source)]
                       for source in ["SNIa", "SNII", "AGB"]])

    # not all elements are individually tracked, so the sum of elements must
    # be less than the total
    assert sources_tot > elts_tot


@all_stars
def test_snia_adding_c_to_cell(star):
    old_density = star.snia["C current"]
    new_density = star.snia["C new"]
    added_mass = star.snia["C ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_snia_adding_n_to_cell(star):
    old_density = star.snia["N current"]
    new_density = star.snia["N new"]
    added_mass = star.snia["N ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_snia_adding_o_to_cell(star):
    old_density = star.snia["O current"]
    new_density = star.snia["O new"]
    added_mass = star.snia["O ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_snia_adding_fe_to_cell(star):
    old_density = star.snia["Fe current"]
    new_density = star.snia["Fe new"]
    added_mass = star.snia["Fe ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_snia_adding_metals_to_cell(star):
    old_density = star.snia["Ia current"]
    new_density = star.snia["Ia new"]
    added_mass = star.snia["metals ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


# ==============================================================================
#
# AGB / SN II
#
# ==============================================================================
@all_stars
def test_end_ms_ave_age_range(star):
    """Check that ave_birth is past particle birth, but that it's not past
    15 Myr, the end of particle formation"""
    diff = star.detail["ave_birth"] - star.detail["birth"]
    assert 0 < diff < 15E6


@all_stars
def test_end_ms_age(star):
    test_age = star.detail["time"] - star.detail["ave_birth"]
    assert star.detail["age"] == test_age
    assert star.detail["age"] > 0


@all_stars
def test_end_ms_dt(star):
    test_dt = star.detail["next"] - star.detail["time"]
    assert star.detail["dt"] == test_dt


@all_stars
def test_end_ms_vol(star):
    assert star.detail["1/vol"] == star.detail["true 1/vol"]


@all_stars
def test_end_ms_mass_factor_consistency(star):
    factor = star.detail["Msol_to_code_mass"]
    inv_factor = star.detail["1/Msol_to_code_mass"]

    assert 1.0 / factor == approx(inv_factor, abs=0, rel=1E-10)


@all_stars
def test_end_ms_mass_factor_value(star):
    factor = star.detail["Msol_to_code_mass"]
    inv_factor = star.detail["1/Msol_to_code_mass"]

    assert (1.0 * u.Msun).to(code_mass).value == approx(factor, abs=0, rel=1E-7)
    assert (1.0 * code_mass).to(u.Msun).value == approx(inv_factor, abs=0, rel=1E-7)


@all_stars
def test_end_ms_mass_range(star):
    assert 1 < star.detail["stellar mass Msun"] < 1E8


@all_stars
def test_end_ms_mass_conversion(star):
    msun = star.detail["stellar mass Msun"]
    code = star.detail["stellar mass code"]

    assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
    assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)


@all_stars
def test_end_ms_metallicity_range(star):
    # just a check that we're getting the correct thing
    assert 0 < star.detail["metallicity II"] < 1
    assert 0 < star.detail["metallicity Ia"] < 1
    assert 0 < star.detail["metallicity AGB"] < 1
    assert 0 < star.detail["metallicity"] < 1
    assert 0 < star.detail["metallicity C"] < star.detail["metallicity"]
    assert 0 < star.detail["metallicity N"] < star.detail["metallicity"]
    assert 0 < star.detail["metallicity O"] < star.detail["metallicity"]
    assert 0 < star.detail["metallicity Fe"] < star.detail["metallicity"]
    sum_elts = star.detail["metallicity C"] + star.detail["metallicity N"] + \
               star.detail["metallicity O"] + star.detail["metallicity Fe"]
    assert sum_elts < star.detail["metallicity"]


@all_stars
def test_end_ms_total_metallicity(star):
    test_total_z = star.detail["metallicity II"] + \
                   star.detail["metallicity Ia"] + \
                   star.detail["metallicity AGB"]
    assert star.detail["metallicity"] == test_total_z

# Simple test of the ejecta depending on the timing
@all_stars
def test_end_ms_simple(star):
    if star.detail["age"] < 3E6: # winds only
        assert star.detail["AGB C ejecta Msun"] == 0
        assert star.detail["AGB N ejecta Msun"] == 0
        assert star.detail["AGB O ejecta Msun"] == 0
        assert star.detail["AGB initial Fe ejecta Msun"] == 0
        assert star.detail["AGB initial metals ejecta Msun"] == 0
        assert star.detail["AGB total ejecta Msun"] == 0

        assert star.detail["SNII C ejecta Msun"] == 0
        assert star.detail["SNII N ejecta Msun"] == 0
        assert star.detail["SNII O ejecta Msun"] == 0
        assert star.detail["SNII Fe ejecta Msun"] == 0
        assert star.detail["SNII metals ejecta Msun"] == 0

        assert star.detail["Winds total ejecta Msun"] > 0

    elif 5E6 < star.detail["age"] < 30E6:  # SN and winds
        assert star.detail["AGB C ejecta Msun"] == 0
        assert star.detail["AGB N ejecta Msun"] == 0
        assert star.detail["AGB O ejecta Msun"] == 0
        assert star.detail["AGB initial Fe ejecta Msun"] == 0
        assert star.detail["AGB initial metals ejecta Msun"] == 0
        assert star.detail["AGB total ejecta Msun"] == 0

        assert star.detail["SNII C ejecta Msun"] > 0
        assert star.detail["SNII N ejecta Msun"] > 0
        assert star.detail["SNII O ejecta Msun"] > 0
        assert star.detail["SNII Fe ejecta Msun"] > 0
        assert star.detail["SNII metals ejecta Msun"] > 0

        assert star.detail["Winds total ejecta Msun"] > 0

    elif star.detail["age"] > 50E6:  # AGB only
        assert star.detail["AGB C ejecta Msun"] > 0
        assert star.detail["AGB N ejecta Msun"] > 0
        assert star.detail["AGB O ejecta Msun"] > 0
        assert star.detail["AGB initial Fe ejecta Msun"] > 0
        assert star.detail["AGB initial metals ejecta Msun"] > 0
        assert star.detail["AGB total ejecta Msun"] > 0

        assert star.detail["SNII C ejecta Msun"] == 0
        assert star.detail["SNII N ejecta Msun"] == 0
        assert star.detail["SNII O ejecta Msun"] == 0
        assert star.detail["SNII Fe ejecta Msun"] == 0
        assert star.detail["SNII metals ejecta Msun"] == 0

        assert star.detail["Winds total ejecta Msun"] == 0


# Testing the ejected masses. I have tested the C functions separately, so what
# I'll do here is to check that the values are what's returned by those
# functions. I will also compare against the ejected masses my Python code
# gives, but with a larger tolerance, since the interpolation is done a bit
# differently between those two methods.
@all_stars
def test_agb_ejecta_c_msun_c_code(star):
    true_c = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
                                               star.detail["metallicity"],
                                               star.detail["stellar mass Msun"],
                                               star.detail["dt"])[0]
    assert star.detail["AGB C ejecta Msun"] == approx(true_c, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_n_msun_c_code(star):
    true_n = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
                                               star.detail["metallicity"],
                                               star.detail["stellar mass Msun"],
                                               star.detail["dt"])[1]
    assert star.detail["AGB N ejecta Msun"] == approx(true_n, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_o_msun_c_code(star):
    true_o = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
                                               star.detail["metallicity"],
                                               star.detail["stellar mass Msun"],
                                               star.detail["dt"])[2]
    assert star.detail["AGB O ejecta Msun"] == approx(true_o, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_fe_msun_c_code(star):
    true_fe = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
                                                star.detail["metallicity"],
                                                star.detail["stellar mass Msun"],
                                                star.detail["dt"])[3]
    assert star.detail["AGB initial Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_met_msun_c_code(star):
    true_met = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
                                                 star.detail["metallicity"],
                                                 star.detail["stellar mass Msun"],
                                                 star.detail["dt"])[4]
    assert star.detail["AGB initial metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_tot_msun_c_code(star):
    true_ej = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
                                                 star.detail["metallicity"],
                                                 star.detail["stellar mass Msun"],
                                                 star.detail["dt"])[5]
    assert star.detail["AGB total ejecta Msun"] == approx(true_ej, abs=0, rel=1E-6)


# @all_stars
# def test_agb_ejecta_c_msun_py_code(star):
#     true_c = mass_loss_agb(star.detail["age"], star.detail["dt"],
#                            star.detail["metallicity"], "C") * star.detail["stellar mass Msun"]
#
#     assert star.detail["AGB C ejecta Msun"] == approx(true_c, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_agb_ejecta_n_msun_py_code(star):
#     true_n = mass_loss_agb(star.detail["age"], star.detail["dt"],
#                            star.detail["metallicity"], "N") * star.detail["stellar mass Msun"]
#
#     assert star.detail["AGB N ejecta Msun"] == approx(true_n, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_agb_ejecta_o_msun_py_code(star):
#     true_o = mass_loss_agb(star.detail["age"], star.detail["dt"],
#                            star.detail["metallicity"], "O") * star.detail["stellar mass Msun"]
#
#     assert star.detail["AGB O ejecta Msun"] == approx(true_o, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_agb_ejecta_fe_msun_py_code(star):
#     true_fe = mass_loss_agb(star.detail["age"], star.detail["dt"],
#                             star.detail["metallicity"], "Fe") * star.detail["stellar mass Msun"]
#
#     assert star.detail["AGB initial Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_agb_ejecta_met_py_code(star):
#     true_met = mass_loss_agb(star.detail["age"], star.detail["dt"],
#                              star.detail["metallicity"], "total_metals") * star.detail["stellar mass Msun"]
#
#     assert star.detail["AGB initial metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_agb_ejecta_tot_msun_py_code(star):
#     true_ej = mass_loss_agb(star.detail["age"], star.detail["dt"],
#                              star.detail["metallicity"], "total") * star.detail["stellar mass Msun"]
#
#     assert star.detail["AGB total ejecta Msun"] == approx(true_ej, abs=0, rel=1E-1)


@all_stars
def test_sn_ii_ejecta_c_msun_c_code(star):
    true_c = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
                                                star.detail["metallicity"],
                                                star.detail["stellar mass Msun"],
                                                star.detail["dt"])[0]
    assert star.detail["SNII C ejecta Msun"] == approx(true_c, abs=0, rel=1E-6)


@all_stars
def test_sn_ii_ejecta_n_msun_c_code(star):
    true_n = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
                                                star.detail["metallicity"],
                                                star.detail["stellar mass Msun"],
                                                star.detail["dt"])[1]
    assert star.detail["SNII N ejecta Msun"] == approx(true_n, abs=0, rel=1E-6)


@all_stars
def test_sn_ii_ejecta_o_msun_c_code(star):
    true_o = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
                                                star.detail["metallicity"],
                                                star.detail["stellar mass Msun"],
                                                star.detail["dt"])[2]
    assert star.detail["SNII O ejecta Msun"] == approx(true_o, abs=0, rel=1E-6)


@all_stars
def test_sn_ii_ejecta_fe_msun_c_code(star):
    true_fe = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
                                                 star.detail["metallicity"],
                                                 star.detail["stellar mass Msun"],
                                                 star.detail["dt"])[3]
    assert star.detail["SNII Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-6)


@all_stars
def test_sn_ii_ejecta_met_msun_c_code(star):
    true_met = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
                                                  star.detail["metallicity"],
                                                  star.detail["stellar mass Msun"],
                                                  star.detail["dt"])[4]
    assert star.detail["SNII metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-6)


# @all_stars
# def test_sn_ii_ejecta_c_msun_py_code(star):
#     true_c = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
#                              star.detail["metallicity"], "C") * star.detail["stellar mass Msun"]
#
#     assert star.detail["SNII C ejecta Msun"] == approx(true_c, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_sn_ii_ejecta_n_msun_py_code(star):
#     true_n = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
#                              star.detail["metallicity"], "N") * star.detail["stellar mass Msun"]
#
#     assert star.detail["SNII N ejecta Msun"] == approx(true_n, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_sn_ii_ejecta_o_msun_py_code(star):
#     true_o = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
#                              star.detail["metallicity"], "O") * star.detail["stellar mass Msun"]
#
#     assert star.detail["SNII O ejecta Msun"] == approx(true_o, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_sn_ii_ejecta_fe_msun_py_code(star):
#     true_fe = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
#                               star.detail["metallicity"], "Fe") * star.detail["stellar mass Msun"]
#
#     assert star.detail["SNII Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-1)
#
#
# @all_stars
# def test_sn_ii_ejecta_met_py_code(star):
#     true_met = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
#                                star.detail["metallicity"], "total_metals") * star.detail["stellar mass Msun"]
#
#     assert star.detail["SNII metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-1)


@all_stars
def test_winds_ejecta_total_msun_c_code(star):
    true_ej = c_code.get_ejecta_timestep_winds_py(star.detail["age"],
                                                  star.detail["metallicity"],
                                                  star.detail["stellar mass Msun"],
                                                  star.detail["dt"])[0]
    assert star.detail["Winds total ejecta Msun"] == approx(true_ej, abs=0, rel=1E-6)


@all_stars
def test_end_ms_code_mass_conversion_ejecta(star):
    for source in ["AGB C", "AGB N", "AGB O",
                   "SNII C", "SNII N", "SNII O", "SNII Fe", "SNII metals"]:
        msun = star.detail["{} ejecta Msun".format(source)]
        code = star.detail["{} ejecta code".format(source)]
        assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
        assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)


@all_stars
def test_agb_fe_mass_conversion(star):
    true_fe_msun = star.detail["metallicity Fe"] * star.detail["AGB total ejecta Msun"]
    true_fe_code = (true_fe_msun*u.Msun).to(code_mass).value
    test_fe_code = star.detail["AGB Fe ejecta code"]
    assert true_fe_code == approx(test_fe_code, abs=0, rel=1E-7)


@all_stars
def test_agb_metals_mass_conversion(star):
    initial_fe_msun = star.detail["AGB initial Fe ejecta Msun"]
    true_fe_msun = star.detail["metallicity Fe"] * star.detail["AGB total ejecta Msun"]
    change_in_fe_msun = initial_fe_msun - true_fe_msun
    true_metals_msun = star.detail["AGB initial metals ejecta Msun"] - change_in_fe_msun

    true_metals_code = (true_metals_msun * u.Msun).to(code_mass).value
    test_metals_code = star.detail["AGB metals ejecta code"]
    assert true_metals_code == approx(test_metals_code, abs=0, rel=1E-7)

@all_stars
def test_winds_split_into_metals(star):
    total_ejecta_msun = star.detail["Winds total ejecta Msun"]
    total_ejecta_code = (total_ejecta_msun*u.Msun).to(code_mass).value
    assert total_ejecta_code * star.detail["metallicity C"] == \
           approx(star.detail["Winds C ejecta code"])
    assert total_ejecta_code * star.detail["metallicity N"] == \
           approx(star.detail["Winds N ejecta code"])
    assert total_ejecta_code * star.detail["metallicity O"] == \
           approx(star.detail["Winds O ejecta code"])
    assert total_ejecta_code * star.detail["metallicity Fe"] == \
           approx(star.detail["Winds Fe ejecta code"])
    assert total_ejecta_code * star.detail["metallicity Ia"] == \
           approx(star.detail["Winds SNIa ejecta code"])
    assert total_ejecta_code * star.detail["metallicity II"] == \
           approx(star.detail["Winds SNII ejecta code"])
    assert total_ejecta_code * star.detail["metallicity AGB"] == \
           approx(star.detail["Winds AGB ejecta code"])


@all_stars
def test_end_ms_initial_densities_nonzero(star):
    for elt in ["C", "N", "O", "Fe", "AGB", "SNIa", "SNII"]:
        assert star.detail["{} current".format(elt)] > 0


@all_stars
def test_end_ms_initial_densities_sun(star):
    elts_tot = sum([star.detail["{} current".format(elt)]
                   for elt in ["C", "N", "O", "Fe"]])
    sources_tot = sum([star.detail["{} current".format(source)]
                   for source in ["SNIa", "SNII", "AGB"]])

    # not all elements are individually tracked, so the sum of elements must
    # be less than the total
    assert sources_tot > elts_tot



@all_stars
def test_end_ms_adding_c_to_cell(star):
    old_density = star.detail["C current"]
    new_density = star.detail["C new"]
    added_mass = star.detail["AGB C ejecta code"] + \
                 star.detail["SNII C ejecta code"] + \
                 star.detail["Winds C ejecta code"]
    added_density = added_mass * star.detail["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_end_ms_adding_n_to_cell(star):
    old_density = star.detail["N current"]
    new_density = star.detail["N new"]
    added_mass = star.detail["AGB N ejecta code"] + \
                 star.detail["SNII N ejecta code"] + \
                 star.detail["Winds N ejecta code"]
    added_density = added_mass * star.detail["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_end_ms_adding_o_to_cell(star):
    old_density = star.detail["O current"]
    new_density = star.detail["O new"]
    added_mass = star.detail["AGB O ejecta code"] + \
                 star.detail["SNII O ejecta code"] + \
                 star.detail["Winds O ejecta code"]
    added_density = added_mass * star.detail["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_end_ms_adding_fe_to_cell(star):
    old_density = star.detail["Fe current"]
    new_density = star.detail["Fe new"]
    added_mass = star.detail["AGB Fe ejecta code"] + \
                 star.detail["SNII Fe ejecta code"] + \
                 star.detail["Winds Fe ejecta code"]
    added_density = added_mass * star.detail["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_agb_adding_met_to_cell(star):
    old_density = star.detail["AGB current"]
    new_density = star.detail["AGB new"]
    added_mass = star.detail["AGB metals ejecta code"] + \
                 star.detail["Winds AGB ejecta code"]
    added_density = added_mass * star.detail["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_sn_ia_adding_met_to_cell(star):
    old_density = star.detail["SNIa current"]
    new_density = star.detail["SNIa new"]
    added_mass = star.detail["Winds SNIa ejecta code"]
    added_density = added_mass * star.detail["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_sn_ii_adding_met_to_cell(star):
    old_density = star.detail["SNII current"]
    new_density = star.detail["SNII new"]
    added_mass = star.detail["SNII metals ejecta code"] + \
                 star.detail["Winds SNII ejecta code"]
    added_density = added_mass * star.detail["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
