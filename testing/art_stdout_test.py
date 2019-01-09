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


# create the code units to check against
code_length = u.def_unit("code_length", 4 * u.Mpc / 128)
h = 0.6814000010490417
H_0 = 100 * h * u.km / (u.second * u.Mpc)
omega_m = 0.3035999834537506
code_mass = u.def_unit("code_mass", 3 * H_0**2 * omega_m / (8 * np.pi * c.G) *
                       code_length**3)
code_time = u.def_unit("code_time", 2.0 / (H_0 * np.sqrt(omega_m)))


number_to_check = 100
timesteps_to_check = parse_stdout.parse_file(10**4)
timesteps_to_check = random.sample(timesteps_to_check, number_to_check)

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
def test_snia_age(star):
    test_age = star.snia["time"] - star.snia["birth"]
    assert star.snia["age"] == test_age


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
def test_snia_ejecta_c(star):
    true_c = sn_ia_yields_interp["C"](star.snia["metallicity"])
    assert star.snia["C ejecta Msun"] == approx(true_c, abs=0, rel=1E-6)


@all_stars
def test_snia_ejecta_n(star):
    true_n = sn_ia_yields_interp["N"](star.snia["metallicity"])
    assert star.snia["N ejecta Msun"] == approx(true_n, abs=0, rel=1E-6)


@all_stars
def test_snia_ejecta_o(star):
    true_o = sn_ia_yields_interp["O"](star.snia["metallicity"])
    assert star.snia["O ejecta Msun"] == approx(true_o, abs=0, rel=1E-6)


@all_stars
def test_snia_ejecta_fe(star):
    true_fe = sn_ia_yields_interp["Fe"](star.snia["metallicity"])
    assert star.snia["Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-6)


@all_stars
def test_snia_ejecta_metals(star):
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
        msun = star.snia["{} ejecta Msun".format(element)] * u.Msun
        code = star.snia["{} ejecta code".format(element)]
        assert msun.to(code_mass).value == approx(code, abs=0, rel=1E-7)

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
# AGB
#
# ==============================================================================
@all_stars
def test_agb_age(star):
    test_age = star.agb["time"] - star.agb["birth"]
    assert star.agb["age"] == test_age


@all_stars
def test_agb_dt(star):
    test_dt = star.agb["next"] - star.agb["time"]
    assert star.snia["dt"] == test_dt


@all_stars
def test_agb_vol(star):
    assert star.agb["1/vol"] == star.agb["true 1/vol"]


@all_stars
def test_agb_mass_factor_consistency(star):
    factor = star.agb["Msol_to_code_mass"]
    inv_factor = star.agb["1/Msol_to_code_mass"]

    assert 1.0 / factor == approx(inv_factor, abs=0, rel=1E-10)


@all_stars
def test_agb_mass_factor_value(star):
    factor = star.agb["Msol_to_code_mass"]
    inv_factor = star.agb["1/Msol_to_code_mass"]

    assert (1.0 * u.Msun).to(code_mass).value == approx(factor, abs=0, rel=1E-7)
    assert (1.0 * code_mass).to(u.Msun).value == approx(inv_factor, abs=0, rel=1E-7)


@all_stars
def test_agb_mass_conversion(star):
    msun = star.snia["stellar mass Msun"]
    code = star.snia["stellar mass code"]

    assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
    assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)


@all_stars
def test_agb_total_metallicity(star):
    test_total_z = star.agb["metallicity II"] + \
                   star.agb["metallicity Ia"] + \
                   star.agb["metallicity AGB"]
    assert star.agb["metallicity"] == test_total_z

# Testing the ejected masses. I have tested the C functions separately, so what
# I'll do here is to check that the values are what's returned by those
# functions. I will also compare against the ejected masses my Python code
# gives, but with a larger tolerance, since the interpolation is done a bit
# differently between those two methods.
@all_stars
def test_agb_ejecta_c_c_code(star):
    true_c = c_code.get_ejecta_timestep_agb_py(star.agb["age"],
                                               star.agb["metallicity"],
                                               star.agb["stellar mass Msun"],
                                               star.agb["dt"])[0]
    assert star.agb["C ejecta Msun"] == approx(true_c, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_n_c_code(star):
    true_n = c_code.get_ejecta_timestep_agb_py(star.agb["age"],
                                               star.agb["metallicity"],
                                               star.agb["stellar mass Msun"],
                                               star.agb["dt"])[1]
    assert star.agb["N ejecta Msun"] == approx(true_n, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_o_c_code(star):
    true_o = c_code.get_ejecta_timestep_agb_py(star.agb["age"],
                                               star.agb["metallicity"],
                                               star.agb["stellar mass Msun"],
                                               star.agb["dt"])[2]
    assert star.agb["O ejecta Msun"] == approx(true_o, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_fe_c_code(star):
    true_fe = c_code.get_ejecta_timestep_agb_py(star.agb["age"],
                                                star.agb["metallicity"],
                                                star.agb["stellar mass Msun"],
                                                star.agb["dt"])[3]
    assert star.agb["initial Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_met_c_code(star):
    true_met = c_code.get_ejecta_timestep_agb_py(star.agb["age"],
                                                 star.agb["metallicity"],
                                                 star.agb["stellar mass Msun"],
                                                 star.agb["dt"])[4]
    assert star.agb["initial metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-6)


@all_stars
def test_agb_ejecta_tot_c_code(star):
    true_ej = c_code.get_ejecta_timestep_agb_py(star.agb["age"],
                                                 star.agb["metallicity"],
                                                 star.agb["stellar mass Msun"],
                                                 star.agb["dt"])[5]
    assert star.agb["total ejecta Msun"] == approx(true_ej, abs=0, rel=1E-6)
# =======

@all_stars
def test_agb_ejecta_c_py_code(star):
    true_c = mass_loss_agb(star.agb["age"], star.agb["dt"],
                           star.agb["metallicity"], "C") * star.agb["stellar mass Msun"]

    assert star.agb["C ejecta Msun"] == approx(true_c, abs=0, rel=1E-1)


@all_stars
def test_agb_ejecta_n_py_code(star):
    true_n = mass_loss_agb(star.agb["age"], star.agb["dt"],
                           star.agb["metallicity"], "N") * star.agb["stellar mass Msun"]

    assert star.agb["N ejecta Msun"] == approx(true_n, abs=0, rel=1E-1)


@all_stars
def test_agb_ejecta_o_py_code(star):
    true_o = mass_loss_agb(star.agb["age"], star.agb["dt"],
                           star.agb["metallicity"], "O") * star.agb["stellar mass Msun"]

    assert star.agb["O ejecta Msun"] == approx(true_o, abs=0, rel=1E-1)


@all_stars
def test_agb_ejecta_fe_py_code(star):
    true_fe = mass_loss_agb(star.agb["age"], star.agb["dt"],
                            star.agb["metallicity"], "Fe") * star.agb["stellar mass Msun"]

    assert star.agb["initial Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-1)


@all_stars
def test_agb_ejecta_met_py_code(star):
    true_met = mass_loss_agb(star.agb["age"], star.agb["dt"],
                             star.agb["metallicity"], "total_metals") * star.agb["stellar mass Msun"]

    assert star.agb["initial metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-1)


@all_stars
def test_agb_ejecta_tot_py_code(star):
    true_ej = mass_loss_agb(star.agb["age"], star.agb["dt"],
                             star.agb["metallicity"], "total") * star.agb["stellar mass Msun"]

    assert star.agb["total ejecta Msun"] == approx(true_ej, abs=0, rel=1E-1)