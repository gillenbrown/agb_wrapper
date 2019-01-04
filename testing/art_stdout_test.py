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

"""
This file tests the implementation of the code directly in ART by parsing the
stdout file generated as it runs and checking that what's written is correct.
"""

imf = tabulation.IMF("Kroupa", 0.1, 50)
lifetimes = tabulation.Lifetimes("Raiteri_96")
number_sn_ia = 1.6E-3
sn_ia_check = tabulation.SNIa("ART power law", "Nomoto_18", lifetimes, imf,
                              number_sn_ia=number_sn_ia)
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
def test_snia_rate_code_units(star):
    true_rate = sn_ia_check.sn_dtd(star.snia["age"],
                                   star.snia["metallicity"]) / u.year
    assert star.snia["Ia rate code"] == approx(true_rate.to(1/code_time).value,
                                               abs=0, rel=1E-7)


@all_stars
def test_snia_num_sn(star):
    true_rate = sn_ia_check.sn_dtd(star.snia["age"],
                                   star.snia["metallicity"])
    true_num = true_rate * star.snia["dt"] * star.snia["stellar mass"]
    assert star.snia["num Ia"] == approx(true_num, abs=0, rel=1E-7)


@all_stars
def test_ejecta_c(star):
    true_c = sn_ia_yields_interp["C"](star.snia["metallicity"])
    assert star.snia["C ejecta Msun"] == approx(true_c, abs=0, rel=1E-6)


@all_stars
def test_ejecta_n(star):
    true_n = sn_ia_yields_interp["N"](star.snia["metallicity"])
    assert star.snia["N ejecta Msun"] == approx(true_n, abs=0, rel=1E-6)


@all_stars
def test_ejecta_o(star):
    true_o = sn_ia_yields_interp["O"](star.snia["metallicity"])
    assert star.snia["O ejecta Msun"] == approx(true_o, abs=0, rel=1E-6)


@all_stars
def test_ejecta_fe(star):
    true_fe = sn_ia_yields_interp["Fe"](star.snia["metallicity"])
    assert star.snia["Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-6)


@all_stars
def test_ejecta_metals(star):
    true_met = sn_ia_yields_interp["total_metals"](star.snia["metallicity"])
    assert star.snia["metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-6)

@all_stars
def test_code_conversion_constant(star):
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
def test_adding_c_to_cell(star):
    old_density = star.snia["C current"]
    new_density = star.snia["C new"]
    added_mass = star.snia["C ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_adding_n_to_cell(star):
    old_density = star.snia["N current"]
    new_density = star.snia["N new"]
    added_mass = star.snia["N ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_adding_o_to_cell(star):
    old_density = star.snia["O current"]
    new_density = star.snia["O new"]
    added_mass = star.snia["O ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_adding_fe_to_cell(star):
    old_density = star.snia["Fe current"]
    new_density = star.snia["Fe new"]
    added_mass = star.snia["Fe ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


@all_stars
def test_adding_metals_to_cell(star):
    old_density = star.snia["Ia current"]
    new_density = star.snia["Ia new"]
    added_mass = star.snia["metals ejecta code"]  # per SN
    added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
    assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)

#TODO: check that I've done all the testing of everything in SNIa
#TODO: need to check that the conversion to code masses and times is working okay
# ==============================================================================
#
# AGB
#
# ==============================================================================
@all_stars
def test_agb_age(star):
    test_age = star.agb["time"] - star.agb["birth"]
    assert star.agb["age"] == test_age
