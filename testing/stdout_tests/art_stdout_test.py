import sys
from pathlib import Path
import pytest
from pytest import approx
import random
from scipy import interpolate
from astropy import units as u
from astropy import constants as c
import numpy as np

import tabulation

import parse_stdout

# add directory of compiled C code to my path so it can be imported
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent))

from core_elts import lib as art_enrich_core
art_enrich_core.detailed_enrichment_init()

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

# ==============================================================================
#
# pick the stars to test
#
# ==============================================================================
number_to_check = 1000
# We want some that have winds, some that have SN, and some that have AGB
good = False
all_source_timesteps = parse_stdout.parse_file(str(this_dir/"snii_stdout.txt"),
                                               10**4)
wind_timesteps_all = all_source_timesteps[0]
snii_timesteps_inactive_all = all_source_timesteps[1]
snii_timesteps_active_all = all_source_timesteps[2]
snia_timesteps_inactive_all = all_source_timesteps[3]
snia_timesteps_active_all = all_source_timesteps[4]
agb_timesteps_all  = all_source_timesteps[5]

def test_agb_present():
    assert len(agb_timesteps_all) > 0
        # raise RuntimeError("stdout file appears to lack any stars in the "
        #                    "AGB phase. Regenerate and try again.")
def test_snii_inactive_present():
    assert len(snii_timesteps_inactive_all) > 0
        # raise RuntimeError("stdout file appears to lack any stars in the "
        #                    "SNII phase. Regenerate and try again.")
def test_snii_active_present():
    assert len(snii_timesteps_active_all) > 0
        # raise RuntimeError("stdout file appears to lack any stars in the "
        #                    "SNII phase. Regenerate and try again.")
def test_wind_present():
    assert len(wind_timesteps_all) > 0
        # raise RuntimeError("stdout file appears to lack any stars in the "
        #                    "stellar wind phase. Regenerate and try again.")
def test_snia_inactive_present():
    assert len(snia_timesteps_inactive_all) > 0
        # raise RuntimeError("stdout file appears to lack any stars that are "
        #                    "undergoing SNIa. Regenerate and try again.")
def test_snia_active_present():
    assert len(snia_timesteps_active_all) > 0
        # raise RuntimeError("stdout file appears to lack any stars that are "
        #                    "undergoing SNIa. Regenerate and try again.")

# draw a random sample of these to compare against, to make tests faster
def random_sample(full_sample, n_desired):
    if len(full_sample) < n_desired:
        return full_sample
    return random.sample(full_sample, number_to_check)

# winds and AGB are continuous, so any random sample is okay
wind_to_check = random_sample(wind_timesteps_all, number_to_check)
agb_to_check  = random_sample(agb_timesteps_all,  number_to_check)
snia_active_to_check = random_sample(snia_timesteps_active_all, number_to_check)
snia_inactive_to_check = random_sample(snia_timesteps_inactive_all, number_to_check)
snii_active_to_check = random_sample(snii_timesteps_active_all, number_to_check)
snii_inactive_to_check = random_sample(snii_timesteps_inactive_all, number_to_check)

# Convert masses into solar masses. We want to do this for ease of comparison
for source in [wind_to_check, snia_active_to_check, snia_inactive_to_check,
               agb_to_check, snii_active_to_check, snii_inactive_to_check]:
    for star in source:
        for star_dict in [star.snia, star.snii, star.wind, star.agb]:
            for key in star_dict.copy():
                new_key = key.replace("code", "Msun")
                if new_key not in star_dict:
                    m_sun_val = (star_dict[key] * code_mass).to(u.Msun).value
                    star_dict[new_key + " local"] = m_sun_val

# Make a wrapper to test functions on all stars, but without using pytests
# parameterize. I don't want to do this because it counts each iteration as an
# extra test, making the number of tests go crazy. This doesn't do that.
def all_stars_wind(func):
    def internal():
        for star in wind_to_check:
            func(star)
    return internal

def all_stars_snii(func):
    def internal():
        for star in snii_active_to_check + snii_inactive_to_check:
            func(star)
    return internal

def all_stars_snii_active(func):
    def internal():
        for star in snii_active_to_check :
            func(star)
    return internal

def all_stars_snii_inactive(func):
    def internal():
        for star in snii_inactive_to_check:
            func(star)
    return internal

def all_stars_snia(func):
    def internal():
        for star in snia_active_to_check + snia_inactive_to_check:
            func(star)
    return internal

def all_stars_snia_active(func):
    def internal():
        for star in snia_active_to_check :
            func(star)
    return internal

def all_stars_snia_inactive(func):
    def internal():
        for star in snia_inactive_to_check:
            func(star)
    return internal

def all_stars_agb(func):
    def internal():
        for star in agb_to_check:
            func(star)
    return internal


# ==============================================================================
#
# SN Ia
#
# ==============================================================================
# @all_stars_snia
# def test_snia_ave_age_range(star):
#     """Check that ave_birth is past particle birth, but that it's not past
#     15 Myr, the end of particle formation"""
#     diff = star.snia["ave_birth"] - star.snia["birth"]
#     assert 0 < diff < 15E6
#
#
# @all_stars_snia
# def test_snia_age(star):
#     test_age = star.snia["time"] - star.snia["ave_birth"]
#     assert star.snia["age"] == test_age
#     # The first timestep a star forms, it's ave_birth will be set to half of
#     # the current timestep after the current time, so in this first timestep
#     # the star can have negative ages.
#     assert star.snia["age"] > -0.6 * star.snia["dt"]
#
#
# @all_stars_snia
# def test_snia_dt(star):
#     test_dt = star.snia["next"] - star.snia["time"]
#     assert star.snia["dt"] == test_dt
#
#
# @all_stars_snia
# def test_snia_vol(star):
#     assert star.snia["1/vol"] == star.snia["true 1/vol"]
#
#
# @all_stars_snia
# def test_snia_mass_factor_consistency(star):
#     factor = star.snia["Msol_to_code_mass"]
#     inv_factor = star.snia["1/Msol_to_code_mass"]
#
#     assert 1.0 / factor == approx(inv_factor, abs=0, rel=1E-10)
#
#
# @all_stars_snia
# def test_snia_mass_factor_value(star):
#     factor = star.snia["Msol_to_code_mass"]
#     inv_factor = star.snia["1/Msol_to_code_mass"]
#
#     assert (1.0 * u.Msun).to(code_mass).value == approx(factor, abs=0, rel=1E-7)
#     assert (1.0 * code_mass).to(u.Msun).value == approx(inv_factor, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_mass_conversion(star):
#     msun = star.snia["stellar mass Msun"]
#     code = star.snia["stellar mass code"]
#
#     assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
#     assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)
#
# @all_stars_snia
# def test_snia_mass_range(star):
#     assert 1 < star.snia["stellar mass Msun"] < 1E8
#
#
# @all_stars_snia
# def test_snia_metallicity_range(star):
#     # just a check that we're getting the correct thing
#     assert 0 < star.snia["metallicity II"] < 1
#     assert 0 < star.snia["metallicity Ia"] < 1
#     assert 0 < star.snia["metallicity AGB"] < 1
#     assert 0 < star.snia["metallicity"] < 1
#
#
# @all_stars_snia
# def test_snia_total_metallicity(star):
#     test_total_z = star.snia["metallicity II"] + \
#                    star.snia["metallicity Ia"] + \
#                    star.snia["metallicity AGB"]
#     assert star.snia["metallicity"] == test_total_z
#
#
# @all_stars_snia
# def test_snia_early_rate(star):
#     if star.snia["age"] < 20E6:
#         assert star.snia["Ia rate"] == 0
#
#
# @all_stars_snia
# def test_snia_late_rate(star):
#     if star.snia["age"] > 50E6:
#         true_rate = sn_ia_check.sn_dtd(star.snia["age"],
#                                        star.snia["metallicity"])
#         assert star.snia["Ia rate"] == approx(true_rate, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_all_rate(star):
#     true_rate = sn_ia_check.sn_dtd(star.snia["age"],
#                                    star.snia["metallicity"])
#     assert star.snia["Ia rate"] == approx(true_rate, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_num_sn(star):
#     true_rate = sn_ia_check.sn_dtd(star.snia["age"],
#                                    star.snia["metallicity"])
#     true_num = true_rate * star.snia["dt"] * star.snia["stellar mass Msun"]
#     assert star.snia["num Ia"] == approx(true_num, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_c_c_code_msun(star):
#     true_c = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[0]
#     assert star.snia["C ejecta Msun local"] == approx(true_c, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_n_c_code_msun(star):
#     true_n = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[1]
#     assert star.snia["N ejecta Msun local"] == approx(true_n, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_o_c_code_msun(star):
#     true_o = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[2]
#     assert star.snia["O ejecta Msun local"] == approx(true_o, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_mg_c_code_msun(star):
#     true_mg = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[3]
#     assert star.snia["Mg ejecta Msun local"] == approx(true_mg, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_s_c_code_msun(star):
#     true_s = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[4]
#     assert star.snia["S ejecta Msun local"] == approx(true_s, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_ca_c_code_msun(star):
#     true_ca = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[5]
#     assert star.snia["Ca ejecta Msun local"] == approx(true_ca, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_fe_c_code_msun(star):
#     true_fe = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[6]
#     assert star.snia["Fe ejecta Msun local"] == approx(true_fe, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_metals_c_code_msun(star):
#     true_met = c_code.get_yields_sn_ia_py(star.snia["metallicity"])[7]
#     assert star.snia["metals ejecta Msun local"] == approx(true_met, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_ejecta_c_py_code_msun(star):
#     true_c = sn_ia_yields_interp["C"](star.snia["metallicity"])
#     assert star.snia["C ejecta Msun local"] == approx(true_c, abs=0, rel=1E-6)
#
#
# @all_stars_snia
# def test_snia_ejecta_n_py_code_msun(star):
#     true_n = sn_ia_yields_interp["N"](star.snia["metallicity"])
#     assert star.snia["N ejecta Msun local"] == approx(true_n, abs=0, rel=1E-6)
#
#
# @all_stars_snia
# def test_snia_ejecta_o_py_code_msun(star):
#     true_o = sn_ia_yields_interp["O"](star.snia["metallicity"])
#     assert star.snia["O ejecta Msun local"] == approx(true_o, abs=0, rel=1E-6)
#
#
# @all_stars_snia
# def test_snia_ejecta_fe_py_code_msun(star):
#     true_fe = sn_ia_yields_interp["Fe"](star.snia["metallicity"])
#     assert star.snia["Fe ejecta Msun local"] == approx(true_fe, abs=0, rel=1E-6)
#
#
# @all_stars_snia
# def test_snia_ejecta_metals_py_code_msun(star):
#     true_met = sn_ia_yields_interp["total_metals"](star.snia["metallicity"])
#     assert star.snia["metals ejecta Msun local"] == approx(true_met, abs=0, rel=1E-6)
#
# # @all_stars
# # def test_snia_code_conversion_constant(star):
# #     c_ratio = star.snia["C ejecta Msun"] / star.snia["C ejecta code"]
# #     n_ratio = star.snia["N ejecta Msun"] / star.snia["N ejecta code"]
# #     o_ratio = star.snia["O ejecta Msun"] / star.snia["O ejecta code"]
# #     fe_ratio = star.snia["Fe ejecta Msun"] / star.snia["Fe ejecta code"]
# #     metals_ratio = star.snia["metals ejecta Msun"] / star.snia["metals ejecta code"]
# #     assert n_ratio == approx(c_ratio, abs=0, rel=1E-7)
# #     assert o_ratio == approx(c_ratio, abs=0, rel=1E-7)
# #     assert fe_ratio == approx(c_ratio, abs=0, rel=1E-7)
# #     assert metals_ratio == approx(c_ratio, abs=0, rel=1E-7)
#
# # @all_stars
# # def test_snia_code_mass_conversion_ejecta(star):
# #     for element in ["C", "N", "O", "Fe", "metals"]:
# #         msun = star.snia["{} ejecta Msun".format(element)]
# #         code = star.snia["{} ejecta code".format(element)]
# #         assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
# #         assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_initial_densities_nonzero(star):
#     for elt in ["C", "N", "O", "Fe", "Ia"]:
#         assert star.snia["{} current".format(elt)] > 0
#
#
# @all_stars_snia
# def test_snia_adding_c_to_cell(star):
#     old_density = star.snia["C current"]
#     new_density = star.snia["C new"]
#     added_mass = star.snia["C ejecta code"]  # per SN
#     added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_adding_n_to_cell(star):
#     old_density = star.snia["N current"]
#     new_density = star.snia["N new"]
#     added_mass = star.snia["N ejecta code"]  # per SN
#     added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_adding_o_to_cell(star):
#     old_density = star.snia["O current"]
#     new_density = star.snia["O new"]
#     added_mass = star.snia["O ejecta code"]  # per SN
#     added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_adding_mg_to_cell(star):
#     old_density = star.snia["Mg current"]
#     new_density = star.snia["Mg new"]
#     added_mass = star.snia["Mg ejecta code"]  # per SN
#     added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_adding_s_to_cell(star):
#     old_density = star.snia["S current"]
#     new_density = star.snia["S new"]
#     added_mass = star.snia["S ejecta code"]  # per SN
#     added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_adding_ca_to_cell(star):
#     old_density = star.snia["Ca current"]
#     new_density = star.snia["Ca new"]
#     added_mass = star.snia["Ca ejecta code"]  # per SN
#     added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_adding_fe_to_cell(star):
#     old_density = star.snia["Fe current"]
#     new_density = star.snia["Fe new"]
#     added_mass = star.snia["Fe ejecta code"]  # per SN
#     added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snia
# def test_snia_adding_metals_to_cell(star):
#     old_density = star.snia["Ia current"]
#     new_density = star.snia["Ia new"]
#     added_mass = star.snia["metals ejecta code"]  # per SN
#     added_density = added_mass * star.snia["num Ia"] * star.snia["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)


# ==============================================================================
#
# SN II
#
# ==============================================================================
@all_stars_snii
def test_snii_ave_age_range(star):
    """Check that ave_birth is past particle birth, but that it's not past
    15 Myr, the end of particle formation"""
    diff = star.snii["ave_birth"] - star.snii["birth"]
    assert 0 < diff < 15E6


@all_stars_snii
def test_snii_age(star):
    test_age = star.snii["time"] - star.snii["ave_birth"]
    assert star.snii["age"] == test_age
    # The first timestep a star forms, it's ave_birth will be set to half of
    # the current timestep after the current time, so in this first timestep
    # the star can have negative ages.
    assert star.snii["age"] > -0.6 * star.snii["dt"]


@all_stars_snii
def test_snii_dt(star):
    test_dt = star.snii["next"] - star.snii["time"]
    assert star.snii["dt"] == test_dt


@all_stars_snii
def test_snii_vol(star):
    assert star.snii["1/vol"] == star.snii["true 1/vol"]


# @all_stars_snii
# def test_snii_mass_factor_value(star):
#     factor = star.snii["Msol_to_code_mass"]
#
#     assert (1.0 * u.Msun).to(code_mass).value == approx(factor, abs=0, rel=1E-7)
#     assert (1.0 * code_mass).to(u.Msun).value == approx(1.0 / factor, abs=0, rel=1E-7)


@all_stars_snii
def test_snii_time_factor_value(star):
    factor = star.snii["code time"]

    code_time = code_time_func(star.snii["abox[level]"])
    assert (1.0 * code_time).to(u.second).value == approx(factor, abs=0, rel=1E-7)


@all_stars_snii
def test_snii_length_factor_value(star):
    factor = star.snii["code length"]
    code_length = code_length_func(star.snii["abox[level]"])
    assert (1.0 * code_length).to(u.cm).value == approx(factor, abs=0, rel=1E-7)


@all_stars_snii
def test_snii_energy_factor_value(star):
    ergs_1E51 = star.snii["1E51 ergs"]

    code_energy = code_energy_func(star.snii["abox[level]"])

    assert (1E51 * u.erg).to(code_energy).value == approx(ergs_1E51, abs=0, rel=1E-4)


@all_stars_snii
def test_snii_mass_range(star):
    assert 1 < star.snii["stellar mass Msun"] < 1E8


@all_stars_snii
def test_end_ms_mass_conversion(star):
    msun = star.snii["stellar mass Msun"]
    code = star.snii["stellar mass code"]

    assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
    assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)


@all_stars_snii
def test_snii_metallicity_range(star):
    # just a check that we're getting the correct thing
    assert 0 < star.snii["metallicity II"] < 1
    assert 0 < star.snii["metallicity Ia"] < 1
    assert 0 < star.snii["metallicity AGB"] < 1
    assert 0 < star.snii["metallicity"] < 1


@all_stars_snii
def test_end_ms_total_metallicity(star):
    test_total_z = star.snii["metallicity II"] + \
                   star.snii["metallicity Ia"] + \
                   star.snii["metallicity AGB"]
    assert star.snii["metallicity"] == test_total_z


@all_stars_snii
def test_turnoff_mass_now(star):
    true_m_turnoff_now = lifetimes.turnoff_mass(star.snii["age"],
                                                star.snii["metallicity"])
    test_m_turnoff_now = star.snii["m_turnoff_now"]
    # early lifetimes are unreliable, and don't matter as long as they're above
    # our maximum mass, which is 50
    if true_m_turnoff_now > 60:
        assert test_m_turnoff_now > 59
    else:
        assert test_m_turnoff_now == approx(true_m_turnoff_now, abs=0, rel=1E-5)

@all_stars_snii
def test_turnoff_mass_next(star):
    new_age = star.snii["age"] + star.snii["dt"]
    true_m_turnoff_next = lifetimes.turnoff_mass(new_age,
                                                 star.snii["metallicity"])
    test_m_turnoff_next = star.snii["m_turnoff_next"]
    # early lifetimes are unreliable, and don't matter as long as they're above
    # our maximum mass, which is 50
    if true_m_turnoff_next > 60:
        assert test_m_turnoff_next > 59
    else:
        assert test_m_turnoff_next == approx(true_m_turnoff_next, abs=0, rel=1E-5)

@all_stars_snii
def test_unexploded_sn_current(star):
    assert 0 <= star.snii["unexploded_sn current"] < 1.0

@all_stars_snii
def test_unexploded_sn_new(star):
    assert 0 <= star.snii["unexploded_sn new"] < 1.0

@all_stars_snii_inactive
def test_snii_inactive_counter_increment(star):
    m_low  = max(min(star.snii["m_turnoff_next"], 50.0), 8.0)
    m_high = max(min(star.snii["m_turnoff_now"],  50.0), 8.0)
    imf_integral = art_enrich_core.imf_integral_py(m_low, m_high)
    n_stars = imf_integral * star.snii["stellar mass Msun"]
    old_counter = star.snii["unexploded_sn current"]
    new_counter = star.snii["unexploded_sn new"]
    assert new_counter == approx(old_counter + n_stars, abs=1E-5, rel=0)

@all_stars_snii_active
def test_snii_active_counter_increment(star):
    m_low  = max(min(star.snii["m_turnoff_next"], 50.0), 8.0)
    m_high = max(min(star.snii["m_turnoff_now"],  50.0), 8.0)
    imf_integral = art_enrich_core.imf_integral_py(m_low, m_high)
    n_stars = imf_integral * star.snii["stellar mass Msun"]
    old_counter = star.snii["unexploded_sn current"]
    correct_counter = (old_counter + n_stars) % 1.0
    new_counter = star.snii["unexploded_sn new"]
    assert new_counter == approx(correct_counter, abs=1E-5, rel=0)

# Simple test of the ejecta depending on the timing
@all_stars_snii_inactive
def test_snii_simple_inactive(star):
    for elt in ["C", "N", "O", "Mg", "S", "Ca", "Fe", "AGB", "SNIa", "SNII"]:
        current = star.snii["{} current".format(elt)]
        new = star.snii["{} new".format(elt)]
        assert current == new

@all_stars_snii_active
def test_snii_simple_active(star):
    for elt in ["C", "N", "O", "Mg", "S", "Ca", "Fe", "SNII"]:
        current = star.snii["{} current".format(elt)]
        new = star.snii["{} new".format(elt)]
        assert current < new
    assert star.snii["AGB current"] == star.snii["AGB new"]
    assert star.snii["SNIa current"] == star.snii["SNIa new"]

@all_stars_snii_active
def test_snii_active_addition(star):
    for elt in ["C", "N", "O", "Mg", "S", "Ca", "Fe", "SNII"]:
        current = star.snii["{} current".format(elt)]
        added = star.snii["{}_added".format(elt)]
        new_expected = current + added
        new = star.snii["{} new".format(elt)]
        assert new_expected == approx(new, abs=1E-5, rel=1E-5)
    assert star.snii["AGB current"] == star.snii["AGB new"]
    assert star.snii["SNIa current"] == star.snii["SNIa new"]

# ------------------------------------------------------------------------------
# SN II yields
# ------------------------------------------------------------------------------
# These are a bit tricky, since we have to figure out whether we have a
# supernovae or hypernova. I will first write a function to determine how many
# supernovae or hypernovae happened based on the energy injected
def sn_and_hn(code_energy_added, ergs_1E51_code, sn_mass):
    ergs = (code_energy_added / ergs_1E51_code) * 1E51
    # boost it every so slightly to account for floating point errors, since
    # in the upcoming code I"ll subtract from this initial value
    ergs *= 1.0001
    if sn_mass > 20.0:  # we can have HN
        hn_energy_this_mass = hn_energy(sn_mass)
        # hn are always dominant over SN, so the dominant fraction of energy
        # will be from HN, if they exist
        n_hn = ergs // hn_energy_this_mass
        ergs -= n_hn * hn_energy_this_mass
    else:
        n_hn = 0
    # then everything that's left over is SN
    n_sn = round(ergs / 1E51, 0)  # each SN is 1E51 ergs
    return n_sn, n_hn

def hn_energy(sn_mass):
    if sn_mass < 25.0:
        return 10E51
    if sn_mass < 30.0:
        interp = interpolate.interp1d(x=[25, 30], y=[10, 20], kind="linear")
        return interp(sn_mass) * 1E51
    if sn_mass < 40.0:
        interp = interpolate.interp1d(x=[30, 40], y=[20, 30], kind="linear")
        return interp(sn_mass) * 1E51
    else:
        return 30E51

@all_stars_snii_active
def test_snii_number(star):
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(star.snii["energy added"],
                           star.snii["1E51 ergs"],
                           star.snii["m_turnoff_next"])
    # then get the number expected by the IMF integration
    m_low = max(min(star.snii["m_turnoff_next"], 50.0), 8.0)
    m_high = max(min(star.snii["m_turnoff_now"], 50.0), 8.0)
    imf_integral = art_enrich_core.imf_integral_py(m_low, m_high)
    n_stars = imf_integral * star.snii["stellar mass Msun"]
    old_counter = star.snii["unexploded_sn current"]
    expected_n_sn = (old_counter + n_stars) // 1.0

    assert n_sn + n_hn == expected_n_sn

@all_stars_snii_active
def test_snii_yields_c(star):
    m = star.snii["m_turnoff_next"]
    z = star.snii["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(star.snii["energy added"], star.snii["1E51 ergs"], m)

    yields_snii = art_enrich_core.get_yields_raw_sn_ii_py(z, m)[0]
    yields_hnii = art_enrich_core.get_yields_raw_hn_ii_py(z, m)[0]

    total_c = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code density
    total_c_code = (total_c * u.Msun).to(code_mass).value
    c_density = total_c_code * star.snii["1/vol"]
    assert c_density == approx(star.snii["C_added"], abs=0, rel=1E-5)

@all_stars_snii_active
def test_snii_yields_n(star):
    m = star.snii["m_turnoff_next"]
    z = star.snii["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(star.snii["energy added"], star.snii["1E51 ergs"], m)

    yields_snii = art_enrich_core.get_yields_raw_sn_ii_py(z, m)[1]
    yields_hnii = art_enrich_core.get_yields_raw_hn_ii_py(z, m)[1]

    total_n = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code density
    total_n_code = (total_n * u.Msun).to(code_mass).value
    n_density = total_n_code * star.snii["1/vol"]
    assert n_density == approx(star.snii["N_added"], abs=0, rel=1E-5)

@all_stars_snii_active
def test_snii_yields_o(star):
    m = star.snii["m_turnoff_next"]
    z = star.snii["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(star.snii["energy added"], star.snii["1E51 ergs"], m)

    yields_snii = art_enrich_core.get_yields_raw_sn_ii_py(z, m)[2]
    yields_hnii = art_enrich_core.get_yields_raw_hn_ii_py(z, m)[2]

    total_o = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code density
    total_o_code = (total_o * u.Msun).to(code_mass).value
    o_density = total_o_code * star.snii["1/vol"]
    assert o_density == approx(star.snii["O_added"], abs=0, rel=1E-5)

@all_stars_snii_active
def test_snii_yields_mg(star):
    m = star.snii["m_turnoff_next"]
    z = star.snii["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(star.snii["energy added"], star.snii["1E51 ergs"], m)

    yields_snii = art_enrich_core.get_yields_raw_sn_ii_py(z, m)[3]
    yields_hnii = art_enrich_core.get_yields_raw_hn_ii_py(z, m)[3]

    total_mg = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code density
    total_mg_code = (total_mg * u.Msun).to(code_mass).value
    mg_density = total_mg_code * star.snii["1/vol"]
    assert mg_density == approx(star.snii["Mg_added"], abs=0, rel=1E-5)

@all_stars_snii_active
def test_snii_yields_s(star):
    m = star.snii["m_turnoff_next"]
    z = star.snii["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(star.snii["energy added"], star.snii["1E51 ergs"], m)

    yields_snii = art_enrich_core.get_yields_raw_sn_ii_py(z, m)[4]
    yields_hnii = art_enrich_core.get_yields_raw_hn_ii_py(z, m)[4]

    total_s = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code density
    total_s_code = (total_s * u.Msun).to(code_mass).value
    s_density = total_s_code * star.snii["1/vol"]
    assert s_density == approx(star.snii["S_added"], abs=0, rel=1E-5)

@all_stars_snii_active
def test_snii_yields_ca(star):
    m = star.snii["m_turnoff_next"]
    z = star.snii["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(star.snii["energy added"], star.snii["1E51 ergs"], m)

    yields_snii = art_enrich_core.get_yields_raw_sn_ii_py(z, m)[5]
    yields_hnii = art_enrich_core.get_yields_raw_hn_ii_py(z, m)[5]

    total_ca = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code density
    total_ca_code = (total_ca * u.Msun).to(code_mass).value
    ca_density = total_ca_code * star.snii["1/vol"]
    assert ca_density == approx(star.snii["Ca_added"], abs=0, rel=1E-5)

@all_stars_snii_active
def test_snii_yields_fe(star):
    m = star.snii["m_turnoff_next"]
    z = star.snii["metallicity"]
    # get the number of supernovae based on yields
    n_sn, n_hn = sn_and_hn(star.snii["energy added"], star.snii["1E51 ergs"], m)

    yields_snii = art_enrich_core.get_yields_raw_sn_ii_py(z, m)[6]
    yields_hnii = art_enrich_core.get_yields_raw_hn_ii_py(z, m)[6]

    total_fe = yields_snii * n_sn + yields_hnii * n_hn
    # this is in stellar masses, have to convert to code density
    total_fe_code = (total_fe * u.Msun).to(code_mass).value
    fe_density = total_fe_code * star.snii["1/vol"]
    assert fe_density == approx(star.snii["Fe_added"], abs=0, rel=1E-5)
#
# # # Testing the ejected masses. I have tested the C functions separately, so what
# # I'll do here is to check that the values are what's returned by those
# # functions. I will also compare against the ejected masses my Python code
# # gives, but with a larger tolerance, since the interpolation is done a bit
# # differently between those two methods.
# @all_stars_snii
# def test_agb_ejecta_c_c_code(star):
#     true_c = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                star.detail["metallicity"],
#                                                star.detail["stellar mass code"],
#                                                star.detail["dt"])[0]
#     assert star.detail["AGB C ejecta code"] == approx(true_c, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_agb_ejecta_n_c_code(star):
#     true_n = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                star.detail["metallicity"],
#                                                star.detail["stellar mass code"],
#                                                star.detail["dt"])[1]
#     assert star.detail["AGB N ejecta code"] == approx(true_n, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_agb_ejecta_o_c_code(star):
#     true_o = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                star.detail["metallicity"],
#                                                star.detail["stellar mass code"],
#                                                star.detail["dt"])[2]
#     assert star.detail["AGB O ejecta code"] == approx(true_o, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_agb_ejecta_mg_c_code(star):
#     true_mg = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                 star.detail["metallicity"],
#                                                 star.detail["stellar mass code"],
#                                                 star.detail["dt"])[3]
#     assert star.detail["AGB Mg ejecta code"] == approx(true_mg, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_agb_ejecta_s_c_code(star):
#     true_s = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                star.detail["metallicity"],
#                                                star.detail["stellar mass code"],
#                                                star.detail["dt"])[4]
#     assert star.detail["AGB initial S ejecta code"] == approx(true_s, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_agb_ejecta_ca_c_code(star):
#     true_ca = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                 star.detail["metallicity"],
#                                                 star.detail["stellar mass code"],
#                                                 star.detail["dt"])[5]
#     assert star.detail["AGB initial Ca ejecta code"] == approx(true_ca, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_agb_ejecta_fe_c_code(star):
#     true_fe = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                 star.detail["metallicity"],
#                                                 star.detail["stellar mass code"],
#                                                 star.detail["dt"])[6]
#     assert star.detail["AGB initial Fe ejecta code"] == approx(true_fe, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_agb_ejecta_met_c_code(star):
#     true_met = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                  star.detail["metallicity"],
#                                                  star.detail["stellar mass code"],
#                                                  star.detail["dt"])[7]
#     assert star.detail["AGB initial metals ejecta code"] == approx(true_met, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_agb_ejecta_tot_c_code(star):
#     true_ej = c_code.get_ejecta_timestep_agb_py(star.detail["age"],
#                                                  star.detail["metallicity"],
#                                                  star.detail["stellar mass code"],
#                                                  star.detail["dt"])[8]
#     assert star.detail["AGB total ejecta code"] == approx(true_ej, abs=0, rel=1E-6)
#
#
# # @all_stars
# # def test_agb_ejecta_c_msun_py_code(star):
# #     true_c = mass_loss_agb(star.detail["age"], star.detail["dt"],
# #                            star.detail["metallicity"], "C") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["AGB C ejecta Msun"] == approx(true_c, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_agb_ejecta_n_msun_py_code(star):
# #     true_n = mass_loss_agb(star.detail["age"], star.detail["dt"],
# #                            star.detail["metallicity"], "N") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["AGB N ejecta Msun"] == approx(true_n, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_agb_ejecta_o_msun_py_code(star):
# #     true_o = mass_loss_agb(star.detail["age"], star.detail["dt"],
# #                            star.detail["metallicity"], "O") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["AGB O ejecta Msun"] == approx(true_o, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_agb_ejecta_fe_msun_py_code(star):
# #     true_fe = mass_loss_agb(star.detail["age"], star.detail["dt"],
# #                             star.detail["metallicity"], "Fe") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["AGB initial Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_agb_ejecta_met_py_code(star):
# #     true_met = mass_loss_agb(star.detail["age"], star.detail["dt"],
# #                              star.detail["metallicity"], "total_metals") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["AGB initial metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_agb_ejecta_tot_msun_py_code(star):
# #     true_ej = mass_loss_agb(star.detail["age"], star.detail["dt"],
# #                              star.detail["metallicity"], "total") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["AGB total ejecta Msun"] == approx(true_ej, abs=0, rel=1E-1)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_c_c_code(star):
#     true_c = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                 star.detail["metallicity"],
#                                                 star.detail["stellar mass code"],
#                                                 star.detail["dt"])[0]
#     assert star.detail["SNII C ejecta code"] == approx(true_c, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_n_c_code(star):
#     true_n = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                 star.detail["metallicity"],
#                                                 star.detail["stellar mass code"],
#                                                 star.detail["dt"])[1]
#     assert star.detail["SNII N ejecta code"] == approx(true_n, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_o_c_code(star):
#     true_o = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                 star.detail["metallicity"],
#                                                 star.detail["stellar mass code"],
#                                                 star.detail["dt"])[2]
#     assert star.detail["SNII O ejecta code"] == approx(true_o, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_mg_c_code(star):
#     true_mg = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                  star.detail["metallicity"],
#                                                  star.detail["stellar mass code"],
#                                                  star.detail["dt"])[3]
#     assert star.detail["SNII Mg ejecta code"] == approx(true_mg, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_s_c_code(star):
#     true_s = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                 star.detail["metallicity"],
#                                                 star.detail["stellar mass code"],
#                                                 star.detail["dt"])[4]
#     assert star.detail["SNII S ejecta code"] == approx(true_s, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_ca_c_code(star):
#     true_ca = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                  star.detail["metallicity"],
#                                                  star.detail["stellar mass code"],
#                                                  star.detail["dt"])[5]
#     assert star.detail["SNII Ca ejecta code"] == approx(true_ca, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_fe_c_code(star):
#     true_fe = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                  star.detail["metallicity"],
#                                                  star.detail["stellar mass code"],
#                                                  star.detail["dt"])[6]
#     assert star.detail["SNII Fe ejecta code"] == approx(true_fe, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_met_c_code(star):
#     true_met = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                   star.detail["metallicity"],
#                                                   star.detail["stellar mass code"],
#                                                   star.detail["dt"])[7]
#     assert star.detail["SNII metals ejecta code"] == approx(true_met, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_total_c_code(star):
#     true_tot = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                   star.detail["metallicity"],
#                                                   star.detail["stellar mass code"],
#                                                   star.detail["dt"])[8]
#     assert star.detail["SNII total ejecta code"] == approx(true_tot, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_nsn_c_code(star):
#     # In runtime the units of N_SN have be changed, since it's
#     # N_SN / (year * Msun). My wrapper keeps the original units, so by
#     # using by the mass in Msun we should get the same thing we get in
#     # runtime when it does the conversion to code masses.
#     true_n = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                 star.detail["metallicity"],
#                                                 star.detail["stellar mass Msun"],
#                                                 star.detail["dt"])[9]
#     assert star.detail["N_SN"] == approx(true_n, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_energy_ergs_c_code(star):
#     # In runtime the units of N_SN have be changed, since it's
#     # ergs / (year * Msun). My wrapper keeps the original units, so by
#     # using by the mass in Msun we should get the same thing we get in
#     # runtime when it does the conversion to code masses.
#     true_eng = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                   star.detail["metallicity"],
#                                                   star.detail["stellar mass Msun"],
#                                                   star.detail["dt"])[10]
#     assert star.detail["E_0 ergs"] == approx(true_eng, abs=0, rel=1E-6)
#
#
# @all_stars_snii
# def test_sn_ii_ejecta_energy_code_c_code(star):
#     # In runtime the units of N_SN have be changed, since it's
#     # ergs / (year * Msun). My wrapper keeps the original units, so by
#     # using by the mass in Msun we should get the same thing we get in
#     # runtime when it does the conversion to code masses.
#     true_eng = c_code.get_ejecta_timestep_snii_py(star.detail["age"],
#                                                   star.detail["metallicity"],
#                                                   star.detail["stellar mass Msun"],
#                                                   star.detail["dt"])[10]
#     # This energy is in ergs, and the reported value is in code units
#     code_energy = code_energy_func(star.detail["abox[level]"])
#     true_eng_code = (true_eng * u.erg).to(code_energy).value
#     assert star.detail["E_0 code"] == approx(true_eng_code, abs=0, rel=1E-4)
#
#
# # @all_stars
# # def test_sn_ii_ejecta_c_msun_py_code(star):
# #     true_c = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
# #                              star.detail["metallicity"], "C") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["SNII C ejecta Msun"] == approx(true_c, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_sn_ii_ejecta_n_msun_py_code(star):
# #     true_n = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
# #                              star.detail["metallicity"], "N") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["SNII N ejecta Msun"] == approx(true_n, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_sn_ii_ejecta_o_msun_py_code(star):
# #     true_o = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
# #                              star.detail["metallicity"], "O") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["SNII O ejecta Msun"] == approx(true_o, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_sn_ii_ejecta_fe_msun_py_code(star):
# #     true_fe = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
# #                               star.detail["metallicity"], "Fe") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["SNII Fe ejecta Msun"] == approx(true_fe, abs=0, rel=1E-1)
# #
# #
# # @all_stars
# # def test_sn_ii_ejecta_met_py_code(star):
# #     true_met = mass_loss_sn_ii(star.detail["age"], star.detail["dt"],
# #                                star.detail["metallicity"], "total_metals") * star.detail["stellar mass Msun"]
# #
# #     assert star.detail["SNII metals ejecta Msun"] == approx(true_met, abs=0, rel=1E-1)
#
#
# @all_stars_snii
# def test_winds_ejecta_total_c_code(star):
#     true_ej = c_code.get_ejecta_timestep_winds_py(star.detail["age"],
#                                                   star.detail["metallicity"],
#                                                   star.detail["stellar mass code"],
#                                                   star.detail["dt"])[0]
#     assert star.detail["Winds total ejecta code"] == approx(true_ej, abs=0, rel=1E-6)
#
#
# # @all_stars
# # def test_end_ms_code_mass_conversion_ejecta(star):
# #     for source in ["AGB C", "AGB N", "AGB O",
# #                    "SNII C", "SNII N", "SNII O", "SNII Fe", "SNII metals"]:
# #         msun = star.detail["{} ejecta Msun".format(source)]
# #         code = star.detail["{} ejecta code".format(source)]
# #         assert (msun * u.Msun).to(code_mass).value == approx(code, abs=0, rel=1E-7)
# #         assert (code * code_mass).to(u.Msun).value == approx(msun, abs=0, rel=1E-7)
#
#
# # @all_stars
# # def test_agb_fe_mass_conversion(star):
# #     true_fe_msun = star.detail["metallicity Fe"] * star.detail["AGB total ejecta Msun"]
# #     true_fe_code = (true_fe_msun*u.Msun).to(code_mass).value
# #     test_fe_code = star.detail["AGB Fe ejecta code"]
# #     assert true_fe_code == approx(test_fe_code, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_agb_metals_mass_conversion_msun(star):
#     true_metals_msun = star.detail["AGB initial metals ejecta Msun local"]
#     for elt in ["Fe", "Ca", "S"]:
#         initial_msun = star.detail["AGB initial {} ejecta Msun local".format(elt)]
#         true_msun = star.detail["metallicity {}".format(elt)] * star.detail["AGB total ejecta Msun local"]
#         change_msun = initial_msun - true_msun
#         true_metals_msun -= change_msun
#
#         true_code = (true_msun * u.Msun).to(code_mass).value
#         assert true_code == approx(star.detail["AGB {} ejecta code".format(elt)])
#
#     true_metals_code = (true_metals_msun * u.Msun).to(code_mass).value
#     test_metals_code = star.detail["AGB metals ejecta code"]
#     assert true_metals_code == approx(test_metals_code, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_agb_metals_mass_conversion_code(star):
#     true_metals_code = star.detail["AGB initial metals ejecta code"]
#     for elt in ["Fe", "Ca", "S"]:
#         initial_code = star.detail["AGB initial {} ejecta code".format(elt)]
#         true_code = star.detail["metallicity {}".format(elt)] * star.detail["AGB total ejecta code"]
#         change_code = initial_code - true_code
#         true_metals_code -= change_code
#
#         assert true_code == approx(star.detail["AGB {} ejecta code".format(elt)])
#
#     test_metals_code = star.detail["AGB metals ejecta code"]
#     assert true_metals_code == approx(test_metals_code, abs=0, rel=1E-7)
#
# @all_stars_snii
# def test_winds_split_into_metals(star):
#     total_ejecta_code = star.detail["Winds total ejecta code"]
#     assert total_ejecta_code * star.detail["metallicity C"] == \
#            approx(star.detail["Winds C ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity N"] == \
#            approx(star.detail["Winds N ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity O"] == \
#            approx(star.detail["Winds O ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity Mg"] == \
#            approx(star.detail["Winds Mg ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity S"] == \
#            approx(star.detail["Winds S ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity Ca"] == \
#            approx(star.detail["Winds Ca ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity Fe"] == \
#            approx(star.detail["Winds Fe ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity Ia"] == \
#            approx(star.detail["Winds SNIa ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity II"] == \
#            approx(star.detail["Winds SNII ejecta code"])
#     assert total_ejecta_code * star.detail["metallicity AGB"] == \
#            approx(star.detail["Winds AGB ejecta code"])
#
#
# @all_stars_snii
# def test_end_ms_initial_densities_nonzero(star):
#     for elt in ["C", "N", "O", "Mg", "S", "Ca", "Fe", "AGB", "SNIa", "SNII"]:
#         assert star.detail["{} current".format(elt)] > 0
#
#
# @all_stars_snii
# def test_end_ms_initial_densities_sum(star):
#     elts_tot = sum([star.detail["{} current".format(elt)]
#                    for elt in ["C", "N", "O", "Mg", "S", "Ca", "Fe"]])
#     sources_tot = sum([star.detail["{} current".format(source)]
#                    for source in ["SNIa", "SNII", "AGB"]])
#
#     # not all elements are individually tracked, so the sum of elements must
#     # be less than the total
#     assert sources_tot > elts_tot
#
#
#
# @all_stars_snii
# def test_end_ms_adding_c_to_cell(star):
#     old_density = star.detail["C current"]
#     new_density = star.detail["C new"]
#     added_mass = star.detail["AGB C ejecta code"] + \
#                  star.detail["SNII C ejecta code"] + \
#                  star.detail["Winds C ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_end_ms_adding_n_to_cell(star):
#     old_density = star.detail["N current"]
#     new_density = star.detail["N new"]
#     added_mass = star.detail["AGB N ejecta code"] + \
#                  star.detail["SNII N ejecta code"] + \
#                  star.detail["Winds N ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_end_ms_adding_o_to_cell(star):
#     old_density = star.detail["O current"]
#     new_density = star.detail["O new"]
#     added_mass = star.detail["AGB O ejecta code"] + \
#                  star.detail["SNII O ejecta code"] + \
#                  star.detail["Winds O ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_end_ms_adding_o_to_cell(star):
#     old_density = star.detail["Mg current"]
#     new_density = star.detail["Mg new"]
#     added_mass = star.detail["AGB Mg ejecta code"] + \
#                  star.detail["SNII Mg ejecta code"] + \
#                  star.detail["Winds Mg ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_end_ms_adding_o_to_cell(star):
#     old_density = star.detail["S current"]
#     new_density = star.detail["S new"]
#     added_mass = star.detail["AGB S ejecta code"] + \
#                  star.detail["SNII S ejecta code"] + \
#                  star.detail["Winds S ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_end_ms_adding_o_to_cell(star):
#     old_density = star.detail["Ca current"]
#     new_density = star.detail["Ca new"]
#     added_mass = star.detail["AGB Ca ejecta code"] + \
#                  star.detail["SNII Ca ejecta code"] + \
#                  star.detail["Winds Ca ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_end_ms_adding_fe_to_cell(star):
#     old_density = star.detail["Fe current"]
#     new_density = star.detail["Fe new"]
#     added_mass = star.detail["AGB Fe ejecta code"] + \
#                  star.detail["SNII Fe ejecta code"] + \
#                  star.detail["Winds Fe ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_agb_adding_met_to_cell(star):
#     old_density = star.detail["AGB current"]
#     new_density = star.detail["AGB new"]
#     added_mass = star.detail["AGB metals ejecta code"] + \
#                  star.detail["Winds AGB ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_sn_ia_adding_met_to_cell(star):
#     old_density = star.detail["SNIa current"]
#     new_density = star.detail["SNIa new"]
#     added_mass = star.detail["Winds SNIa ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
#
#
# @all_stars_snii
# def test_sn_ii_adding_met_to_cell(star):
#     old_density = star.detail["SNII current"]
#     new_density = star.detail["SNII new"]
#     added_mass = star.detail["SNII metals ejecta code"] + \
#                  star.detail["Winds SNII ejecta code"]
#     added_density = added_mass * star.detail["1/vol"]
#     assert old_density + added_density == approx(new_density, abs=0, rel=1E-7)
