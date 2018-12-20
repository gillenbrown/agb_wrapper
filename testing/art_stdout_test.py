import sys, os

import pytest
import random

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

@all_stars
def test_snia_age(star):
    test_age = star.snia["time"] - star.snia["birth"]
    assert star.snia["age"] == test_age

@all_stars
def test_snia_dt(star):
    test_dt = star.snia["next"] - star.snia["time"]
    assert star.snia["dt"] == test_dt

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
def test_snia_rate(star):
    true_rate = sn_ia_check.sn_dtd(star.snia["age"], star.snia["metallicity"])
    assert true_rate == star.snia["Ia rate"]