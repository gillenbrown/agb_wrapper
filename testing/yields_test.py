import sys, os
sys.path.append(os.path.abspath("../"))

import pytest
import numpy as np

from art_enrich import lib as tab

# We need to initialize the table by doing the read in.
tab.init_detailed_enrichment()

r_tol = 1E-8
a_tol = 0

n_fields_agb = 6
n_fields_sn_ii = 5
n_fields_winds = 1
z_values_agb = [0.0001, 0.001, 0.006, 0.01, 0.02]
z_values_sn_ii = [0, 0.001, 0.004, 0.02]

# ------------------------------------------------------------------------------
#
# metallicity index checking
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("z, true_idx_0, true_idx_1",
                         [[-0.5,    0, 0],
                          [0,       0, 0],
                          [0.00001, 0, 0],
                          [0.0001,  0, 0],
                          [0.0005,  0, 1],
                          [0.001,   1, 1],
                          [0.005,   1, 2],
                          [0.006,   2, 2],
                          [0.009,   2, 3],
                          [0.01,    3, 3],
                          [0.015,   3, 4],
                          [0.02,    4, 4],
                          [0.03,    4, 4],
                          [1.4,     4, 4]])
def test_z_idxs_agb(z, true_idx_0, true_idx_1):
    z_idx = tab.find_z_bound_idxs_agb_py(z)
    assert z_idx[0] == true_idx_0
    assert z_idx[1] == true_idx_1

@pytest.mark.parametrize("func", [tab.find_z_bound_idxs_winds_py,
                                  tab.find_z_bound_idxs_sn_ii_py])
@pytest.mark.parametrize("z, true_idx_0, true_idx_1",
                         [[-0.5,   0, 0],
                          [0,      0, 0],
                          [0.0005, 0, 1],
                          [0.001,  1, 1],
                          [0.003,  1, 2],
                          [0.004,  2, 2],
                          [0.01,   2, 3],
                          [0.02,   3, 3],
                          [0.03,   3, 3],
                          [1.4,    3, 3]])
def test_z_idxs_winds_sn(func, z, true_idx_0, true_idx_1):
    z_idx = func(z)
    assert z_idx[0] == true_idx_0
    assert z_idx[1] == true_idx_1

# ------------------------------------------------------------------------------
#
# Age index checking
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("age, true_idx_0, true_idx_1",
                         [[-0.5,  0, 0],
                          [0,     0, 0],
                          [1,     0, 1],
                          [3E7,   1, 1],
                          [3.2E7, 11, 12],
                          [1E20,  1000, 1000]])
def test_age_idx_known_agb(age, true_idx_0, true_idx_1):
    age_idxs = tab.find_age_bound_idxs_agb_py(age)
    assert age_idxs[0] == true_idx_0
    assert age_idxs[1] == true_idx_1

@pytest.mark.parametrize("age, true_idx_0, true_idx_1",
                         [[-0.5,  0, 0],
                          [0,     0, 0],
                          [1,     0, 1],
                          [1E6,   1, 1],
                          [1.2E6, 47, 48],
                          [1E20,  1000, 1000]])
def test_age_idx_known_sn_ii(age, true_idx_0, true_idx_1):
    age_idxs = tab.find_age_bound_idxs_sn_ii_py(age)
    assert age_idxs[0] == true_idx_0
    assert age_idxs[1] == true_idx_1

@pytest.mark.parametrize("age, true_idx_0, true_idx_1",
                         [[-0.5,  0, 0],
                          [0,     0, 0],
                          [1,     0, 1],
                          [1E6,   1, 1],
                          [1.2E6, 47, 48],
                          [1E20,  1000, 1000]])
def test_age_idx_known_sn_ii(age, true_idx_0, true_idx_1):
    age_idxs = tab.find_age_bound_idxs_winds_py(age)
    assert age_idxs[0] == true_idx_0
    assert age_idxs[1] == true_idx_1

@pytest.mark.parametrize("idx_func, age_func",
                         [[tab.find_age_bound_idxs_agb_py, tab.get_ages_agb],
                          [tab.find_age_bound_idxs_sn_ii_py, tab.get_ages_sn_ii],
                          [tab.find_age_bound_idxs_winds_py, tab.get_ages_winds]])
def test_age_idx_bounds(idx_func, age_func):
    """Test that the boundaries returned actually span the input value"""
    for age in np.random.uniform(0, 50E6, 100):
        age_idxs = idx_func(age)
        age_0 = age_func(age_idxs[0])
        age_1 = age_func(age_idxs[1])
        assert age_0 <= age <= age_1


# ------------------------------------------------------------------------------
#
# Rate checking
#
# ------------------------------------------------------------------------------
# define some exact values to check
exact_rates = {"AGB":   {z:dict() for z in z_values_agb},
               "SN":    {z:dict() for z in z_values_sn_ii},
               "winds": {z:dict() for z in z_values_sn_ii}}

a1_agb = 4.87362e+07
a2_agb = 5.17732e+09
a3_agb = 1.5e+10

a1_big = 5.36522e+06
a2_big = 1.99992e+07
a3_big = 5e+07

exact_rates["AGB"][0.0001][a1_agb] = [7.09127e-14, 5.69972e-13, 5.4067e-14,
                                      1.69338e-15, 7.20228e-13, 1.17717e-09]
exact_rates["AGB"][0.0001][a2_agb] = [1.25127e-13, 1.11558e-15, 2.63708e-14,
                                      9.27275e-18, 1.52877e-13, 6.7587e-12]
exact_rates["AGB"][0.0001][a3_agb] = [4.45291e-14, 3.97002e-16, 9.38457e-15,
                                      3.2999e-18, 5.44045e-14, 2.40522e-12]
exact_rates["AGB"][0.02][a1_agb] = [4.72688e-13, 9.11285e-12, 8.09145e-12,
                                    1.63682e-12, 2.44475e-11, 1.13936e-09]
exact_rates["AGB"][0.02][a2_agb] = [2.04009e-14, 1.01252e-14, 6.39979e-14,
                                    9.54084e-15, 1.33143e-13, 6.63505e-12]
exact_rates["AGB"][0.02][a3_agb] = [6.24391e-15, 3.09892e-15, 1.95872e-14,
                                    2.92007e-15, 4.07498e-14, 2.03073e-12]

exact_rates["SN"][0][a1_big] = [1.99356e-10, 1.33188e-14, 2.66105e-09,
                                7.26859e-11, 3.72347e-09]
exact_rates["SN"][0][a2_big] = [2.09617e-11, 5.17694e-13, 1.27298e-10,
                                2.02901e-11, 2.32181e-10]
exact_rates["SN"][0][a3_big] = [0, 0, 0, 0, 0]
exact_rates["SN"][0.02][a1_big] = [1.12158e-10, 5.28931e-11, 1.54796e-09,
                                   5.64704e-11, 2.53759e-09]
exact_rates["SN"][0.02][a2_big] = [2.86939e-11, 1.27637e-11, 5.90797e-11,
                                   2.3237e-11, 1.78902e-10]
exact_rates["SN"][0.02][a3_big] = [0, 0, 0, 0, 0]

exact_rates["winds"][0][a1_big] = [0]
exact_rates["winds"][0][a2_big] = [0]
exact_rates["winds"][0][a3_big] = [0]
exact_rates["winds"][0.02][a1_big] = [1.31147e-09]
exact_rates["winds"][0.02][a2_big] = [3.33282e-11]
exact_rates["winds"][0.02][a3_big] = [0]

# then store the ages and metallicities we used
exact_value_ages_agb = [a1_agb, a2_agb, a3_agb]
exact_value_ages_big = [a1_big, a2_big, a3_big]
exact_value_zs_agb = [0.0001, 0.02]
exact_value_zs_big = [0, 0.02]

# then some values to be used for error testing
testing_ages_agb = [0, 30E6, 100E6, 1E9, 13.9E9, 14E9]
testing_zs = z_values_agb + [0, 0.00001, 0.0005, 0.005, 0.009, 0.015, 0.03,
                             -0.005, 0.5, 1.5]
# ------------------------------------------------------------------------------
#
# Error checking and edge cases
#
# ------------------------------------------------------------------------------
# For these first ones, we are checking how it responds to bad parameter values
# Some of these will be independent bad paramter checks, but others will
# involve combinations of bad parameters. One parameter should govern in these
# combinations so that results re predictable.
# ------------------------------------------------------------------------------
#
# Negative Age
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("age", [-1, -1E6, -1E12])
@pytest.mark.parametrize("z", testing_zs)
@pytest.mark.parametrize("func, n_fields",
                         [[tab.get_ejecta_rate_agb_py, n_fields_agb],
                          [tab.get_ejecta_rate_winds_py, n_fields_winds],
                          [tab.get_ejecta_rate_sn_ii_py, n_fields_sn_ii]])
def test_negative_age_rate(func, n_fields, age, z):
    rates = func(age, z)
    for idx in range(n_fields):
        assert rates[idx] == 0

# ------------------------------------------------------------------------------
#
# Early age
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("func, min_age",
                         [[tab.get_ejecta_rate_agb_py, 30E6],
                          [tab.get_ejecta_rate_sn_ii_py, 3E6]])
@pytest.mark.parametrize("z", testing_zs)
def test_early_age(func, min_age, z):
    """Rates should be zero at early times before AGB are active.
    This checks all metallicities"""
    for age in np.random.uniform(0, min_age, 100):
        rates = func(age, z)
        for idx in range(n_fields_agb):
            assert rates[idx] == 0

# ------------------------------------------------------------------------------
#
# Late Times
#
# ------------------------------------------------------------------------------
# If the above tests pass, we are confident that the early time yields
# are working properly at all metallicities. Now we can test at late times.
# We'll first check at the known metallicity values
@pytest.mark.parametrize("age", [14E9, 14.0001E9, 20E9])
@pytest.mark.parametrize("z,answer",
                         [(0.0001, exact_rates["AGB"][0.0001][a3_agb]),
                          (0.02, exact_rates["AGB"][0.02][a3_agb])])
def test_very_late_age_agb(age, z, answer):
    """Rates should be equal to the last timestep"""
    rates = tab.get_ejecta_rate_agb_py(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("age", [50E6, 51E6, 20E9])
@pytest.mark.parametrize("z", exact_value_zs_big)
def test_very_late_age_sn_ii(age, z):
    rates = tab.get_ejecta_rate_sn_ii_py(age, z)
    for idx in range(n_fields_sn_ii):
        assert rates[idx] == 0

@pytest.mark.parametrize("age", [50E6, 51E6, 20E9])
@pytest.mark.parametrize("z", exact_value_zs_big)
def test_very_late_age_winds(age, z):
    rates = tab.get_ejecta_rate_winds_py(age, z)
    for idx in range(n_fields_winds):
        assert rates[idx] == 0


# ------------------------------------------------------------------------------
#
# Low Metallicities
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("z", [-0.5, 0, 1E-8, 1E-4, 9.99E-5])
@pytest.mark.parametrize("age,answer",
                         [(a1_agb, exact_rates["AGB"][0.0001][a1_agb]),
                          (a2_agb, exact_rates["AGB"][0.0001][a2_agb])])
def test_low_metallicity_agb(age, z, answer):
    """Metallicity less than the minimum should use the yields for minimum z"""
    rates = tab.get_ejecta_rate_agb_py(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("z", [-0.5, -0.1, -1E-5, 0])
@pytest.mark.parametrize("age,answer",
                         [(a1_big, exact_rates["SN"][0][a1_big]),
                          (a2_big, exact_rates["SN"][0][a2_big])])
def test_low_metallicity_sn(age, z, answer):
    """Metallicity less than the minimum should use the yields for minimum z"""
    rates = tab.get_ejecta_rate_sn_ii_py(age, z)
    for idx in range(n_fields_sn_ii):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("z", [-0.5, -0.1, -1E-5, 0])
@pytest.mark.parametrize("age,answer",
                         [(a1_big, exact_rates["winds"][0][a1_big]),
                          (a2_big, exact_rates["winds"][0][a2_big])])
def test_low_metallicity_winds(age, z, answer):
    """Metallicity less than the minimum should use the yields for minimum z"""
    rates = tab.get_ejecta_rate_winds_py(age, z)
    for idx in range(n_fields_winds):
        assert rates[idx] == answer[idx]

# ------------------------------------------------------------------------------
#
# High Metallicities
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("z", [0.02, 0.05, 0.5])
@pytest.mark.parametrize("age,answer",
                         [(a1_agb, exact_rates["AGB"][0.02][a1_agb]),
                          (a2_agb, exact_rates["AGB"][0.02][a2_agb])])
def test_high_metallicity_agb(age, z, answer):
    """Metallicity more than the maximum should use the yields for maximum z"""
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == answer[idx]


@pytest.mark.parametrize("z", [0.02, 0.05, 0.5])
@pytest.mark.parametrize("age,answer",
                         [(a1_big, exact_rates["SN"][0.02][a1_big]),
                          (a2_big, exact_rates["SN"][0.02][a2_big])])
def test_high_metallicity_sn(age, z, answer):
    """Metallicity more than the maximum should use the yields for maximum z"""
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_sn_ii):
        assert rates[idx] == answer[idx]


@pytest.mark.parametrize("z", [0.02, 0.05, 0.5])
@pytest.mark.parametrize("age,answer",
                         [(a1_big, exact_rates["winds"][0.02][a1_big]),
                          (a2_big, exact_rates["winds"][0.02][a2_big])])
def test_high_metallicity_winds(age, z, answer):
    """Metallicity more than the maximum should use the yields for maximum z"""
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_winds):
        assert rates[idx] == answer[idx]


# ------------------------------------------------------------------------------
#
# Late times with low metallicities
#
# ------------------------------------------------------------------------------
# Now we are confident that it works at late times with known metallicity values
# and at arbitrary times with high and low metallicity. Now we need to test
# the cases when both of these might be true.
@pytest.mark.parametrize("age", [14E9, 14.0001E9, 20E9])
@pytest.mark.parametrize("z", [0, 0.00001, 0.0001])
def test_low_z_late_time_agb(age, z):
    """No matter what, if we are past the last time and at low metallicity,
    we should have the same rate as the low z late time output."""
    answer = exact_rates["AGB"][0.0001][a3_agb]
    rates = tab.get_ejecta_rate_agb_py(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("age", [50E6, 50.0001E6, 20E9])
@pytest.mark.parametrize("z", [0, -0.0001, -5])
def test_low_z_late_time_sn(age, z):
    """No matter what, if we are past the last time and at low metallicity,
    we should have the same rate as the low z late time output."""
    answer = exact_rates["SN"][0][a3_big]
    rates = tab.get_ejecta_rate_sn_ii_py(age, z)
    for idx in range(n_fields_sn_ii):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("age", [50E6, 50.0001E6, 20E9])
@pytest.mark.parametrize("z", [0, -0.0001, -5])
def test_low_z_late_time_agb(age, z):
    """No matter what, if we are past the last time and at low metallicity,
    we should have the same rate as the low z late time output."""
    answer = exact_rates["winds"][0][a3_big]
    rates = tab.get_ejecta_rate_winds_py(age, z)
    for idx in range(n_fields_winds):
        assert rates[idx] == answer[idx]


# ------------------------------------------------------------------------------
#
# Late times with high metallicities
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("age", [14E9, 14.0001E9, 20E9])
@pytest.mark.parametrize("z", [0.02, 0.03, 0.5])
def test_high_z_late_time_agb(age, z):
    """No matter what, if we are past the last time and at low metallicity,
    we should have the same rate as the low z late time output."""
    answer = exact_rates["AGB"][0.02][a3_agb]
    rates = tab.get_ejecta_rate_agb_py(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == answer[idx]


@pytest.mark.parametrize("age", [50E6, 50.0001E6, 20E9])
@pytest.mark.parametrize("z", [0.02, 0.03, 0.5])
def test_high_z_late_time_sn(age, z):
    """No matter what, if we are past the last time and at low metallicity,
    we should have the same rate as the low z late time output."""
    answer = exact_rates["SN"][0.02][a3_big]
    rates = tab.get_ejecta_rate_sn_ii_py(age, z)
    for idx in range(n_fields_sn_ii):
        assert rates[idx] == answer[idx]


@pytest.mark.parametrize("age", [50E6, 50.0001E6, 20E9])
@pytest.mark.parametrize("z", [0.02, 0.03, 0.5])
def test_high_z_late_time_agb(age, z):
    """No matter what, if we are past the last time and at low metallicity,
    we should have the same rate as the low z late time output."""
    answer = exact_rates["winds"][0.02][a3_big]
    rates = tab.get_ejecta_rate_winds_py(age, z)
    for idx in range(n_fields_winds):
        assert rates[idx] == answer[idx]

# ------------------------------------------------------------------------------
#
# Checking cases in normal range
#
# ------------------------------------------------------------------------------
# We have exhausted the parameter space of when age or metallicity is outside
# the range set by the models. We can then test the times when the metallicity
# and age is inside the range of the models. We first check cases when the
# age and metallicity exactly line up with one of the tabulated points
@pytest.mark.parametrize("age", exact_value_ages_agb)
@pytest.mark.parametrize("z", exact_value_zs_agb)
def test_double_alignment_agb(age, z):
    """
    Check the yields when they line up exactly with one of the time and
    metallicity steps.
    """
    answer = exact_rates["AGB"][z][age]
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("age", exact_value_ages_big)
@pytest.mark.parametrize("z", exact_value_zs_big)
def test_double_alignment_sn(age, z):
    """
    Check the yields when they line up exactly with one of the time and
    metallicity steps.
    """
    answer = exact_rates["SN"][z][age]
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_sn_ii):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("age", exact_value_ages_big)
@pytest.mark.parametrize("z", exact_value_zs_big)
def test_double_alignment_sinds(age, z):
    """
    Check the yields when they line up exactly with one of the time and
    metallicity steps.
    """
    answer = exact_rates["winds"][z][age]
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_winds):
        assert rates[idx] == answer[idx]

# ------------------------------------------------------------------------------
#
# Checking cases where age aligns but not metallicity
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("z,answer",
                         [(0.00055, [3.09117e-13, 1.597165e-12, 7.01507e-13,
                                     9.038095e-15, 2.715225e-12]),
                          (0.005, [2.394706e-13, 2.896638e-12, 2.505184e-12,
                                   7.985132e-14, 6.4357940e-12]),
                          (0.007, [2.762215e-13, 3.8769875e-12, 3.0566825e-12,
                                   2.67147225e-13, 8.811905e-12]),
                          (0.015, [4.177815e-13, 7.12888e-12, 5.439345e-12,
                                   1.1479675e-12, 1.78976e-11])])
def test_age_alignment(z,answer):
    """
    Test when the age matches one of the ages, but the metallicity does not.
    These are calculated by hand.
    """
    age = 4.87735e+07
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == pytest.approx(answer[idx], rel=r_tol, abs=a_tol)

# ------------------------------------------------------------------------------
#
# Checking cases where metallicity aligns but not age
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("age,answer",
                         [(1E8, [1.11721345e-13, 9.4988731e-13, 1.82078847e-12,
                                 4.4600222442673606e-14, 3.26713668076109e-12]),
                          (1.38116e9, [2.30691e-13, 1.216195e-14, 2.287855e-13,
                                       2.666265e-15, 5.0098e-13])])
def test_z_alignment(age, answer):
    """
    Test when the age matches one of the metallicities, but the age does not.
    These are calculated by hand.
    """
    z = 0.006
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == pytest.approx(answer[idx], rel=r_tol, abs=a_tol)

# ------------------------------------------------------------------------------
#
# Checking cases where neither age or metallicity aligns
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("age,z,answer",
                         [(1.18941e10, 0.00055, [2.46099725e-14, 2.53411425e-16,
                                                 6.2606475e-15, 1.669699250e-17,
                                                 3.13173275e-14]),
                          (1E9, 0.017, [5.934036074380165e-13,
                                        9.672938628099173e-14,
                                        5.528954140495868e-13,
                                        5.3083009256198344e-14,
                                        1.4929106446280992e-12])])
def test_nonalignment(age, z, answer):
    """Test when neither age or metallicity are aligned, as will typically
    be the case. I only do a few tests here since it's a lot of work to
    calculate by hand."""
    rates = tab.get_ejecta_rate(age, z)
    for idx in range(n_fields_agb):
        assert rates[idx] == pytest.approx(answer[idx], rel=r_tol, abs=a_tol)
