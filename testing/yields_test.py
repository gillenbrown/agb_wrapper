import pytest
from _agb.lib import get_ejecta_rate, read_in_check

# We need to initialize the table by doing the read in.
read_in_check()

n_fields = 5
z_values = [0.0001, 0.001, 0.006, 0.01, 0.02]

testing_ages = [0, 30E6, 100E6, 1E9, 13.9E9, 14E9]
testing_zs = z_values + [0, 0.00001, 0.0005, 0.005, 0.009, 0.015, 0.03]

# define some exact values to check
exact_values = {z:dict() for z in z_values}
arb_age_1 = 6.55276e+07
arb_age_2 = 4.43123e+09

exact_values[0.0001][arb_age_1] = [5.09667e-14, 4.09653e-13, 3.88593e-14,
                                   1.21708e-15, 5.17646e-13]
exact_values[0.0001][arb_age_2] = [1.49565e-13, 1.33345e-15, 3.1521e-14,
                                   1.10837e-17, 1.82734e-13]
exact_values[0.0001][14E9] = [3.76005e-14, 3.35229e-16, 7.92434e-15,
                              2.78644e-18, 4.59393e-14]
exact_values[0.02][arb_age_1] = [3.17607e-13, 6.20672e-12, 5.55177e-12,
                                 1.11052e-12, 1.6666e-11]
exact_values[0.02][arb_age_2] = [3.68107e-14, 1.80335e-14, 7.75804e-14,
                                 1.11294e-14, 1.786e-13]
exact_values[0.02][14E9] = [6.14806e-15, 3.05135e-15, 1.92865e-14,
                            2.87525e-15, 4.01243e-14]

# then store the ages and metallicities we used
exact_value_ages = [arb_age_1, arb_age_2, 14E9]
exact_value_zs = [0.0001, 0.02]
# ------------------------------------------------------------------------------
#
# Error checking and edge cases
#
# ------------------------------------------------------------------------------
# For these first ones, we are checking how it responds to one bad parameter
# value. We assume a correct value for the other parameter, unless the value
# of this other parameter doesn't matter at all. Combinations
# of bad parameter values will be checked later.
@pytest.mark.parametrize("age", [-1, -1E6, -1E12])
@pytest.mark.parametrize("z", testing_zs + [-0.005, 0.5])
def test_negative_age(age, z):
    rates = get_ejecta_rate(age, z)
    for idx in range(n_fields):
        assert rates[idx] == 0

@pytest.mark.parametrize("age", [0, 1E5, 30E6])
@pytest.mark.parametrize("z", testing_zs + [-0.005, 0.5])
def test_early_age(age, z):
    """Rates should be zero at early times before AGB are active."""
    rates = get_ejecta_rate(age, z)
    for idx in range(n_fields):
        assert rates[idx] == 0

# If the above tests pass, we are confident that the early time yields
# are working properly at all metallicities. Now we can test at late times.
# We'll first check at the known metallicity values
@pytest.mark.parametrize("age", [14E9, 14.0001E9, 20E9])
@pytest.mark.parametrize("z,answer",
                         [(0.0001, exact_values[0.0001][14E9]),
                          (0.02,   exact_values[0.02][14E9])])
def test_very_late_age(age, z, answer):
    """Rates should be equal to the last timestep"""
    rates = get_ejecta_rate(age, z)
    for idx in range(n_fields):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("z", [0, 1E-8, 1E-4, 9.99E-5])
@pytest.mark.parametrize("age,answer",
                         [(arb_age_1, exact_values[0.0001][arb_age_1]),
                          (arb_age_2, exact_values[0.0001][arb_age_2])])
def test_low_metallicity(age, z, answer):
    """Metallicity less than the minimum should use the yields for minimum z"""
    rates = get_ejecta_rate(age, z)
    for idx in range(n_fields):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("z", [0.02, 0.05, 0.5])
@pytest.mark.parametrize("age,answer",
                         [(arb_age_1, exact_values[0.02][arb_age_1]),
                          (arb_age_2, exact_values[0.02][arb_age_2])])
def test_high_metallicity(age, z, answer):
    """Metallicity more than the maximum should use the yields for maximum z"""
    rates = get_ejecta_rate(age, z)
    for idx in range(n_fields):
        assert rates[idx] == answer[idx]

# Now we are confident that it works at late times with known metallicity values
# and at arbitrary times with high and low metallicity. Now we need to test
# the cases when both of these might be true.
@pytest.mark.parametrize("age", [14E9, 14.0001E9, 20E9])
@pytest.mark.parametrize("z", [0, 0.00001, 0.0001])
def test_low_z_late_time(age, z):
    """No matter what, if we are past the last time and at low metallicity,
    we should have the same rate as the low z late time output."""
    answer = exact_values[0.0001][14E9]
    rates = get_ejecta_rate(age, z)
    for idx in range(n_fields):
        assert rates[idx] == answer[idx]

@pytest.mark.parametrize("age", [14E9, 14.0001E9, 20E9])
@pytest.mark.parametrize("z", [0.02, 0.03, 0.5])
def test_high_z_late_time(age, z):
    """No matter what, if we are past the last time and at low metallicity,
    we should have the same rate as the low z late time output."""
    answer = exact_values[0.02][14E9]
    rates = get_ejecta_rate(age, z)
    for idx in range(n_fields):
        assert rates[idx] == answer[idx]

# ------------------------------------------------------------------------------
#
# Checking normal cases
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("age", exact_value_ages)
@pytest.mark.parametrize("z", exact_value_zs)
def test_double_alignment(age, z):
    """
    Check the yields when they line up exactly with one of the time and
    metallicity steps.
    """
    answer = exact_values[z][age]
    rates = get_ejecta_rate(age, z)
    for idx in range(n_fields):
        assert rates[idx] == answer[idx]
#
# def test_age_alignment():
#
#
# def test_z_alignment():
#
#
# def test_nonalignment():