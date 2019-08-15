import sys, os
sys.path.append(os.path.abspath("../"))
import itertools

import pytest
import numpy as np
from scipy import interpolate

import betterplotlib as bpl
bpl.presentation_style()

from art_enrich import lib as tab

# We need to initialize the table by doing the read in.
tab.detailed_enrichment_init()


# convenience function
def is_between(x, a, b):
    if a <= x:
        return a <= x <= b
    else:
        return b <= x <= a

r_tol = 1E-8
a_tol = 0

n_elements = 8
n_returned_agb = 6
n_returned_sn_ii = 9
n_returned_hn_ii = 9

ejecta_table = dict()

# Just read the table, it will make things easier than copying many values
# by hand
for source in ["AGB", "SN", "HN"]:
    if source == "AGB":
        file_loc = "/Users/gillenb/code/art_cluster/src/sf/models/detail_enrich_agb_ejecta.txt"
    elif source == "SN":
        file_loc = "/Users/gillenb/code/art_cluster/src/sf/models/detail_enrich_sn_ii_ejecta.txt"
    elif source == "HN":
        file_loc = "/Users/gillenb/code/art_cluster/src/sf/models/detail_enrich_hn_ii_ejecta.txt"
    else:
        raise ValueError("Wrong name!")

    if source not in ejecta_table:
        ejecta_table[source] = dict()
    with open(file_loc, "r") as table:
        for row in table:
            if row.startswith("#"):
                continue

            split_line = [float(item) for item in row.split()]
            m = split_line[0]
            z = split_line[1]
            ejecta = split_line[2:]

            if z not in ejecta_table[source]:
                ejecta_table[source][z] = dict()

            ejecta_table[source][z][m] = ejecta

def test_ejecta_table():
    # This checks that I read in the table correctly, and reuses the many rows
    # I copy-pasted before just reading the table directly
    assert ejecta_table["AGB"][0.0001][1.0] == pytest.approx([0.0087079, 7.76358e-05, 0.0018352, 1.00483e-06, 0.0106379, 0.470354])
    assert ejecta_table["AGB"][0.0001][6.0] == pytest.approx([0.00029367, 0.00236042, 0.000223907, 1.9373e-05, 0.00296963, 4.87499])
    assert ejecta_table["AGB"][0.0001][7.0] == pytest.approx([0.0017969, 0.00801823, 0.00324965, 4.9072e-05, 0.0132547, 5.72732])
    assert ejecta_table["AGB"][0.02][1.0] == pytest.approx([0.0013525, 0.000671261, 0.00424281, 0.00033083, 0.00797528, 0.439878])
    assert ejecta_table["AGB"][0.02][6.0] == pytest.approx([0.0019763, 0.0411218, 0.0379833, 0.0041146, 0.101123, 5.03487])
    assert ejecta_table["AGB"][0.02][7.0] == pytest.approx([0.002462, 0.0474643, 0.0421444, 0.0049799, 0.115856, 5.93435])

    assert ejecta_table["SN"][0][13] == pytest.approx([0.0741001, 0.00183006, 0.450002, 0.0864277, 0.0240688, 0.00293784, 0.071726, 0.820768, 11.43])
    assert ejecta_table["SN"][0][20] == pytest.approx([0.211, 5.42113e-05, 2.11, 0.150354, 0.053788, 0.00624737, 0.072287, 3.63389, 18.34])
    assert ejecta_table["SN"][0][40] == pytest.approx([0.429, 1.218e-06, 8.38, 0.478554, 0.3754, 0.0373304, 0.080001, 11.1967, 37.11])
    assert ejecta_table["SN"][0.02][13] == pytest.approx([0.108, 0.0480409, 0.222368, 0.02994, 0.0391454, 0.00498225, 0.087461, 0.673363, 11.13])
    assert ejecta_table["SN"][0.02][20] == pytest.approx([0.24645, 0.072124, 1.05617, 0.09487, 0.0300352, 0.00382078, 0.093756, 2.11036, 16.81])
    assert ejecta_table["SN"][0.02][40] == pytest.approx([0.596431, 0.0581057, 7.34327, 0.4562, 0.112995, 0.0157017, 0.089675, 11.3629, 19.62])

    assert ejecta_table["HN"][0][20] == pytest.approx([0.19, 5.43295e-05, 2.03, 0.165317, 0.043028, 0.00489967, 0.084898, 3.38732, 18.12])
    assert ejecta_table["HN"][0][30] == pytest.approx([0.316, 4.202e-05, 3.92, 0.217228, 0.085472, 0.00840379, 0.16384, 5.50342, 26.73])
    assert ejecta_table["HN"][0][40] == pytest.approx([0.372, 4.044e-06, 6.32, 0.337702, 0.261935, 0.0287948, 0.26354, 8.63498, 34.47])
    assert ejecta_table["HN"][0.02][20] == pytest.approx([0.21045, 0.072153, 0.984929, 0.08752, 0.051811, 0.0050832, 0.041134, 1.89223, 16.59])
    assert ejecta_table["HN"][0.02][30] == pytest.approx([0.18092, 0.102015, 2.74447, 0.2135, 0.123579, 0.00955079, 0.11391, 4.41294, 21.53])
    assert ejecta_table["HN"][0.02][40] == pytest.approx([0.490431, 0.0581426, 7.06137, 0.4466, 0.158714, 0.014751, 0.29315, 10.842, 19.17])


zs_agb   = sorted(list(ejecta_table["AGB"].keys()))
zs_sn_ii = sorted(list(ejecta_table["SN"].keys()))
zs_hn_ii = sorted(list(ejecta_table["HN"].keys()))
ms_agb   = sorted(list(ejecta_table["AGB"][zs_agb[0]].keys()))
ms_sn_ii = sorted(list(ejecta_table["SN"][zs_sn_ii[0]].keys()))
ms_hn_ii = sorted(list(ejecta_table["HN"][zs_hn_ii[0]].keys()))

zs_winds = zs_sn_ii
zs_snia = [0.002, 0.02]

# I want to keep a log of the points checked, so that I can verify that I'm
# checking everything I should be
points_checked = {"AGB": [], "SN": [], "HN": []}
points_passed  = {"AGB": [], "SN": [], "HN": []}

def make_inbetween_values(values, min_allowed_value, max_allowed_value):
    return_values = []
    # get one below the minimum point, if desired
    if min_allowed_value is not None:
        return_values.append(np.random.uniform(min_allowed_value, min(values), 1))

    # then get the ones between the points
    for idx in range(len(values) - 1):
        this_min = values[idx]
        this_max = values[idx+1]
        return_values.append(np.random.uniform(this_min, this_max, 1))

    # then the highest value
    if max_allowed_value is not None:
        return_values.append(np.random.uniform(max(values), max_allowed_value, 1))

    return return_values

# ------------------------------------------------------------------------------
#
# metallicity checking
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("z_vals_truth, func",
                         [(zs_agb, tab.get_z_agb),
                          (zs_sn_ii, tab.get_z_sn_ii),
                          (zs_winds, tab.get_z_winds),
                          (zs_snia, tab.get_z_sn_ia)])
def test_metallicities(z_vals_truth, func):
    for idx in range(len(z_vals_truth)):
        truth = z_vals_truth[idx]
        test = func(idx)
        assert truth == test

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
def test_z_idxs_winds_sn_ii(func, z, true_idx_0, true_idx_1):
    z_idx = func(z)
    assert z_idx[0] == true_idx_0
    assert z_idx[1] == true_idx_1


@pytest.mark.parametrize("z, true_idx_0, true_idx_1",
                         [[-0.5,   0, 0],
                          [0,      0, 0],
                          [0.001,  0, 0],
                          [0.002,  0, 0],
                          [0.004,  0, 1],
                          [0.01,   0, 1],
                          [0.02,   1, 1],
                          [0.03,   1, 1],
                          [1.4,    1, 1]])
def test_z_idxs_sn_ia(z, true_idx_0, true_idx_1):
    z_idx = tab.find_z_bound_idxs_sn_ia_py(z)
    assert z_idx[0] == true_idx_0
    assert z_idx[1] == true_idx_1

# ------------------------------------------------------------------------------
#
# mass model index checking
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("m, true_idx_0, true_idx_1",
                         [[0.8, 0, 0],
                          [1.0, 0, 0],
                          [1.3, 0, 1],
                          [1.65, 1, 1],
                          [1.7, 1, 2],
                          [2.0, 2, 2],
                          [2.7, 2, 3],
                          [3.0, 3, 3],
                          [3.7, 3, 4],
                          [4.0, 4, 4],
                          [4.7, 4, 5],
                          [5.0, 5, 5],
                          [5.7, 5, 6],
                          [6.0, 6, 6],
                          [6.7, 6, 7],
                          [7.0, 7, 7],
                          [7.7, 7, 7],
                          [8.0, 7, 7]
                          ])
def test_m_idxs_agb(m, true_idx_0, true_idx_1):
    m_idx = tab.find_mass_bound_idxs_agb_py(m)
    assert m_idx[0] == true_idx_0
    assert m_idx[1] == true_idx_1


@pytest.mark.parametrize("m, true_idx_0, true_idx_1",
                         [[12, 0, 0],
                          [13, 0, 0],
                          [14, 0, 1],
                          [15, 1, 1],
                          [16, 1, 2],
                          [18, 2, 2],
                          [19, 2, 3],
                          [20, 3, 3],
                          [22, 3, 4],
                          [25, 4, 4],
                          [27, 4, 5],
                          [30, 5, 5],
                          [35, 5, 6],
                          [40, 6, 6],
                          [45, 6, 6],
                          [50, 6, 6]
                          ])
def test_m_idxs_sn(m, true_idx_0, true_idx_1):
    m_idx = tab.find_mass_bound_idxs_sn_ii_py(m)
    assert m_idx[0] == true_idx_0
    assert m_idx[1] == true_idx_1

@pytest.mark.parametrize("m, true_idx_0, true_idx_1",
                         [[18, 0, 0],
                          [20, 0, 0],
                          [22, 0, 1],
                          [25, 1, 1],
                          [27, 1, 2],
                          [30, 2, 2],
                          [35, 2, 3],
                          [40, 3, 3],
                          [40, 3, 3]
                          ])
def test_m_idxs_hn(m, true_idx_0, true_idx_1):
    m_idx = tab.find_mass_bound_idxs_hn_ii_py(m)
    assert m_idx[0] == true_idx_0
    assert m_idx[1] == true_idx_1



# ------------------------------------------------------------------------------
#
# Yield checking
#
# ------------------------------------------------------------------------------
# See that the code returns the correct values with the values requested exactly
# align with the input table
@pytest.mark.parametrize("z", zs_agb)
@pytest.mark.parametrize("m", ms_agb)
def test_yields_agb_m_align_z_align(z, m):
    points_checked["AGB"].append([m, z])
    true_ejecta = ejecta_table["AGB"][z][m]
    code_ejecta = tab.get_yields_raw_agb_py(z, m)
    for idx in range(len(true_ejecta)):
        assert code_ejecta[idx] == pytest.approx(true_ejecta[idx], rel=r_tol, abs=a_tol)
    points_passed["AGB"].append([m, z])


@pytest.mark.parametrize("z", zs_sn_ii)
@pytest.mark.parametrize("m", ms_sn_ii)
def test_yields_sn_ii_m_align_z_align(z, m):
    points_checked["SN"].append([m, z])
    true_ejecta = ejecta_table["SN"][z][m]
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)
    for idx in range(len(true_ejecta)):
        assert code_ejecta[idx] == pytest.approx(true_ejecta[idx], rel=r_tol, abs=a_tol)
    points_passed["SN"].append([m, z])


@pytest.mark.parametrize("z", zs_hn_ii)
@pytest.mark.parametrize("m", ms_hn_ii)
def test_yields_hn_ii_m_align_z_align(z, m):
    points_checked["HN"].append([m, z])
    true_ejecta = ejecta_table["HN"][z][m]
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)
    for idx in range(len(true_ejecta)):
        assert code_ejecta[idx] == pytest.approx(true_ejecta[idx], rel=r_tol, abs=a_tol)
    points_passed["HN"].append([m, z])


# ------------------------------------------------------------------------------
#
# Metallicity interpolation yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the mass lines up with the grid, but the
# metallicity value doesn't. This is all inside the range of given metallicity,
# the tests for outside this range are done later
@pytest.mark.parametrize("z_idx", range(len(zs_agb) - 1))
@pytest.mark.parametrize("m", ms_agb)
def test_yields_agb_m_align_z_interp(z_idx, m):
    # get a random value between two endpoints on the table
    z_low = zs_agb[z_idx]
    z_high = zs_agb[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    points_checked["AGB"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["AGB"][z_low][m]
    ejecta_high = ejecta_table["AGB"][z_high][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])


@pytest.mark.parametrize("z_idx", range(len(zs_sn_ii) - 1))
@pytest.mark.parametrize("m", ms_sn_ii)
def test_yields_sn_ii_m_align_z_interp(z_idx, m):
    # get a random value between two endpoints on the table
    z_low = zs_sn_ii[z_idx]
    z_high = zs_sn_ii[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    points_checked["SN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["SN"][z_low][m]
    ejecta_high = ejecta_table["SN"][z_high][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)
    points_passed["SN"].append([m, z])


@pytest.mark.parametrize("z_idx", range(len(zs_hn_ii) - 1))
@pytest.mark.parametrize("m", ms_hn_ii)
def test_yields_hn_ii_m_align_z_interp(z_idx, m):
    # get a random value between two endpoints on the table
    z_low = zs_hn_ii[z_idx]
    z_high = zs_hn_ii[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    points_checked["HN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["HN"][z_low][m]
    ejecta_high = ejecta_table["HN"][z_high][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)
    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# Metallicity that's too high error checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity value is above the maximum
# allowed value, for masses in the grid
@pytest.mark.parametrize("m", ms_agb)
def test_yields_agb_m_align_z_high(m):
    z_max = max(zs_agb)
    # get a random value between two endpoints on the table
    z = np.random.uniform(z_max, 0.025, 1)
    assert z_max < z

    points_checked["AGB"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_max = ejecta_table["AGB"][z_max][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_max)):
        # first do sanity check that it's equal to this value
        assert code_ejecta[idx] == pytest.approx(ejecta_max[idx], rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])


@pytest.mark.parametrize("m", ms_sn_ii)
def test_yields_sn_ii_m_align_z_high(m):
    z_max = max(zs_sn_ii)
    # get a random value between two endpoints on the table
    z = np.random.uniform(z_max, 0.025, 1)
    assert z_max < z

    points_checked["SN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_max = ejecta_table["SN"][z_max][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_max)):
        # first do sanity check that it's equal to this value
        assert code_ejecta[idx] == pytest.approx(ejecta_max[idx], rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])


@pytest.mark.parametrize("m", ms_hn_ii)
def test_yields_hn_ii_m_align_z_high(m):
    z_max = max(zs_hn_ii)
    # get a random value between two endpoints on the table
    z = np.random.uniform(z_max, 0.025, 1)
    assert z_max < z

    points_checked["HN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_max = ejecta_table["HN"][z_max][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_max)):
        # first do sanity check that it's equal to this value
        assert code_ejecta[idx] == pytest.approx(ejecta_max[idx], rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# Metallicity that's too low error checking
#
# ------------------------------------------------------------------------------
# These check the values for when the mass lines up with the grid, but the
# metallicity value is below the minimum allowed value
@pytest.mark.parametrize("m", ms_agb)
def test_yields_agb_m_align_z_low(m):
    z_min = min(zs_agb)
    # get a random value between two endpoints on the table
    z = np.random.uniform(-0.005, z_min, 1)
    assert z < z_min

    points_checked["AGB"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_min = ejecta_table["AGB"][z_min][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_min)):
        # first do sanity check that it's equal to this value
        assert code_ejecta[idx] == pytest.approx(ejecta_min[idx], rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])


@pytest.mark.parametrize("m", ms_sn_ii)
def test_yields_sn_ii_m_align_z_low(m):
    z_min = min(zs_sn_ii)  # should be zero, so we choose a negative value
    # get a random value between two endpoints on the table
    z = np.random.uniform(-0.005, z_min, 1)
    assert z < z_min

    points_checked["SN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_min = ejecta_table["SN"][z_min][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_min)):
        # first do sanity check that it's equal to this value
        assert code_ejecta[idx] == pytest.approx(ejecta_min[idx], rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])


@pytest.mark.parametrize("m", ms_hn_ii)
def test_yields_hn_ii_m_align_z_low(m):
    z_min = min(zs_hn_ii)
    # get a random value between two endpoints on the table
    z = np.random.uniform(-0.005, z_min, 1)
    assert z < z_min

    points_checked["HN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_min = ejecta_table["HN"][z_min][m]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_min)):
        # first do sanity check that it's equal to this value
        assert code_ejecta[idx] == pytest.approx(ejecta_min[idx], rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# Mass interpolation yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity lines up with the grid, but
# the mass value doesn't. This is all inside the range of given mass, the tests
# for outside this range are done later
@pytest.mark.parametrize("m_idx", range(len(ms_agb) - 1))
@pytest.mark.parametrize("z", zs_agb)
def test_yields_agb_m_interp_z_align(m_idx, z):
    # get a random value between two endpoints on the table
    m_low = ms_agb[m_idx]
    m_high = ms_agb[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    points_checked["AGB"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["AGB"][z][m_low]
    ejecta_high = ejecta_table["AGB"][z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("m_idx", range(len(ms_sn_ii) - 1))
@pytest.mark.parametrize("z", zs_sn_ii)
def test_yields_sn_m_interp_z_align(m_idx, z):
    # get a random value between two endpoints on the table
    m_low = ms_sn_ii[m_idx]
    m_high = ms_sn_ii[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    points_checked["SN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["SN"][z][m_low]
    ejecta_high = ejecta_table["SN"][z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])

@pytest.mark.parametrize("m_idx", range(len(ms_hn_ii) - 1))
@pytest.mark.parametrize("z", zs_hn_ii)
def test_yields_hn_m_interp_z_align(m_idx, z):
    # get a random value between two endpoints on the table
    m_low = ms_hn_ii[m_idx]
    m_high = ms_hn_ii[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    points_checked["HN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["HN"][z][m_low]
    ejecta_high = ejecta_table["HN"][z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])


# ------------------------------------------------------------------------------
#
# Mass interpolation, metallicity high yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the mass value needs interpolating, and the
# metallicity value is higher than any of the models
@pytest.mark.parametrize("m_idx", range(len(ms_agb) - 1))
def test_yields_agb_m_interp_z_high(m_idx):
    # get a random value between two endpoints on the table
    m_low = ms_agb[m_idx]
    m_high = ms_agb[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # Then get a metallicity above the maximum value
    max_z = max(zs_agb)
    z = np.random.uniform(max_z, 0.025, 1)

    points_checked["AGB"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["AGB"][max_z][m_low]
    ejecta_high = ejecta_table["AGB"][max_z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("m_idx", range(len(ms_sn_ii) - 1))
def test_yields_sn_m_interp_z_high(m_idx):
    # get a random value between two endpoints on the table
    m_low = ms_sn_ii[m_idx]
    m_high = ms_sn_ii[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # Then get a metallicity above the maximum value
    max_z = max(zs_sn_ii)
    z = np.random.uniform(max_z, 0.025, 1)

    points_checked["SN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["SN"][max_z][m_low]
    ejecta_high = ejecta_table["SN"][max_z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])

@pytest.mark.parametrize("m_idx", range(len(ms_hn_ii) - 1))
def test_yields_hn_m_interp_z_high(m_idx):
    # get a random value between two endpoints on the table
    m_low = ms_hn_ii[m_idx]
    m_high = ms_hn_ii[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # Then get a metallicity above the maximum value
    max_z = max(zs_hn_ii)
    z = np.random.uniform(max_z, 0.025, 1)

    points_checked["HN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["HN"][max_z][m_low]
    ejecta_high = ejecta_table["HN"][max_z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# Mass interpolation, metallicity low yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the mass value needs interpolating, and the
# metallicity value is lower than any of the models
@pytest.mark.parametrize("m_idx", range(len(ms_agb) - 1))
def test_yields_agb_m_interp_z_low(m_idx):
    # get a random value between two endpoints on the table
    m_low = ms_agb[m_idx]
    m_high = ms_agb[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # Then get a metallicity below the minimum value
    min_z = min(zs_agb)
    z = np.random.uniform(-0.005, min_z, 1)

    points_checked["AGB"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["AGB"][min_z][m_low]
    ejecta_high = ejecta_table["AGB"][min_z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("m_idx", range(len(ms_sn_ii) - 1))
def test_yields_sn_m_interp_z_low(m_idx):
    # get a random value between two endpoints on the table
    m_low = ms_sn_ii[m_idx]
    m_high = ms_sn_ii[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # Then get a metallicity below the minimum value
    min_z = min(zs_sn_ii)
    z = np.random.uniform(-0.005, min_z, 1)

    points_checked["SN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["SN"][min_z][m_low]
    ejecta_high = ejecta_table["SN"][min_z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])

@pytest.mark.parametrize("m_idx", range(len(ms_hn_ii) - 1))
def test_yields_hn_m_interp_z_low(m_idx):
    # get a random value between two endpoints on the table
    m_low = ms_hn_ii[m_idx]
    m_high = ms_hn_ii[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # Then get a metallicity below the minimum value
    min_z = min(zs_hn_ii)
    z = np.random.uniform(-0.005, min_z, 1)

    points_checked["HN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_low = ejecta_table["HN"][min_z][m_low]
    ejecta_high = ejecta_table["HN"][min_z][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[m_low, m_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(m), rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# Mass below models yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity lines up with the grid, but
# the mass value is below the minimum modeled mass.
@pytest.mark.parametrize("z", zs_agb)
def test_yields_agb_m_low_z_align(z):
    m_min = min(ms_agb)
    # get a random value below the minimum
    m = np.random.uniform(0, m_min, 1)
    assert m < m_min

    # the ejecta will be scaled by the ratio of the masses
    factor = m / m_min

    points_checked["AGB"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_min = ejecta_table["AGB"][z][m_min]

    # and the values the code says for the mass of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_min)):
        # the yield will be scaled by the ratio of the masses
        assert code_ejecta[idx] == pytest.approx(factor * ejecta_min[idx], rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("z", zs_sn_ii)
def test_yields_sn_ii_m_low_z_align(z):
    m_min = min(ms_sn_ii)
    # get a random value below the minimum
    m = np.random.uniform(8, m_min, 1)
    assert m < m_min

    # the ejecta will be scaled by the ratio of the masses
    factor = m / m_min

    points_checked["SN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_min = ejecta_table["SN"][z][m_min]

    # and the values the code says for the mass of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_min)):
        # the yield will be scaled by the ratio of the masses
        assert code_ejecta[idx] == pytest.approx(factor * ejecta_min[idx], rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])

# HN don't need this test, since their cutoff is their minimum mass

# ------------------------------------------------------------------------------
#
# Mass above models yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity lines up with the grid, but
# the mass value is below the minimum modeled mass.
@pytest.mark.parametrize("z", zs_agb)
def test_yields_agb_m_high_z_align(z):
    m_max = max(ms_agb)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 8, 1)
    assert m_max < m

    # the ejecta will be scaled by the ratio of the masses
    factor = m / m_max

    points_checked["AGB"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_max = ejecta_table["AGB"][z][m_max]

    # and the values the code says for the mass of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_max)):
        # the yield will be scaled by the ratio of the masses
        assert code_ejecta[idx] == pytest.approx(factor * ejecta_max[idx], rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("z", zs_sn_ii)
def test_yields_sn_ii_m_high_z_align(z):
    m_max = max(ms_sn_ii)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 50, 1)
    assert m_max < m

    # the ejecta will be scaled by the ratio of the masses
    factor = m / m_max

    points_checked["SN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_max = ejecta_table["SN"][z][m_max]

    # and the values the code says for the mass of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_max)):
        # the yield will be scaled by the ratio of the masses
        assert code_ejecta[idx] == pytest.approx(factor * ejecta_max[idx], rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])

@pytest.mark.parametrize("z", zs_hn_ii)
def test_yields_hn_ii_m_high_z_align(z):
    m_max = max(ms_hn_ii)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 50, 1)
    assert m_max < m

    # the ejecta will be scaled by the ratio of the masses
    factor = m / m_max

    points_checked["HN"].append([m, z])

    # get the ejecta at the endpoints
    ejecta_max = ejecta_table["HN"][z][m_max]

    # and the values the code says for the mass of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_max)):
        # the yield will be scaled by the ratio of the masses
        assert code_ejecta[idx] == pytest.approx(factor * ejecta_max[idx], rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# Mass below models - interpolate z yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity lines up with the grid, but
# the mass value is below the minimum modeled mass.
@pytest.mark.parametrize("z_idx", range(len(zs_agb) - 1))
def test_yields_agb_m_low_z_interp(z_idx):
    # get a random value between two endpoints on the table
    z_low = zs_agb[z_idx]
    z_high = zs_agb[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    m_min = min(ms_agb)
    # get a random value below the minimum
    m = np.random.uniform(0, m_min, 1)
    assert m < m_min

    points_checked["AGB"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_min

    # get the ejecta at the endpoints
    ejecta_low = factor * ejecta_table["AGB"][z_low][m_min]
    ejecta_high = factor * ejecta_table["AGB"][z_high][m_min]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)
    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("z_idx", range(len(zs_sn_ii) - 1))
def test_yields_sn_ii_m_low_z_interp(z_idx):
    # get a random value between two endpoints on the table
    z_low = zs_sn_ii[z_idx]
    z_high = zs_sn_ii[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    m_min = min(ms_sn_ii)
    # get a random value below the minimum
    m = np.random.uniform(8, m_min, 1)
    assert m < m_min

    points_checked["SN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_min

    # get the ejecta at the endpoints
    ejecta_low = factor * ejecta_table["SN"][z_low][m_min]
    ejecta_high = factor * ejecta_table["SN"][z_high][m_min]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)
    points_passed["SN"].append([m, z])

# Hypernovae don't need this test, since their cutoff mass is the mass of their
# smallest model

# ------------------------------------------------------------------------------
#
# Mass above models - interpolate z yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity lines up with the grid, but
# the mass value is below the minimum modeled mass.
@pytest.mark.parametrize("z_idx", range(len(zs_agb) - 1))
def test_yields_agb_m_high_z_interp(z_idx):
    # get a random value between two endpoints on the table
    z_low = zs_agb[z_idx]
    z_high = zs_agb[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    m_max = max(ms_agb)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 8, 1)
    assert m > m_max

    points_checked["AGB"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_low = factor * ejecta_table["AGB"][z_low][m_max]
    ejecta_high = factor * ejecta_table["AGB"][z_high][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)
    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("z_idx", range(len(zs_sn_ii) - 1))
def test_yields_sn_ii_m_high_z_interp(z_idx):
    # get a random value between two endpoints on the table
    z_low = zs_sn_ii[z_idx]
    z_high = zs_sn_ii[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    m_max = max(ms_sn_ii)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 50, 1)
    assert m > m_max

    points_checked["SN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_low = factor * ejecta_table["SN"][z_low][m_max]
    ejecta_high = factor * ejecta_table["SN"][z_high][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)
    points_passed["SN"].append([m, z])

@pytest.mark.parametrize("z_idx", range(len(zs_hn_ii) - 1))
def test_yields_hn_ii_m_high_z_interp(z_idx):
    # get a random value between two endpoints on the table
    z_low = zs_hn_ii[z_idx]
    z_high = zs_hn_ii[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    m_max = max(ms_hn_ii)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 50, 1)
    assert m > m_max

    points_checked["HN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_low = factor * ejecta_table["HN"][z_low][m_max]
    ejecta_high = factor * ejecta_table["HN"][z_high][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_low)):
        # first do sanity check that it's between the two endpoints
        assert is_between(code_ejecta[idx], ejecta_low[idx], ejecta_high[idx])

        # then do the actual interpolation
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_low[idx], ejecta_high[idx]],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)
    points_passed["HN"].append([m, z])


# ------------------------------------------------------------------------------
#
# Mass below models, z too high yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity is too high, and
# the mass value is below the minimum modeled mass.
def test_yields_agb_m_low_z_high():
    # Then get a metallicity above the maximum value
    max_z = max(zs_agb)
    z = np.random.uniform(max_z, 0.025, 1)
    assert z > 0.02

    m_min = min(ms_agb)
    # get a random value below the minimum
    m = np.random.uniform(0, m_min, 1)
    assert m < m_min

    points_checked["AGB"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_min

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["AGB"][max_z][m_min]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

def test_yields_sn_ii_m_low_z_high():
    # Then get a metallicity above the maximum value
    max_z = max(zs_sn_ii)
    z = np.random.uniform(max_z, 0.025, 1)
    assert z > 0.02

    m_min = min(ms_sn_ii)
    # get a random value below the minimum
    m = np.random.uniform(8, m_min, 1)
    assert m < m_min

    points_checked["SN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_min

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["SN"][max_z][m_min]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])

# Hypernovae don't need this test, since their cutoff mass is the mass of their
# smallest model

# ------------------------------------------------------------------------------
#
# Mass below models, z too low yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity is too low, and
# the mass value is below the maximum modeled mass.
def test_yields_agb_m_low_z_low():
    # Then get a metallicity below the minimum value
    min_z = min(zs_agb)
    z = np.random.uniform(-0.005, min_z, 1)
    assert z < min_z

    m_min = min(ms_agb)
    # get a random value below the minimum
    m = np.random.uniform(0, m_min, 1)
    assert m < m_min

    points_checked["AGB"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_min

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["AGB"][min_z][m_min]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

def test_yields_sn_ii_m_low_z_low():
    # Then get a metallicity below the minimum value
    min_z = min(zs_sn_ii)
    z = np.random.uniform(-0.005, min_z, 1)
    assert z < min_z

    m_min = min(ms_sn_ii)
    # get a random value below the minimum
    m = np.random.uniform(8, m_min, 1)
    assert m < m_min

    points_checked["SN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_min

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["SN"][min_z][m_min]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])

# Hypernovae don't need this test, since their cutoff mass is the mass of their
# smallest model

# ------------------------------------------------------------------------------
#
# Mass above models, z too low yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity is too low, and
# the mass value is above the maximum modeled mass.
def test_yields_agb_m_high_z_low():
    # Then get a metallicity below the minimum value
    min_z = min(zs_agb)
    z = np.random.uniform(-0.005, min_z, 1)
    assert z < min_z

    m_max = max(ms_agb)
    # get a random value below the minimum
    m = np.random.uniform(m_max, 8, 1)
    assert m > m_max

    points_checked["AGB"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["AGB"][min_z][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

def test_yields_sn_ii_m_high_z_low():
    # Then get a metallicity below the minimum value
    min_z = min(zs_sn_ii)
    z = np.random.uniform(-0.005, min_z, 1)
    assert z < min_z

    m_max = max(ms_sn_ii)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 50, 1)
    assert m > m_max

    points_checked["SN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["SN"][min_z][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])


def test_yields_hn_ii_m_high_z_low():
    # Then get a metallicity below the minimum value
    min_z = min(zs_hn_ii)
    z = np.random.uniform(-0.005, min_z, 1)
    assert z < min_z

    m_max = max(ms_hn_ii)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 50, 1)
    assert m > m_max

    points_checked["HN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["HN"][min_z][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# Mass above models, z too high yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the metallicity is too high, and
# the mass value is above the maximum modeled mass.
def test_yields_agb_m_high_z_high():
    # Then get a metallicity above the maximum value
    max_z = max(zs_agb)
    z = np.random.uniform(max_z, 0.025, 1)
    assert z > max_z

    m_max = max(ms_agb)
    # get a random value below the minimum
    m = np.random.uniform(m_max, 8, 1)
    assert m > m_max

    points_checked["AGB"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["AGB"][max_z][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

def test_yields_sn_ii_m_high_z_high():
    # Then get a metallicity above the maximum value
    max_z = max(zs_sn_ii)
    z = np.random.uniform(max_z, 0.025, 1)
    assert z > max_z

    m_max = max(ms_sn_ii)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 50, 1)
    assert m > m_max

    points_checked["SN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["SN"][max_z][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])


def test_yields_hn_ii_m_high_z_high():
    # Then get a metallicity above the maximum value
    max_z = max(zs_hn_ii)
    z = np.random.uniform(max_z, 0.025, 1)
    assert z > max_z

    m_max = max(ms_hn_ii)
    # get a random value above the maximum
    m = np.random.uniform(m_max, 50, 1)
    assert m > m_max

    points_checked["HN"].append([m, z])

    # We'll interpolate between these two, then scale by the fractional mass
    factor = m / m_max

    # get the ejecta at the endpoints
    ejecta_true = factor * ejecta_table["HN"][max_z][m_max]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_true)):
        assert code_ejecta[idx] == pytest.approx(ejecta_true[idx], rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# Mass outside models yield checking
#
# ------------------------------------------------------------------------------
# These check the values for when the mass is below the range for that source,
# no matter the metallicity
@pytest.mark.parametrize("z",zs_agb + make_inbetween_values(zs_agb, -0.005, 0.025))
@pytest.mark.parametrize("direction", ["low", "high"])
def test_yields_agb_m_outside(z, direction):
    # get a random value below the minimum
    if direction == "low":
        m = np.random.uniform(-1, 0, 1)
        assert m < 0
    else:
        m = np.random.uniform(8, 9, 1)
        assert m > 8.0

    points_checked["AGB"].append([m, z])

    # and the values the code says for the mass of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(n_returned_agb):
        # the yield should always be zero
        assert code_ejecta[idx] == 0

    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("z",zs_sn_ii + make_inbetween_values(zs_sn_ii, -0.005, 0.025))
@pytest.mark.parametrize("direction", ["low", "high"])
def test_yields_sn_ii_m_outside(z, direction):
    # get a random value below the minimum
    if direction == "low":
        m = np.random.uniform(5, 8, 1)
        assert m < 8.0
    else:
        m = np.random.uniform(50, 55, 1)
        assert m > 50.0
    points_checked["SN"].append([m, z])

    # and the values the code says for the mass of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(n_returned_sn_ii):
        # the yield should always be zero
        assert code_ejecta[idx] == 0

    points_passed["SN"].append([m, z])

@pytest.mark.parametrize("z",zs_hn_ii + make_inbetween_values(zs_hn_ii, -0.005, 0.025))
@pytest.mark.parametrize("direction", ["low", "high"])
def test_yields_hn_ii_m_outside(z, direction):
    # get a random value below the minimum
    if direction == "low":
        m = np.random.uniform(15, 20, 1)
        assert m < 20
    else:
        m = np.random.uniform(50, 55, 1)
        assert m > 50.0
    points_checked["HN"].append([m, z])

    # and the values the code says for the mass of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(n_returned_hn_ii):
        # the yield should always be zero
        assert code_ejecta[idx] == 0

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# double interpolation yield checking
#
# ------------------------------------------------------------------------------
# both the quantities need interpolating. This can only be done inside the grid
@pytest.mark.parametrize("m_idx", range(len(ms_agb) - 1))
@pytest.mark.parametrize("z_idx", range(len(zs_agb) - 1))
def test_yields_agb_m_interp_z_interp(m_idx, z_idx):
    # get a random value between two endpoints in mass
    m_low = ms_agb[m_idx]
    m_high = ms_agb[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # and in metallicity
    z_low = zs_agb[z_idx]
    z_high = zs_agb[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    points_checked["AGB"].append([m, z])

    # get the ejecta at the four corners
    ejecta_z_low_m_low   = ejecta_table["AGB"][z_low][m_low]
    ejecta_z_low_m_high  = ejecta_table["AGB"][z_low][m_high]
    ejecta_z_high_m_low  = ejecta_table["AGB"][z_high][m_low]
    ejecta_z_high_m_high = ejecta_table["AGB"][z_high][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_agb_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_z_low_m_low)):

        # do the interpolation in mass first
        interp_z_low = interpolate.interp1d(x=[m_low, m_high],
                                            y=[ejecta_z_low_m_low[idx],
                                               ejecta_z_low_m_high[idx]],
                                            kind="linear")
        interp_z_high = interpolate.interp1d(x=[m_low, m_high],
                                             y=[ejecta_z_high_m_low[idx],
                                                ejecta_z_high_m_high[idx]],
                                             kind="linear")

        ejecta_z_low = interp_z_low(m)[0]
        ejecta_z_high = interp_z_high(m)[0]

        # quick sanity check that the ejecta is in this range
        assert is_between(code_ejecta[idx], ejecta_z_low, ejecta_z_high)

        # then do the final interpolation in metallicity
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_z_low, ejecta_z_high],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)

    points_passed["AGB"].append([m, z])

@pytest.mark.parametrize("m_idx", range(len(ms_sn_ii) - 1))
@pytest.mark.parametrize("z_idx", range(len(zs_sn_ii) - 1))
def test_yields_sn_ii_m_interp_z_interp(m_idx, z_idx):
    # get a random value between two endpoints in mass
    m_low = ms_sn_ii[m_idx]
    m_high = ms_sn_ii[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # and in metallicity
    z_low = zs_sn_ii[z_idx]
    z_high = zs_sn_ii[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    points_checked["SN"].append([m, z])

    # get the ejecta at the four corners
    ejecta_z_low_m_low   = ejecta_table["SN"][z_low][m_low]
    ejecta_z_low_m_high  = ejecta_table["SN"][z_low][m_high]
    ejecta_z_high_m_low  = ejecta_table["SN"][z_high][m_low]
    ejecta_z_high_m_high = ejecta_table["SN"][z_high][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_sn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_z_low_m_low)):

        # do the interpolation in mass first
        interp_z_low = interpolate.interp1d(x=[m_low, m_high],
                                            y=[ejecta_z_low_m_low[idx],
                                               ejecta_z_low_m_high[idx]],
                                            kind="linear")
        interp_z_high = interpolate.interp1d(x=[m_low, m_high],
                                             y=[ejecta_z_high_m_low[idx],
                                                ejecta_z_high_m_high[idx]],
                                             kind="linear")

        ejecta_z_low = interp_z_low(m)[0]
        ejecta_z_high = interp_z_high(m)[0]

        # quick sanity check that the ejecta is in this range
        assert is_between(code_ejecta[idx], ejecta_z_low, ejecta_z_high)

        # then do the final interpolation in metallicity
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_z_low, ejecta_z_high],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)

    points_passed["SN"].append([m, z])


@pytest.mark.parametrize("m_idx", range(len(ms_hn_ii) - 1))
@pytest.mark.parametrize("z_idx", range(len(zs_hn_ii) - 1))
def test_yields_hn_ii_m_interp_z_interp(m_idx, z_idx):
    # get a random value between two endpoints in mass
    m_low = ms_hn_ii[m_idx]
    m_high = ms_hn_ii[m_idx + 1]
    m = np.random.uniform(m_low, m_high, 1)
    assert m_low < m < m_high

    # and in metallicity
    z_low = zs_hn_ii[z_idx]
    z_high = zs_hn_ii[z_idx + 1]
    z = np.random.uniform(z_low, z_high, 1)
    assert z_low < z < z_high

    points_checked["HN"].append([m, z])

    # get the ejecta at the four corners
    ejecta_z_low_m_low   = ejecta_table["HN"][z_low][m_low]
    ejecta_z_low_m_high  = ejecta_table["HN"][z_low][m_high]
    ejecta_z_high_m_low  = ejecta_table["HN"][z_high][m_low]
    ejecta_z_high_m_high = ejecta_table["HN"][z_high][m_high]

    # and the values the code says for the metallicity of interest
    code_ejecta = tab.get_yields_raw_hn_ii_py(z, m)

    # iterate through all fields
    for idx in range(len(ejecta_z_low_m_low)):

        # do the interpolation in mass first
        interp_z_low = interpolate.interp1d(x=[m_low, m_high],
                                            y=[ejecta_z_low_m_low[idx],
                                               ejecta_z_low_m_high[idx]],
                                            kind="linear")
        interp_z_high = interpolate.interp1d(x=[m_low, m_high],
                                             y=[ejecta_z_high_m_low[idx],
                                                ejecta_z_high_m_high[idx]],
                                             kind="linear")

        ejecta_z_low = interp_z_low(m)[0]
        ejecta_z_high = interp_z_high(m)[0]

        # quick sanity check that the ejecta is in this range
        assert is_between(code_ejecta[idx], ejecta_z_low, ejecta_z_high)

        # then do the final interpolation in metallicity
        interp = interpolate.interp1d(x=[z_low, z_high],
                                      y=[ejecta_z_low, ejecta_z_high],
                                      kind="linear")
        assert code_ejecta[idx] == pytest.approx(interp(z), rel=r_tol, abs=a_tol)

    points_passed["HN"].append([m, z])

# ------------------------------------------------------------------------------
#
# SN Ia yields
#
# ------------------------------------------------------------------------------
# first get the exact yields
exact_yields_snia_01_solar = [0.0667, 1.3983e-08, 0.0995, 0.00868375, 0.076732,
                              0.0142031, 0.882064, 1.38398]
exact_yields_snia_solar = [0.0475001, 1.10546e-05, 0.0500047, 0.0048054,
                           0.0818503, 0.00969994, 0.899624, 1.37164]

# the yields only depend on metallicity, so we only have to test a few ranges
@pytest.mark.parametrize("z", [-0.5, 0, 1E-4, 0.0019999])
def test_low_metallicity_sn_ia(z):
    """Metallicity less than the minimum should use the yields for minimum z"""
    yields = tab.get_yields_sn_ia_py(z)
    for idx in range(n_elements):
        assert yields[idx] == exact_yields_snia_01_solar[idx]

@pytest.mark.parametrize("z", [0.020001, 0.05, 0.5, 1.5])
def test_low_metallicity_sn_ia(z):
    """Metallicity less than the minimum should use the yields for minimum z"""
    yields = tab.get_yields_sn_ia_py(z)
    for idx in range(n_elements):
        assert yields[idx] == exact_yields_snia_solar[idx]

# ------------------------------------------------------------------------------
#
# Test exact values for SN Ia
#
# ------------------------------------------------------------------------------
def test_exact_low_metallicity_sn_ia():
    """Metallicity less than the minimum should use the yields for minimum z"""
    yields = tab.get_yields_sn_ia_py(0.002)
    for idx in range(n_elements):
        assert yields[idx] == exact_yields_snia_01_solar[idx]

def test_exact_high_metallicity_sn_ia():
    """Metallicity less than the minimum should use the yields for minimum z"""
    yields = tab.get_yields_sn_ia_py(0.02)
    for idx in range(n_elements):
        assert yields[idx] == exact_yields_snia_solar[idx]

# ------------------------------------------------------------------------------
#
# Test the range at which we interpolate
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("z, answers",
                         [(0.006, [0.06243335555555555, 2.467453444444444e-06,
                                   0.08850104444444444, 0.007821894444444444,
                                   0.07786939999999999, 0.013202397777777777,
                                   0.8859662222222222, 1.3812377777777776]),
                          (0.01, [0.0581667111111111, 4.920923888888888e-06,
                                  0.07750208888888888, 0.006960038888888888,
                                  0.07900679999999999, 0.012201695555555555,
                                  0.8898684444444444, 1.3784955555555554])])
def test_interpolate_sn_ia(z, answers):
    """Metallicity less than the minimum should use the yields for minimum z"""
    yields = tab.get_yields_sn_ia_py(z)
    for idx in range(n_elements):
        assert yields[idx] == pytest.approx(answers[idx], rel=r_tol, abs=a_tol)

# ------------------------------------------------------------------------------
#
# direct interpolation checking
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("x, x_0, x_1, y_0, y_1, answer",
                         [[2, 2, 2, 5, 5, 5],
                          [2, 1, 3, 4, 4, 4]])
def test_interpolate_single_value(x, x_0, x_1, y_0, y_1, answer):
    code_answer = tab.interpolate_py(x, x_0, x_1, y_0, y_1)
    assert code_answer == pytest.approx(answer, rel=r_tol, abs=a_tol)


@pytest.mark.parametrize("x, x_0, x_1, y_0, y_1, answer",
                         [[2, 2, 4, 4, 6, 4],
                          [3, 1, 3, 0, 8, 8]])
def test_interpolate_edge_cases(x, x_0, x_1, y_0, y_1, answer):
    code_answer = tab.interpolate_py(x, x_0, x_1, y_0, y_1)
    assert code_answer == pytest.approx(answer, rel=r_tol, abs=a_tol)


@pytest.mark.parametrize("x, x_0, x_1, y_0, y_1, answer",
                         [[2, 1, 3, 5, 7, 6],
                          [2, 0, 4, 4, 6, 5],
                          [7, 5, 11, 0, 3, 1]])
def test_interpolate_simple_positive_slope(x, x_0, x_1, y_0, y_1, answer):
    code_answer = tab.interpolate_py(x, x_0, x_1, y_0, y_1)
    assert code_answer == pytest.approx(answer, rel=r_tol, abs=a_tol)


@pytest.mark.parametrize("x, x_0, x_1, y_0, y_1, answer",
                         [[2, 1, 3, 7, 5, 6],
                          [2, 0, 4, 6, 4, 5],
                          [7, 5, 11, 3, 0, 2]])
def test_interpolate_simple_negative_slope(x, x_0, x_1, y_0, y_1, answer):
    code_answer = tab.interpolate_py(x, x_0, x_1, y_0, y_1)
    assert code_answer == pytest.approx(answer, rel=r_tol, abs=a_tol)



# Then make plots showing what I checked
@pytest.fixture(scope="session", autouse=True)
def plot_points():
    yield True  # to make the rest of the code run at teardown
    for source in ["AGB", "SN", "HN"]:
        fig, ax = bpl.subplots()
        zs_grid = sorted(list(ejecta_table[source].keys()))
        ms_grid = ejecta_table[source][zs_grid[0]].keys()
        for m in ms_grid:
            ax.axvline(m, lw=0.5, c=bpl.almost_black, ls=":")
            for z in zs_grid:
                ax.axhline(z, lw=0.5, c=bpl.almost_black, ls=":")
                ax.scatter(m, z, s=100, facecolor="none", color="k", zorder=0)

        points_failed = [item for item in points_checked[source] if
                         item not in points_passed[source]]
        ms = [item[0] for item in points_failed]
        zs = [item[1] for item in points_failed]
        ax.scatter(ms, zs, s=20, c="firebrick", zorder=5, linewidth=0)

        ms = [item[0] for item in points_passed[source]]
        zs = [item[1] for item in points_passed[source]]
        ax.scatter(ms, zs, s=10, color="forestgreen",
                   linewidth=0, zorder=4)

        if source == "AGB":
            ax.axvline(0, lw=0.5, ls='-')
            ax.axvline(8.0, lw=0.5, ls='-')
        elif source == "SN":
            ax.axvline(8.0, lw=0.5, ls='-')
            ax.axvline(50.0, lw=0.5, ls='-')
        elif source == "HN":
            ax.axvline(20.0, lw=0.5, ls='-')
            ax.axvline(50.0, lw=0.5, ls='-')

        ax.add_labels("Stellar Mass [$M_\odot$]", "Metallicity")

        fig.savefig("plots/grid_{}_checked.pdf".format(source))
