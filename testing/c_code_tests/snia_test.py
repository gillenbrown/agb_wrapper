import pytest
from pytest import approx

# add directory of compiled C code to my path so it can be imported
import sys
from pathlib import Path
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent))

import numpy as np
import tabulation

from snia_enrich_ia_elts_cluster_discrete import lib as snia_discrete
from snia_enrich_ia_elts_cluster import lib as snia_continuous
from snia_enrich_ia_elts_discrete import lib as snia_e_i_e_d
from snia_enrich_ia_elts import lib as snia_e_i_e_c
from snia_enrich_ia_discrete import lib as snia_e_i_d
from snia_enrich_ia import lib as snia_e_i_c
from snia_enrich_discrete import lib as snia_e_d
from snia_enrich import lib as snia_e_c
from snia_discrete import lib as snia_d
from snia_none import lib as snia_c

from core_enrich_ia_elts_cluster_discrete import lib as core

snias_discrete = [snia_discrete, snia_e_i_e_d, snia_e_i_d, snia_e_d, snia_d]
snias_continuous = [snia_continuous, snia_e_i_e_c, snia_e_i_c, snia_e_c, snia_c]
snias_all = snias_discrete + snias_continuous

core.detailed_enrichment_init()
for snia in snias_all:
    snia.detailed_enrichment_init()

# tolerances for tests
rel = 1E-4

# energy per SN
E_0 = 2E51

# initialize lifetimes
lt = tabulation.Lifetimes("Raiteri_96")

# we want the possibility of having many timesteps to check against
dts = [10**(np.random.uniform(2, 3, 1)[0]), 10**(np.random.uniform(3, 6, 1)[0])]
ages = np.random.uniform(60E6, 14E9, 2)
ms = 10**np.random.uniform(3, 8, 2)
zs = 10**np.random.uniform(-3, np.log10(0.05), 2)
n_sn_lefts = np.random.uniform(0, 1, 2)

# ------------------------------------------------------------------------------
#
# Cumulative ejecta analytic function
#
# ------------------------------------------------------------------------------
# simple check that rates are zero before time starts
@pytest.mark.parametrize("snia", snias_all)
def test_rate_zero_early_times(snia):
    t_start = lt.lifetime(8.0, 0.02)
    t_early = np.random.uniform(0, t_start, 100)
    for age in t_early:
        dt = t_start - age
        assert 0 == snia.get_sn_ia_number_py(age, dt, t_start)

# test that at late times it has the expected form. In the tests the parameter
# for number of supernovae per mass can't be adjusted, so we assume the default
@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_all)
def test_late_time_values(age, dt, z, snia):
    # see https://nbviewer.jupyter.org/github/gillenbrown/agb_wrapper/blob/master/testing/informal_tests/snia.ipynb
    # and
    # https://nbviewer.jupyter.org/github/gillenbrown/Tabulation/blob/master/notebooks/sn_Ia.ipynb
    t_start = lt.lifetime(8.0, z)
    true = (1.6E-3 * 2.3480851917 / 0.13) * (age**(-0.13) - (age+dt)**(-0.13))
    test = snia.get_sn_ia_number_py(age, dt, t_start)
    assert test == approx(true, abs=0, rel=rel)

# test that it integrates as expected
@pytest.mark.parametrize("snia", snias_all)
def test_rate_integration(snia):
    # it should integrate to 1.6E-3 when taken from the lifetime of an 8 solar
    # mass, solar metallicity star to 13.79 Gyr.
    t_hubble = 13.791485359241204E9  # taken from notebook above
    t_start = lt.lifetime(8.0, 0.02)
    # the start value of the integral doesn't matter, as long as its lower
    # than t_start
    n_sn = snia.get_sn_ia_number_py(0, t_hubble, t_start)
    assert n_sn == approx(1.6E-3, abs=0, rel=rel)

# ------------------------------------------------------------------------------
#
# Number of supernovae
#
# ------------------------------------------------------------------------------
# Check that there are only an integer number of explosions. We can check this
# by using the energy. To get this, we need to write out the indices use to
# access the return values of this function.
idxs = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6, "Z": 7,
        "E": 8, "N_SN_left": 9}

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_discrete)
def test_number_sn_ejected_discrete(n_sn_left, age, dt, m, z, snia):
    # Discrete SN should have an integer number of SN
    t_start = lt.lifetime(8.0, z)
    yields = snia.sn_ia_core_py(n_sn_left, age, dt, m, z, t_start)
    this_e = yields[idxs["E"]]
    # get the number of supernovae
    n_sn = this_e / E_0
    assert int(n_sn) == approx(n_sn, abs=1E-10, rel=0)

@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_continuous)
def test_number_sn_ejected_continuous_not_discrete(age, dt, m, z, snia):
    # Check that continuous SN do not exactly match 1 SN. Note that this test
    # will occasionally fail due to really lucky values
    t_start = lt.lifetime(8.0, z)
    yields = snia.sn_ia_core_py(0, age, dt, m, z, t_start)
    this_e = yields[idxs["E"]]
    # get the number of supernovae
    n_sn = this_e / E_0
    assert int(n_sn) != approx(n_sn, abs=1E-10, rel=0)

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_discrete)
def test_reasonable_leftover_sn(n_sn_left, age, dt, m, z, snia):
    # We check that the leftover SN is always between zero and one.
    t_start = lt.lifetime(8.0, z)
    yields = snia.sn_ia_core_py(n_sn_left, age, dt, m, z, t_start)
    leftover = yields[idxs["N_SN_left"]]
    assert 0 < leftover < 1

@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_continuous)
def test_continuous_leftover_sn_zero(age, dt, m, z, snia):
    # Continuous SN should always have zero leftover
    t_start = lt.lifetime(8.0, z)
    yields = snia.sn_ia_core_py(0, age, dt, m, z, t_start)
    leftover = yields[idxs["N_SN_left"]]
    assert 0 == leftover


@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_continuous)
def test_continuous_sn_matches_rate(age, dt, m, z, snia):
    # Continuous SN should match the expected rate
    t_start = lt.lifetime(8.0, z)
    n_sn_expected = snia.get_sn_ia_number_py(age, dt, t_start) * m
    n_sn = snia.sn_ia_core_py(0, age, dt, m, z, t_start)[idxs["E"]] / E_0
    assert n_sn == approx(n_sn_expected, rel=rel, abs=0)


@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia_c", snias_continuous)
@pytest.mark.parametrize("snia_d", snias_discrete)
def test_discrete_continuous_close(age, dt, m, z, snia_c, snia_d):
    # continuous and discrete SN should always be within 1 for N_SN. Here
    # we assume 0 leftover SN, so that continuous should always be larger
    # than discrete
    t_start = lt.lifetime(8.0, z)
    yields_c = snia_c.sn_ia_core_py(0, age, dt, m, z, t_start)
    yields_d = snia_d.sn_ia_core_py(0, age, dt, m, z, t_start)

    n_sn_c = yields_c[idxs["E"]] / E_0
    n_sn_d = yields_d[idxs["E"]] / E_0

    assert 0 <= n_sn_c - n_sn_d < 1.0


@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia_c", snias_continuous)
@pytest.mark.parametrize("snia_d", snias_discrete)
def test_discrete_continuous_close_leftover(n_sn_left, age, dt, m, z, snia_c, snia_d):
    # continuous and discrete SN should always be within 1 for N_SN. Here
    # we assume nonzero leftover SN, so that either could be larger
    t_start = lt.lifetime(8.0, z)
    yields_c = snia_c.sn_ia_core_py(0, age, dt, m, z, t_start)
    yields_d = snia_d.sn_ia_core_py(n_sn_left, age, dt, m, z, t_start)

    n_sn_c = yields_c[idxs["E"]] / E_0
    n_sn_d = yields_d[idxs["E"]] / E_0

    assert -1.0 < n_sn_c - n_sn_d < 1.0

@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia_c", snias_continuous)
@pytest.mark.parametrize("snia_d", snias_discrete)
def test_discrete_continuous_difference(age, dt, m, z, snia_c, snia_d):
    # check that the difference between discrete and continuous matches the
    # leftovers in the discrete prescription

    t_start = lt.lifetime(8.0, z)
    yields_c = snia_c.sn_ia_core_py(0, age, dt, m, z, t_start)
    yields_d = snia_d.sn_ia_core_py(0, age, dt, m, z, t_start)

    n_sn_c = yields_c[idxs["E"]] / E_0
    n_sn_d = yields_d[idxs["E"]] / E_0

    leftover = yields_d[idxs["N_SN_left"]]

    assert n_sn_d + leftover == approx(n_sn_c, abs=0, rel=rel)

# ------------------------------------------------------------------------------
#
# Test the metals ejected
#
# ------------------------------------------------------------------------------
test_zs = [-0.01, 0, 0.0000001, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.7]

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", test_zs)
@pytest.mark.parametrize("sn", snias_all)
def test_ejecta_z_variation(n_sn_left, age, dt, m, z, sn):
    # check against the returned
    t_start = lt.lifetime(8.0, z)

    yields = sn.sn_ia_core_py(n_sn_left, age, dt, m, z, t_start)

    n_sn = yields[idxs["E"]] / E_0

    # check against the individual yields, which were tested in the core
    # testing file
    individual_yield = core.get_yields_sn_ia_py(z)

    for elt in ["C", "N", "O", "Mg", "S", "Ca", "Fe", "Z"]:
        this_yield = yields[idxs[elt]]
        true_yield = n_sn * individual_yield[idxs[elt]]
        assert this_yield == approx(true_yield, abs=0, rel=rel)

# ------------------------------------------------------------------------------
#
# Test the rate in practice
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_all)
def test_early_rates(dt, m, z, snia):
    t_start = lt.lifetime(8.0, z)
    t_now = np.random.uniform(0, t_start-dt, 100)
    for t in t_now:
        # at early times we should have 0 SN left
        yields = snia.sn_ia_core_py(0, t, dt, m, z, t_start)

        for idx in idxs.values():
            assert yields[idx] == 0

@pytest.mark.parametrize("z", test_zs)
@pytest.mark.parametrize("snia", snias_all)
def test_declining_with_time_ejecta(z, snia):
    # here we get the ejecta in a system with huge mass, so there will be SN
    # each timestep. We then check that each timestep is less than the one
    # before it, as expected from the DTD
    t_start = lt.lifetime(8.0, z)
    dt = 1E6
    m_big = 1E10
    times = np.arange(50E6, 14E9, dt)
    for idx in range(len(times) - 1):
        t_now = times[idx]
        t_next = times[idx+1]

        # here we use 0 for SN leftover for a cleaner result
        yields_now = snia.sn_ia_core_py(0, t_now, dt, m_big, z, t_start)
        n_sn_now = yields_now[idxs["E"]] / E_0
        yields_next = snia.sn_ia_core_py(0, t_next, dt, m_big, z, t_start)
        n_sn_next = yields_next[idxs["E"]] / E_0

        # if the sn rate, is proportional to time^{-1.13}, then
        # time^1.13 * sn should be a constant
        product_now = t_now **1.13 * n_sn_now
        product_next = t_next **1.13 * n_sn_next
        # With discrete SN this isn't perfect, to leave a wider relative tol
        assert product_now == approx(product_next, abs=0, rel=1E-2)

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("z", test_zs)
@pytest.mark.parametrize("snia", snias_all)
def test_positive_values(n_sn_left, age, dt, z, snia):
    # some of the values are very large when given large masses, and in my
    # experimentation became negative sometimes! Check against this
    t_start = lt.lifetime(8.0, z)
    yields = snia.sn_ia_core_py(n_sn_left, age, dt, 1E15, z, t_start)
    for idx in idxs.values():
        assert yields[idx] >= 0

# ------------------------------------------------------------------------------
#
# Test that there is no difference amongst the discrete snias and among the
# continuous snias. We'll compare to the one with all defines
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_discrete)
def test_discrete_same(n_sn_left, age, dt, m, z, snia):
    t_start = lt.lifetime(8.0, z)
    yields_ref  = snia_discrete.sn_ia_core_py(n_sn_left, age, dt, m, z, t_start)
    yields_test = snia.sn_ia_core_py(n_sn_left, age, dt, m, z, t_start)
    for idx in idxs.values():
        assert yields_ref[idx] == approx(yields_test[idx],
                                                rel=rel, abs=0)

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("age", ages)
@pytest.mark.parametrize("dt", dts)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snia", snias_continuous)
def test_continuous_same(n_sn_left, age, dt, m, z, snia):
    t_start = lt.lifetime(8.0, z)
    yields_ref  = snia_continuous.sn_ia_core_py(n_sn_left, age, dt, m, z, t_start)
    yields_test = snia.sn_ia_core_py(n_sn_left, age, dt, m, z, t_start)
    for idx in idxs.values():
        assert yields_ref[idx] == approx(yields_test[idx],
                                                 rel=rel, abs=0)
