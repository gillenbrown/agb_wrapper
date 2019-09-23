import pytest

import numpy as np
from scipy import integrate

import tabulation
from snia_discrete_elements import lib as snia_elts
from snia_discrete_no_elements import lib as snia_no_elts
from core_elts import lib as core_elts

snia_elts.detailed_enrichment_init()
core_elts.detailed_enrichment_init()

# tolerances for tests
rtol = 1E-5
atol = 0

# initialize lifetimes
lt = tabulation.Lifetimes("Raiteri_96")
# ------------------------------------------------------------------------------
#
# Rates
#
# ------------------------------------------------------------------------------
# simple check that rates are zero before time starts
@pytest.mark.parametrize("snia", [snia_elts, snia_no_elts])
def test_rate_zero_early_times(snia):
    t_starts = np.random.uniform(20E6, 100E6, 100)
    for t_start in t_starts:
        age = t_start / 2.0
        assert 0 == snia.get_sn_ia_rate_py(age, t_start)

# test that at late times it has the expected form. In the tests the parameter
# for number of supernovae per mass can't be adjusted, so we assume the default
@pytest.mark.parametrize("snia", [snia_elts, snia_no_elts])
def test_late_time_values(snia):
    ages = np.random.uniform(40E6, 14E9, 100)
    for a in ages:
        # see https://nbviewer.jupyter.org/github/gillenbrown/Tabulation/blob/master/notebooks/sn_Ia.ipynb
        true = a**(-1.13) * 1.6E-3 * 2.3480851917
        test = snia.get_sn_ia_rate_py(a, 40E6)
        assert test == pytest.approx(true, abs=atol, rel=rtol)

# test that it integrates as expected
@pytest.mark.parametrize("snia", [snia_elts, snia_no_elts])
def test_rate_integration(snia):
    # it should integrate to 1.6E-3 when taken from the lifetime of an 8 solar
    # mass, solar metallicity star to 13.79 Gyr.
    t_hubble = 13.791485359241204E9  # taken from notebook above
    t_start = lt.lifetime(8.0, 0.02)
    # the start value of the integral doesn't matter, as long as its lower
    # than t_start
    integral = integrate.quad(snia.get_sn_ia_rate_py, 30E6, t_hubble,
                              args=(t_start))[0]
    assert integral == pytest.approx(1.6E-3, abs=atol, rel=rtol)

# ------------------------------------------------------------------------------
#
# Ejecta in a given timestep
#
# ------------------------------------------------------------------------------
# Check that there are only an integer number of explosions. We can check this
# by using the energy. To get this, we need to write out the indices use to
# access the return values of this function.
idxs_elts = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6,
             "Z": 7, "E": 8, "N_SN_left": 9}
idxs_no_elts = {"Z": 0, "E": 1, "N_SN_left": 2}

# we want the possibility of having many timesteps to check against
uniform_dts = np.logspace(2, 6, 9)

@pytest.mark.parametrize("dt", uniform_dts)
@pytest.mark.parametrize("snia,idxs", ([snia_elts, idxs_elts],
                                       [snia_no_elts, idxs_no_elts]))
def test_number_sn_ejected(dt, snia, idxs):
    # I'll choose many random timesteps of varying lengths at a given time
    # when SN are active
    yields = snia.sn_ia_core_py(0, 1E8, dt, 1E6, 0.02, 40E6)
    this_e = yields[idxs["E"]]
    # get the remainder when we divide by the energy of 1 SN
    remainder = this_e % 2E51
    assert remainder == pytest.approx(0, abs=1E47, rel=0) or \
           remainder == pytest.approx(2E51, abs=0, rel=1E-5)

@pytest.mark.parametrize("dt", uniform_dts)
@pytest.mark.parametrize("snia,idxs", ([snia_elts, idxs_elts],
                                       [snia_no_elts, idxs_no_elts]))
def test_reasonable_leftover_sn(dt, snia, idxs):
    # I'll choose many random timesteps of varying lengths at a given time
    # when SN are active. We check that the leftover SN is always between
    # zero and one.
    yields = snia.sn_ia_core_py(0, 1E8, dt, 1E6, 0.02, 40E6)
    leftover = yields[idxs["N_SN_left"]]
    assert 0 < leftover < 1

@pytest.mark.parametrize("dt", uniform_dts)
def test_energy_consistent_between_defines(dt):
    # test that in a given timestep, the versions with and without elements
    # should give the same energy.
    yields_elts= snia_elts.sn_ia_core_py(0, 1E8, dt, 1E6, 0.02, 40E6)
    yields_no_elts = snia_no_elts.sn_ia_core_py(0, 1E8, dt, 1E6, 0.02,
                                                     40E6)
    e_elts = yields_elts[idxs_elts["E"]]
    e_no_elts = yields_no_elts[idxs_no_elts["E"]]

    assert e_elts == pytest.approx(e_no_elts, abs=1, rel=1E-5)


@pytest.mark.parametrize("dt", uniform_dts)
def test_leftover_consistent_between_defines(dt):
    # test that in a given timestep, the versions with and without elements
    # should give the same leftover SN. Again use timesteps where SN should
    #  be active
    yields_elts = snia_elts.sn_ia_core_py(0, 1E8, dt, 1E6, 0.02, 40E6)
    yields_no_elts = snia_no_elts.sn_ia_core_py(0, 1E8, dt, 1E6, 0.02,
                                                     40E6)
    leftover_elts = yields_elts[idxs_elts["N_SN_left"]]
    leftover_no_elts = yields_no_elts[idxs_no_elts["N_SN_left"]]

    assert leftover_elts == pytest.approx(leftover_no_elts, rel=1E-5)

# ------------------------------------------------------------------------------
#
# Test the metals ejected
#
# ------------------------------------------------------------------------------
test_zs = [-0.01, 0, 0.0000001, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.7]
@pytest.mark.parametrize("dt", uniform_dts)
@pytest.mark.parametrize("z", test_zs)
def test_metals_consistent_no_elts_with_z(dt, z):
    # When all elements are disabled, the yield should be the same at all z
    yields_no_elts = snia_no_elts.sn_ia_core_py(0, 1E8, dt, 1E6,
                                                     z, 40E6)
    n_sn = yields_no_elts[idxs_no_elts["E"]] / 2E51
    metals = yields_no_elts[idxs_no_elts["Z"]]
    assert metals == pytest.approx(n_sn * 1.37164)

@pytest.mark.parametrize("dt", uniform_dts)
@pytest.mark.parametrize("elt", ["C", "N", "O", "Mg", "S", "Ca", "Fe", "Z"])
@pytest.mark.parametrize("z", test_zs)
def test_all_elts_ejecta_z_variation(dt, elt, z):
    # check against the returned
    yields_elts = snia_elts.sn_ia_core_py(0, 1E8, dt, 1E6, z, 40E6)

    idx = idxs_elts[elt]
    this_yield = yields_elts[idx]

    n_sn = yields_elts[idxs_elts["E"]] / 2E51


    # check against the individual yields, which were tested in the core
    # testing file
    individual_yield = core_elts.get_yields_sn_ia_py(z)


    true = n_sn * individual_yield[idx]
    assert this_yield == pytest.approx(true, abs=1E-4, rel=1E-5)

# ------------------------------------------------------------------------------
#
# Test the rate in practice
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", uniform_dts)
@pytest.mark.parametrize("snia,idxs", ([snia_elts, idxs_elts],
                                       [snia_no_elts, idxs_no_elts]))
def test_early_rates(dt, snia, idxs):
    t_now = np.random.uniform(0, 40E6, 100)
    for t in t_now:
        yields = snia.sn_ia_core_py(0, t, dt, 1E5, 0.02, 40E6)

        for idx in range(idxs["E"]):
            assert yields[idx] == 0

@pytest.mark.parametrize("snia,idxs", ([snia_elts, idxs_elts],
                                       [snia_no_elts, idxs_no_elts]))
def test_declining_with_time_ejecta(snia, idxs):
    # here we get the ejecta in a system with huge mass, so there will be SN
    # each timestep. We then check that each timestep is less than the one
    # before it, as expected from the DTD
    dt = 1E8
    times = np.arange(50E6, 14E9, dt)
    for idx in range(len(times) - 1):
        t_now = times[idx]
        t_next = times[idx+1]

        yields_now = snia.sn_ia_core_py(0, t_now, dt, 1E10, 0.02, 40E6)
        n_sn_now = yields_now[idxs["E"]] / 2E51
        yields_next = snia.sn_ia_core_py(0, t_next, dt, 1E10, 0.02, 40E6)
        n_sn_next = yields_next[idxs["E"]] / 2E51

        # if the sn rate, is proportional to time^{-1.13}, then
        # time^1.13 * sn should be a constant
        product_now = t_now **1.13 *n_sn_now
        product_next = t_next **1.13 * n_sn_next
        # With discrete SN this isn't perfect, to leave a wider relative tol
        assert product_now == pytest.approx(product_next, abs=1, rel=1E-2)


@pytest.mark.parametrize("dt", uniform_dts)
@pytest.mark.parametrize("elt", idxs_no_elts.keys())
def test_positive_values_no_elts(dt, elt):
    # some of the values are very large when given large masses, and in my
    # experimentation became negative sometimes! Check against this
    yields = snia_no_elts.sn_ia_core_py(0, 1E8, dt, 1E15, 0.02, 40E6)
    assert yields[idxs_no_elts[elt]] >= 0

@pytest.mark.parametrize("dt", uniform_dts)
@pytest.mark.parametrize("elt", idxs_elts.keys())
def test_positive_values_no_elts(dt, elt):
    # some of the values are very large when given large masses, and in my
    # experimentation became negative sometimes! Check against this
    yields = snia_elts.sn_ia_core_py(0, 1E8, dt, 1E15, 0.02, 40E6)
    assert yields[idxs_elts[elt]] >= 0





