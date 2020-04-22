import pytest
from pytest import approx

import numpy as np
from scipy import integrate, interpolate

import tabulation

# add directory of compiled C code to my path so it can be imported
import sys
from pathlib import Path
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent/"build"))

from snii_enrich_ia_elts_cluster_discrete import lib as snii_default
from snii_enrich_ia_elts_cluster import lib as snii_e_i_e_c_c
from snii_enrich_ia_elts_discrete import lib as snii_e_i_e_d
from snii_enrich_ia_elts import lib as snii_e_i_e_c
from snii_enrich_ia_discrete import lib as snii_e_i_d
from snii_enrich_ia import lib as snii_e_i_c
from snii_enrich_discrete import lib as snii_e_d
from snii_enrich import lib as snii_e_c
from snii_discrete import lib as snii_d
from snii_none import lib as snii_c

from core_enrich_ia_elts_cluster_discrete import lib as core

sniis_continuous = [snii_e_i_e_c_c, snii_e_i_e_c, snii_e_i_c, snii_e_c, snii_c]
sniis_discrete = [snii_default, snii_e_i_e_d, snii_e_i_d, snii_e_d, snii_d]
sniis_all = sniis_continuous + sniis_discrete

core.detailed_enrichment_init()
for snii in sniis_all:
    snii.detailed_enrichment_init()
    snii.init_rand()

# set up indices for accessing the results of the SNII calculations
idxs = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6, "Z": 7,
        "total": 8, "E": 9, "N_SN": 10, "N_SN_left": 11}

# tolerances for tests
rel = 1E-4

# energy per SN
E_0 = 1E51

# initialize lifetimes
lt = tabulation.Lifetimes("Raiteri_96")
imf = tabulation.IMF("Kroupa", 0.08, 50)

# we want the possibility of having many timesteps to check against
m_stars_1 = np.concatenate([np.random.uniform(9, 20, 2),
                            np.random.uniform(20, 50, 3)])
m_clusters = 10 ** np.random.uniform(3, 8, 2)
zs = 10**np.random.uniform(-3, np.log10(0.05), 2)
n_sn_lefts = np.random.uniform(0, 1, 2)

def m2(m1):
    return np.random.uniform(8, m1, 1)[0]  # m2 < m1

# ------------------------------------------------------------------------------
#
# Number of supernovae and leftover SN
#
# ------------------------------------------------------------------------------
# simple check that rates are zero before time starts
@pytest.mark.parametrize("snii", sniis_all)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("m_cluster", m_clusters)
def test_rate_zero_early_times(snii, z, m_cluster):
    for m1 in np.random.uniform(50, 200, 100):
        m2 = np.random.uniform(50, m1, 1)[0]  # m2 < m1
        ejecta = snii.get_ejecta_sn_ii_py(0, m1, m2, m_cluster, z)
        for idx in idxs.values():
            assert 0 == ejecta[idx]

# simple check that rates are zero at late times
@pytest.mark.parametrize("snii", sniis_all)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("m_cluster", m_clusters)
def test_rate_zero_late_times(snii, z, m_cluster):
    for m1 in np.random.uniform(5, 8, 100):
        m2 = np.random.uniform(5, m1, 1)[0]  # m2 < m1
        ejecta = snii.get_ejecta_sn_ii_py(0, m1, m2, m_cluster, z)
        for idx in idxs.values():
            assert 0 == ejecta[idx]

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_discrete)
def test_number_sn_ejected_discrete(n_sn_left, m1, m_cluster, z, snii):
    # Discrete SN should have an integer number of SN
    ejecta = snii.get_ejecta_sn_ii_py(n_sn_left, m1, m2(m1), m_cluster, z)
    n_sn = ejecta[idxs['N_SN']]
    assert int(n_sn) == approx(n_sn, abs=1E-10, rel=0)

@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_continuous)
def test_number_sn_ejected_continuous(m1, m_cluster, z, snii):
    # Continuous SN should not have an integer number of SN
    ejecta = snii.get_ejecta_sn_ii_py(0, m1, m2(m1), m_cluster, z)
    n_sn = ejecta[idxs['N_SN']]
    assert int(n_sn) != approx(n_sn, abs=1E-10, rel=0)

@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_continuous)
def test_number_of_sn_from_imf_continuous(m1, m_cluster, z, snii):
    # check that the number of SN in the continuous case matches what we expect
    # from the IMF exactly
    this_m2 = m2(m1)

    n_sn_true = integrate.quad(imf.normalized_dn_dm, this_m2, m1)[0] * m_cluster
    ejecta = snii.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)
    n_sn_test = ejecta[idxs['N_SN']]

    assert n_sn_test == approx(n_sn_true, abs=0, rel=rel)

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_discrete)
def test_number_of_sn_from_imf_discrete(n_sn_left, m1, m_cluster, z, snii):
    # check that the number of SN in the discrete case matches what we expect
    # from the IMF. Here we check whether we're within 1 of the correct answer.
    # we also add the leftovers to make sure they contribute to the total
    this_m2 = m2(m1)

    n_sn_true = integrate.quad(imf.normalized_dn_dm, this_m2, m1)[0] * m_cluster
    n_sn_true += n_sn_left
    ejecta = snii.get_ejecta_sn_ii_py(n_sn_left, m1, this_m2, m_cluster, z)
    n_sn_test = ejecta[idxs['N_SN']]
    n_leftover = ejecta[idxs["N_SN_left"]]

    assert n_sn_test == approx(n_sn_true, abs=1, rel=0)
    assert n_sn_test + n_leftover == approx(n_sn_true, abs=0, rel=rel)


@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii_d", sniis_discrete)
@pytest.mark.parametrize("snii_c", sniis_continuous)
def test_number_of_sn_discrete_continuous_close(m1, m_cluster, z, snii_c, snii_d):
    # continuous and discrete SN should always be within 1 for N_SN. Here
    # we assume 0 leftover SN, so that continuous should always be larger
    # than discrete
    this_m2 = m2(m1)

    ejecta_c = snii_c.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)
    ejecta_d = snii_d.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)

    n_sn_c = ejecta_c[idxs['N_SN']]
    n_sn_d = ejecta_d[idxs['N_SN']]

    assert 0 <= n_sn_c - n_sn_d < 1.0

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii_d", sniis_discrete)
@pytest.mark.parametrize("snii_c", sniis_continuous)
def test_number_of_sn_discrete_continuous_close_leftover(n_sn_left, m1, m_cluster, z, snii_c, snii_d):
    # continuous and discrete SN should always be within 1 for N_SN. Here
    # we assume nonzero leftover SN, so that either could be larger
    this_m2 = m2(m1)

    ejecta_c = snii_c.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)
    ejecta_d = snii_d.get_ejecta_sn_ii_py(n_sn_left, m1, this_m2, m_cluster, z)

    n_sn_c = ejecta_c[idxs['N_SN']]
    n_sn_d = ejecta_d[idxs['N_SN']]

    assert -1.0 < n_sn_c - n_sn_d < 1.0

@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii_d", sniis_discrete)
@pytest.mark.parametrize("snii_c", sniis_continuous)
def test_number_of_sn_discrete_continuous_difference(m1, m_cluster, z, snii_c, snii_d):
    # check that the difference between discrete and continuous matches the
    # leftovers in the discrete prescription
    this_m2 = m2(m1)

    ejecta_c = snii_c.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)
    ejecta_d = snii_d.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)

    n_sn_c = ejecta_c[idxs['N_SN']]
    n_sn_d = ejecta_d[idxs['N_SN']]
    leftover = ejecta_d[idxs["N_SN_left"]]

    assert n_sn_d + leftover == approx(n_sn_c, abs=0, rel=rel)

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_discrete)
def test_reasonable_leftover_sn_discrete(n_sn_left, m1, m_cluster, z, snii):
    # The number of leftover supernovae should be between zero and one
    ejecta = snii.get_ejecta_sn_ii_py(n_sn_left, m1, m2(m1), m_cluster, z)
    leftover = ejecta[idxs['N_SN_left']]
    assert 0 < leftover < 1

@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_continuous)
def test_reasonable_leftover_sn_continuous(m1, m_cluster, z, snii):
    # continuous SN should have no leftover SN
    ejecta = snii.get_ejecta_sn_ii_py(0, m1, m2(m1), m_cluster, z)
    leftover = ejecta[idxs['N_SN_left']]
    assert 0 == leftover

# ------------------------------------------------------------------------------
#
# Energy injection
#
# ------------------------------------------------------------------------------
def supernova_partition(energy, total_n_sn, sn_mass):
    # here's we'll try out all combos of SN and HN to see if any of them
    # produce the energy output. We have to do this because the specific
    # number of SN or HN is sampled from a binomial distribution
    hn_energy = snii.hn_energy_py(sn_mass)

    for n_hn in range(total_n_sn + 1):  # need to include an all HN iteration
        n_sn = total_n_sn - n_hn
        this_E = n_sn * E_0 + n_hn * hn_energy
        if this_E == approx(energy, abs=0, rel=rel):
            return n_sn, n_hn

    # if we got here we did not find a match
    assert False


@pytest.mark.parametrize("m_star,true_energy", [[0, 1E52],
                                                [19, 1E52],
                                                [20, 1E52],
                                                [22.5, 1E52],
                                                [25, 1E52],
                                                [26, 1.2E52],
                                                [27.5, 1.5E52],
                                                [29, 1.8E52],
                                                [30, 2E52],
                                                [31, 2.1E52],
                                                [32, 2.2E52],
                                                [35, 2.5E52],
                                                [37, 2.7E52],
                                                [39, 2.9E52],
                                                [40, 3E52],
                                                [41, 3E52],
                                                [45, 3E52],
                                                [50, 3E52],
                                                [55, 3E52],
                                                [100, 3E52]])
@pytest.mark.parametrize("snii", sniis_all)
def test_hn_energy(snii, m_star, true_energy):
    test_energy = snii.hn_energy_py(m_star)
    assert test_energy == approx(true_energy, abs=0, rel=rel)

@pytest.mark.parametrize("snii", sniis_all)
@pytest.mark.parametrize("m_star", np.random.uniform(20, 50, 10))
def test_hn_energy_full(snii, m_star):
    interp = interpolate.interp1d(x=[20, 25, 30, 40],
                                  y=[10E51, 10E51, 20E51, 30E51],
                                  kind="linear", bounds_error=False,
                                  fill_value=(10E51, 30E51))

    test_energy = snii.hn_energy_py(m_star)
    true_energy = interp(m_star)
    assert test_energy == approx(true_energy, abs=0, rel=rel)

@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_continuous)
def test_energy_continuous(snii, m1, m_cluster, z):
    # continuous SN can be tested in a given timestep, since there is no
    # stochasticity in the SN yields
    this_m2 = m2(m1)

    sn_ejecta = snii.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)
    n_sn = sn_ejecta[idxs["N_SN"]]

    assert n_sn > 0

    # get the energy at the mass leaving halfway through the timestep
    # if we have HN, the energy is weighted exactly halfway between the two
    mass = 0.5 * (m1 + this_m2)
    if mass > 20.0:
        energy_hn = snii.hn_energy_py(mass)
        energy = 0.5 * E_0 + 0.5 * energy_hn
    else:
        energy = E_0

    true_E = n_sn * energy
    test_E = sn_ejecta[idxs["E"]]

    assert approx(true_E, abs=0, rel=rel) == test_E

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_discrete)
def test_energy_discrete(n_sn_left, m1, m_cluster, z, snii):
    # Discrete SN is much more difficult to test. What we can do instead it to
    # use the number of SN to validate that there is at least one combination
    # of SN and HN that replicates the energy injected
    n_sn_total = 0
    while n_sn_total == 0:
        this_m2 = m2(m1)
        sn_ejecta = snii.get_ejecta_sn_ii_py(n_sn_left, m1, this_m2, m_cluster, z)
        n_sn_total = int(sn_ejecta[idxs["N_SN"]])
    ejected_E = sn_ejecta[idxs["E"]]

    # get the energy at the mass leaving halfway through the timestep
    mass = 0.5 * (m1 + this_m2)
    if mass > 20.0:  # we have HN
        supernova_partition(ejected_E, n_sn_total, mass)
        # this will assert False if it cannot be done
    else:  # just regular SN, easy to check
        true_E = n_sn_total * E_0
        assert ejected_E == approx(true_E, abs=0, rel=rel)

# ------------------------------------------------------------------------------
#
# Elemental yields
#
# ------------------------------------------------------------------------------
elts = ["C", "N", "O", "Mg", "S", "Ca", "Fe", "Z", "total"]

@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_discrete)
@pytest.mark.parametrize("elt", elts)
def test_yields_discrete(n_sn_left, m1, m_cluster, z, snii, elt):
    # check the individual yields, which we do by comparing to the yield table
    # which we've already validated. Discrete has to do the thing to see how
    # many SN and HN there are
    n_sn_total = 0
    while n_sn_total == 0:
        this_m2 = m2(m1)
        sn_ejecta = snii.get_ejecta_sn_ii_py(n_sn_left, m1, this_m2, m_cluster, z)
        n_sn_total = int(sn_ejecta[idxs["N_SN"]])
    ejected_E = sn_ejecta[idxs["E"]]

    # get the energy at the mass leaving halfway through the timestep
    mass = 0.5 * (m1 + this_m2)
    if mass > 20.0:  # we have HN
        n_sn, n_hn = supernova_partition(ejected_E, n_sn_total, mass)
    else:  # just regular SN, easy to check
        n_sn = n_sn_total
        n_hn = 0

    true_yields_per_sn = core.get_yields_raw_sn_ii_py(z, mass)
    true_yields_per_hn = core.get_yields_raw_hn_ii_py(z, mass)

    test_yield = sn_ejecta[idxs[elt]]
    true_yield = n_sn * true_yields_per_sn[idxs[elt]] + \
                 n_hn * true_yields_per_hn[idxs[elt]]

    assert test_yield == approx(true_yield, abs=0, rel=rel)

@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_continuous)
@pytest.mark.parametrize("elt", elts)
def test_yields_continuous(snii, m1, m_cluster, z, elt):
    this_m2 = m2(m1)
    sn_ejecta = snii.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)
    n_sn_total = sn_ejecta[idxs["N_SN"]]

    assert n_sn_total > 0

    # get the energy at the mass leaving halfway through the timestep
    # if we have HN, the energy is weighted exactly halfway between the two
    mass = 0.5 * (m1 + this_m2)

    if mass > 20:
        n_hn = 0.5 * n_sn_total
        n_sn = n_hn
    else:
        n_sn = n_sn_total
        n_hn = 0

    true_yields_per_sn = core.get_yields_raw_sn_ii_py(z, mass)
    true_yields_per_hn = core.get_yields_raw_hn_ii_py(z, mass)

    test_yield = sn_ejecta[idxs[elt]]
    true_yield = n_sn * true_yields_per_sn[idxs[elt]] + \
                 n_hn * true_yields_per_hn[idxs[elt]]

    assert test_yield == approx(true_yield, abs=0, rel=rel)

# ------------------------------------------------------------------------------
#
# Test that there is no difference amongst the discrete and among the
# continuous. We'll compare to the one with all defines
#
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("n_sn_left", n_sn_lefts)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_discrete)
def test_consistency_discrete(n_sn_left, m1, m_cluster, z, snii):
    # this does not work if there are hypernovae, unfortunately, since there is
    # randomness there
    if m1 > 20:
        m1 = np.random.uniform(9, 20, 1)[0]
    this_m2 = m2(m1)
    sn_ejecta_test =        snii.get_ejecta_sn_ii_py(n_sn_left, m1, this_m2, m_cluster, z)
    sn_ejecta_ref = snii_default.get_ejecta_sn_ii_py(n_sn_left, m1, this_m2, m_cluster, z)

    for idx in idxs.values():
        assert sn_ejecta_test[idx] == approx(sn_ejecta_ref[idx], abs=0, rel=rel)


@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("snii", sniis_continuous)
def test_consistency_continuous(m1, m_cluster, z, snii):
    this_m2 = m2(m1)
    sn_ejecta_test =          snii.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)
    sn_ejecta_ref = snii_e_i_e_c_c.get_ejecta_sn_ii_py(0, m1, this_m2, m_cluster, z)

    for idx in idxs.values():
        assert sn_ejecta_test[idx] == approx(sn_ejecta_ref[idx], abs=0, rel=rel)