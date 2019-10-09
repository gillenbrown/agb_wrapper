import pytest

import numpy as np
from scipy import integrate

import tabulation
from winds_elements import lib as winds_elements
from winds_no_elements import lib as winds_no_elements
from core_elts import lib as core_elts

winds_no_elements.detailed_enrichment_init()
winds_elements.detailed_enrichment_init()

lt = tabulation.Lifetimes("Raiteri_96")

def age_50(z):
    return lt.lifetime(50.0, z)

def test_compare_elts_no_elts_cumulative():
    # should be the same at all times, masses, metallicities
    elt_func = winds_elements.get_cumulative_mass_winds_py
    no_elt_func = winds_no_elements.get_cumulative_mass_winds_py
    for z in 10 ** np.random.uniform(-4, np.log10(0.05), 100):
        age_50 = lt.lifetime(50.0, z)
        for m_star in np.random.uniform(7.5, 50.5, 100):
            age = lt.lifetime(m_star, z)

            ejecta_elts = elt_func(age, m_star, z, age_50)
            ejecta_no_elts = no_elt_func(age, m_star, z, age_50)
            assert ejecta_elts == pytest.approx(ejecta_no_elts, rel=1E-5, abs=1E-15)

def test_compare_elts_no_elts_timestep():
    # should be the same at all times, masses, metallicities
    elt_func = winds_elements.get_ejecta_winds_py
    no_elt_func = winds_no_elements.get_ejecta_winds_py

    dm = 0.1
    for z in 10 ** np.random.uniform(-4, np.log10(0.05), 10):
        age_50 = lt.lifetime(50.0, z)
        for m_now in np.random.uniform(7.5, 50.5, 10):
            m_next = m_now + dm

            age_now = lt.lifetime(m_now, z)
            age_next = lt.lifetime(m_next, z)

            for m_cluster in 10**np.random.uniform(3, 8, 10):

                ejecta_elts = elt_func(age_now, age_next, m_now, m_next,
                                       m_cluster, z, age_50)
                ejecta_no_elts = no_elt_func(age_now, age_next, m_now, m_next,
                                             m_cluster, z, age_50)
                assert ejecta_elts == pytest.approx(ejecta_no_elts, rel=1E-5, abs=1E-15)

# Now that i've shown that both are identical I can just test one

