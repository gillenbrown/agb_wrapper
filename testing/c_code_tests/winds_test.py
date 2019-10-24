import pytest

import numpy as np
from scipy import integrate

import tabulation

# add directory of compiled C code to my path so it can be imported
import sys
from pathlib import Path
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent))

from winds_elements import lib as winds_elements
from winds_no_elements import lib as winds_no_elements
from core_elts import lib as core_elts

core_elts.detailed_enrichment_init()
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
                assert ejecta_elts == pytest.approx(ejecta_no_elts, rel=1E-8, abs=1E-15)


def test_winds_calculation():
    # Check that the subtraction is happening as expected
    func = winds_elements.get_ejecta_winds_py
    for z in 10 ** np.random.uniform(-4, np.log10(0.05), 10):
        age_50 = lt.lifetime(50.0, z)
        for m_now in np.random.uniform(7.5, 50.5, 10):
            m_next = np.random.uniform(7.5, m_now, 1)

            age_now = lt.lifetime(m_now, z)
            age_next = lt.lifetime(m_next, z)

            for m_cluster in 10 ** np.random.uniform(3, 8, 10):
                ejecta_test = func(age_now, age_next, m_now, m_next, m_cluster,
                                   z, age_50)

                ejecta_0 = core_elts.get_cumulative_mass_winds_py(age_now,
                                                                  m_now, z,
                                                                  age_50)
                ejecta_1 = core_elts.get_cumulative_mass_winds_py(age_next,
                                                                  m_next, z,
                                                                  age_50)
                ejecta_true = (ejecta_1 - ejecta_0) * m_cluster

                assert ejecta_test == pytest.approx(ejecta_true, rel=1E-8, abs=1E-15)


@pytest.mark.parametrize("z", [0.02, 0.01, 0.004, 0.001, 0.0005, 0.0001, 0])
def test_winds_conservation(z):
    # Check that the total ejecta is the same regardless of how its split up
    # into individual timesteps
    func = winds_elements.get_ejecta_winds_py
    age_50 = lt.lifetime(50.0, z)

    m_cluster = 10 ** np.random.uniform(3, 8, 1)

    lt_old = lt.lifetime(7, z)

    ages = np.linspace(0, lt_old, 100)
    masses = [lt.turnoff_mass(a, z) for a in ages]
    cumulative_ejecta = 0
    for idx in range(len(ages) - 1):
        age_now = ages[idx]
        age_next = ages[idx+1]
        m_now = masses[idx]
        m_next = masses[idx+1]

        cumulative_ejecta += func(age_now, age_next, m_now, m_next, m_cluster,
                                  z, age_50)


    cumulative_ejecta_true = core_elts.get_cumulative_mass_winds_py(lt_old,
                                                                    7, z,
                                                                    age_50)
    cumulative_ejecta_true *= m_cluster
    assert cumulative_ejecta == pytest.approx(cumulative_ejecta_true, rel=1E-8, abs=1E-15)

