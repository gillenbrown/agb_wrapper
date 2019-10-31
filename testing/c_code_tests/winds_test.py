import pytest

import numpy as np
from scipy import integrate

import tabulation

# add directory of compiled C code to my path so it can be imported
import sys
from pathlib import Path
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent))

from wind_enrich_ia_elts_cluster_discrete import lib as wind_default
from wind_enrich_ia_elts_cluster import lib as wind_e_i_e_c_c
from wind_enrich_ia_elts_discrete import lib as wind_e_i_e_d
from wind_enrich_ia_elts import lib as wind_e_i_e_c
from wind_enrich_ia_discrete import lib as wind_e_i_d
from wind_enrich_ia import lib as wind_e_i_c
from wind_enrich_discrete import lib as wind_e_d
from wind_enrich import lib as wind_e_c
from wind_discrete import lib as wind_d
from wind_none import lib as wind_c
all_winds = [wind_default, wind_e_i_e_c_c, wind_e_i_e_d, wind_e_i_e_c,
             wind_e_i_d, wind_e_i_c, wind_e_d, wind_e_c, wind_d, wind_c]

from core_enrich_ia_elts_cluster_discrete import lib as core

core.detailed_enrichment_init()
for wind in all_winds:
    wind.detailed_enrichment_init()

lt = tabulation.Lifetimes("Raiteri_96")

# we want the possibility of having many timesteps to check against
dts = [10**(np.random.uniform(2, 3, 1)[0]), 10**(np.random.uniform(3, 6, 1)[0])]
dms = np.random.uniform(0.1, 10, 2)
ages = np.random.uniform(60E6, 14E9, 2)
m_clusters = 10**np.random.uniform(3, 8, 2)
m_stellars = np.random.uniform(7.5, 50.5, 2)
zs = 10**np.random.uniform(-3, np.log10(0.05), 2)

@pytest.mark.parametrize("m_star", m_stellars)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("wind", all_winds)
def test_compare_def_options_cumulative(wind, m_star, z):
    # should be the same at all times, masses, metallicities
    # will compare to default value
    ref_func  = wind_default.get_cumulative_mass_winds_py
    test_func = wind.get_cumulative_mass_winds_py
    age = lt.lifetime(m_star, z)
    age_50 = lt.lifetime(50, z)

    ejecta_ref  =  ref_func(age, m_star, z, age_50)
    ejecta_test = test_func(age, m_star, z, age_50)
    assert ejecta_ref == pytest.approx(ejecta_test, rel=1E-5, abs=1E-15)


@pytest.mark.parametrize("m_now", m_stellars)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("wind", all_winds)
def test_compare_elts_no_elts_timestep(m_now, z, m_cluster, wind):
    # should be the same at all times, masses, metallicities
    ref_func = wind_default.get_ejecta_winds_py
    test_func = wind.get_ejecta_winds_py

    age_50 = lt.lifetime(50.0, z)
    m_next = np.random.uniform(7.5, m_now, 1)

    age_now = lt.lifetime(m_now, z)
    age_next = lt.lifetime(m_next, z)

    ejecta_ref = ref_func(age_now, age_next, m_now, m_next, m_cluster, z, age_50)
    ejecta_test = test_func(age_now, age_next, m_now, m_next, m_cluster, z, age_50)
    assert ejecta_ref == pytest.approx(ejecta_test, rel=1E-8, abs=1E-15)

# ==============================================================================
#
# Now that we know all variations are behaving the same we can just teset
# the default one
#
# ==============================================================================
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("m_now", m_stellars)
@pytest.mark.parametrize("m_cluster", m_clusters)
def test_winds_calculation(z, m_now, m_cluster):
    # Check that the subtraction is happening as expected
    func = wind_default.get_ejecta_winds_py
    age_50 = lt.lifetime(50.0, z)
    m_next = np.random.uniform(7.5, m_now, 1)

    age_now = lt.lifetime(m_now, z)
    age_next = lt.lifetime(m_next, z)

    ejecta_test = func(age_now, age_next, m_now, m_next, m_cluster, z, age_50)

    ejecta_0 = core.get_cumulative_mass_winds_py(age_now, m_now, z, age_50)
    ejecta_1 = core.get_cumulative_mass_winds_py(age_next, m_next, z, age_50)
    ejecta_true = (ejecta_1 - ejecta_0) * m_cluster

    assert ejecta_test == pytest.approx(ejecta_true, rel=1E-8, abs=1E-15)

@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", [0.02, 0.01, 0.004, 0.001, 0.0005, 0.0001, 0])
def test_winds_conservation(m_cluster, z):
    # Check that the total ejecta is the same regardless of how its split up
    # into individual timesteps
    func = wind_default.get_ejecta_winds_py
    age_50 = lt.lifetime(50.0, z)
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


    cumulative_ejecta_true = core.get_cumulative_mass_winds_py(lt_old, 7,
                                                               z, age_50)
    cumulative_ejecta_true *= m_cluster
    assert cumulative_ejecta == pytest.approx(cumulative_ejecta_true, rel=1E-8, abs=1E-15)

