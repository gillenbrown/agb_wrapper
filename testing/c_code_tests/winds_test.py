import pytest
from pytest import approx

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

rel = 1E-4

# we want the possibility of having many timesteps to check against
m_clusters = 10**np.random.uniform(3, 8, 3)
m_early = np.random.uniform(50.0, 120.0, 3)
m_late = np.random.uniform(7.5, 50.0, 3)
m_stellars = np.concatenate([m_early, m_late])
zs = 10**np.random.uniform(-3, np.log10(0.05), 3)

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
    assert ejecta_ref == approx(ejecta_test, rel=rel, abs=0)


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
    assert ejecta_ref == approx(ejecta_test, rel=rel, abs=0)

# ==============================================================================
#
# Much of the functionality was tested in the core tests, so not too much to
# test here
#
# ==============================================================================
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("m_now", m_stellars)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("wind", all_winds)
def test_winds_calculation(z, m_now, m_cluster, wind):
    # Check that the subtraction is happening as expected
    func = wind.get_ejecta_winds_py
    age_50 = lt.lifetime(50.0, z)
    m_next = np.random.uniform(7.5, m_now, 1)

    age_now = lt.lifetime(m_now, z)
    age_next = lt.lifetime(m_next, z)

    ejecta_test = func(age_now, age_next, m_now, m_next, m_cluster, z, age_50)

    ejecta_0 = core.get_cumulative_mass_winds_py(age_now, m_now, z, age_50)
    ejecta_1 = core.get_cumulative_mass_winds_py(age_next, m_next, z, age_50)
    ejecta_true = (ejecta_1 - ejecta_0) * m_cluster

    assert ejecta_test == approx(ejecta_true, rel=rel, abs=0)

@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("m_now", m_early)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("wind", all_winds)
def test_winds_calculation_early_times(z, m_now, m_cluster, wind):
    # the early times have a scale
    func = wind.get_ejecta_winds_py
    age_50 = lt.lifetime(50.0, z)
    m_next = np.random.uniform(50, m_now, 1)

    age_now = lt.lifetime(m_now, z)
    age_next = lt.lifetime(m_next, z)

    ejecta_test = func(age_now, age_next, m_now, m_next, m_cluster, z, age_50)

    ejecta_50 = core.get_cumulative_mass_winds_py(age_50, 50, z, age_50)
    ejecta_true = ejecta_50 * m_cluster * (age_next - age_now) / age_50

    assert ejecta_test == approx(ejecta_true, rel=rel, abs=0)

@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", [0.03, 0.02, 0.015, 0.01, 0.004, 0.001, 0.0005,
                               0.0001, 0])
def test_winds_conservation(m_cluster, z):
    # Check that the total ejecta is the same regardless of how its split up
    # into individual timesteps. This is slow, so we only test it on the
    # default (which is fine because we've already validated that they're all
    # the same
    func = wind_default.get_ejecta_winds_py
    age_50 = lt.lifetime(50.0, z)
    lt_old = lt.lifetime(7, z)

    ages = sorted(np.concatenate([[0],
                                  np.random.uniform(0, lt_old, 100),
                                  [lt_old]]))
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
    assert cumulative_ejecta == approx(cumulative_ejecta_true, rel=rel, abs=0)


@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("z", [0.03, 0.02, 0.015, 0.01, 0.004, 0.001, 0.0005,
                               0.0001, 0])
def test_winds_conservation_up_to_50(m_cluster, z):
    # Check that the total ejecta is the same regardless of how its split up
    # into individual timesteps. This is slow, so we only test it on the
    # default (which is fine because we've already validated that they're all
    # the same
    func = wind_default.get_ejecta_winds_py
    age_50 = lt.lifetime(50.0, z)

    ages = sorted(np.concatenate([[0],
                                  np.random.uniform(0, age_50, 100),
                                  [age_50]]))
    masses = [lt.turnoff_mass(a, z) for a in ages]
    cumulative_ejecta = 0
    for idx in range(len(ages) - 1):
        age_now = ages[idx]
        age_next = ages[idx+1]
        m_now = masses[idx]
        m_next = masses[idx+1]

        cumulative_ejecta += func(age_now, age_next, m_now, m_next, m_cluster,
                                  z, age_50)


    cumulative_ejecta_true = core.get_cumulative_mass_winds_py(age_50, 50,
                                                               z, age_50)
    cumulative_ejecta_true *= m_cluster
    assert cumulative_ejecta == approx(cumulative_ejecta_true, rel=rel, abs=0)

