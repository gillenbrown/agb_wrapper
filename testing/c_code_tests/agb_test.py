import pytest

import numpy as np
from scipy import integrate

import tabulation

# add directory of compiled C code to my path so it can be imported
import sys
from pathlib import Path
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent))

from agb_enrich_ia_elts_cluster_discrete import lib as agb_default
from agb_enrich_ia_elts_cluster import lib as agb_e_i_e_c_c
from agb_enrich_ia_elts_discrete import lib as agb_e_i_e_d
from agb_enrich_ia_elts import lib as agb_e_i_e_c
from agb_enrich_ia_discrete import lib as agb_e_i_d
from agb_enrich_ia import lib as agb_e_i_c
from agb_enrich_discrete import lib as agb_e_d
from agb_enrich import lib as agb_e_c
from agb_discrete import lib as agb_d
from agb_none import lib as agb_c

from core_enrich_ia_elts_cluster_discrete import lib as core

agbs_all = [agb_default, agb_e_i_e_d, agb_e_i_d, agb_e_d, agb_d,
            agb_e_i_e_c_c, agb_e_i_e_c, agb_e_i_c, agb_e_c, agb_c]

core.detailed_enrichment_init()
for agb in agbs_all:
    agb.detailed_enrichment_init()

# initialize lifetimes
imf = tabulation.IMF("Kroupa", 0.08, 50)

# set up indices for accessing the results of the SNII calculations
idxs_ejecta = {"C": 0, "N": 1, "O":2, "Mg":3, "S":4, "Ca": 5, "Fe": 6, "Z": 7,
                 "total": 8}
idxs_raw_yields = {"C": 0, "N":1, "O":2, "Mg": 3, "Z": 4, "total":5}
full_elts = ["C", "N", "O", "Mg", "total"]
scale_elts = ["S", "Ca", "Fe"]
idxs_elts = {"S":0, "Ca":1, "Fe":2}

# tolerances for tests
rtol = 1E-4
atol = 0

# we want the possibility of having many timesteps to check against
m_stars_1 = np.random.uniform(0.08, 8, 5)
m_clusters = 10 ** np.random.uniform(3, 8, 2)
zs = 10**np.random.uniform(-3, np.log10(0.05), 2)
z_eltss = np.reshape(10**np.random.uniform(-4, -1, 6), (2, 3))

def m2(m1):
    return np.random.uniform(0.08, m1, 1)[0]  # m2 < m1

# There should be no differences in return ejects for any. We'll compare them
# to the default
@pytest.mark.parametrize("agb", agbs_all)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("z_elts", z_eltss)
@pytest.mark.parametrize("m_cluster", m_clusters)
def test_defines_no_changes(agb, m1, z, z_elts, m_cluster):
    this_m2 = m2(m1)
    ejecta_test = agb.get_ejecta_agb_py(m1, this_m2, m_cluster, z, *z_elts)
    ejecta_ref = agb_default.get_ejecta_agb_py(m1, this_m2, m_cluster, z, *z_elts)
    for idx in idxs_ejecta.values():
        assert ejecta_test[idx] == ejecta_ref[idx]

# simple check that rates are zero before time starts
@pytest.mark.parametrize("agb", agbs_all)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("z_elts", z_eltss)
@pytest.mark.parametrize("m_cluster", m_clusters)
def test_rate_zero_early_times(agb, z, z_elts, m_cluster):
    for m1 in np.random.uniform(8, 50, 100):
        m2 = np.random.uniform(8, m1, 1)[0]  # m2 < m1
        ejecta = agb.get_ejecta_agb_py(m1, m2, m_cluster, z, *z_elts)
        for idx in idxs_ejecta.values():
            assert 0 == ejecta[idx]

@pytest.mark.parametrize("agb", agbs_all)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("z_elts", z_eltss)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("elt", full_elts)
def test_yields_actually_calculated(agb, m1, z, z_elts, m_cluster, elt):
    # There is a lot that goes into this check. We get the number of AGBs by
    # integrating the real IMF, then multiplying by the single yields to get
    # the expected return, which we can then compare. This really tests a lot.
    # This test only does this for the yields that are fully calculated, rather
    # than the ones that simple scale based on the stellar mass and metallicity
    this_m2 = m2(m1)
    mass = 0.5 * (m1 + this_m2)

    n_agb_true = integrate.quad(imf.normalized_dn_dm, this_m2, m1)[0] * m_cluster
    true_yields_per_agb = core.get_yields_raw_agb_py(z, mass)
    true_yield = n_agb_true * true_yields_per_agb[idxs_raw_yields[elt]]

    ejecta = agb.get_ejecta_agb_py(m1, this_m2, m_cluster, z, *z_elts)
    test_yield = ejecta[idxs_ejecta[elt]]

    assert test_yield == pytest.approx(true_yield, abs=atol, rel=rtol)

@pytest.mark.parametrize("agb", agbs_all)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("z_elts", z_eltss)
@pytest.mark.parametrize("m_cluster", m_clusters)
@pytest.mark.parametrize("elt", scale_elts)
def test_yields_scaled(agb, m1, z, z_elts, m_cluster, elt):
    # There is a lot that goes into this check. We get the number of AGBs by
    # integrating the real IMF, then multiplying by the single yields to get
    # the expected return, which we can then compare. This really tests a lot.
    # This test only does the elements that are not modified in AGBs, so are
    # just scaled based on the metallicity
    this_m2 = m2(m1)
    mass = 0.5 * (m1 + this_m2)

    n_agb_true = integrate.quad(imf.normalized_dn_dm, this_m2, m1)[0] * m_cluster
    true_yields_per_agb = core.get_yields_raw_agb_py(z, mass)
    true_yield = n_agb_true * true_yields_per_agb[idxs_raw_yields["total"]] * z_elts[idxs_elts[elt]]

    ejecta = agb.get_ejecta_agb_py(m1, this_m2, m_cluster, z, *z_elts)
    test_yield = ejecta[idxs_ejecta[elt]]

    assert test_yield == pytest.approx(true_yield, abs=atol, rel=rtol)

@pytest.mark.parametrize("agb", agbs_all)
@pytest.mark.parametrize("m1", m_stars_1)
@pytest.mark.parametrize("z", zs)
@pytest.mark.parametrize("z_elts", z_eltss)
@pytest.mark.parametrize("m_cluster", m_clusters)
def test_ejecta_total_z(agb, m1, z, z_elts, m_cluster):
    # There is a lot that goes into this check. We get the number of AGBs by
    # integrating the real IMF, then multiplying by the single yields to get
    # the expected return, which we can then compare. This really tests a lot.
    # The Z calculation is done separately
    this_m2 = m2(m1)
    mass = 0.5 * (m1 + this_m2)

    ejecta = agb.get_ejecta_agb_py(m1, this_m2, m_cluster, z, *z_elts)
    test_yield = ejecta[idxs_ejecta["Z"]]

    n_agb_true = integrate.quad(imf.normalized_dn_dm, this_m2, m1)[0] * m_cluster
    true_yields_per_agb = core.get_yields_raw_agb_py(z, mass)
    # first get the yield table, which does not contain S, Ca, Fe
    true_z = n_agb_true * true_yields_per_agb[idxs_raw_yields["Z"]]
    for elt in scale_elts:
        true_z += n_agb_true * true_yields_per_agb[idxs_raw_yields["total"]] * z_elts[idxs_elts[elt]]


    assert test_yield == pytest.approx(true_z, abs=atol, rel=rtol)