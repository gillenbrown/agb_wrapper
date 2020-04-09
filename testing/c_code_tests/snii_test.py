import pytest

import numpy as np
from scipy import integrate

import tabulation

# add directory of compiled C code to my path so it can be imported
import sys
from pathlib import Path
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent))

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

imf = tabulation.IMF("Kroupa", 0.08, 50)


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
    # should be the same at all times, masses, metallicities
    # will compare to default value
    test_energy = snii.hn_energy_py(m_star)
    assert test_energy == pytest.approx(true_energy, abs=0, rel=1E-10)

@pytest.mark.parametrize("snii", sniis_continuous)
@pytest.mark.parametrize("m_1,m_2", [[8.0, 19.0],
                                     [9.0, 10.0],
                                     [19.0, 20.0],
                                     [20.0, 21.0],
                                     [24.0, 25.0],
                                     [21.0, 50.0]])
def test_continuous_energy_timestep(snii, m_1, m_2):
    """m1 and m2 here have to be close, since in my implementation there is only
    one supernova energy per timestep. """
    sn_ejecta = snii.get_ejecta_sn_ii_py(0, m_2, m_1, 1.0, 0.02)
    n_sn = sn_ejecta[10]

    mass = 0.5 * (m_1 + m_2)
    energy_sn = 1E51
    if mass > 20.0:
        energy_hn = snii.hn_energy_py(mass)
        energy = 0.5 * energy_sn + 0.5 * energy_hn
    else:
        energy = energy_sn

    true_E = n_sn * energy
    test_E = sn_ejecta[9]

    assert pytest.approx(true_E, abs=0, rel=1E-5) == test_E

# test n_sn_continuous