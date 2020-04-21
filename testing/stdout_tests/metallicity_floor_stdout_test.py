# add directory of compiled C code to my path so it can be imported
import sys
from pathlib import Path
this_dir = Path(__file__).absolute().parent
sys.path.append(str(this_dir.parent.parent))
from core_enrich_ia_elts_cluster_discrete import lib as core_elts
core_elts.detailed_enrichment_init()

import pytest

elts = ["C", "N", "O", "Mg", "S", "Ca", "Fe", "Z"]

# get the values from the file
floor_values = []
stdout_file = this_dir/"stdout_metallicity_floor.txt"
with stdout_file.open("r") as in_file:
    for line in in_file:
        if "SF_METALLICITY_FLOOR" in line:
            densities = dict()
            line = line.replace(",", "")
            for item in line.split():
                if "=" not in item:
                    continue
                else:
                    elt, value = item.split("=")
                    densities[elt] = float(value)
            floor_values.append(densities)

# get the expected ejecta
sn_metal_fractions = dict()
for idx, elt in enumerate(elts):
    metals = core_elts.get_yields_raw_sn_ii_py(0, 40)[7]
    this_elt = core_elts.get_yields_raw_sn_ii_py(0, 40)[idx]
    sn_metal_fractions[elt] = this_elt / metals

# Make a wrapper to test functions on all outputs, but without using pytests
# parameterize. I don't want to do this because it counts each iteration as an
# extra test, making the number of tests go crazy. This doesn't do that.

@pytest.mark.parametrize("output", floor_values)
def test_nonzero(output):
    # everything should be greater than one
    for density in output.values():
        assert density > 0

@pytest.mark.parametrize("output", floor_values)
def test_metal_fractions(output):
    # make sure the densities match the raw SN yield
    for elt in elts:
        this_metal_fraction = output[elt] / output["Z"]
        assert this_metal_fraction == pytest.approx(sn_metal_fractions[elt],
                                                    abs=0, rel=1E-12)