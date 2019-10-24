import os
from cffi import FFI

dir = "/Users/gillenb/code/art_cluster/src/sf/models/"
# in the .c file, order does matter for this compilation. In the full context of
# ART it doesn't matter, since the .h file will be imported by the SF recipe.
# But here, I'm only compiling this snippet, and so the order does matter. I
# could include the .h file in the .c file, but I think that could lead to
# issues in the full ART compilation. I don't want to risk that. It gets
# included below anyway.

# some other helpful things
defines_discrete_elts = ["PY_TESTING", "ENRICHMENT_ELEMENTS", "DISCRETE_SN"]
defines_continuous_elts = ["PY_TESTING", "ENRICHMENT_ELEMENTS"]
defines_discrete_no_elts = ["PY_TESTING", "DISCRETE_SN"]
defines_continuous_no_elts = ["PY_TESTING",]

def build_extension(cdef, include_str, name, defines, sources, need_gsl=False):
    ffibuilder = FFI()
    ffibuilder.cdef(cdef)

    # parse the defines into the right format
    macros = [(define, None) for define in defines]
    if need_gsl:
        ffibuilder.set_source(name, include_str, define_macros=macros,
                              sources=sources,
                              libraries=["gsl"])
    else:
        ffibuilder.set_source(name, include_str, define_macros=macros,
                              sources=sources)
    ffibuilder.compile(verbose=True)

    # after this is build, it will have a weird name. We want to rename it so
    # it has a cleaner name for importing
    file_name = name + ".cpython-37m-darwin.so"
    os.rename(file_name, name + ".so")


# ==============================================================================
#
# SN Ia
#
# ==============================================================================
# SNIa will be tested in all four configurations

snia_elts_cdef = """
void detailed_enrichment_init(void);  // from core file
double *sn_ia_core_py(double, double, double, double, double, double);
double get_sn_ia_rate_py(double, double);"""
snia_no_elts_cdef = """
double *sn_ia_core_py(double, double, double, double, double, double);
double get_sn_ia_rate_py(double, double);"""
# the include will be the same for both
snia_include_str = '''
#include <math.h> 
#include "{0}feedback.detailed_enrich.h" 
#include "{0}feedback.snIa-detailed.h"
'''.format(dir)
# and the needed sources
sources = [dir + "feedback.detailed_enrich.c", dir + "feedback.snIa-detailed.c"]

# then we can build things!
build_extension(snia_elts_cdef, snia_include_str, "snia_discrete_elements",
                defines_discrete_elts, sources)
build_extension(snia_no_elts_cdef, snia_include_str, "snia_discrete_no_elements",
                defines_discrete_no_elts, sources)
build_extension(snia_elts_cdef, snia_include_str, "snia_continuous_elements",
                defines_continuous_elts, sources)
build_extension(snia_no_elts_cdef, snia_include_str, "snia_continuous_no_elements",
                defines_continuous_no_elts, sources)

# ==============================================================================
#
# SN II
#
# ==============================================================================
# I'll test SNII with and without my detailed enrichmen prescription

# The functions to include will be the same for both
snii_cdef = """
void detailed_enrichment_init(void);  // from core file
void init_rand(void);
int random_binomial_py(double, unsigned int);
double *get_ejecta_sn_ii_py(double, double, double, double, double);
"""
# the include will be the same for both
snii_include_str = '''
#include "{0}feedback.detailed_enrich.h" 
#include "{0}feedback.snII-detailed.h"
'''.format(dir)
# and the needed sources
sources = [dir + "feedback.detailed_enrich.c", dir + "feedback.snII-detailed.c"]

# then we can build things!
build_extension(snii_cdef, snii_include_str, "snii_discrete_elements",
                defines_discrete_elts, sources, need_gsl=True)
build_extension(snii_cdef, snii_include_str, "snii_discrete_no_elements",
                defines_discrete_no_elts, sources, need_gsl=True)
build_extension(snii_cdef, snii_include_str, "snii_continuous_elements",
                defines_continuous_elts, sources, need_gsl=True)
build_extension(snii_cdef, snii_include_str, "snii_continuous_no_elements",
                defines_continuous_no_elts, sources, need_gsl=True)

# ==============================================================================
#
# AGB
#
# ==============================================================================
# I'll test AGB with and without my detailed enrichment prescription

# The functions to include will be the same for both
agb_elts_cdef = """
void detailed_enrichment_init(void);  // from core file
double *get_ejecta_agb_py(double, double, double, double, double, double, double);
"""
agb_no_elts_cdef = """
void detailed_enrichment_init(void);  // from core file
double *get_ejecta_agb_py(double, double, double, double, double, double, double);
"""
# the include will be the same for both
agb_include_str = '''
#include "{0}feedback.detailed_enrich.h" 
#include "{0}feedback.AGB-detailed.h"
'''.format(dir)
# and the needed sources
sources = [dir + "feedback.detailed_enrich.c", dir + "feedback.AGB-detailed.c"]

# then we can build things!
build_extension(agb_elts_cdef, agb_include_str, "agb_elements",
                defines_discrete_elts, sources)
build_extension(agb_no_elts_cdef, agb_include_str, "agb_no_elements",
                defines_discrete_no_elts, sources)

# ==============================================================================
#
# Winds
#
# ==============================================================================
# I'll test winds with and without my detailed enrichment prescription. The
# results should not change, but it will be a nice consistency check

# The functions to include will be the same for both
winds_cdef = """
void detailed_enrichment_init(void);  // from core file
double get_ejecta_winds_py(double, double, double, double, double, double, double);
double get_cumulative_mass_winds_py(double, double, double, double);
"""
# the include will be the same for both
wind_include_str = '''
#include "{0}feedback.detailed_enrich.h" 
#include "{0}feedback.winds-detailed.h"
'''.format(dir)
# and the needed sources
sources = [dir + "feedback.detailed_enrich.c", dir + "feedback.winds-detailed.c"]

# then we can build things!
build_extension(winds_cdef, wind_include_str, "winds_elements",
                defines_discrete_elts, sources)
build_extension(winds_cdef, wind_include_str, "winds_no_elements",
                defines_discrete_no_elts, sources)

# ==============================================================================
#
# Background detailed_enrichment machinery
#
# ==============================================================================
core_cdef = """
void detailed_enrichment_init(void);

int *find_z_bound_idxs_agb_py(double);
int *find_z_bound_idxs_winds_py(double);
int *find_z_bound_idxs_sn_ii_py(double);
int *find_z_bound_idxs_sn_ia_py(double);
int guess_mass_idx_winds_py(double);
int *find_mass_bound_idxs_agb_py(double);
int *find_mass_bound_idxs_sn_ii_py(double);
int *find_mass_bound_idxs_hn_ii_py(double);
double *get_yields_sn_ia_py(double);
double get_masses_agb(int);
double get_masses_sn_ii(int);
double get_masses_hn_ii(int);
double get_masses_winds(int);
double get_z_winds(int);
double get_z_agb(int);
double get_z_sn_ii(int);
double get_z_sn_ia(int);
double *get_yields_raw_sn_ii_py(double, double);
double *get_yields_raw_hn_ii_py(double, double);
double *get_yields_raw_agb_py(double, double);
double imf_integral_py(double, double);
double interpolate_py(double, double, double, double, double);
double extrapolate_py(double, double, double);
double get_cumulative_mass_winds_py(double, double, double, double);"""


core_include_str = '#include "{}feedback.detailed_enrich.h"'.format(dir)

# The discrete flag has no effect here, so we only need two compilations
build_extension(core_cdef, core_include_str, "core_elts",
                defines_discrete_elts, [dir + "feedback.detailed_enrich.c"])
build_extension(core_cdef, core_include_str, "core_no_elts",
                defines_discrete_no_elts, [dir + "feedback.detailed_enrich.c"])

