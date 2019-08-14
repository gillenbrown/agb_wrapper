from cffi import FFI
ffibuilder = FFI()

# any functions that you want to import in Python must be here. 
ffibuilder.cdef(
"""
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
double *get_yields_sn_ii_py(double, double, double, double, double);
double *get_yields_raw_sn_ii_py(double, double);
double *get_yields_raw_hn_ii_py(double, double);
double *get_yields_raw_agb_py(double, double);
double imf_integral_py(double, double);
double extrapolate_py(double, double, double);"""
)
# in the .c file, order does matter for this compilation. In the full context of
# ART it doesn't matter, since the .h file will be imported by the SF recipe.
# But here, I'm only compiling this snippet, and so the order does matter. I 
# could include the .h file in the .c file, but I think that could lead to 
# issues in the full ART compilation. I don't want to risk that. It gets 
# included below anyway.
dir = "/Users/gillenb/code/art_cluster/src/sf/models/"

ffibuilder.set_source("art_enrich",  # name of the output C extension
                      '#include "{}/feedback.detailed_enrich.h"'.format(dir),
                      define_macros=[("PY_TESTING", None),
                                     ("ENRICHMENT_ELEMENTS", None)],
                      sources=[dir + "feedback.detailed_enrich.c"])

ffibuilder.compile(verbose=True)
