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
int guess_age_idx_agb_py(double);
int guess_age_idx_winds_py(double);
int guess_age_idx_sn_ii_py(double);
int *find_age_bound_idxs_agb_py(double);
int *find_age_bound_idxs_winds_py(double);
int *find_age_bound_idxs_sn_ii_py(double);
double *get_ejecta_rate_agb_py(double, double);
double *get_ejecta_rate_winds_py(double, double);
double *get_ejecta_rate_sn_ii_py(double, double);
double *get_ejecta_timestep_agb_py(double, double, double, double);
double *get_ejecta_timestep_winds_py(double, double, double, double);
double *get_ejecta_timestep_snii_py(double, double, double, double);
double *get_yields_sn_ia_py(double);
double get_ages_agb(int);
double get_ages_sn_ii(int);
double get_ages_winds(int);
double get_z_winds(int);
double get_z_agb(int);
double get_z_sn_ii(int);
double get_z_sn_ia(int);"""
)
# in the .c file, order does matter for this compilation. In the full context of
# ART it doesn't matter, since the .h file will be imported by the SF recipe.
# But here, I'm only compiling this snippet, and so the order does matter. I 
# could include the .h file in the .c file, but I think that could lead to 
# issues in the full ART compilation. I don't want to risk that. It gets 
# included below anyway. 

ffibuilder.set_source("art_enrich",  # name of the output C extension
                      '#include "feedback.detailed_enrich.h"',
                      define_macros=[("PY_TESTING", None),
                                     ("ENRICHMENT_ELEMENTS", None)],
                      sources=['feedback.detailed_enrich.c'])   

ffibuilder.compile(verbose=True)
