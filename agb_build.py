from cffi import FFI
ffibuilder = FFI()

# any functions that you want to import in Python must be here. 
ffibuilder.cdef("void init_agb(void);"
                "double *get_ejecta_rate(double, double);"
                "int *find_z_bound_idxs(double);"
                "int *find_age_bound_idxs(double);"
                "double read_in_check(void);"
                "double get_ages(int);"
                "int guess_age_idx(double);"
                "double *get_ejecta_timestep(double, double, double, double);")
# in the .c file, order does matter for this compilation. In the full context of
# ART it doesn't matter, since the .h file will be imported by the SF recipe.
# But here, I'm only compiling this snippet, and so the order does matter. I 
# could include the .h file in the .c file, but I think that could lead to 
# issues in the full ART compilation. I don't want to risk that. It gets 
# included below anyway. 

ffibuilder.set_source("art_enrich",  # name of the output C extension
                      '#include "feedback.detailed_enrich.h"',
                      define_macros=[("TEST_FLAG", None)],    
                      sources=['feedback.detailed_enrich.c'])   

ffibuilder.compile(verbose=True)
