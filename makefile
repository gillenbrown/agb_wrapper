home_dir = /Users/gillenb/google_drive/research/agb/agb_wrapper/
source_dir = /Users/gillenb/code/art_cluster/src/sf/models/

file_names = feedback.detailed_enrich \
             feedback.snIa-detailed \
             feedback.snII-detailed \
             feedback.AGB-detailed
source_files_c = $(foreach item,$(file_names),$(source_dir)$(item).c)
source_files_h = $(foreach item,$(file_names),$(source_dir)$(item).h)
source_files = $(source_files_c) $(source_files_h)

py_build_file = build.py

output_names = snia_discrete_elements \
               snia_discrete_no_elements \
               snia_continuous_elements \
               snia_continuous_no_elements \
               snii_discrete_elements \
               snii_discrete_no_elements \
               snii_continuous_elements \
               snii_continuous_no_elements \
               agb_elements \
               agb_no_elements

targets = $(foreach item,$(output_names),$(item).so)

all: $(targets)

$(targets): $(source_files) $(home_dir)$(py_build_file)
	cd $(home_dir)
	python $(py_build_file)

clean:
	cd $(home_dir)
	rm *.c
	rm *.o
	rm *.so
