home_dir = /Users/gillenb/google_drive/research/agb/agb_wrapper/
source_dir = /Users/gillenb/code/art_cluster/src/sf/models/

file_name = feedback.detailed_enrich
source_files = $(source_dir)$(file_name).c $(source_dir)$(file_name).h

py_build_file = agb_build.py
output_prefix = art_enrich
# ^ found in $(py_build_file)

lib_ext = cpython-37m-darwin
libfile = $(output_prefix).$(lib_ext).so
final_libfile = $(output_prefix).so

target_names = $(file_name).o $(output_prefix).c $(output_prefix).o $(final_libfile)
compilation_outputs = $(file_name).o $(output_prefix).c $(output_prefix).o $(libfile)

targets_home = $(foreach item,$(target_names),$(home_dir)$(item))
targets_source = $(foreach item,$(compilation_outputs),$(source_dir)$(item))

$(targets_home): $(source_files) $(home_dir)$(py_build_file)
	cp $(home_dir)$(py_build_file) $(source_dir)$(py_build_file)
	cd $(source_dir) && python $(source_dir)$(py_build_file)
	rm $(source_dir)$(py_build_file)
	mv $(targets_source) $(home_dir)
	mv $(libfile) $(final_libfile)

clean:
	rm $(targets_home)