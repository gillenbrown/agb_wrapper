home_dir = /Users/gillenb/code/agb_wrapper/
source_dir = /Users/gillenb/code/art_cluster/src/sf/models/

file_name = feedback.detailed_enrich
source_files = $(source_dir)$(file_name).c $(source_dir)$(file_name).h

py_build_file = agb_build.py
output_prefix = _agb
# ^ found in $(py_build_file)

target_names = $(file_name).o $(output_prefix).c $(output_prefix).cpython-35m-darwin.so $(output_prefix).o  

$(info $(target_names))

targets_home = $(foreach item,$(target_names),$(home_dir)$(item))
targets_source = $(foreach item,$(target_names),$(source_dir)$(item))

$(targets_home): $(source_files) $(home_dir)$(py_build_file)
	cp $(home_dir)$(py_build_file) $(source_dir)$(py_build_file)
	cd $(source_dir) && python $(source_dir)$(py_build_file)
	rm $(source_dir)$(py_build_file)
	mv $(targets_source) $(home_dir)

clean:
	rm $(targets_home)