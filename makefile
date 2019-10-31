home_dir = /Users/gillenb/google_drive/research/agb/agb_wrapper/
source_dir = /Users/gillenb/code/art_cluster/src/sf/models/

file_extensions = c h
prefix_to_files = $(foreach ext,$(file_extensions),$(source_dir)$(1).$(ext))

sources_core = $(call prefix_to_files,feedback.detailed_enrich)
sources_snia = $(call prefix_to_files,feedback.snIa-detailed) $(sources_core)
sources_snii = $(call prefix_to_files,feedback.snII-detailed) $(sources_core)
sources_agb = $(call prefix_to_files,feedback.AGB-detailed) $(sources_core)
sources_wind = $(call prefix_to_files,feedback.winds-detailed) $(sources_core)

py_build_file = $(home_dir)build.py

# There are a bazillion names to check against, so here I'll only copy the
# first and last created by the build script
output_core = core_enrich_ia_elts_cluster_discrete core_none
output_snia = snia_enrich_ia_elts_cluster_discrete snia_none
output_snii = snii_enrich_ia_elts_cluster_discrete snii_none
output_agb  = agb_enrich_ia_elts_cluster_discrete  agb_none
output_wind = wind_enrich_ia_elts_cluster_discrete wind_none

outputs_to_targets = $(foreach item,$(1),$(item).so)

targets_core = $(call outputs_to_targets,$(output_core))
targets_snia = $(call outputs_to_targets,$(output_snia))
targets_snii = $(call outputs_to_targets,$(output_snii))
targets_agb  = $(call outputs_to_targets,$(output_agb))
targets_wind = $(call outputs_to_targets,$(output_wind))

$(info $(targets_core))
all: $(targets_core) $(targets_snia) $(targets_snii) $(targets_agb) $(targets_wind)

$(targets_core): $(sources_core) $(py_build_file)
	cd $(home_dir)
	python $(py_build_file) core

$(targets_snia): $(sources_snia) $(py_build_file)
	cd $(home_dir)
	python $(py_build_file) snia

$(targets_snii): $(sources_snii) $(py_build_file)
	cd $(home_dir)
	python $(py_build_file) snii

$(targets_agb): $(sources_agb) $(py_build_file)
	cd $(home_dir)
	python $(py_build_file) agb

$(targets_wind): $(sources_wind) $(py_build_file)
	cd $(home_dir)
	python $(py_build_file) winds

clean:
	cd $(home_dir)
	rm *.c
	rm *.o
	rm *.so
