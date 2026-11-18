#!/bin/bash

# This script downloads the outputs from Aire to local (temp)
#   -only downloads outputs with base dist
#   -ignores molecule and self analysis
#   -recurses through all subdirectories
#   -does not overwrite files which haven't been updated
#   -does not delete any files
#   -shows progress for each file
rsync -avh \
    --exclude='/*_dist=1.0/molecule/' \
    --exclude='/*_dist=1.0/self_analysis/' \
    --include='/*_dist=1.0/' \
    --exclude='/*/' \
    --progress \
    aire:/users/vzwt2244/coherent-states/mol_el_struct/outputs/ \
    /home/michal/Documents/APtemp/
