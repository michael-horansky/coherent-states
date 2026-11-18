#!/bin/bash

# This script downloads the outputs from Aire to local (temp)
#   -recurses through all subdirectories
#   -does not overwrite files which haven't been updated
#   -does not delete any files
#   -shows progress for each file
rsync -avh \
    --progress \
    --dry-run \
    aire:/users/vzwt2244/coherent-states/mol_el_struct/outputs/ \
    /home/michal/Documents/APtemp/
