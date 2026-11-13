#!/bin/bash

# This script updates the project_management_utils folder in Aire from local
#   -the package is installed as an editable installation in the conda environment mol-el-struct-env
#   -no manual updating needed on Aire's side
rsync -avh --delete \
    /home/michal/Documents/Projects/coherent-states/project_management_utils \
    aire:/users/vzwt2244/coherent-states/
