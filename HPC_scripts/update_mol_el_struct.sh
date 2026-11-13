#!/bin/bash

# This script updates the mol_el_struct script directory in Aire from local
#   -recurses through all subdirectories
#   -ignores directories and files based on the UMES_exc.txt ignore file
#   -deletes files which do not exist on local
rsync -avh --delete \
    --exclude-from=UMES_exc.txt \
    /home/michal/Documents/Projects/coherent-states/mol_el_struct/ \
    aire:/users/vzwt2244/coherent-states/mol_el_struct/
