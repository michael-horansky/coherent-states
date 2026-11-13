#!/bin/bash

decode_index() {
    local idx=$1
    shift

    local dims=("$@")
    local n=${#dims[@]}

    local coords=()

    for ((i=n-1; i>=0; i--)); do
        coords[$i]=$(( idx % dims[$i] ))
        idx=$(( idx / dims[$i] ))
    done

    echo "${coords[@]}"
}
