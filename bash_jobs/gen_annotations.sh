#!/bin/bash

# Define the list of directories
base="b1/2_validated_w_sugs"
directories=("proc" "med" "dis" "symp")
lang="nl"

# Loop through each directory
for dir in "${directories[@]}"; do

    python cardioner/pubscience/ner_caster.py \
        --txt_dir "assets/$base/$lang/$dir/txt" \
        --ann_dir "assets/$base/$lang/$dir/ann" \
        --out_path "assets/$base/$lang/$dir/annotations.jsonl"
done
