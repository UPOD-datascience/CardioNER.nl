#!/bin/bash

# Define the list of directories
directories=("proc" "med" "dis" "symp")
lang="es"

# Loop through each directory
for dir in "${directories[@]}"; do

    python cardioner/pubscience/ner_caster.py \
        --txt_dir "assets/train/$lang/$dir/txt" \
        --ann_dir "assets/train/$lang/$dir/ann" \
        --out_path "assets/train/$lang/$dir/annotations.jsonl"

    python cardioner/pubscience/ner_caster.py \
        --txt_dir "assets/validation/$lang/$dir/txt" \
        --ann_dir "assets/validation/$lang/$dir/ann" \
        --out_path "assets/validation/$lang/$dir/annotations.jsonl"

    python cardioner/pubscience/ner_caster.py \
        --txt_dir "assets/test/$lang/$dir/txt" \
        --ann_dir "assets/test/$lang/$dir/ann" \
        --out_path "assets/test/$lang/$dir/annotations.jsonl"
done