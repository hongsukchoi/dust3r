#!/bin/bash

# activate conda environment
export PYTHONPATH=/home/hongsuk/projects/dust3r
source /home/hongsuk/anaconda3/etc/profile.d/conda.sh
conda activate dust3r


# Read the input file line by line
while IFS=',' read -r dust3r_path prefit_path input_data_path; do
    # Skip empty lines
    if [ -z "$dust3r_path" ]; then
        continue
    fi

    # Run the Python script with the paths
    python hongsuk_egoexo_test.py \
        --dust3r-network-output-path "$dust3r_path" \
        --vitpose-and-gt-path "$input_data_path" \
        --output-dir "./outputs/egoexo/nov11/sota_comparison_trial1" \
        --vis 

done < egoexo4d_result_paths_val.txt
