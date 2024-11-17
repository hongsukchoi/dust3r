#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# activate conda environment
export PYTHONPATH=/home/hongsuk/projects/dust3r
source /home/hongsuk/anaconda3/etc/profile.d/conda.sh
conda activate dust3r

python hongsuk_egohumans_align_dust3r_hmr2hamer_scale1.py \
    --sel-big-seqs 07_tennis \
    --sel-small-seq-range 6 13