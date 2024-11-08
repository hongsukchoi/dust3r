#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

# Activate conda environment
source activate dust3r

python hongsuk_egohumans_align_dust3r_hmr2hamer.py \
    --sel-big-seqs 01_tagging \
    --sel-small-seq-range 1 1