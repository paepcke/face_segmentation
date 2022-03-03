#!/usr/bin/env bash
PROJ_ROOT=`pwd`
time python ${PROJ_ROOT}/face-segmentation/training/main.py --batchsize=32 --gpuId=1 dataset_subsamples exp_results
