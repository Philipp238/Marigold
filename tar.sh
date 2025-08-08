#!/bin/sh

cd /scratch/scholl/diffusion/marigold_data/hypersim/processed/train
tar -cf ../../hypersim_processed_train.tar .
cd /scratch/scholl/diffusion/marigold_data/hypersim/processed/val
tar -cf ../../hypersim_processed_val.tar .
cd /scratch/scholl/diffusion/marigold_data/hypersim/processed/test
tar -cf ../../hypersim_processed_test.tar .