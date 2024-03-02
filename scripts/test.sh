#!/bin/bash

# Author: Taha Bouhsine
# Email: contact@tahabouhsine.com
# Description: 
# This script runs the test.py script for image classification training 
# with predefined or customizable settings.

# Set script arguments
MODEL_PATH="./last_exp_model.keras"
# Path to the test dataset
TEST_DATASET_PATH="./test"
dataset_path='./test' # Path to the dataset directory

# Configuration parameters
NUM_CLASSES=5
IMG_HEIGHT=224
IMG_WIDTH=224
SEED=42
GPU="1"
BATCH_SIZE=256
NUM_IMG_LIM=100000

# Run test.py with the specified arguments
nohup python /workspaces/beyond-the-jupyter-notebook/src/test.py --model_path $MODEL_PATH \
            --dataset_path $dataset_path \
            --num_classes $NUM_CLASSES \
            --img_height $IMG_HEIGHT \
            --img_width $IMG_WIDTH \
            --seed $SEED \
            --gpu $GPU \
            --batch_size $BATCH_SIZE \
            --num_img_lim $NUM_IMG_LIM > test_logs.out
