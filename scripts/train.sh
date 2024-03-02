#!/bin/bash

# Author: Taha Bouhsine
# Email: contact@tahabouhsine.com
# Description: 
# This script runs the train.py script for image classification training 
# with predefined or customizable settings.

# Default parameters
learning_rate=0.001
batch_size=32
epochs=100
img_height=224
img_width=224
seed=420
gpu='0'
dataset_path='/content/beyond-the-jupyter-notebook/data/processed/Rice_Image_Dataset' # Path to the dataset directory
test_dataset_path='/content/beyond-the-jupyter-notebook/data/processed/Rice_Image_Dataset' # Path to the test dataset directory
num_img_lim=100000 # Number of images per class
val_split=0.2 # Validation split
n_cross_validation=5 # Number of bins for cross-validation
num_classes=5 # Number of classes
trainable_epochs=0 # Number of epochs before the backbone becomes trainable
project='beyond-jup'
entity='skywolf'
model_name='BasicCNN'

# Running the training script with parameters
nohup python /content/beyond-the-jupyter-notebook/src/train.py \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --epochs $epochs \
  --dataset_path $dataset_path \
  --img_height $img_height \
  --img_width $img_width \
  --seed $seed \
  --gpu $gpu \
  --test_dataset_path $test_dataset_path \
  --num_img_lim $num_img_lim \
  --val_split $val_split \
  --num_classes $num_classes \
  --trainable_epochs $trainable_epochs \
  --project $project \
  --entity $entity \
  --model_name $model_name \
  > train_logs.out

