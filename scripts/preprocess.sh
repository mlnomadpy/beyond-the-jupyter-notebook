#!/bin/bash

# Set default values for source and destination folders
source_folder="/kaggle/input/rice-image-dataset/Rice_Image_Dataset"
destination_folder="/kaggle/working/beyond-the-jupyter-notebook/data/processed"

# Run the Python script to copy folders
python /kaggle/working/beyond-the-jupyter-notebook/src/utils/preprocessing/preprocess_data.py "$source_folder" "$destination_folder"
