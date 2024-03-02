#!/bin/bash

# Set default values for source and destination folders
source_folder="/kaggle/input/rice-image-dataset/Rice_Image_Dataset"
destination_folder="/kaggle/working/beyond-the-jupyter-notebook/data/raw"

# Run the Python script to copy folders
python script_name.py "$source_folder" "$destination_folder"
