#!/bin/bash

# Set default values for source and destination folders
unzip /content/rice-image-dataset.zip -d ./dataset
source_folder="./dataset"
destination_folder="/content/beyond-the-jupyter-notebook/data/processed"

# Run the Python script to copy folders
python /content/beyond-the-jupyter-notebook/src/utils/preprocessing/preprocess_data.py "$source_folder" "$destination_folder"
