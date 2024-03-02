import os
import shutil
import argparse

def copy_folder(source_folder, destination_folder):
    try:
        shutil.copytree(source_folder, destination_folder)
        print(f"Folder '{source_folder}' copied to '{destination_folder}' successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a folder and its sub-contents from source to destination.")
    parser.add_argument("source", help="Source folder path")
    parser.add_argument("destination", help="Destination folder path")
    args = parser.parse_args()

    source_folder = args.source
    destination_folder = args.destination

    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
    else:
        copy_folder(source_folder, destination_folder)
