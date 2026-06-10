#!/usr/bin/env python3
import glob
import os
import subprocess
from pathlib import Path


def convert_fvecs_to_bin(fvecs_file, bin_file, data_type="float"):
    cmd = ["../utils/build/fvecs_to_bin", data_type, fvecs_file, bin_file]
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully converted: {fvecs_file} -> {bin_file}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed for {fvecs_file}: {e}")


def process_directory(data_dir):
    fvecs_files = glob.glob(os.path.join(data_dir, "**/*.fvecs"), recursive=True)

    if not fvecs_files:
        print(f"No .fvecs files found in {data_dir}")
        return

    print(f"Found {len(fvecs_files)} .fvecs files")

    for fvecs_file in fvecs_files:
        bin_file = fvecs_file.replace(".fvecs", ".bin")

        if os.path.exists(bin_file):
            print(f"Skipping existing file: {bin_file}")
            continue

        os.makedirs(os.path.dirname(bin_file), exist_ok=True)
        convert_fvecs_to_bin(fvecs_file, bin_file)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(script_dir)

    if not os.path.exists("../utils/build/fvecs_to_bin"):
        print("Error: fvecs_to_bin executable not found")
        print("Please make sure you have compiled the code in utils directory")
        return

    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    if not os.path.exists(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        return

    print(f"Starting to process directory: {data_dir}")
    process_directory(data_dir)
    print("Processing completed")


if __name__ == "__main__":
    main()
