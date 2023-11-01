#!/usr/bin/env python3

import os
import sys
import platform
import subprocess
import argparse

def main():
    # Set up argparse for command line arguments
    parser = argparse.ArgumentParser(
        description="Wrapper script for fnt-to-swc conversion"
    )
    parser.add_argument("folder", 
                        help="Directory containing .fnt files")
    parser.add_argument("-f", "--force",
                        action="store_true",
                        help="Force overwrite of existing .swc files")
    args = parser.parse_args()

    # 1. Check if fnt-to-swc or fnt-to-swc.exe exists in the current folder.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if platform.system() == "Windows":
        exe_name = "fnt-to-swc.exe"
    else:
        exe_name = "fnt-to-swc"

    exe_path = os.path.join(script_dir, exe_name)
    if not os.path.exists(exe_path):
        print(f"Executable {exe_name} not found in the script's directory.")
        sys.exit(1)

    # 2. Find all the .fnt files in the given folder.
    fnt_files = [f for f in os.listdir(args.folder) if f.endswith('.fnt')]
    if not fnt_files:
        print("No .fnt files found in the provided directory.")
        sys.exit(1)

    print("Found the following .fnt files:")
    for f in fnt_files:
        print(f" - {f}")

    # 3. Convert each .fnt file to .swc as required.
    for fnt_file in fnt_files:
        base_name = os.path.splitext(fnt_file)[0]
        swc_file = os.path.join(args.folder, f"{base_name}.swc")

        if os.path.exists(swc_file) and not args.force:
            print(f"SWC file {base_name}.swc already exists. Skipping...")
            continue

        fnt_path = os.path.join(args.folder, fnt_file)
        subprocess.run([exe_path, fnt_path, swc_file])

        if os.path.exists(swc_file):
            print(f"Generated {base_name}.swc from {fnt_file}.")
        else:
            print(f"Failed to generate {base_name}.swc from {fnt_file}.")

if __name__ == "__main__":
    main()
