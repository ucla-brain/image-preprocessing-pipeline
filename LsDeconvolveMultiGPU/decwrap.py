import os
import argparse
import subprocess


def construct_cache_drive_folder_name(brain_name, lambda_ex, lambda_em):
    """Construct name using brain name and wavelengths."""
    return f"cache_deconvolution_{brain_name}_Ex_{lambda_ex}_Em_{lambda_em}"


def main():
    magnifications = {'15x': (400, 1200), '9x': (700, 1400), '4x': (1800, 1800)}
    lambda_ex_choices = [488, 561, 642]
    lambda_em_choices = [525, 600, 690]

    parser = argparse.ArgumentParser(
        description='Python wrapper for MATLAB deconvolution.')
    parser.add_argument('folderPath', type=str, 
                        help='Path to the folder containing the images.')
    parser.add_argument('magnification', type=str, 
                        choices=magnifications.keys(), 
                        help='Magnification used, either 15x, 9x, or 4x.')
    parser.add_argument('numit', type=int, 
                        choices=range(1, 51), metavar="[1-50]", 
                        help='Number of iterations, between 1 and 50.')
    parser.add_argument('lambda_ex', type=int, 
                        choices=lambda_ex_choices, 
                        help='Excitation wavelength.')
    parser.add_argument('lambda_em', type=int, 
                        choices=lambda_em_choices, 
                        help='Emission wavelength.')
    parser.add_argument('brain_name', type=str, 
                        help='Brain name for cache path construction.')
    args = parser.parse_args()

    if not (os.path.exists(args.folderPath) and os.path.isdir(args.folderPath)):
        print(f"ERROR: {args.folderPath} does not exist or is not a folder")
        exit(1)
    
    # set dxy and dz from the map based on magnification
    dxy, dz = magnifications[args.magnification]
    
    # Construct cache drive path
    cache_drive_folder = construct_cache_drive_folder_name(
        args.brain_name, str(args.lambda_ex), str(args.lambda_em))

    # Match lambda_ex and lambda_em by their positions in the lists
    if lambda_ex_choices.index(args.lambda_ex) != lambda_em_choices.index(args.lambda_em):
        print(f"ERROR: Ex {args.lambda_ex} and Em {args.lambda_em} wavelengths do not match.")
        exit(1)

    # Construct the MATLAB command
    matlab_command = (f"matlab -nodisplay -r \"deconvolve('{args.folderPath}', '{dxy}', '{dz}', "
                      f"'{args.numit}', '{args.lambda_ex}', '{args.lambda_em}', "
                      f"'{cache_drive_folder}'); exit;\"")
    
    try:
        print("decwrap.py running deconvolution...")
        subprocess.run(matlab_command, shell=True, check=True)
        print("decwrap.py done.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the deconvolution: {e}")


if __name__ == "__main__":
    main()
