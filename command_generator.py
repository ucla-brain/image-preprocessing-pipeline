from colorama import Fore, Style
import subprocess
import sys
import os
import re


def validate_paths(composite_path, imaris_output_path, fnt_output_path, fnt_channel_path):
    if imaris_output_path[-3:] != 'ims':
        print(Fore.RED + 'ERROR: --imaris output should have.ims extension' + Style.RESET_ALL)
        sys.exit(1)
    # if composite_path:

    if fnt_output_path[-1:] != '\\':
        fnt_output_path += '\\'
    

    return 

def build_imaris_conversion_cmd(composite_path, imaris_output_path):
    return

def build_fnt_conversion_cmd(fnt_output_path, fnt_channel_path):
    return

def build_merge_cmd(composite_path, imaris_output_path):
    return

def compute_node_access(user, passwd):
    return 0;

def main():
    GOALS = [0, 1, 2, 3]
    COMPOSITE_PATH = ''
    IMARIS_OUTPUT_PATH = ''
    FNT_OUTPUT_PATH = ''
    FNT_CHANNEL_PATH = ''
    X_VOXEL = ''
    Y_VOXEL = ''
    Z_VOXEL = ''
    mergeChannelsCMD = ''
    imarisConversionCMD = ''
    fntConversionCMD = ''
    localCredentialsPath = 'tmp/creds.txt'

    # TODO: Update this script to also perform FNT conversions (will read from qnap3 temp drive and use a tiff channel), creat documentation, and update README.md

    print(Fore.BLUE + '\nRunning command_generator.py to generate commands for merging and or Imaris conversion scripts on Windows...\n' + Style.RESET_ALL)
    print('Enter [0] if target command is merging + imaris conversion + FNT conversion')
    print('Enter [1] if target command is merging only')
    print('Enter [2] if target command is imaris conversion only')
    print('Enter [3] if target command is FNT conversion only\n')

    goal = int(input(Fore.MAGENTA + "Goal (0, 1, 2, or 3): " + Style.RESET_ALL))

    if goal not in GOALS:
        print(Fore.RED + 'Invalid goal entered: ' + str(goal) + Style.RESET_ALL)
        sys.exit()

    if goal in (0,3):
        print('\nBuilding FNT conversion command...\n')
        # TODO: add mitch login step here...
        # if saved pass file not found
        # check if localCredentialsPath file exists
        if (os.path.isfile(localCredentialsPath)):
            userName = open(localCredentialsPath, 'r').read().split(':')[0]
            password = open(localCredentialsPath, 'r').read().split(':')[1]
            creds_choice = input(f'Credentials to access BMAP cluster found. Proceed with "{userName}" credentials? [0 for yes, 1 for no]: ')
        
        if (not (os.path.isfile(localCredentialsPath))) or (creds_choice == '1'):
            print('FNT conversion requires access to BMAP cluster. Please enter your username and password to access the BMAP cluster to perform FNT conversion (Will be saved for future use): ')
            userName = input('Username: ')
            password = input('Password: ')
            # create a file to store credentials with the text added in userName:password
            with open(localCredentialsPath, 'w') as credsFile:
                credsFile.write(f'{userName}:{password}')

        # execute FNT conversion on compute node here


        if (goal == 3):
            FNT_CHANNEL_PATH = input(Fore.MAGENTA + "Enter the path to the tiff channel to convert to FNT (W:\path\Ex_xxx_Chx): " + Style.RESET_ALL)
            IMARIS_OUTPUT_PATH = input(Fore.MAGENTA + "Enter the destination path for the converted imaris image (include imaris name in path: SWxxxxxx-xx.ims). Required for FNT script: " + Style.RESET_ALL)
            X_VOXEL = input(Fore.MAGENTA +"Enter x voxel value in microns (x.x um): " + Style.RESET_ALL)
            Y_VOXEL = input(Fore.MAGENTA +"Enter y voxel value in microns (x.x um): " + Style.RESET_ALL)
            Z_VOXEL = input(Fore.MAGENTA +"Enter z voxel value in microns (x.x um): " + Style.RESET_ALL)
            tgt_channel = 0
        else:
            print(Fore.MAGENTA + "Use red, green, or blue channel for FNT conversion: " + Style.RESET_ALL)
            print('Enter [0] for '+ Fore.RED + 'red' + Style.RESET_ALL + f' channel: {redChannelPath}')
            print('Enter [1] for '+ Fore.GREEN + 'green' + Style.RESET_ALL + f' channel: {greenChannelPath}')
            print('Enter [2] for '+ Fore.BLUE + 'blue' + Style.RESET_ALL + f' channel: {blueChannelPath}')
            tgt_channel = int(input(Fore.MAGENTA + "Target channel (0, 1, or 2): " + Style.RESET_ALL))
            if (tgt_channel == 0):
                FNT_CHANNEL_PATH = redChannelPath
            elif (tgt_channel == 1):
                FNT_CHANNEL_PATH = greenChannelPath
            elif (tgt_channel == 2):
                FNT_CHANNEL_PATH = blueChannelPath
            else:
                print(Fore.RED + 'Invalid channel entered: ' + str(tgt_channel) + Style.RESET_ALL)
                sys.exit()
        
        match = re.search(f'Ex_.*', str(FNT_CHANNEL_PATH))
        channel_folder = match.group()
        FNT_OUTPUT_PATH = input(Fore.MAGENTA + "Enter the destination path for the converted FNT output ('Ex_xxx_Chx_FNT_tiff' will be added automatically to output path): " + Style.RESET_ALL)


        fntConversionCMD = f"python .\\convert.py -i {IMARIS_OUTPUT_PATH} -t {FNT_OUTPUT_PATH}\{channel_folder}_FNT_tiff\ -fnt {FNT_OUTPUT_PATH}\{channel_folder}_FNT\ --channel {FNT_CHANNEL_PATH} -dx {X_VOXEL} -dy {Y_VOXEL} -dz {Z_VOXEL}"

    if goal in (0, 1):
        print ('\nBuilding merge command. Enter individual RGB tiff channel paths. Use 0 for empty channel...\n')
        redChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.RED + "RED" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        greenChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.GREEN + "GREEN" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        blueChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.BLUE + "BLUE" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        COMPOSITE_PATH = input(Fore.MAGENTA + "Enter the destination path to save the composite (merged tiffs) folder to save to ('\\composite\\' will be added to this path): " + Style.RESET_ALL)
        COMPOSITE_PATH = COMPOSITE_PATH + '\\composite\\'
        
        mergeChannelsCMD = f"python .\\merge_channels.py" + (f" --red {redChannelPath}" if (redChannelPath != '0') else "") + (f" --green {greenChannelPath}" if (greenChannelPath != '0') else "") + (f" --blue {blueChannelPath}" if (blueChannelPath != '0') else "") + (f" --output_path {COMPOSITE_PATH}") 

    if goal in (0,2):
        print('\nBuilding imaris conversion command...\n')
        if (goal == 2):
            COMPOSITE_PATH = input(Fore.MAGENTA + "Enter the path to the composite (merged/RGB) tiff channel: " + Style.RESET_ALL)

        IMARIS_OUTPUT_PATH = input(Fore.MAGENTA + "Enter the destination path for the converted imaris image (include imaris name in path: SWxxxxxx-xx.ims): " + Style.RESET_ALL)
        X_VOXEL = input(Fore.MAGENTA +"Enter x voxel value in microns (x.x um): " + Style.RESET_ALL)
        Y_VOXEL = input(Fore.MAGENTA +"Enter y voxel value in microns (x.x um): " + Style.RESET_ALL)
        Z_VOXEL = input(Fore.MAGENTA +"Enter z voxel value in microns (x.x um): " + Style.RESET_ALL)
        imarisConversionCMD = f"python .\\convert.py -i {COMPOSITE_PATH} -o {IMARIS_OUTPUT_PATH} -dx {X_VOXEL} -dy {Y_VOXEL} -dz {Z_VOXEL}"
    
    
    # TODO: perform validations of the variables that require folder slashes... do this for all three
    COMPOSITE_PATH, IMARIS_OUTPUT_PATH, FNT_OUTPUT_PATH, FNT_CHANNEL_PATH = validate_paths(COMPOSITE_PATH, IMARIS_OUTPUT_PATH, FNT_OUTPUT_PATH, FNT_CHANNEL_PATH)

    print('\nCommand generated below. Copy, paste into Anaconda Prompt, and run to begin processing:\n')
    print(Fore.GREEN + (mergeChannelsCMD if (goal == 0) or (goal == 1) else "") + (" && " if goal == 0 else "") + (fntConversionCMD if (goal == 0) or (goal == 3) else "") + (" && " if goal == 0 else "") + (imarisConversionCMD if (goal == 0) or (goal == 2) else "") + Style.RESET_ALL + '\n')

if __name__ == "__main__":
    main()
