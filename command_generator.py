from colorama import Fore, Style
import subprocess
import pathlib
import sys
import os
import re

def linux_path(path):
    windows_linux_drive_mappings = {
        'Z:\\': '/panfs/dong/',
        'Y:\\': '/qnap/',
        'X:\\': '/qnap2/',
        'W:\\': '/qnap3/data/',
    }
    l_path = pathlib.PureWindowsPath(path)
    tmp = l_path.parts
    if tmp[0] in windows_linux_drive_mappings:
        newDrive = windows_linux_drive_mappings[tmp[0]]
    # print(newDrive)
    l_path = pathlib.PurePosixPath(newDrive, *tmp[1:])
    # print(l_path)
    return l_path

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
    GOALS = [0, 1, 2, 3, 4]
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
    print('Enter [3] if target command is FNT conversion only')
    print('Enter [4] if target command is merging + imaris conversion\n')

    goal = int(input(Fore.MAGENTA + "Goal (0, 1, 2, 3, or 4): " + Style.RESET_ALL))

    if goal not in GOALS:
        print(Fore.RED + 'Invalid goal entered: ' + str(goal) + Style.RESET_ALL)
        sys.exit()

    if goal in (0, 1, 4):
        print ('\nBuilding merge command. Enter individual RGB tiff channel paths. Use 0 for empty channel...\n')
        redChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.RED + "RED" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        greenChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.GREEN + "GREEN" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        blueChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.BLUE + "BLUE" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        COMPOSITE_PATH = input(Fore.MAGENTA + "Enter the destination path to save the composite (merged tiffs) folder to save to ('\\composite\\' will be added to this path): " + Style.RESET_ALL)
        COMPOSITE_PATH = COMPOSITE_PATH + '\\composite\\'
        
        mergeChannelsCMD = f"python .\\merge_channels.py" + (f" --red {redChannelPath}" if (redChannelPath != '0') else "") + (f" --green {greenChannelPath}" if (greenChannelPath != '0') else "") + (f" --blue {blueChannelPath}" if (blueChannelPath != '0') else "") + (f" --output_path {COMPOSITE_PATH}") 

    if goal in (0, 2, 4):
        print('\nBuilding imaris conversion command...\n')
        if (goal == 2):
            COMPOSITE_PATH = input(Fore.MAGENTA + "Enter the path to the composite (merged/RGB) tiff channel: " + Style.RESET_ALL)

        IMARIS_OUTPUT_PATH = input(Fore.MAGENTA + "Enter the destination path for the converted imaris image (include imaris name in path: SWxxxxxx-xx.ims): " + Style.RESET_ALL)
        X_VOXEL = input(Fore.MAGENTA +"Enter x voxel value in microns (x.x um): " + Style.RESET_ALL)
        Y_VOXEL = input(Fore.MAGENTA +"Enter y voxel value in microns (x.x um): " + Style.RESET_ALL)
        Z_VOXEL = input(Fore.MAGENTA +"Enter z voxel value in microns (x.x um): " + Style.RESET_ALL)
        imarisConversionCMD = f"python .\\convert.py -i {COMPOSITE_PATH} -o {IMARIS_OUTPUT_PATH} -dx {X_VOXEL} -dy {Y_VOXEL} -dz {Z_VOXEL}"
    
    if goal in (0,3):
        print('\nBuilding FNT conversion command...\n')
        print('The FNT command will be Generated at the end and another terminal window will be opened. Enter your BMAP password and copy and paste the command into the terminal window to perform the FNT conversion.\n')

        if (goal == 3):
            FNT_CHANNEL_PATH = input(Fore.MAGENTA + "Enter the path to the tiff channel to convert to FNT (W:\path\Ex_xxx_Chx): " + Style.RESET_ALL)
            # IMARIS_OUTPUT_PATH = input(Fore.MAGENTA + "Enter the destination path for the placeholder imaris image (include imaris name in path: SWxxxxxx-xx.ims). Required for FNT script: " + Style.RESET_ALL)
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

        # fntConversionCMD = f"python convert.py -i {IMARIS_OUTPUT_PATH} -t {FNT_OUTPUT_PATH}/{channel_folder}_FNT_tiff/ -fnt {FNT_OUTPUT_PATH}/{channel_folder}_FNT/ --channel {FNT_CHANNEL_PATH}/ -dx {X_VOXEL} -dy {Y_VOXEL} -dz {Z_VOXEL}"

        fntConversionCMD = f"""
srun -p bigmem --mem=800G --pty bash <<EOF 
cd image-preprocessing-pipeline && conda activate stitching && mkdir -p {linux_path(FNT_OUTPUT_PATH)} && chmod -R 777 {linux_path(FNT_OUTPUT_PATH)} && python convert.py -i {linux_path(FNT_CHANNEL_PATH)}/ -fnt {linux_path(FNT_OUTPUT_PATH)}/{channel_folder}_FNT/ -dx {X_VOXEL} -dy {Y_VOXEL} -dz {Z_VOXEL} > {linux_path(FNT_OUTPUT_PATH)}/output_fnt_log.txt
exit
EOF"""
        # print(fntConversionCMD)


        # TODO: add mitch login step here...
        # if saved pass file not found
        # check if localCredentialsPath file exists
        if (os.path.isfile(localCredentialsPath)):
            userName = open(localCredentialsPath, 'r').read().split(':')[0]
            # password = open(localCredentialsPath, 'r').read().split(':')[1]
            creds_choice = input(f'Connection to BMAP compute nodes is need. Proceed with "{userName}" credentials? [1 for yes, 2 for no]: ')
        
        if (not (os.path.isfile(localCredentialsPath))) or (creds_choice == '2'):
            print('FNT conversion requires access to BMAP cluster. Please enter your username to access the BMAP cluster to perform FNT conversion (Will be saved for future use): ')
            userName = input('Username: ')
            # password = input('Password: ')
            directory = os.path.dirname(localCredentialsPath)
            os.makedirs(directory, exist_ok=True)
            with open(localCredentialsPath, 'w') as credsFile:
                credsFile.write(f'{userName}')

        ssh_command = f'ssh {userName}:@cl.bmap.ucla.edu'
    
        os.system(f'start "Anaconda Prompt" cmd.exe /k "{ssh_command}"')
    # TODO: perform validations of the variables that require folder slashes... and replace the drive letters with mount paths 
    # COMPOSITE_PATH, IMARIS_OUTPUT_PATH, FNT_OUTPUT_PATH, FNT_CHANNEL_PATH = validate_paths(COMPOSITE_PATH, IMARIS_OUTPUT_PATH, FNT_OUTPUT_PATH, FNT_CHANNEL_PATH)

    # Print the command statement with FNT specified as running concurrently with merge/Imaris conversion, print FNT and merge/imaris seperately
    if (goal == 0):
        print('\nMerging channels + Imaris conversion commands generated below. Copy, paste into Anaconda Prompt (current window), and run to begin processing:\n')
        print(Fore.GREEN + (mergeChannelsCMD if (goal == 0) or (goal == 1) else "") + (" && " if goal == 0 else "") + (" && " if goal == 0 else "") + (imarisConversionCMD if (goal == 0) or (goal == 2) else "") + Style.RESET_ALL + '\n')
        print('\nFNT conversion command generated below. Copy, paste into Compute Node terminal, and run to begin processing (can be run at the same time as above):')
        print(Fore.GREEN + fntConversionCMD + Style.RESET_ALL + '\n')
    elif (goal == 3):
        print('\nFNT conversion command generated below. Copy, paste into Compute Node terminal, and run to begin processing:')
        print(Fore.GREEN + fntConversionCMD + Style.RESET_ALL + '\n')
    elif (goal == 4):
        print('\nMerging channels + Imaris conversion commands generated below. Copy, paste into Anaconda Prompt (current window), and run to begin processing:\n')
        print(Fore.GREEN + (mergeChannelsCMD)  + (" && ") + (imarisConversionCMD) + Style.RESET_ALL + '\n')
    else: 
        print('\n ' + ('Merge' if (goal == 1) else 'Imaris conversion') + ' command generated below. Copy, paste into Anaconda Prompt (current window), and run to begin processing:\n')
        print(Fore.GREEN + (mergeChannelsCMD if (goal == 0) or (goal == 1) else "") + (" && " if goal == 0 else "") + (" && " if goal == 0 else "") + (imarisConversionCMD if (goal == 0) or (goal == 2) else "") + Style.RESET_ALL + '\n')

if __name__ == "__main__":
    main()
    
