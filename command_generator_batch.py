from colorama import Fore, Style, init
init(convert=True)
import subprocess
from pathlib import Path
import sys
import os
import re


def fnt_confirmation(subpath, x_y_voxels, z_voxel):
    while True:
        print('Channel: ' + Fore.YELLOW + subpath.name + Style.RESET_ALL + f' || Voxels found: {x_y_voxels, x_y_voxels, z_voxel}')
        convert_channel = input(f'Convert to FNT? [Enter 0 for false, 1 for true]: ')
        if convert_channel in ['0', '1']:
            return convert_channel
        else: 
            print("Invalid input: Please enter 0 or 1")


def sanitize_paths(paths):
    chars_to_remove = '"'
    return [''.join(ch for ch in string if ch not in chars_to_remove) for string in paths if string]

def merge_channel_color(path, channel_count):
    cmd = ''
    color_dict = {
        0: 'cyan',
        1: 'magenta',
        2: 'yellow',
        3: 'black'
    }
    cmd = f'--{color_dict[channel_count]} {path}'
    return cmd

def main():
    ### DEFAULTS ###
    GOALS = [0, 1, 2, 3]
    COMPOSITE_PATH = 'V:\\Merged_Data'
    FNT_OUTPUT_PATH = 'W:\\3D_stitched_LS\\'
    IMS_OUTPUT_PATH = 'W:\\3D_stitched_LS\\'

    ### FINAL CMDS ###
    STITCHED_PATHS = ''
    BATCH_MERGE_CMDS = ''
    BATCH_FNT_CMDS = ''
    BATCH_IMS_CMDS = ''

    ### Temporary variables; values updated per channel, per case ###
    mergeChannelsCMD = ''
    fntConversionCMD = ''
    imsConversionCMD = ''

    print(Fore.BLUE + '\nRunning command_generator_batch.py to generate/run commands for performing batch processing of Tiff Channel Merging, Imaris, and/or FNT Conversions on Windows.\n' + Style.RESET_ALL)
    print('Enter [0] if target command is merging + imaris conversion + FNT conversion')
    print('Enter [1] if target command is merging only')
    print('Enter [2] if target command is imaris conversion only - DEPRECATED...')
    print('Enter [3] if target command is FNT conversion only')
    # print('Enter [4] if target command is merging + imaris conversion\n')
    # print('Enter [5] if target command is merging + imaris conversion\n')

    goal = int(input(Fore.MAGENTA + "Goal (0, 1, 2, or 3): \n" + Style.RESET_ALL))

    if goal not in GOALS:
        print(Fore.RED + 'Invalid goal entered: ' + str(goal) + Style.RESET_ALL)
        sys.exit()


    print(Fore.MAGENTA + "Enter all stitched paths for batch processing, use Control + Z once complete\n" + Style.RESET_ALL)
    rawInput = sys.stdin.read()
    STITCHED_PATHS = sanitize_paths(rawInput.splitlines())
    totalPaths = len(STITCHED_PATHS)
    # print(Fore.BLUE + f"\nProcessing {totalPaths} total datasets:" + Style.RESET_ALL)
    # for item in STITCHED_PATHS:
    #     print(f"- {item}")


    if goal in (0, 1):
        print(Fore.LIGHTGREEN_EX + '\n################ MERGING ################' + Style.RESET_ALL)
        print(Fore.BLUE + '\nMerging all stitched channels (Yellow) to respective (Green) output paths\n' + Style.RESET_ALL)

        for path in STITCHED_PATHS:
            mergeChannelsCMD = f"python .\\merge_channels.py "
            channel_count = 0
            valid_merge = False
            print('Stitched path: ' + Fore.MAGENTA + path + Style.RESET_ALL)
            stitched_path = Path(path)
            output_path = COMPOSITE_PATH + '\\' + stitched_path.name + '\\'
            os.makedirs(output_path, exist_ok=True)

            for subpath in stitched_path.iterdir():
                if (str(subpath.name).startswith('Ex_') and 'mip' not in str(subpath.name).lower()):
                    print('Channel: ' + Fore.YELLOW + subpath.name + Style.RESET_ALL)
                    channel_cmd = merge_channel_color(subpath, channel_count)
                    mergeChannelsCMD += channel_cmd + ' '
                    channel_count +=1
                    if channel_count > 1:
                        valid_merge = True

            if valid_merge:
                print(Fore.BLUE + 'Merged output will be saved to: ' + Fore.GREEN + output_path + Style.RESET_ALL)
                mergeChannelsCMD += '--output_path ' + output_path
                # print(Fore.RED + mergeChannelsCMD + Style.RESET_ALL)
                if len(BATCH_MERGE_CMDS) > 0:
                    BATCH_MERGE_CMDS += ' && ' + mergeChannelsCMD
                else:
                    BATCH_MERGE_CMDS += mergeChannelsCMD
        
        print('\n '+ BATCH_MERGE_CMDS + '\n')

    if goal in (0, 3):
        print(Fore.LIGHTGREEN_EX + '\n################ FNT CONVERSIONS ################' + Style.RESET_ALL)
        print(Fore.BLUE + '\nProcessing stitched channels (Yellow) to respective (Green) FNT output paths. Confirm FNT conversions' + Style.RESET_ALL)

        for path in STITCHED_PATHS:
            print('\nStitched path: ' + Fore.MAGENTA + path + Style.RESET_ALL)
            stitched_path = Path(path)
            z_voxel = ''
            x_y_voxels = ''
            for file in stitched_path.iterdir():
                if 'metadata' in file.name.lower():
                    with open(file, "r") as mf:
                        content = mf.readlines()
                    
                    if len(content) >= 2:
                        second_line = content[1]
                        words = second_line.split()
                        
                        if len(words) >= 4:
                            x_y_voxels = words[2]  # 3rd word
                            z_voxel = words[3]  # 4th word
                            x_y_voxels = round(float(x_y_voxels), 1)
                            z_voxel = round(float(z_voxel), 1)

            for subpath in stitched_path.iterdir():
                if (str(subpath.name).startswith('Ex_') and 'mip' not in str(subpath.name).lower() and z_voxel != ''):
                    convert_channel = fnt_confirmation(subpath, x_y_voxels, z_voxel)
                    if convert_channel == '1':
                        output_path = FNT_OUTPUT_PATH + stitched_path.name + '\\' + subpath.name + '_FNT\\'
                        os.makedirs(output_path, exist_ok=True)
                        print(Fore.BLUE + 'FNT output will be saved to: ' + Fore.GREEN + output_path + Style.RESET_ALL)
                        fntConversionCMD = f"python .\\convert.py -i {subpath} --fnt {output_path} -dx {x_y_voxels} -dy {x_y_voxels} -dz {z_voxel}"
                        if len(BATCH_FNT_CMDS) > 0:
                            BATCH_FNT_CMDS += ' && ' + fntConversionCMD
                        else:
                            BATCH_FNT_CMDS += fntConversionCMD
                    elif convert_channel == '0':
                        print(f'Skipping FNT processing for [{subpath.name}] channel')  

        print('\n '+ BATCH_FNT_CMDS + '\n')          

    if goal in (0, 2):
        if (goal == 2):
            # Copy / pasting merged paths is not user friendly, skipping implementation atm
            print('Direct Batch Imaris not yet implemented...')
            sys.exit
        print(Fore.LIGHTGREEN_EX + '\n################ IMARIS CONVERSIONS ################' + Style.RESET_ALL)
        print(Fore.BLUE + '\nProcessing Merged channels (Yellow) to respective (Green) Imaris output paths. Confirm Imaris conversions' + Style.RESET_ALL)

        for path in STITCHED_PATHS:
            x_y_voxels = ''
            z_voxel = ''
            stitched_path = Path(path)
            for file in stitched_path.iterdir():
                if 'metadata' in file.name.lower():
                    with open(file, "r") as mf:
                        content = mf.readlines()
                    
                    if len(content) >= 2:
                        second_line = content[1]
                        words = second_line.split()
                        
                        if len(words) >= 4:
                            x_y_voxels = words[2] 
                            z_voxel = words[2]  # 3rd word
                            # z_voxels = words[3]  # 3rd word
                            x_y_voxels = round(float(x_y_voxels), 1)
                            z_voxel = round(float(z_voxel), 1)
            if z_voxel != '':
                merged_path = COMPOSITE_PATH + '\\' + stitched_path.name + '\\'
                print('\nStitched path: ' + Fore.MAGENTA + path + Style.RESET_ALL)
                print('(Planned) Merged data path: ' + Fore.YELLOW + merged_path + Style.RESET_ALL)
                tmp_string = stitched_path.name.replace('_stitched', '')
                parts = tmp_string.split('_')
                ims_filename = '_'.join(parts[:1] + parts[4:])
                ims_filename = ims_filename + '.ims'

                output_folder = IMS_OUTPUT_PATH + stitched_path.name
                os.makedirs(output_folder, exist_ok=True)

                output_path = IMS_OUTPUT_PATH + stitched_path.name + '\\' + ims_filename
                print(Fore.BLUE + 'Imaris output will be saved to: ' + Fore.GREEN + output_path + Style.RESET_ALL)
                imsConversionCMD = f"python .\\convert.py -i {merged_path} -o {output_path} -dx {x_y_voxels} -dy {x_y_voxels} -dz {z_voxel}"
                if len(BATCH_IMS_CMDS) > 0:
                    BATCH_IMS_CMDS += ' && ' + imsConversionCMD
                else:
                    BATCH_IMS_CMDS += imsConversionCMD
            # Merged data won't exist yet at the the time of this log
            # else:
            #     print('No Merged data found in path: ' + output_path + ' Skipping Imaris Conversion for this case')

        print('\n' + BATCH_IMS_CMDS + ' \n')   

    # print('Batch Merging Commands: ')
    # print(BATCH_MERGE_CMDS)
    # print('Batch FNT Commands: ')
    # print(BATCH_FNT_CMDS)
    # print('Batch Imaris Commands: ')
    # print(BATCH_IMS_CMDS)
    batch_confirmation = input(Fore.CYAN + f'Confirm details above to begin batch processing (1 to continue, 0 to exit): ' + Style.RESET_ALL)

    if batch_confirmation == '1':
        print(Fore.YELLOW + '\nBatch Processing has started...' + Style.RESET_ALL)
        os.system(BATCH_MERGE_CMDS)
        os.system(BATCH_FNT_CMDS)
        os.system(BATCH_IMS_CMDS)
    else:
        print(Fore.YELLOW + '\nBatch Processing has been aborted.' + Style.RESET_ALL)
        sys.exit(0)

 
if __name__ == "__main__":
    main()
