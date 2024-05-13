from colorama import Fore, Back, Style
import sys




def main():
    GOALS = [0, 1, 2]
    COMPOSITE_PATH = ''
    IMARIS_OUTPUT_PATH = ''
    X_VOXEL = ''
    Y_VOXEL = ''
    Z_VOXEL = ''
    mergeChannelsCMD = ''
    imarisConversionCMD = ''

    print(Fore.BLUE + '\nRunning command_generator.py to generate commands for merging and or Imaris conversion scripts on Windows...\n' + Style.RESET_ALL)
    print('Enter [0] if target command is merging and imaris conversion')
    print('Enter [1] if target command is merging only')
    print('Enter [2] if target command is imaris conversion only\n')

    goal = int(input(Fore.MAGENTA + "Goal (0, 1, or 2): " + Style.RESET_ALL))

    if goal not in GOALS:
        print(Fore.RED + 'Invalid goal entered: ' + str(goal) + Style.RESET_ALL)
        sys.exit()


    if (goal == 0) or (goal == 1):
        print ('\nBuilding merge command. Enter individual RGB tiff channel paths. Use 0 for empty channel...\n')
        redChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.RED + "RED" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        greenChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.GREEN + "GREEN" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        blueChannelPath = input(Fore.MAGENTA + "Enter the path of the " + Fore.BLUE + "BLUE" + Fore.MAGENTA + " channel to merge: "+ Style.RESET_ALL)
        COMPOSITE_PATH = input(Fore.MAGENTA + "Enter the destination path to save the composite folder to save to ('\\composite\\' will be added to this path): " + Style.RESET_ALL)
        COMPOSITE_PATH = COMPOSITE_PATH + '\\composite\\'
        
        mergeChannelsCMD = f"python .\\merge_channels.py" + (f" --red {redChannelPath}" if (redChannelPath != '0') else "") + (f" --green {greenChannelPath}" if (greenChannelPath != '0') else "") + (f" --blue {blueChannelPath}" if (blueChannelPath != '0') else "") + (f" --output_path {COMPOSITE_PATH}") 

    if (goal == 0) or (goal == 2):
        print('\nBuilding imaris conversion command...\n')
        if (goal == 2):
            COMPOSITE_PATH = input(Fore.MAGENTA + "Enter the path to the composite (merged/RGB) tiff channel: " + Style.RESET_ALL)

        IMARIS_OUTPUT_PATH = input(Fore.MAGENTA + "Enter the destination path for the converted imaris image (include imaris name in path): " + Style.RESET_ALL)
        X_VOXEL = input(Fore.MAGENTA +"Enter x voxel value in microns (x.x um): " + Style.RESET_ALL)
        Y_VOXEL = input(Fore.MAGENTA +"Enter y voxel value in microns (x.x um): " + Style.RESET_ALL)
        Z_VOXEL = input(Fore.MAGENTA +"Enter z voxel value in microns (x.x um): " + Style.RESET_ALL)
        imarisConversionCMD = f"python .\\convert.py -i {COMPOSITE_PATH} -o {IMARIS_OUTPUT_PATH} -dx {X_VOXEL} -dy {Y_VOXEL} -dz {Z_VOXEL}"


    print('\nCommand generated below. Copy, paste, then continue to begin processing:\n')
    print(Fore.GREEN + (mergeChannelsCMD if (goal == 0) or (goal == 1) else "") + (" && " if goal == 0 else "") +  (imarisConversionCMD if (goal == 0) or (goal == 2) else "") + Style.RESET_ALL + '\n')

if __name__ == "__main__":
    main()
