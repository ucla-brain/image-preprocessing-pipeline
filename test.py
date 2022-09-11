# from pystripe.core import batch_filter
# from pathlib import Path
# from multiprocessing import freeze_support
# from supplements.cli_interface import PrintColors
#
#
# def main():
#     batch_filter(
#         Path(r"E:\sm20220717_01_40X_ZStep2um_20220813_175015_17_F50\Ex_642_Em_680"),
#         Path(r"D:\test_interruption"),
#         lightsheet=True,
#         compression=("ZLIB", 0)
#     )
#
#
# if __name__ == '__main__':
#     freeze_support()
#     main()

from process_images import run_command
run_command(r"mpiexec -np 6 python -m mpi4py TeraStitcher/pyscripts/Parastitcher.py -2 --oH=185 --oV=185 --sH=184 --sV=184 --sD=200 --subvoldim=8400 --threshold=0.95 --projin=/data/20220725_17_41_55_SW220510_02_LS_15x_1000z_lightsheet_cleaned_tif_bitshift.g0.r0.b0_stitched/Ex_642_Em_680_xml_import_step_1.xml --projout=/data/20220725_17_41_55_SW220510_02_LS_15x_1000z_lightsheet_cleaned_tif_bitshift.g0.r0.b0_stitched/Ex_642_Em_680_xml_import_step_2.xml",
            need_progress_dot=True)

