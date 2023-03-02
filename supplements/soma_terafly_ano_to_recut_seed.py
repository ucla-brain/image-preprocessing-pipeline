# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:14:36 2023
Recut seeds: soma location in file content is in um (ignore the file name)
TeraFly apo rading: uses voxel
Imaris reading: uses voxel

"""

from pathlib import Path
from pandas import read_csv
from shutil import rmtree
from math import pi
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace
from tqdm import tqdm


def main(args: Namespace):
    voxel_size_x = float(args.voxel_size_x)
    voxel_size_y = float(args.voxel_size_y)
    voxel_size_z = float(args.voxel_size_z)
    soma_radius_um = args.default_radius if args.default_radius > 0 else None

    apo_file = Path(args.apo_file)
    annotations_df = read_csv(apo_file).drop_duplicates().reset_index(drop=True)

    # create a folder to store recut marker files converted from apo files
    recut = apo_file.parent / 'recut_seeds_from_marker'
    imaris = recut / 'seeds_for_Imaris_proofread.swc'
    if recut.exists():
        rmtree(recut)
    recut.mkdir(exist_ok=True)

    # make a copy of xyz in voxel, to generate the consolidated SWC file for proofread in Imaris
    annotations_df['x_in_voxel'] = annotations_df['x']
    annotations_df['y_in_voxel'] = annotations_df['y']
    annotations_df['z_in_voxel'] = annotations_df['z']

    # convert from voxel to um (recut marker files in um for further reconstruction, if proofread is done in Terafly)
    annotations_df['x'] = annotations_df['x'] * voxel_size_x
    annotations_df['y'] = annotations_df['y'] * voxel_size_y
    annotations_df['z'] = annotations_df['z'] * voxel_size_z

    # convert to integer (in um)
    for column in ("x", "y", "z", "volsize", 'x_in_voxel', 'y_in_voxel', 'z_in_voxel'):
        annotations_df[column] = annotations_df[column].round(decimals=0).astype(int)

    # create a consolidated .swc file to store all the somata, to be imported to imaris
    with imaris.open('w') as imaris_file:
        for row in annotations_df.itertuples():
            soma_radius_um_each_point = (row.volsize * 3 / 4 / pi) ** (1 / 3)
            volume = round(4 / 3 * pi * soma_radius_um_each_point ** 3, 3)
            if soma_radius_um:
                soma_radius_um_each_point = soma_radius_um
                volume = round(4 / 3 * pi * soma_radius_um_each_point ** 3, 3)

            with open(recut / f"marker_{row.x_in_voxel}_{row.y_in_voxel}_{row.z_in_voxel}_{int(volume)}", 'w') as \
                    marker_file:
                marker_file.write("# x,y,z,radius_um\n")
                # unit should be in um
                marker_file.write(f"{row.x},{row.y},{row.z},{soma_radius_um_each_point}")

            # unit should be in voxels
            imaris_file.write(f"{row.Index} 0 {row.x_in_voxel} {row.y_in_voxel} {row.z_in_voxel} "
                            f"{soma_radius_um_each_point} {-1}\n")

    print(f"Marker files (in um) saved in {recut.__str__()}")
    print(f"Consolidated SWC file containing all somata that can be imported in Imaris {imaris.__str__()}")


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert TeraFly apo file to recut seeds (marker files) \n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hong Wei Dong Lab (B.R.A.I.N) \n"
    )
    parser.add_argument("--apo_file", "-a", type=str, required=True,
                        help="Marker (apo) file containing all seed locations.")
    parser.add_argument("--default_radius", "-r", type=float, default=0, required=False,
                        help="Default radius of the seeds to apply uniformly.  "
                             "If not given, marker volume will be used to calculate the radius.")
    parser.add_argument("--voxel_size_x", "-dx", type=float, default=0.4,
                        help="Image voxel size on x-axis (µm).")
    parser.add_argument("--voxel_size_y", "-dy", type=float, default=0.4,
                        help="Image voxel size on y-axis (µm).")
    parser.add_argument("--voxel_size_z", "-dz", type=float, default=0.4,
                        help="Image voxel size on z-axis (µm).")
    main(parser.parse_args())
