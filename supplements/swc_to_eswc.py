from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from pandas import read_csv


def main(args: Namespace):
    reconstructions_path = Path(args.reconstructions)
    assert reconstructions_path.exists()
    for swc_file in reconstructions_path.rglob("*.swc"):
        ano_file = swc_file.parent / (swc_file.name[0:-3] + "ano")
        apo_file = ano_file.parent / (ano_file.name + ".apo")
        eswc_file = ano_file.parent / (ano_file.name + ".eswc")
        if ano_file.exists() or apo_file.exists() or eswc_file.exists():
            print(f"{swc_file.name} is already converted. "
                  f"Please delete ano, apo or eswc files if reconversion is needed.")
            continue

        swc_df = read_csv(swc_file, sep=r"\s+", comment="#", index_col=0,
                          names=("id", "type", "x", "y", "z", "radius", "parent_id"))

        # convert from um unit to voxel unit
        swc_df['x'] *= args.voxel_size_x
        swc_df['y'] *= args.voxel_size_y
        swc_df['z'] *= args.voxel_size_z

        # flipping operation should be in voxel unit
        if args.x_axis_length > 0:
            swc_df['x'] = args.x_axis_length - swc_df['x']
        if args.y_axis_length > 0:
            swc_df['y'] = args.y_axis_length - swc_df['y']
        if args.z_axis_length > 0:
            swc_df['z'] = args.z_axis_length - swc_df['z']

        for col_name, value in (("seg_id", 0), ("level", 1), ("mode", 0), ("timestamp", 1), ("TFresindex", 1)):
            swc_df[col_name] = value
        # print(swc_df.head())

        with ano_file.open('w') as ano:
            ano.write(f"APOFILE={apo_file.name}\n")
            ano.write(f"SWCFILE={eswc_file.name}\n")

        with apo_file.open('w'):
            apo_file.write_text(
                "##n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b")

        with open(eswc_file, 'a'):
            eswc_file.write_text("#")
            swc_df.to_csv(eswc_file, sep=" ", mode="a")


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert swcs exported from Imaris to eswcs that can be read in TeraFly images\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hongwei Dong Lab (B.R.A.I.N) \n"
    )
    parser.add_argument("--reconstructions", "-r", type=str, required=True,
                        help="Path folder containing all swc files.")
    parser.add_argument("--voxel_size_x", "-dx", type=float, required=False, default=1.0,
                        help="voxel size on the x-axis. Default value is 1, i.e. no for voxel size.")
    parser.add_argument("--voxel_size_y", "-dy", type=float, required=False, default=1.0,
                        help="voxel size on the y-axis. Default value is 1, i.e. no for voxel size.")
    parser.add_argument("--voxel_size_z", "-dz", type=float, required=False, default=1.0,
                        help="voxel size on the z-axis. Default value is 1, i.e. no for voxel size.")
    parser.add_argument("--x_axis_length", "-x", type=int, required=False, default=0,
                        help="The length of x-axis in pixels. "
                             "If x>0 is provided x-axis will be flipped. Default is 0 --> no x-axis flipping")
    parser.add_argument("--y_axis_length", "-y", type=int, required=False, default=0,
                        help="The length of y-axis in pixels. "
                             "If y>0 is provided y-axis will be flipped. Default is 0 --> no y-axis flipping")
    parser.add_argument("--z_axis_length", "-z", type=int, required=False, default=0,
                        help="The length of z-axis in pixels. "
                             "If z>0 is provided z-axis will be flipped. Default is 0 --> no z-axis flipping")
    main(parser.parse_args())
