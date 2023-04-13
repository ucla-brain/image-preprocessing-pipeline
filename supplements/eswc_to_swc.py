from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from pandas import read_csv


def main(args: Namespace):
    reconstructions_path = Path(args.reconstructions)
    assert reconstructions_path.exists()
    for eswc_file in reconstructions_path.rglob("*.eswc"):
        swc_file = eswc_file.parent / (eswc_file.name[0:-len(".eswc")] + ".swc")
        if swc_file.exists():
            print(f"{eswc_file.name} is already converted. "
                  f"Please delete swc files if reconversion is needed.")
            continue

        eswc_df = read_csv(eswc_file, sep=r"\s+", comment="#", index_col=0,
                           names=("id", "type", "x", "y", "z", "radius", "parent_id", "seg_id", "level", "mode",
                                  "timestamp", "TFresindex"))

        swc_df = eswc_df[["type", "x", "y", "z", "radius", "parent_id"]].copy()

        swc_df['x'] /= args.voxel_size_x
        swc_df['y'] /= args.voxel_size_y
        swc_df['z'] /= args.voxel_size_z

        swc_df['x'] = swc_df['x'].astype(int)
        swc_df['y'] = swc_df['y'].astype(int)
        swc_df['z'] = swc_df['z'].astype(int)

        if args.x_axis_length > 0:
            swc_df['x'] = args.x_axis_length - swc_df['x']
        if args.y_axis_length > 0:
            swc_df['y'] = args.y_axis_length - swc_df['y']
        if args.z_axis_length > 0:
            swc_df['z'] = args.z_axis_length - swc_df['z']

        with open(swc_file, 'a'):
            swc_file.write_text("#")
            swc_df.to_csv(swc_file, sep=" ", mode="a")

        print(swc_file)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert eswcs exported from TeraFly to swcs that can be read in Imaris\n\n",
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
