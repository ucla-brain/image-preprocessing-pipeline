from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from pandas import read_csv


def main(args: Namespace):
    assert args.extension in ("swc", "eswc")

    input_path = Path(args.input)
    assert input_path.exists()

    output_path = input_path
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        assert output_path.exists()

    for eswc_file in input_path.rglob(f"*.{args.extension}"):
        swc_file = output_path / eswc_file.relative_to(input_path)
        swc_file = swc_file.parent / (swc_file.name[0:-len(swc_file.suffix)] + ".swc")
        if swc_file.exists():
            print(f"{swc_file.name} is already existed. "
                  f"Please use a different output path or delete swc files if eswc to swc reconversion is needed.")
            continue

        if args.extension == "eswc":
            eswc_df = read_csv(eswc_file, sep=r"\s+", comment="#", index_col=0,
                               names=("id", "type", "x", "y", "z", "radius", "parent_id", "seg_id", "level", "mode",
                                      "timestamp", "TFresindex"))
            swc_df = eswc_df[["type", "x", "y", "z", "radius", "parent_id"]].copy()
        else:
            swc_df = read_csv(eswc_file, sep=r"\s+", comment="#", index_col=0,
                              names=("id", "type", "x", "y", "z", "radius", "parent_id"))

        if args.x_axis_length > 0:
            swc_df['x'] = args.x_axis_length - swc_df['x']
        if args.y_axis_length > 0:
            swc_df['y'] = args.y_axis_length - swc_df['y']
        if args.z_axis_length > 0:
            swc_df['z'] = args.z_axis_length - swc_df['z']

        swc_df['x'] *= args.voxel_size_x_source / args.voxel_size_x_target
        swc_df['y'] *= args.voxel_size_y_source / args.voxel_size_y_target
        swc_df['z'] *= args.voxel_size_z_source / args.voxel_size_z_target

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
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path folder containing all swc or eswc files.")
    parser.add_argument("--output", "-o", type=str, required=False,
                        help="Output folder for converted files.")
    parser.add_argument("--extension", "-e", type=str, required=False, default="eswc",
                        help="Input extension options are eswc and swc. Default is eswc")
    parser.add_argument("--voxel_size_x_source", "-dxs", type=float, required=False, default=1.0,
                        help="The voxel size on the x-axis of the image used for reconstruction. "
                             "Default value is 1.")
    parser.add_argument("--voxel_size_y_source", "-dys", type=float, required=False, default=1.0,
                        help="The voxel size on the y-axis of the image used for reconstruction. "
                             "Default value is 1.")
    parser.add_argument("--voxel_size_z_source", "-dzs", type=float, required=False, default=1.0,
                        help="The voxel size on the z-axis of the image used for reconstruction. "
                             "Default value is 1.")
    parser.add_argument("--voxel_size_x_target", "-dxt", type=float, required=False, default=1.0,
                        help="The voxel size on the x-axis of the target image. "
                             "Default value is 1.")
    parser.add_argument("--voxel_size_y_target", "-dyt", type=float, required=False, default=1.0,
                        help="The voxel size on the y-axis of the target image. "
                             "Default value is 1.")
    parser.add_argument("--voxel_size_z_target", "-dzt", type=float, required=False, default=1.0,
                        help="The voxel size on the z-axis of the target image. "
                             "Default value is 1.")
    parser.add_argument("--x_axis_length", "-x", type=int, required=False, default=0,
                        help="The length of x-axis in pixels of the image used for reconstruction. "
                             "If x>0 is provided x-axis will be flipped. Default is 0 --> no x-axis flipping")
    parser.add_argument("--y_axis_length", "-y", type=int, required=False, default=0,
                        help="The length of y-axis in pixels of the image used for reconstruction. "
                             "If y>0 is provided y-axis will be flipped. Default is 0 --> no y-axis flipping")
    parser.add_argument("--z_axis_length", "-z", type=int, required=False, default=0,
                        help="The length of z-axis in pixels of the image used for reconstruction. "
                             "If z>0 is provided z-axis will be flipped. Default is 0 --> no z-axis flipping")
    main(parser.parse_args())
