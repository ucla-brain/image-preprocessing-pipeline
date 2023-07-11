from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace, BooleanOptionalAction
from pathlib import Path
from pandas import read_csv, DataFrame
from numpy import where, empty, vstack, append
SWC_COLUMNS = ["id", "type", "x", "y", "z", "radius", "parent_id"]


def is_overwrite_needed(file: Path, overwrite: bool) -> bool:
    if file.exists():
        if overwrite:
            file.unlink()
            return True
        else:
            print(f"{file.name} is already existed. "
                  f"Please use a different output path or selectively delete this file if reconversion is needed.")
            return False
    else:
        return True


def sort_swc(swc_df: DataFrame) -> DataFrame:
    unsorted_swc = swc_df.to_numpy()

    sorted_swc = empty((0, 7), float)
    Px = where(unsorted_swc[:, 6] == -1)
    Px = list(Px[0])
    while len(Px) > 0:
        P = Px[0]
        Px = Px[1:]
        while P.size > 0:
            P = int(P)
            sorted_swc = vstack((sorted_swc, unsorted_swc[P, :]))
            child = where(unsorted_swc[:, 6] == unsorted_swc[P, 0])
            child = list(child[0])
            if len(child) == 0:
                break
            if len(child) > 1:
                Px = append(child[1:], Px)
            P = child[0]

    sRe = sorted_swc[:, 6]
    Li = list(range(1, (len(sorted_swc[:, 1]) + 1)))
    Li1 = Li[:-1]
    for i in Li1:
        if sorted_swc[i, 6] != -1:
            pids = where(sorted_swc[:, 0] == sorted_swc[i, 6])
            pids = float(pids[0])
            sRe[i] = pids + 1
    sorted_swc[:, 6] = sRe
    sorted_swc[:, 0] = Li

    swc_df = DataFrame(sorted_swc, columns=SWC_COLUMNS)
    for column in ("id", "type", "parent_id"):
        swc_df[column] = swc_df[column].astype(int)
    return swc_df


def main(args: Namespace):
    assert args.input_extension in ("swc", "eswc")
    if not args.output_extension:
        args.output_extension = "swc"
        if args.input_extension == "swc":
            args.output_extension = "eswc"
    assert args.output_extension in ("swc", "eswc")

    input_path = Path(args.input)
    assert input_path.exists()

    output_path = input_path
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        assert output_path.exists()

    for input_file in input_path.rglob(f"*.{args.input_extension}"):
        output_file = output_path / input_file.relative_to(input_path)
        if args.input_extension == "eswc" and output_file.name.lower().endswith("ano.eswc"):
            output_file = output_file.parent / (output_file.name[0:-len("ano.eswc")] + "eswc")
        if args.sort:
            output_file = output_file.parent / (
                    output_file.name[0:-len(output_file.suffix)] + f"_sorted" + output_file.suffix)
        if args.output_extension == "swc":
            output_file = output_file.parent / (
                    output_file.name[0:-len(output_file.suffix)] + "." + args.output_extension)
        else:
            output_file = output_file.parent / (
                    output_file.name[0:-len(output_file.suffix)] + ".ano." + args.output_extension)

        if not is_overwrite_needed(output_file, args.overwrite):
            continue

        if args.input_extension == "eswc":
            eswc_df = read_csv(input_file, sep=r"\s+", comment="#",
                               names=SWC_COLUMNS + ["seg_id", "level", "mode", "timestamp", "TFresindex"])
            swc_df = eswc_df[SWC_COLUMNS].copy()
        else:
            swc_df = read_csv(input_file, sep=r"\s+", comment="#", names=SWC_COLUMNS)

        if args.x_axis_length > 0:
            swc_df['x'] = args.x_axis_length - swc_df['x']
        if args.y_axis_length > 0:
            swc_df['y'] = args.y_axis_length - swc_df['y']
        if args.z_axis_length > 0:
            swc_df['z'] = args.z_axis_length - swc_df['z']

        swc_df['x'] *= args.voxel_size_x_source / args.voxel_size_x_target
        swc_df['y'] *= args.voxel_size_y_source / args.voxel_size_y_target
        swc_df['z'] *= args.voxel_size_z_source / args.voxel_size_z_target

        if args.sort:
            swc_df = sort_swc(swc_df)

        if args.output_extension == "swc":
            with open(output_file, 'a'):
                output_file.write_text("#")
                swc_df.to_csv(output_file, sep=" ", mode="a", index=False)
        else:
            apo_file = output_file.parent / (output_file.name[0:-len(".eswc")] + ".apo")
            ano_file = output_file.parent / (output_file.name[0:-len(".ano.eswc")] + ".ano")

            if is_overwrite_needed(ano_file, args.overwrite):
                with ano_file.open('w') as ano:
                    ano.write(f"APOFILE={apo_file.name}\n")
                    ano.write(f"SWCFILE={output_file.name}\n")

            if is_overwrite_needed(apo_file, args.overwrite):
                with apo_file.open('w'):
                    apo_file.write_text(
                        "##n,orderinfo,name,comment,z,x,y,pixmax,intensity,sdev,volsize,mass,,,, "
                        "color_r,color_g,color_b")

            for col_name, value in (("seg_id", 0), ("level", 1), ("mode", 0), ("timestamp", 1), ("TFresindex", 1)):
                swc_df[col_name] = value
            with open(output_file, 'a'):
                output_file.write_text("#")
                swc_df.to_csv(output_file, sep=" ", mode="a", index=False)

        print(f"{args.input_extension} to {args.output_extension} -> {output_file}")


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert flip and scale swc and eswc files\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi and Sumit Nanda at UCLA, Hongwei Dong Lab (B.R.A.I.N.) \n"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path folder containing all swc or eswc files.")
    parser.add_argument("--output", "-o", type=str, required=False,
                        help="Output folder for converted files.")
    parser.add_argument("--input_extension", "-ie", type=str, required=False, default="eswc",
                        help="Input extension options are eswc and swc. Default is eswc")
    parser.add_argument("--output_extension", "-oe", type=str, required=False,
                        help="Output extension options are eswc and swc. "
                             "Default is swc if input_extension is eswc and vice versa.")
    parser.add_argument("--overwrite", default=False, action=BooleanOptionalAction,
                        help="Overwrite outputs. Default is --no-overwrite")
    parser.add_argument("--sort", default=False, action=BooleanOptionalAction,
                        help="Sort reconstructions. Default is --no-sort. "
                             "Makes sure if a node is upstream to another node, "
                             "it is never below the second node in the (e)swc file.")
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
