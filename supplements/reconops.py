from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace, BooleanOptionalAction
from pathlib import Path
from pandas import read_csv, DataFrame, concat
from numpy import where, empty, vstack, append
from cli_interface import PrintColors
from math import pi
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
    # print(swc_df.head())
    unsorted_swc = swc_df.sort_values(by=['id'], ascending=True).drop_duplicates().to_numpy()
    # unsorted_swc = unique(unsorted_swc, axis=0)
    sorted_swc = empty((0, 7), float)
    root_nodes = where(unsorted_swc[:, 6] == -1)
    if len(root_nodes) == 1 and len(root_nodes[0]) == 0:
        root_nodes = where(unsorted_swc[:, 6] == 0)
    if len(root_nodes) == 1 and len(root_nodes[0]) == 0:
        root_nodes = where(unsorted_swc[:, 0] == 1)
        unsorted_swc[0, 6] = -1
    root_nodes = list(root_nodes[0])
    # print(root_nodes)
    while len(root_nodes) > 0:
        parent = root_nodes[0]
        root_nodes = root_nodes[1:]
        while parent.size > 0:
            sorted_swc = vstack((sorted_swc, unsorted_swc[int(parent), :]))
            # print(len(sorted_swc))
            child = list(where(unsorted_swc[:, 6] == unsorted_swc[int(parent), 0])[0])
            # print(child)
            if len(child) == 0:
                break
            if len(child) > 1:
                root_nodes = append(child[1:], root_nodes)
            parent = child[0]

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
    args.input_extension = args.input_extension.lower()
    args.output_extension = args.output_extension.lower()
    assert args.input_extension in ("swc", "eswc", "apo")

    if args.input_extension == "apo":
        args.output_extension = "swc"
    if args.output_extension:
        assert args.output_extension in ("swc", "eswc", "seed")
    else:
        args.output_extension = "swc"
        if args.input_extension == "swc":
            args.output_extension = "eswc"

    input_path = Path(args.input)
    assert input_path.exists()
    if input_path.is_file():
        input_list = [input_path]
    else:
        input_list = list(input_path.rglob(f"*.{args.input_extension}"))

    output_path = input_path
    output_path_is_a_file: bool = False
    if args.output:
        output_path = Path(args.output)
        if input_path.is_file() and output_path != input_path and (
                args.output_extension == "swc" and output_path.name.lower().endswith(".swc") or
                args.output_extension == "eswc" and output_path.name.lower().endswith(".eswc")
        ):
            output_path_is_a_file = True
            output_path.parent.mkdir(exist_ok=True, parents=True)
            assert output_path.parent.exists()
        else:
            output_path.mkdir(exist_ok=True, parents=True)
            assert output_path.exists()

    for input_file in input_list:
        # if args.sort and input_file.name.lower().endswith(("_sorted.swc", "_sorted.eswc")):
        #     continue
        if output_path_is_a_file:
            output_file = output_path
        else:
            output_file = output_path / input_file.relative_to(input_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            if output_file == output_path:
                output_file = output_path / input_file.name
        if args.input_extension == "eswc" and output_file.name.lower().endswith("ano.eswc"):
            output_file = output_file.parent / (output_file.name[0:-len("ano.eswc")] + "eswc")
        if args.sort:
            output_file = output_file.parent / (
                    output_file.name[0:-len(output_file.suffix)] + f"_sorted" + output_file.suffix)
        if args.output_extension == "swc":
            output_file = output_file.parent / (
                    output_file.name[0:-len(output_file.suffix)] + "." + args.output_extension)
        elif args.output_extension == "eswc":
            output_file = output_file.parent / (
                    output_file.name[0:-len(output_file.suffix)] + ".ano." + args.output_extension)

        if not is_overwrite_needed(output_file, args.overwrite):
            continue

        if args.input_extension == "eswc":
            swc_df = read_csv(input_file, sep=r"\s+", comment="#",
                               names=SWC_COLUMNS + ["seg_id", "level", "mode", "timestamp", "TFresindex"])
            if args.output_extension == "swc":
                swc_df = swc_df[SWC_COLUMNS].copy()
        elif args.input_extension == "apo":
            swc_df = read_csv(input_file).drop_duplicates().reset_index(drop=True)
            for col_name, value in (
                    ("type", 1), ("radius", 12 if args.radii is None else args.radii), ("parent_id", -1)):
                swc_df[col_name] = value
            swc_df["id"] = swc_df.reset_index().index + 1
            swc_df = swc_df[SWC_COLUMNS].copy()
        else:
            swc_df = read_csv(input_file, sep=r"\s+", comment="#", names=SWC_COLUMNS)

        swc_df['x'] *= args.voxel_size_x_source / args.voxel_size_x_target
        swc_df['y'] *= args.voxel_size_y_source / args.voxel_size_y_target
        swc_df['z'] *= args.voxel_size_z_source / args.voxel_size_z_target

        if args.x_axis_length > 0:
            swc_df['x'] = args.x_axis_length - swc_df['x']
        if args.y_axis_length > 0:
            swc_df['y'] = args.y_axis_length - swc_df['y']
        if args.z_axis_length > 0:
            swc_df['z'] = args.z_axis_length - swc_df['z']

        if args.sort:
            try:
                swc_df = sort_swc(swc_df)
            except Exception as e:
                print(f"{PrintColors.FAIL}sorting failed! --> {input_file}\n"
                      f"error --> {e}{PrintColors.ENDC}")
                continue

        if args.output_extension == "swc":
            with open(output_file, 'a'):
                output_file.write_text("#")
                swc_df.to_csv(output_file, sep=" ", mode="a", index=False)
                print(f"{args.input_extension} to {args.output_extension} -> {output_file}")

            if args.input_extension == "apo" or args.swc_to_seed:
                if args.input_extension == "apo":
                    output_folder = output_file.parent / output_file.name[0:-len('.ano.swc')]
                else:
                    output_folder = output_file.parent / output_file.name[0:-len(output_file.suffix)]
                output_folder.mkdir(exist_ok=True)
                for i in range(0, len(swc_df)):
                    df1: DataFrame = swc_df.iloc[[i]].copy()
                    output_file_new = output_folder / f"x{int(df1['x'])}-y{int(df1['y'])}-z{int(df1['z'])}.swc"
                    if is_overwrite_needed(output_file_new, args.overwrite):
                        with open(output_file_new, 'a'):
                            output_file_new.write_text("#")
                            df2 = df1.copy()
                            df1["id"] = 1
                            df1["parent_id"] = 0
                            df2["id"] = 2
                            df2["type"] = 2
                            df2["parent_id"] = 1
                            concat([df1, df2]).to_csv(output_file_new, sep=" ", mode="a", index=False)
                            print(f"{args.input_extension} to {args.output_extension} -> {output_file_new}")
        elif args.output_extension == "eswc":
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

            if args.input_extension != "eswc":
                for col_name, value in (("seg_id", 0), ("level", 1), ("mode", 0), ("timestamp", 1), ("TFresindex", 1)):
                    swc_df[col_name] = value
            with open(output_file, 'a'):
                output_file.write_text("#")
                swc_df.to_csv(output_file, sep=" ", mode="a", index=False)
            print(f"{args.input_extension} to {args.output_extension} -> {output_file}")
        elif args.output_extension == "seed":
            for row in swc_df.itertuples():
                x = int(row.x + .5)
                y = int(row.y + .5)
                z = int(row.z + .5)
                radii = row.radius
                if args.radii is not None:
                    radii = args.radii
                # radii = float(l_split[5])
                volume = int(4 / 3 * pi * radii ** 3)
                output_file = output_path / f"marker_{x}_{y}_{z}_{volume}"
                with open(output_file, 'w') as marker_file:
                    marker_file.write("# x,y,z,radius_um\n")
                    marker_file.write(f"{x},{y},{z},{radii}")
                    print(f"{args.input_extension} to {args.output_extension} -> {output_file}")


if __name__ == '__main__':
    parser = ArgumentParser(
        description="ReconOps, i.e. reconstruction operations, convert flip and scale swc and eswc files\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi and Sumit Nanda at UCLA, Hongwei Dong Lab (B.R.A.I.N.) \n"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path folder containing all swc or eswc files.")
    parser.add_argument("--output", "-o", type=str, required=False,
                        help="Output folder for converted files.")
    parser.add_argument("--input_extension", "-ie", type=str, required=False, default="eswc",
                        help="Input extension options are eswc, swc, and apo. Default is eswc.")
    parser.add_argument("--output_extension", "-oe", type=str, required=False,
                        help="Output extension options are eswc, swc and seed. "
                             "Default is swc if input_extension is eswc and vice versa. "
                             "Apo files can be converted to swc and seed. "
                             "Two types of swc files are generated for each apo file. "
                             "One of them has all the nodes and can be opened in neuTube. "
                             "The other one is a folder containing one swc per node that can be opened in "
                             "Fast Neurite Tracer (FNT)."
                             "Seed option generates marker files that can be read by recut. Seed files should be in um "
                             "unit and therefore source voxel sizes should be provided as needed.")
    parser.add_argument("--overwrite", default=False, action=BooleanOptionalAction,
                        help="Overwrite outputs. Default is --no-overwrite")
    parser.add_argument("--sort", default=False, action=BooleanOptionalAction,
                        help="Sort reconstructions. Default is --no-sort. "
                             "Makes sure if a node is upstream to another node, "
                             "it is never below the second node in the (e)swc file.")
    parser.add_argument("--swc_to_seed", default=False, action=BooleanOptionalAction,
                        help="If you have a swc file containing only soma location, "
                             "then, each node will be converted to a separate swc file.")
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
                        help="The length of x-axis in pixels of the target image. "
                             "If x>0 is provided x-axis will be flipped. Default is 0 --> no x-axis flipping")
    parser.add_argument("--y_axis_length", "-y", type=int, required=False, default=0,
                        help="The length of y-axis in pixels of the target image. "
                             "If y>0 is provided y-axis will be flipped. Default is 0 --> no y-axis flipping")
    parser.add_argument("--z_axis_length", "-z", type=int, required=False, default=0,
                        help="The length of z-axis in pixels of the target image. "
                             "If z>0 is provided z-axis will be flipped. Default is 0 --> no z-axis flipping")
    parser.add_argument("--radii", "-r", type=float, required=False, default=None,
                        help="Force the radii to be a specific number in um during (e)swc to seed conversion. "
                             "Default value is None which means: "
                             "(1) for swc to seed conversion radius value from swc file will be used."
                             "(2) fro apo to swc conversion the value of 12 will be used.")
    main(parser.parse_args())
