from pyvista import Plotter, PolyData
from pyvista import read as read_mesh
from typing import Tuple, List
from pathlib import Path
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace, BooleanOptionalAction
from pandas import read_csv
from shutil import copy


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def is_point_inside_surface(surface, points: List[Tuple[float, float, float]]):
    points_poly = PolyData(points)
    select = points_poly.select_enclosed_points(surface, check_surface=False)
    return (p == 1 for p in select['SelectedPoints'])


def convert_wrl_to_obj(wrl_file: Path, obj_file: Path):
    pl = Plotter()
    pl.import_vrml(wrl_file.__str__())
    pl.export_obj(obj_file.__str__())


def load_mesh(obj_file: Path):
    return read_mesh(obj_file.__str__())


def get_soma_locations(args):
    reconstructions = Path(args.reconstructions)
    x_axis_length: int = args.x_axis_length * args.voxel_size_x_target
    y_axis_length: int = args.y_axis_length * args.voxel_size_y_target
    z_axis_length: int = args.z_axis_length * args.voxel_size_z_target

    soma_coordinates = []
    for swc_file in list(reconstructions.rglob("*.swc")):
        soma = read_csv(swc_file, sep=r"\s+", comment="#", nrows=1, index_col=False,
                        names=("id", "type_id", "x", "y", "z", "radius", "parent_id")).loc[0]
        if int(soma.type_id) not in (0, 1) and int(soma.parent_id) not in (-1, 0):
            print(f"Warning: skipping {swc_file} --> undetermined soma node: {soma.to_dict()}")
        else:
            # scale everything to um unit
            soma.x *= args.voxel_size_x_source
            soma.y *= args.voxel_size_y_source
            soma.z *= args.voxel_size_z_source
            # print((soma.x, round(soma.y, 1), round(soma.z, 1)))

            # flipping operation
            if x_axis_length > 0:
                soma.x *= -1
                soma.x += x_axis_length

            if y_axis_length > 0:
                soma.y *= -1
                soma.y += y_axis_length

            if z_axis_length > 0:
                soma.z *= -1
                soma.z += z_axis_length

            # axis sign correction
            if args.negate_x > 0:
                soma.x *= -1

            if args.negate_y > 0:
                soma.y *= -1

            if args.negate_z > 0:
                soma.z *= -1

            # print((soma.x, round(soma.y, 1), round(soma.z, 1)))
            soma_coordinates += [DotDict({
                'swc_path': swc_file,
                'point': (soma.x, soma.y, soma.z)
            })]
    return soma_coordinates


def get_surface_meshes(surfaces: Path):
    surface_meshes = []
    for wrl_file in surfaces.rglob("*.wrl"):
        obj_file = wrl_file.parent / (wrl_file.name[0:-len(wrl_file.suffix)] + ".obj")
        if not obj_file.exists():
            convert_wrl_to_obj(wrl_file, obj_file)
        obj_mesh = load_mesh(obj_file)
        print(f"obj: {obj_file.name} --> center: {obj_mesh.center}")
        surface_meshes += [DotDict({
            'file_name': obj_file.name,
            'mesh': obj_mesh
        })]
    return surface_meshes


def main(args: Namespace):
    soma_coordinates = get_soma_locations(args)
    soma_coordinates_list = [soma.point for soma in soma_coordinates]
    surface_meshes = get_surface_meshes(Path(args.surfaces))
    for surface in surface_meshes:
        are_points_inside = is_point_inside_surface(surface.mesh, soma_coordinates_list)
        region = surface.file_name[0:-4]
        for soma, is_point_inside in zip(soma_coordinates, are_points_inside):
            if is_point_inside:
                print(f"{soma.swc_path.name} --> {region}")
                surface_path: Path = soma.swc_path.parent / region
                surface_path.mkdir(exist_ok=True)
                copy(soma.swc_path, surface_path)
                fnt_path: Path = soma.swc_path.parent / (soma.swc_path.name[0:-4] + "_Final.fnt")
                if fnt_path.exists():
                    copy(fnt_path, surface_path)
                fnt_path: Path = soma.swc_path.parent / (soma.swc_path.name[0:-4] + ".fnt")
                if fnt_path.exists():
                    copy(fnt_path, surface_path)

    # from pyvista import UniformGrid
    # regions = Path(r"Y:\3D_stitched_LS\20220725_SW220510_02_LS_6x_1000z\registration\structures")
    # region = regions / "structure_000000000255_surface.wrl"
    #
    # pl = Plotter()
    # pl.import_vrml(region.__str__())
    # pl.add_mesh(
    #     UniformGrid(
    #         origin=(7240.0, -8836.0, -3765.0),
    #         dimensions=(100, 100, 1),
    #         spacing=(10, 10, 10)
    #     ), show_edges=True)
    # pl.camera_position = 'xy'
    # pl.show()


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Classify reconstructions using registrations\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hongwei Dong Lab (B.R.A.I.N) \n"
    )
    parser.add_argument("--reconstructions", "-r", type=str, required=True,
                        help="Path folder containing all swc files.")
    parser.add_argument("--surfaces", "-s", type=str, required=True,
                        help="Path folder containing all surface files.")
    parser.add_argument("--voxel_size_x_source", "-dxs", type=float, required=False, default=1.0,
                        help="The voxel size on the x-axis of the image used for reconstruction. "
                             "Default value is 1, which is correct if swc is in μm unit already.")
    parser.add_argument("--voxel_size_y_source", "-dys", type=float, required=False, default=1.0,
                        help="The voxel size on the y-axis of the image used for reconstruction. "
                             "Default value is 1, which is correct if swc is in μm unit already.")
    parser.add_argument("--voxel_size_z_source", "-dzs", type=float, required=False, default=1.0,
                        help="The voxel size on the z-axis of the image used for reconstruction. "
                             "Default value is 1, which is correct if swc is in μm unit already.")
    parser.add_argument("--voxel_size_x_target", "-dxt", type=float, required=False, default=1.0,
                        help="The voxel size on the x-axis of the target image if wrl is in μm unit. "
                             "Default value is 1.")
    parser.add_argument("--voxel_size_y_target", "-dyt", type=float, required=False, default=1.0,
                        help="The voxel size on the y-axis of the target image if wrl is in μm unit. "
                             "Default value is 1.")
    parser.add_argument("--voxel_size_z_target", "-dzt", type=float, required=False, default=1.0,
                        help="The voxel size on the z-axis of the target image if wrl is in μm unit. "
                             "Default value is 1.")
    parser.add_argument("--x_axis_length", "-x", type=int, required=False, default=0,
                        help="The length of x-axis in pixels of the source image. "
                             "If x>0 is provided x-axis will be flipped. Default is 0 --> no x-axis flipping")
    parser.add_argument("--y_axis_length", "-y", type=int, required=False, default=0,
                        help="The length of y-axis in pixels of the source image. "
                             "If y>0 is provided y-axis will be flipped. Default is 0 --> no y-axis flipping")
    parser.add_argument("--z_axis_length", "-z", type=int, required=False, default=0,
                        help="The length of z-axis in pixels of the source image. "
                             "If z>0 is provided z-axis will be flipped. Default is 0 --> no z-axis flipping")
    parser.add_argument("--negate_x", default=False, action=BooleanOptionalAction,
                        help="Multiply some x value to -1. Default is --no-negate_x.")
    parser.add_argument("--negate_y", default=False, action=BooleanOptionalAction,
                        help="Multiply some y value to -1. Default is --no-negate_y.")
    parser.add_argument("--negate_z", default=False, action=BooleanOptionalAction,
                        help="Multiply some z value to -1. Default is --no-negate_z.")
    main(parser.parse_args())
