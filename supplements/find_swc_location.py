from pyvista import Plotter, PolyData
from pyvista import read as read_mesh
from typing import Tuple, List
from pathlib import Path
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace
from pandas import read_csv
from shutil import copy


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


def correct_point(point: Tuple[float, float, float], y_axis_length: float):
    return tuple((point[0], point[1] - y_axis_length, -point[2]))


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_soma_locations(reconstructions: Path, y_axis_length: float):
    soma_coordinates = []
    for swc_file in reconstructions.rglob("*.swc"):
        swc_df = read_csv(swc_file, sep=r"\s+", comment="#", nrows=1,
                          names=("id", "type_id", "x", "y", "z", "radius", "parent_id"))
        if swc_df.type_id[0] != 1 and swc_df.parent_id[0] != -1:
            print(f"Warning: skipping {swc_file} --> undetermined soma")
        else:
            soma_coordinates += [DotDict({
                'swc_path': swc_file,
                'point': correct_point((swc_df.x[0], swc_df.y[0], swc_df.z[0]), y_axis_length)
            })]
    return soma_coordinates


def get_surface_meshes(surfaces: Path):
    surface_meshes = []
    for wrl_file in surfaces.rglob("*.wrl"):
        obj_file = wrl_file.parent / (wrl_file.name[0:-len(wrl_file.suffix)] + ".obj")
        if not obj_file.exists():
            convert_wrl_to_obj(wrl_file, obj_file)
        surface_meshes += [DotDict({
            'file_name': obj_file.name,
            'mesh': load_mesh(obj_file).flip_z()
        })]
    return surface_meshes


def main(args: Namespace):
    soma_coordinates = get_soma_locations(Path(args.reconstructions), args.y_axis_length)
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
    parser.add_argument("--y_axis_length", "-y", type=float, default=0.0, required=True,
                        help="Y-axis length of image in voxels. Default is 0.")
    main(parser.parse_args())
