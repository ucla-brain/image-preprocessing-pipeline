"""volume.py - representation of a Terastitcher volume"""
import os
import re
import abc
import sys
import itertools

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import enum
else:
    import aenum as enum
from pathlib import Path
from xml.etree import ElementTree
from .raw import raw_imread
from numpy import ndarray, zeros, hstack, inf, arange, arctan2, sin, isinf, ones, float32, newaxis, minimum, finfo, \
    iinfo, around, clip, int32, uint32, uint8, uint16
from numpy import max as np_max
from numpy import min as np_min
from numpy import dtype as np_d_type
from numpy import maximum
from supplements.cli_interface import PrintColors
from tifffile import imread, imwrite
from pystripe.core import glob_re


def get_dim_tuple(element):
    """Given an element, extract the Terastitcher z, y, x dimensions

    Parameter
    ----------
    element: an etree DOM element

    Returns
    -------
    a tuple of the Z, Y and X extracted from the V, H and D attributes of the element
    """
    return tuple([float(element.attrib[_]) for _ in "DVH"])


class Location:
    """A coordinate location"""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, item):
        """Get a coordinate by index"""
        assert 0 <= item <= 2
        if item == 0:
            return self.z
        if item == 1:
            return self.y
        return self.x

    def __repr__(self):
        return "{{x={x:d}, y={y:d}, z={z:d}}}".format(
            x=self.x, y=self.y, z=self.z)


class VExtentBase(abc.ABC):
    """A volume extent (in voxels)"""

    def __init__(self):
        super(abc.ABC, self).__init__()

    def intersects(self, other):
        """Determine whether two extents intersect

        :param other: another VExtent
        :returns: True if the volumes intersect
        """
        return self.x0 < other.x1 and self.x1 > other.x0 and \
            self.y0 < other.y1 and self.y1 > other.y0 and \
            self.z0 < other.z1 and self.z1 > other.z0

    def intersection(self, other):
        """Return the intersection volume of two extents"""
        return VExtent(max(self.x0, other.x0), min(self.x1, other.x1),
                       max(self.y0, other.y0), min(self.y1, other.y1),
                       max(self.z0, other.z0), min(self.z1, other.z1))

    def contains(self, other):
        """Return True if the other volume is wholly within this one

        :param other: a VExtentBase volume extent
        """
        return self.x0 <= other.x0 and self.x1 >= other.x1 and \
            self.y0 <= other.y0 and self.y1 >= other.y1 and \
            self.z0 <= other.z0 and self.z1 >= other.z1

    @property
    @abc.abstractmethod
    def x0(self):
        pass

    @property
    @abc.abstractmethod
    def x1(self):
        pass

    @property
    @abc.abstractmethod
    def y0(self):
        pass

    @property
    @abc.abstractmethod
    def y1(self):
        pass

    @property
    @abc.abstractmethod
    def z0(self):
        pass

    @property
    @abc.abstractmethod
    def z1(self):
        pass

    @property
    def shape(self):
        """The number of voxels in the volume in the z, y and x directions"""
        return self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0

    def start(self, idx):
        """The start coordinate at the given index

        :param idx: an index between 0 and 2 - 0 = z, 1 = y, 2 = x
        :returns: the x0, y0 or z0 of the volume
        """
        assert 0 <= idx <= 2
        if idx == 0:
            return self.z0
        elif idx == 1:
            return self.y0
        return self.x0

    def end(self, idx):
        """The end coordinate at the given index

        :param idx: an index between 0 and 2 - 0 = z, 1 = y, 2 = x
        :returns: the x1, y1 or z1 of the volume
        """
        assert 0 <= idx <= 2
        if idx == 0:
            return self.z1
        elif idx == 1:
            return self.y1
        return self.x1

    def __repr__(self):
        return "VExtent(x0={x0}, x1={x1}, y0={y0}, y1={y1}, z0={z0}, z1={z1})" \
            .format(x0=self.x0, y0=self.y0, z0=self.z0,
                    x1=self.x1, y1=self.y1, z1=self.z1)


class VExtent(VExtentBase):
    """A volume extent"""

    def __init__(self, x0, x1, y0, y1, z0, z1):
        super().__init__()
        self._x0 = x0
        self._x1 = x1
        self._y0 = y0
        self._y1 = y1
        self._z0 = z0
        self._z1 = z1

    @property
    def x0(self):
        return self._x0

    @property
    def x1(self):
        return self._x1

    @property
    def y0(self):
        return self._y0

    @property
    def y1(self):
        return self._y1

    @property
    def z0(self):
        return self._z0

    @property
    def z1(self):
        return self._z1


class TSVStackBase(VExtentBase):
    def __init__(self):
        super().__init__()
        # self.paths = None
        self._height = None
        self._width = None
        self.__x1 = None
        self.__y1 = None
        self.__dtype = None

    def _set_x1y1(self):
        if self.__x1 is None:
            if len(self.paths) == 0:
                self.__x1 = self.x0
                self.__y1 = self.y0
                return
            img = self.read_plane(self.paths[0])
            self.__dtype = img.dtype
            height, width = img.shape[-2:]
            self.set_size(width, height)

    def set_size(self, width: int, height: int):
        """Set the width and height of the stack planes"""
        self._height, self._width = height, width
        self.__x1 = self.x0 + self._width
        self.__y1 = self.y0 + self._height

    @property
    def x0(self):
        return self._x0

    @property
    def x1(self):
        """The block's end coordinate in the x direction"""
        self._set_x1y1()
        return self.__x1

    @property
    def y0(self):
        return self._y0

    @property
    def y1(self):
        """The block's end coordinate in the y direction"""
        self._set_x1y1()
        return self.__y1

    @property
    def z0(self):
        return self._z0

    @property
    def z1(self):
        return self._z0 + self.z1slice - self.z0slice

    @property
    def dtype(self):
        """The numpy dtype of the array data"""
        self._set_x1y1()
        return self.__dtype

    def read_plane(self, path):
        if self.input_plugin == "raw":
            return raw_imread(path)
        else:
            try:
                img = imread(path)
                if img is None:
                    raise
                return img
            except:
                print("Bad file: %s" % path)
                raise

    def imread(self, volume, result=None):
        """
        Read the image data from a block

        volume: the volume to read, a VExtent
        result: if not None, read into this array

        returns: the requested volume
        """
        assert self.contains(volume)
        if result is None:
            result = zeros(volume.shape, self.dtype)
        for z in range(volume.z0, volume.z1):
            plane = self.read_plane(self.paths[z - self.z0])
            result[z - volume.z0] = plane[
                                    volume.y0 - self.y0:volume.y1 - self.y0,
                                    volume.x0 - self.x0:volume.x1 - self.x0]
        return result


class TSVStack(TSVStackBase):
    def __init__(self, element, offset: Location, root_dir: str,
                 ordering_pattern=None,
                 input_plugin=None,
                 z_step: int = None):
        """Initialize a stack from a "Stack" element

        element:
            an ElementTree element (with the "Stack" tag)
        root_dir:
            the root directory of the directory hierarchy
        ordering_pattern:
            how to find the image order # - an expression that extracts a numeric z from the path name.
        input_plugin:
            the input plugin that was used to read the files in TeraStitcher
            that the stage always returns to the same place.
        z_step: int
            used to find missing files if there is any
        """
        TSVStackBase.__init__(self)
        self.root_dir = root_dir
        self.n_chans = int(element.attrib["N_CHANS"])
        self.bytes_per_chan = int(element.attrib["N_BYTESxCHAN"])
        self.row = int(element.attrib["ROW"])
        self.column = int(element.attrib["COL"])
        self._x0 = offset.x
        self._y0 = offset.y
        self._z0 = offset.z
        self.dir_name = element.attrib["DIR_NAME"]
        self.input_plugin = input_plugin
        z_ranges = element.attrib["Z_RANGES"]
        self.__idxs_to_keep = zeros(0, int)
        self.z_step = z_step
        if len(z_ranges) == 0:
            self.z0slice = self.z1slice = 0
        else:
            # format is [AAAA,BBBB);[AAAA,BBBB)
            # never seen (AAAA,BBBB], but it's coded
            for substr in z_ranges.split(";"):
                z0, z1 = map(int, substr[1:-1].split(","))
                if not substr.startswith("["):
                    z0 += 1
                if not z_ranges.endswith(")"):
                    z1 += 1
                self.__idxs_to_keep = \
                    hstack([self.__idxs_to_keep, arange(z0, z1)])
            self.z0slice = 0
            self.z1slice = len(self.__idxs_to_keep)
        self.img_regex = element.attrib["IMG_REGEX"]
        if ordering_pattern is None:
            ordering_pattern = "[^0-9]*(\\d+).*\\.raw" if input_plugin == "raw" else "[^0-9]*(\\d+).*\\.tiff?"
        self.ordering_pattern = ordering_pattern
        self.__paths = None
        self.suffix: str = glob_re(ordering_pattern, Path(root_dir)).__next__().suffix

    @property
    def paths(self):
        """The paths to the individual slices"""
        if self.__paths is None:
            directory = os.path.join(self.root_dir, self.dir_name)
            my_paths = []
            for filename in sorted(os.listdir(directory)):
                match = re.match(self.ordering_pattern, filename)
                if not match:
                    continue
                ordering = int(match.groups()[0])
                if self.img_regex != "":
                    if not re.match(self.img_regex, filename):
                        continue
                my_paths.append((ordering, os.path.join(directory, filename)))
            my_paths = [_[1] for _ in sorted(my_paths)]
            self.__paths = []
            redo_path = False
            for idx in self.__idxs_to_keep:
                try:
                    self.__paths += [my_paths[idx]]
                except IndexError:
                    print(f"{PrintColors.WARNING}missing tif files in:\n\t{directory}{PrintColors.ENDC}")
                    for i in range(self.__idxs_to_keep[-1]):
                        file = os.path.join(directory, f"{int(i * self.z_step[0] * 10):06}{self.suffix}")
                        if not os.path.exists(file):
                            print(f"\t\tthe following missing files is replaced with a dummy (zeros) image:\n"
                                  f"\t\t\t{file}")
                            imwrite(file, zeros(self.shape[1:3], dtype=self.dtype))
                    # raise RuntimeError
                    redo_path = True
            if redo_path:
                self.__paths = []
                for idx in self.__idxs_to_keep:
                    self.__paths += [my_paths[idx]]

        return self.__paths


class TSVSimpleStack(TSVStackBase):
    def __init__(self, row, column, x0, y0, z0, root):
        TSVStackBase.__init__(self)
        self.row = row
        self.column = column
        self._x0 = x0
        self._y0 = y0
        self._z0 = z0
        self.root: Path = root
        self.__paths = None
        self.z0slice = 0

    @property
    def paths(self):
        if self.__paths is None:
            self.__paths = sorted(self.root.glob("*.raw"))
            if len(self.__paths) > 0:
                self.input_plugin = "raw"
            else:
                self.__paths = sorted(self.root.glob("*.tif*"))
                self.input_plugin = "tiff2D"
            self.z1slice = len(self.__paths)
        if getattr(os, "fspath", None) is not None:
            return [os.fspath(_) for _ in self.__paths]
        return [str(_) for _ in self.__paths]


def compute_cosine(volume: VExtentBase, stack: TSVStack, ostack: TSVStack, img: ndarray):
    """
    Given two overlapping stacks, compute the cosine blend between them

    volume:
        the volume being blended
    stack:
        the stack from which the data is being taken
    ostack:
        the stack that overlaps it
    img:
        reduce the intensity using the cosine blend on this image
    """
    if not volume.intersects(ostack):
        return
    iv = volume.intersection(ostack)
    #
    # Find the distance to the edge of the intersecting volume per voxel
    #
    d = get_distance_from_edge(iv, stack, ostack)
    od = get_distance_from_edge(iv, ostack, stack)
    if np_min(d) == inf:
        d[:] = np_max(od)
    elif np_min(od) == inf:
        od[:] = np_max(d)
    #
    # Use the ratio of the two distances to get an "angle". The angle will
    # be 45 degrees if the voxels are the same distance away from the edge
    # and the two stacks will be equally blended. If there is a big difference
    # then the blending will favor the volume that is further away.
    #
    angle = arctan2(d, od)
    blending = sin(angle) ** 2
    img[iv.z0 - volume.z0:iv.z1 - volume.z0,
        iv.y0 - volume.y0:iv.y1 - volume.y0,
        iv.x0 - volume.x0:iv.x1 - volume.x0] *= blending.astype(img.dtype)


class Edge(enum.Flag):
    """Keep track of which edge or edges have some property"""

    """The X0 edge of the volume"""
    XMIN = enum.auto()

    """The X1 edge of the volume"""
    XMAX = enum.auto()

    """The Y0 edge of the volume"""
    YMIN = enum.auto()

    """The Y1 edge of the volume"""
    YMAX = enum.auto()

    """The Z0 edge of the volume"""
    ZMIN = enum.auto()

    """The Z1 edge of the volume"""
    ZMAX = enum.auto()


def get_distance_from_edge(tgt: VExtentBase, stack: VExtentBase, ostack: VExtentBase) -> ndarray:
    """For the volume, get the distance per voxel to the nearest edge

    tgt:
        the target volume to be filled
    stack:
        The stack on which to make the distance estimate
    ostack:
        The stack that is overlapping

    returns:
        an array, similarly sized to the overlap volume, giving the minimum distance to the nearest edge.
    """
    edges = Edge(0)
    if ostack.x1 > stack.x0 > ostack.x0:
        edges = edges | Edge.XMIN
    if ostack.x0 < stack.x1 < ostack.x1:
        edges = edges | Edge.XMAX
    if ostack.y1 > stack.y0 > ostack.y0:
        edges = edges | Edge.YMIN
    if ostack.y0 < stack.y1 < ostack.y1:
        edges = edges | Edge.YMAX
    volume = stack.intersection(ostack)
    assert volume.contains(tgt)
    #
    # Start out with all voxels maximally far from the edge
    #
    max_distance = inf
    if ostack.x1 != stack.x1 and ostack.x0 != stack.x0:
        max_distance = volume.shape[2]
    if ostack.y1 != stack.y1 and ostack.y0 != stack.y0:
        max_distance = min(max_distance, volume.shape[1])
    #
    # Blend z edges if and only if the x and y extents are the
    # entire range
    #
    if isinf(max_distance) and ostack.z1 != stack.z1 and ostack.z0 != stack.z0:
        max_distance = min(max_distance, volume.shape[0])
        if ostack.z1 > stack.z0 > ostack.z0:
            edges = edges | Edge.ZMIN
        if ostack.z0 < stack.z1 < ostack.z1:
            edges = edges | Edge.ZMAX
    result = ones(tgt.shape, float32) * max_distance
    #
    # Process the starting edges
    #
    for idx, flag in enumerate((Edge.ZMIN, Edge.YMIN, Edge.XMIN)):
        if edges & flag:
            slices = [newaxis] * 3
            slices[idx] = slice(0, tgt.shape[idx])
            result = minimum(
                result,
                arange(tgt.start(idx) - volume.start(idx) + 1,
                       tgt.end(idx) - volume.start(idx) + 1)[tuple(slices)])
    #
    # Process the ending edges
    #
    for idx, flag in enumerate((Edge.ZMAX, Edge.YMAX, Edge.XMAX)):
        if edges & flag:
            slices = [newaxis] * 3
            slices[idx] = slice(0, tgt.shape[idx])
            result = minimum(
                result,
                arange(volume.end(idx) - tgt.start(idx),
                       volume.end(idx) - tgt.end(idx), -1)[tuple(slices)])
    return result


class TSVVolumeBase:
    def flattened_stacks(self):
        """
        Return the stacks in row-major order
        """
        return sum(self.stacks, [])

    def imread(self, volume, dtype, cosine_blending: True):
        """Read the given volume

        volume:
            a VExtent delimiting the volume to read
        dtype:
            the numpy dtype of the array to be returned

        returns:
            the array corresponding to the volume (with zeros for data outside the array).
        """

        intersections = []
        for stack in self.flattened_stacks():
            if stack.intersects(volume):
                intersections.append((stack, stack.intersection(volume)))

        if cosine_blending:
            result = zeros(volume.shape, float32)
            multiplier = zeros(volume.shape, float32)
            for stack, intersection in intersections:
                part = stack.imread(intersection).astype(float32)
                mpart = ones(intersection.shape, float32)
                #
                # Look for overlaps and perform a cosine blending
                #
                inter_intersections = []
                for ostack, ointersection in intersections:
                    if ostack == stack:
                        continue
                    if ointersection.intersects(intersection):
                        inter_intersections.append((ostack, ointersection))
                if len(inter_intersections) > 0:
                    for ostack, ointersection in inter_intersections:
                        compute_cosine(intersection, stack, ostack, part)
                        compute_cosine(intersection, stack, ostack, mpart)
                result[
                    intersection.z0 - volume.z0:intersection.z1 - volume.z0,
                    intersection.y0 - volume.y0:intersection.y1 - volume.y0,
                    intersection.x0 - volume.x0:intersection.x1 - volume.x0
                ] += part
                multiplier[
                    intersection.z0 - volume.z0:intersection.z1 - volume.z0,
                    intersection.y0 - volume.y0:intersection.y1 - volume.y0,
                    intersection.x0 - volume.x0:intersection.x1 - volume.x0
                ] += mpart
            result /= (multiplier + finfo(float32).eps)
            if result.dtype != dtype and np_d_type(dtype).kind in ("u", "i"):
                result = around(result, 0)
                clip(result, iinfo(dtype).min, iinfo(dtype).max, out=result)
                result = result.astype(dtype)
        else:
            result = zeros(volume.shape, dtype)
            for stack, intersection in intersections:
                part = stack.imread(intersection)
                res_part = result[
                    intersection.z0 - volume.z0:intersection.z1 - volume.z0,
                    intersection.y0 - volume.y0:intersection.y1 - volume.y0,
                    intersection.x0 - volume.x0:intersection.x1 - volume.x0
                ]
                result[
                    intersection.z0 - volume.z0:intersection.z1 - volume.z0,
                    intersection.y0 - volume.y0:intersection.y1 - volume.y0,
                    intersection.x0 - volume.x0:intersection.x1 - volume.x0
                ] = maximum(res_part, part)

        return result

    def make_diagnostic_img(self, volume: VExtentBase):
        """Create a diagnostic image with separate channels for each stack

        :param volume: The coordinates to use to extract the diagnostic image
        :returns: a 4D array with the last index being the channel and the
        indices of the channel being the stacks intersecting the volume in
        row-major order.
        """
        stacks = [s for s in self.flattened_stacks() if s.intersects(volume)]
        result = zeros(
            (volume.shape[0], volume.shape[1], volume.shape[2], len(stacks)),
            self.dtype)
        for idx, stack in enumerate(stacks):
            intersection = stack.intersection(volume)
            img = stack.imread(intersection)
            result[
                intersection.z0 - volume.z0:intersection.z1 - volume.z0,
                intersection.y0 - volume.y0:intersection.y1 - volume.y0,
                intersection.x0 - volume.x0:intersection.x1 - volume.x0,
                idx
            ] = img.astype(result.dtype)
        return result

    @property
    def volume(self) -> VExtent:
        """The VExtent of the volume"""
        x0 = y0 = z0 = iinfo(int32).max
        x1 = y1 = z1 = 0
        for stack in sum(self.stacks, []):
            x0 = min(x0, stack.x0)
            x1 = max(x1, stack.x1)
            y0 = min(y0, stack.y0)
            y1 = max(y1, stack.y1)
            z0 = min(z0, stack.z0)
            z1 = max(z1, stack.z1)
        return VExtent(x0, x1, y0, y1, z0, z1)



class TSVVolume(TSVVolumeBase):
    def __init__(self, tree_xml_path, ignore_z_offsets=False, alt_stack_dir=None):
        """
        Initialize from a xml.etree.ElementTree

        Parameters
        ----------
        tree_xml_path:
            a xml tree generated by Terastitcher
        ignore_z_offsets:
            if True, ignore Z offsets. (e.g. if the stage always returns to the same Z coordinate repeatably)
        alt_stack_dir:
            alternative directory of stacks for another channel
        """
        tree = ElementTree.parse(tree_xml_path)
        root = tree.getroot()
        assert root.tag == "TeraStitcher"
        dims = root.find("dimensions")
        self.stack_rows = int(dims.attrib["stack_rows"])
        self.stack_columns = int(dims.attrib["stack_columns"])
        self.stacks = [[TSVStack] * self.stack_columns for _ in range(self.stack_rows)]
        self.offsets = [[Location] * self.stack_columns for _ in range(self.stack_rows)]
        self.stack_slices = int(dims.attrib["stack_slices"])
        self.voxel_dims = get_dim_tuple(root.find("voxel_dims"))  # zyx order
        if alt_stack_dir is None:
            self.stacks_dir = root.find("stacks_dir").attrib["value"]
        else:
            self.stacks_dir = alt_stack_dir
        self.ignore_z_offsets = ignore_z_offsets
        self.origin = get_dim_tuple(root.find("origin"))
        self.input_plugin = root.attrib["input_plugin"]
        self.volume_format = root.attrib["volume_format"]

        md = root.find("mechanical_displacements")
        self.mechanical_displacement_x = float(md.attrib["H"])
        self.mechanical_displacement_y = float(md.attrib["V"])
        self.make_stacks(root)

    def make_stacks(self, root):
        """
        Parse and properly offset the stacks

        Parameters
        ----------
        root:
            the root node of the xml
        """
        stacks = root.find("STACKS")
        selems = [[ElementTree.Element] * self.stack_columns for _ in range(self.stack_rows)]
        # self.stacks = [[None] * self.stack_columns for _ in range(self.stack_rows)]
        # self.offsets = [[None] * self.stack_columns for _ in range(self.stack_rows)]
        self.offsets[0][0] = Location(0, 0, 0)
        for child in stacks.iter(tag="Stack"):
            row = int(child.attrib["ROW"])
            column = int(child.attrib["COL"])
            selems[row][column] = child
        for row, elements in enumerate(selems):
            for column, child in enumerate(elements):
                if row > 0:
                    prev = self.offsets[row - 1][column]
                    dn = list(child.find("NORTH_displacements"))[0]
                    xoff = -int(dn.find("H").attrib["displ"])
                    yoff = -int(dn.find("V").attrib["displ"])
                    zoff = 0 if self.ignore_z_offsets \
                        else -int(dn.find("D").attrib["displ"])
                    offset = Location(prev.x + xoff,
                                      prev.y + yoff,
                                      prev.z + zoff)
                    self.offsets[row][column] = offset
                elif column > 0:
                    prev = self.offsets[row][column - 1]
                    dn = list(child.find("WEST_displacements"))[0]
                    xoff = -int(dn.find("H").attrib["displ"])
                    yoff = -int(dn.find("V").attrib["displ"])
                    zoff = 0 if self.ignore_z_offsets \
                        else -int(dn.find("D").attrib["displ"])
                    offset = Location(prev.x + xoff,
                                      prev.y + yoff,
                                      prev.z + zoff)
                    self.offsets[row][column] = offset
        #
        # Find the minimum absolute offset for x, y, z
        #
        min_x = min_y = min_z = iinfo(uint32).max
        for offset in sum(self.offsets, []):
            min_x = min(min_x, offset.x)
            min_y = min(min_y, offset.y)
            min_z = min(min_z, offset.z)
        #
        # Rebase the offsets so that coordinates are all positive and start
        # at zero
        #
        for row, column in itertools.product(range(self.stack_rows),
                                             range(self.stack_columns)):
            offset = self.offsets[row][column]
            offset = Location(offset.x - min_x,
                              offset.y - min_y,
                              offset.z - min_z)
            self.offsets[row][column] = offset
            self.stacks[row][column] = TSVStack(
                selems[row][column],
                offset,
                self.stacks_dir,
                input_plugin=self.input_plugin,
                z_step=self.voxel_dims
            )

    @property
    def dtype(self):
        """The dtype inferred from the stack's bit-depth"""
        if self.stacks[0][0].bytes_per_chan == 1:
            return uint8
        elif self.stacks[0][0].bytes_per_chan < 9:
            return uint16
        else:
            return uint32


class TSVSimpleVolume(TSVVolumeBase):
    """
    A volume created from a directory parse + voxel size
    """

    def __init__(self, root_dir, voxel_size_x, voxel_size_y, voxel_size_z):
        """
        :param root_dir: The root directory of the directory tree. It should
        be laid out as <root_dir>/<x-location-in-tenths-of-microns>/
        <x-location-in-tenths-of-microns>_<y-location-in-tenths-of-microns>
        :param voxel_size_x: The size of a voxel in the x direction
        :param voxel_size_y: The size of a voxel in the Y direction
        :param voxel_size_z: The size of a voxel in the Z direction
        """
        self.stacks_dir = root_dir
        self.voxel_dims = (voxel_size_z, voxel_size_y, voxel_size_x)
        root_path = Path(root_dir)
        xdirs = sorted([_ for _ in root_path.glob("*") if _.is_dir() and re.match("[0-9]+", _.name)])
        ydirs = [sorted([__ for __ in _.glob("*") if __.is_dir() and re.match("[0-9]+_[0-9]+", __.name)])
                 for _ in xdirs]
        self.stack_rows = len(ydirs[0])
        self.stack_columns = len(ydirs)
        self.stacks = [[TSVSimpleStack] * self.stack_columns for _ in range(self.stack_rows)]
        self.offsets = [[(0, 0, 0)] * self.stack_columns for _ in range(self.stack_rows)]
        self.stack_slices = len(list(ydirs[0][0].glob("*.raw")))
        if self.stack_slices == 0:
            self.stack_slices = len(list(ydirs[0][0].glob("*.tif*")))
        #
        # Make the offsets and the stacks
        #
        x0, y0 = 0, 0
        for xi, yd in enumerate(ydirs):
            for yi, ydir in enumerate(yd):
                x, y = [int(_) for _ in ydir.name.split("_")]
                if xi == 0:
                    xloc = 0
                    x0 = x
                else:
                    xloc = int((x - x0) / voxel_size_x / 10.0)
                if yi == 0:
                    yloc = 0
                    y0 = y
                else:
                    yloc = int((y - y0) / voxel_size_y / 10.0)
                self.offsets[yi][xi] = (xloc, yloc, 0)
                self.stacks[yi][xi] = TSVSimpleStack(yi, xi, xloc, yloc, 0, ydir)

    @property
    def dtype(self):
        return uint16
