import itertools
import logging
import multiprocessing
import numpy as np
import os
import pathlib
from scipy.ndimage import zoom, distance_transform_edt
import tifffile
import tqdm
import typing

from .raw import raw_imread
from .volume import TSVStackBase, TSVVolumeBase, VExtent


def imread(path:pathlib.Path) -> np.ndarray:
    if path.name.endswith(".raw"):
        img = raw_imread(os.fspath(path)).astype(np.float32)
    else:
        img = tifffile.imread(os.fspath(path)).astype(np.float32)
    return img


def zcoord(path:pathlib.Path) -> float:
    """
    Return the putative Z coordinate for a image file path name
    """
    return int(path.name.split(".")[0]) / 10

class ScanStack(TSVStackBase):
    """
    A TSVStack that holds a single piezo travel of paths.
    Within the stack, we assume that the frames are not perfectly aligned,
    but they vary with linear offsets per z in x and y. We make the stack
    look square by trimming a bit off the x and y starts and stops. This
    might be a couple pixels - nothing extreme.
    """
    def __init__(self, x0, y0, z0, z0slice, z1slice, paths):
        super(ScanStack, self).__init__()
        self._x0 = self.x0_orig  = x0
        self._y0 = self.y0_orig = y0
        self._z0 = self.z0_orig = z0
        self.x0_trim = 0
        self.x1_trim = 0
        self.y0_trim = 0
        self.y1_trim = 0
        self.x_off_per_z = 0.0
        self.y_off_per_z = 0.0
        self.__paths = paths
        if self.__paths[0].name.endswith(".raw"):
            self.input_plugin = "raw"
        else:
            self.input_plugin = "tiff2D"
        self.z0slice = z0slice
        self.z1slice = z1slice
        self.x_aligned = False
        self.y_aligned = False
        self.z_aligned = False

    @property
    def paths(self):
        return self.__paths

    def x0_getter(self):
        return self._x0 + self.x0_trim

    def x0_setter(self, x0_new):
        self._x0 = int(x0_new) - self.x0_trim
        self.x_aligned = True

    x0 = property(x0_getter, x0_setter)

    @property
    def x1(self):
        self._set_x1y1()
        return self._x0 + self._width  - self.x1_trim

    def y0_getter(self):
        return self._y0 + self.y0_trim

    def y0_setter(self, new_y0):
        self._y0 = int(new_y0) - self.y0_trim
        self.y_aligned = True

    y0 = property(y0_getter, y0_setter)

    @property
    def y1(self):
        return self._y0 + self._height - self.y1_trim

    def z0_getter(self):
        return super(ScanStack, self).z0

    def z0_setter(self, new_z0):
        self._z0 = int(new_z0)
        self.z_aligned = True

    z0 = property(z0_getter, z0_setter)

    def reset(self):
        """Reset to original coordinates"""
        self._x0 = self.x0_orig
        self._y0 = self.y0_orig
        self._z0 = self.z0_orig

    def read_plane(self, path:pathlib.Path):
        z = zcoord(path) - self.z0slice
        x_off = int(self.x_off_per_z * z + .5)
        y_off = int(self.y_off_per_z * z + .5)
        img = super(ScanStack, self).read_plane(os.fspath(path))
        x0 = self.x0_trim - x_off
        x1 = img.shape[1] - self.x1_trim + x_off
        y0 = self.y0_trim - y_off
        y1 = img.shape[1] - self.y1_trim + y_off
        return img[y0:y1, x0:x1]

    def as_dict(self) -> dict:
        path0 = self.paths[0]
        orig_z0 = zcoord(path0)
        folder = os.path.split(os.path.dirname(path0))[-1]
        orig_x0, orig_y0 = [float(_) / 10 for _ in folder.split("_")]
        return dict(x0=self.x0,
                    x1=self.x1,
                    y0=self.y0,
                    y1=self.y1,
                    z0=self.z0,
                    z1=self.z1,
                    orig_x0=orig_x0,
                    orig_y0=orig_y0,
                    orig_z0=orig_z0,
                    path=os.path.dirname(path0))

class AverageDrift:
    """
    The average drift in the x, y and z direction between adjacent x, y and z
    stacks - if the stage axes don't exactly align to the objective axes,
    this will be the chief component in the offsets.
    """
    def __init__(self,
                 xoffx:int, yoffx:int, zoffx:int,
                 xoffy:int, yoffy:int, zoffy:int,
                 xoffz:int, yoffz:int, zoffz:int):
        self.xoffx = xoffx
        self.yoffx = yoffx
        self.zoffx = zoffx
        self.xoffy = xoffy
        self.yoffy = yoffy
        self.zoffy = zoffy
        self.xoffz = xoffz
        self.yoffz = yoffz
        self.zoffz = zoffz

ALIGNMENT_RESULT_T = typing.Tuple[int, typing.Tuple[float, int, int, int]]

class Scanner(TSVVolumeBase):

    def __init__(self,
                 path:pathlib.Path,
                 voxel_size:typing.Tuple[float, float, float],
                 z_stepper_distance=297,
                 piezo_distance=300,
                 z_skip=25,
                 x_slop=5,
                 y_slop=5,
                 z_slop=3,
                 dark=200,
                 drift=None,
                 decimate=1,
                 min_support=5,
                 n_cores=os.cpu_count(),
                 loose_x=False):
        """
        Initialize the scanner with the root path to the directory hierarchy
        and the voxel dimensions
        :param path: the path to the root of the hierarchy. It's assumed
        that the first subdirectory level is the X coordinate of the stack,
        in 10ths of microns and the second level is in the form X_Y (again,
        10ths of microns) and the third level is the Z coordinate (in 10ths
        of microns).
        :param voxel_size: The voxel size in microns, x, y and z
        :param z_stepper_distance: the distance in microns of the (alleged)
        travel of the coarse z-stepper as it takes a single step
        :param piezo_distance: the distance in microns of the (alleged) travel
        of the piezo mini-stepper and the big step size of the Z motor.
        :param z_skip: align every z_skip'th plane.
        :param min_support: don't make a guess unless at least this many alignments are over threshold
        :param path_weight: add an extra weight to 1-score when calculating the
        graph connecting blocks to favor shorter paths.
        :param loose_x: interpret X offsets loosely, with different ones per Y
        """
        self.pool = None
        self.futures_x = {}
        self.futures_y = {}
        self.futures_z = {}
        self.alignments_x = {}
        self.alignments_y = {}
        self.alignments_z = {}
        self._stacks = {}
        self.decimate = decimate
        self.min_support = min_support
        self.x_voxel_size, self.y_voxel_size, self.z_voxel_size = voxel_size
        self.z_skip = z_skip
        self.x_slop = x_slop
        self.y_slop = y_slop
        self.z_slop = z_slop
        self.loose_x = loose_x
        self.dark = dark
        if drift is None:
            self.drift = AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0)
        else:
            self.drift = drift
        self.n_cores = n_cores
        stacks = {}
        for folder in path.iterdir():
            if not folder.is_dir():
                continue
            try:
                x = int(float(folder.name) / self.x_voxel_size / 10)
            except ValueError:
                continue
            for subfolder in folder.iterdir():
                if not subfolder.is_dir():
                    continue
                try:
                    y = int(float(subfolder.name.split("_")[1]) /
                            self.y_voxel_size / 10)
                except:
                    continue
                logging.info("Collecting files for x=%d, y=%d" % (x, y))
                img_paths = sorted(subfolder.glob("*.raw"))
                input_plugin = "raw"
                if len(img_paths) == 0:
                    input_plugin = "tiff2D"
                    img_paths = sorted(subfolder.glob("*.tif*"))
                    if len(img_paths) == 0:
                        continue
                stack_paths = []
                img_path_and_z = sorted(
                    [(int(_.name.rsplit(".", 1)[0]) / 10, _)
                     for _ in img_paths])
                z0 = img_path_and_z[0][0]
                zbase = z0
                current_stack = []
                for z, path in img_path_and_z:
                    if z - z0 >= piezo_distance:
                        stack_paths.append((zbase, current_stack))
                        current_stack = []
                        zbase += z_stepper_distance
                        z0 = z
                    current_stack.append(path)
                stack_paths.append((zbase, current_stack))
                for z, current_stack in stack_paths:
                    z0slice = int(zcoord(current_stack[0]) / self.z_voxel_size)
                    z1slice = int(zcoord(current_stack[-1])/self.z_voxel_size) \
                              + 1
                    stacks[x, y, z] = ScanStack(
                        x, y, int(z / self.z_voxel_size),
                        z0slice, z1slice, current_stack)
        self.xs, self.ys, self.zs = \
            [sorted(set([_[idx] for _ in stacks.keys()]))
                for idx in range(3)]
        self.n_x, self.n_y, self.n_z = [len(_) for _ in (self.xs, self.ys, self.zs)]
        any_stack = next(iter(stacks.values()))
        height, width = imread(any_stack.paths[0]).shape
        for (x, y, z), stack in stacks.items():
            xidx = self.xs.index(x)
            yidx = self.ys.index(y)
            zidx = self.zs.index(z)
            stack.set_size(width, height)
            self._stacks[xidx, yidx, zidx] = stack

    @property
    def stacks(self):
        return [list(self._stacks.values())]

    def reset(self):
        """Set all stacks to original values"""
        for stack in self._stacks.values():
            stack.reset()

    def setup(self, x_slop:int, y_slop:int, z_slop:int, z_skip:int,
              decimate:int, drift:AverageDrift):
        """
        Set up for another round

        :param x_slop: The number of voxels to check in the X direction
        :param y_slop: The number of voxels to check in the Y direction
        :param z_slop: the number of voxels to check in the Z direction
        :param z_skip: Do every z_skip plane in a stack
        :param decimate: the image decimation factor
        :param drift: The calculated mean offsets from the last round
        """
        logging.info("Parameters for next level:")
        logging.info("  x_slop: %d" % x_slop)
        logging.info("  y_slop: %d" % y_slop)
        logging.info("  z_slop: %d" % z_slop)
        logging.info("  decimate: %d" % decimate)
        logging.info("  drift: xx: %d yx: %d zx: %d" %
                     (drift.xoffx, drift.yoffx, drift.zoffx))
        logging.info("         xy: %d yy: %d zy: %d" %
                     (drift.xoffy, drift.yoffy, drift.zoffy))
        logging.info("         xz: %d yz: %d zz: %d" %
                     (drift.xoffz, drift.yoffz, drift.zoffz))

        self.futures_x = {}
        self.futures_y = {}
        self.futures_z = {}
        self.alignments_x = {}
        self.alignments_y = {}
        self.alignments_z = {}
        self.decimate = decimate
        self.z_skip = z_skip
        self.x_slop = x_slop
        self.y_slop = y_slop
        self.z_slop = z_slop
        if drift is None:
            self.drift = AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0)
        else:
            self.drift = drift

    def align_all_stacks(self):
        with multiprocessing.Pool(self.n_cores) as self.pool:
            if len(self.xs) > 1:
                self.align_stacks_x()
            if len(self.ys) > 1:
                self.align_stacks_y()
            if len(self.zs) > 1:
                self.align_stacks_z()
            acc = 0
            for src in (self.futures_x, self.futures_y, self.futures_z):
                for k in src:
                    acc += len(src[k])
            bar = tqdm.tqdm(total=acc)
            for src, dest in ((self.futures_x, self.alignments_x),
                              (self.futures_y, self.alignments_y),
                              (self.futures_z, self.alignments_z)):
                for k in src:
                    dest[k] = []
                    for z, future in src[k]:
                        dest[k].append((z, future.get()))
                        bar.update()

    def align_stacks_x(self):
        """
        Align each stack to the one next to it in the X direction
       """
        for xidx, yidx, zidx in itertools.product(
            range(len(self.xs) - 1), range(len(self.ys)), range(len(self.zs))):
            k0 = (xidx, yidx, zidx)
            k1 = (xidx+1, yidx, zidx)
            if k1 not in self._stacks or k0 not in self._stacks:
                continue
            s0 = self._stacks[k0]
            s1 = self._stacks[k1]
            self.futures_x[xidx, yidx, zidx] = self.align_stack_x(s0, s1)

    def align_stacks_y(self):
        """
        Align each stack to the one next to it in the Y direction
        """
        for xidx, yidx, zidx in itertools.product(
            range(len(self.xs)), range(len(self.ys)-1), range(len(self.zs))):
            k0 = (xidx, yidx, zidx)
            k1 = (xidx, yidx+1, zidx)
            if k1 not in self._stacks or k0 not in self._stacks:
                continue
            s0 = self._stacks[k0]
            s1 = self._stacks[k1]
            self.align_stack_y(s0, s1)
            self.futures_y[xidx, yidx, zidx] = self.align_stack_y(s0, s1)

    def align_stacks_z(self):
        """
        Align each stack to the one next to it in the Z direction
        """
        for xidx, yidx, zidx in itertools.product(
            range(len(self.xs)), range(len(self.ys)), range(len(self.zs)-1)):
            k0 = (xidx, yidx, zidx)
            k1 = (xidx, yidx, zidx+1)
            if k1 not in self._stacks or k0 not in self._stacks:
                continue
            s0 = self._stacks[k0]
            s1 = self._stacks[k1]
            self.futures_z[xidx, yidx, zidx] = self.align_stack_z(s0, s1)

    def align_stack_x(self, s0:ScanStack, s1:ScanStack):
        """
        Align stacks that are overlapping in the X direction
        :param s0:
        :type s0:
        :param s1:
        :type s1:
        """
        xc = s1.x0 - s0.x0 + self.drift.xoffx
        x0 = xc - self.x_slop
        x1 = xc + self.x_slop + 1
        y0 = -self.y_slop + self.drift.yoffx
        y1 = self.y_slop + 1 + self.drift.yoffx
        z0m = max(0, self.z_slop + self.drift.zoffx)
        z1m = min(len(s1.paths),
                  len(s1.paths) - self.z_slop + self.drift.zoffx - 1)
        futures = []
        if self.z_skip == "middle":
            zrange = [(z0m + z1m) // 2]
        else:
            zrange = range(z0m, z1m, self.z_skip)
        for z in zrange:
            z0 = z - self.z_slop - self.drift.zoffx
            z1 = z + self.z_slop + 1 - self.drift.zoffx
            futures.append((z, self.pool.apply_async(
                align_one_x,
                (s1.paths[z],
                 s0.paths[z0:z1],
                 x0, x1, y0, y1, z0 - z,
                 self.dark,
                 self.decimate)

            )))
        return futures

    def align_stack_y(self, s0:ScanStack, s1:ScanStack):
        yc = s1.y0 - s0.y0 + self.drift.yoffy
        y0 = yc - self.y_slop
        y1 = yc + self.y_slop + 1
        x0 = -self.x_slop + self.drift.xoffy
        x1 = self.x_slop + 1 + self.drift.xoffy
        z0m = max(0, self.z_slop + self.drift.zoffx)
        z1m = min(len(s1.paths),
                  len(s1.paths) - self.z_slop + self.drift.zoffx - 1)
        futures = []
        if self.z_skip == "middle":
            zrange = [(z0m + z1m) // 2]
        else:
            zrange = range(z0m, z1m, self.z_skip)
        for z in zrange:
            z0 = z - self.z_slop - self.drift.zoffx
            z1 = z + self.z_slop + 1 - self.drift.zoffx
            logging.info("Aligning %s to %s:%s" %
                         (s1.paths[z], s0.paths[z0], s0.paths[z1]))
            logging.info("x0: %d, x1: %d, y0: %d, y1: %d" %
                         (x0, x1, y0, y1))
            futures.append((z, self.pool.apply_async(
                align_one_y,
                (s1.paths[z],
                 s0.paths[z0:z1],
                 x0, x1, y0, y1, z0 - z,
                 self.dark,
                 self.decimate)
            )))
        return futures

    def align_stack_z(self, s0:ScanStack, s1:ScanStack):
        x0 = -self.x_slop + self.drift.xoffz
        x1 = self.x_slop + 1 + self.drift.xoffz
        y0 = -self.y_slop + self.drift.yoffz
        y1 = self.y_slop + 1 + self.drift.yoffz
        s0_paths = s0.paths[-self.z_slop:]
        s1_path = s1.paths[0]
        future = self.pool.apply_async(
                align_one_z, (s0_paths, s1_path, x0, x1, y0, y1,
                              -self.z_slop, self.dark, self.decimate))
        return [[0, future]]

    def compute_median_min_max_without_outliers(self, offs, stds):
        median = np.median(offs)
        off_std = np.std(offs) * stds
        offs = [_ for _ in offs
                if _ >= median - off_std and _ <= median + off_std]
        median = np.median(offs)
        minimum = np.min(offs)
        maximum = np.max(offs)
        return median, minimum, maximum

    def accumulate_offsets(self, alignments, threshold, stds):
        xoffs = []
        yoffs = []
        zoffs = []
        for xidx, yidx, zidx in alignments:
            for z, (score, xoff, yoff, zoff) in alignments[xidx, yidx, zidx]:
                if score < threshold:
                    continue
                xoffs.append(xoff)
                yoffs.append(yoff)
                zoffs.append(zoff)
        xmedian, xmin, xmax = self.compute_median_min_max_without_outliers(
            xoffs, stds)
        ymedian, ymin, ymax = self.compute_median_min_max_without_outliers(
            yoffs, stds)
        zmedian, zmin, zmax = self.compute_median_min_max_without_outliers(
            zoffs, stds
        )
        return (xmedian, xmin, xmax),\
               (ymedian, ymin, ymax),\
               (zmedian, zmin, zmax)

    def calculate_next_round_parameters(self, threshold=.75, stds=3.0,
                                        slop_factor=1.25):
        (xoffx, xminx, xmaxx), (yoffx, yminx, ymaxx), (zoffx, zminx, zmaxx) = \
        self.accumulate_offsets(self.alignments_x, threshold, stds)
        (xoffy, xminy, xmaxy), (yoffy, yminy, ymaxy), (zoffy, zminy, zmaxy) = \
        self.accumulate_offsets(self.alignments_y, threshold, stds)
        (xoffz, xminz, xmaxz), (yoffz, yminz, ymaxz), (zoffz, zminz, zmaxz) = \
        self.accumulate_offsets(self.alignments_z, threshold, stds)
        x_slop = int(max(xmaxx - xoffx, xoffx - xminx,
                         xmaxy - xoffy, xoffy - xminy,
                         xmaxz - xoffz, xoffz - xminz) *
                     slop_factor) + self.decimate
        y_slop = int(max(ymaxx - yoffx, yoffx - yminx,
                         ymaxy - yoffy, yoffy - yminy,
                         ymaxz - yoffz, yoffz - yminz) *
                     slop_factor) + self.decimate
        z_slop = int(max(zmaxx - zoffx, zoffx - zminx,
                         zmaxy - zoffy, zoffy - zminy,
                         zmaxz - zoffz, zoffz - zminz) *
                     slop_factor) + self.decimate
        drift = AverageDrift(int(xoffx), int(yoffx), int(zoffx),
                             int(xoffy), int(yoffy), int(zoffy),
                             int(xoffz), int(yoffz), int(zoffz))
        self.flat_adjust_stacks(threshold)
        self.setup(int(x_slop), int(y_slop), int(z_slop), self.z_skip,
                   int(self.decimate // 2),
                   AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0))

    KEY_t = typing.Tuple[int, int, int]
    PATH_t = typing.Sequence[KEY_t]
    SCORE_AND_OFFS_t = typing.Tuple[float, int, int, int]
    
    def get_alignment(self, k0:KEY_t, k1:KEY_t) -> SCORE_AND_OFFS_t:
        """
        Get the appropriate alignment (alignment_x, alignment_y or alignment_z) for the edge
        represented by k0 and k1
        
        :param k0: the source in the path
        :param k1: the destination in the path
        :return: a four tuple of score, xoffset, yoffset and zoffset
        """
        if any([ke0 > ke1 for ke0, ke1 in zip(k0, k1)]):
            score, xoff, yoff, zoff = self.get_alignment(k1, k0)
            return score, -xoff, -yoff, -zoff
        for idx, alignment in enumerate((self.alignments_x, self.alignments_y, self.alignments_z)):
            if k0[idx] == k1[idx] - 1:
                best_score = 0
                best_xoff = 0
                best_yoff = 0
                best_zoff = 0
                for z, (score, xoff, yoff, zoff) in alignment[k0]:
                    if score > best_score:
                        best_score = score
                        best_xoff = xoff
                        best_yoff = yoff
                        best_zoff = zoff
                return best_score, best_xoff, best_yoff, best_zoff
        raise ValueError("Maybe %s and %s are not adjacent?" % (k0, k1))

    def get_alignments(self, k0: KEY_t, k1: KEY_t, threshold:float) -> typing.Sequence[SCORE_AND_OFFS_t]:
        """
        Get all alignments above a threshold (alignment_x, alignment_y or alignment_z) for the edge
        represented by k0 and k1

        :param k0: the source in the path
        :param k1: the destination in the path
        :return: a four tuple of score, xoffset, yoffset and zoffset
        """
        if any([ke0 > ke1 for ke0, ke1 in zip(k0, k1)]):
            alignments = self.get_alignments(k1, k0)
            return [(score, -xoff, -yoff, -zoff) for score, xoff, yoff, zoff in alignments]
        alignments = []
        for idx, alignment in enumerate((self.alignments_x, self.alignments_y, self.alignments_z)):
            if k0[idx] == k1[idx] - 1:
                for z, (score, xoff, yoff, zoff) in alignment[k0]:
                    if score > threshold:
                        alignments.append((score, xoff, yoff, zoff))
                return alignments
        raise ValueError("Maybe %s and %s are not adjacent?" % (k0, k1))

    def check_key(self, direction:str, key:KEY_t)->bool:
        """
        Checks to see if a key has an edge in the alignments dictionary

        :param direction: "x", "y" or "z"
        :param key: the starting node
        :return: True if there is another node in the volume in the +1 direction
        """
        xidx, yidx, zidx = key
        xend, yend, zend = [len(_) for _ in (self.xs, self.ys, self.zs)]
        if direction == "x":
            xend -= 1
        elif direction == "y":
            yend -= 1
        else:
            zend -= 1
        return 0 <= xidx < xend and 0 <= yidx < yend and 0 <= zidx < zend

    def check_path(self, path:PATH_t)->bool:
        """
        Check to make sure all elements of a path are in-bounds

        :param path: a path from a source key to a destination key
        :return: True if the path is in the bounds of the volume
        """
        for k0, k1 in zip(path[:-1], path[1:]):
            xidx0, yidx0, zidx0 = k0
            xidx1, yidx1, zidx1 = k1
            if xidx1 == xidx0 + 1:
                test = self.check_key("x", k1)
            elif xidx0 == xidx1 + 1:
                test = self.check_key("x", k0)
            elif yidx1 == yidx0 + 1:
                test = self.check_key("y", k1)
            elif yidx0 == yidx1 + 1:
                test = self.check_key("y", k0)
            elif zidx1 == zidx0 + 1:
                test = self.check_key("z", k1)
            else:
                test = self.check_key("z", k0)
            if not test:
                return False
        return True

    def get_path_score_and_offsets(self, path:PATH_t) -> SCORE_AND_OFFS_t:
        """
        Compute the cumulative score and offsets along a path

        :param path: a sequence of keys from the starting key to the destination key
        :return: the cumulative score and offsets following along the path
        """
        xoff = 0
        yoff = 0
        zoff = 0
        score = 0
        for k0, k1 in zip(path[:-1], path[1:]):
            s, xo, yo, zo = self.get_alignment(k0, k1)
            if s == 0 and np.abs(zo) == 6:
                xo = yo = zo = 0
            xoff += xo
            yoff += yo
            zoff += zo
            score += 1 - s
        return score, xoff, yoff, zoff

    def get_node_z_paths(self, k0:KEY_t) -> typing.Sequence[PATH_t]:
        """
        Return short paths from k0 to the node below it in Z
        The five paths are going straight down and going one in each
        direction in X and Y, then going down, then going in the opposite
        direction
        :param k0: the node's key
        :return: a sequence of valid paths between K0 and the node below.
        """
        xidx0, yidx0, zidx0 = k0
        zidx1 = zidx0 + 1
        return [_ for _ in
                [
                    [(xidx0, yidx0, zidx0), (xidx0, yidx0, zidx1)],
                    [(xidx0, yidx0, zidx0), (xidx0 + 1, yidx0, zidx0), (xidx0 + 1, yidx0, zidx1),
                     (xidx0, yidx0, zidx1)],
                    [(xidx0, yidx0, zidx0), (xidx0 - 1, yidx0, zidx0), (xidx0 - 1, yidx0, zidx1),
                     (xidx0, yidx0, zidx1)],
                    [(xidx0, yidx0, zidx0), (xidx0, yidx0 + 1, zidx0), (xidx0, yidx0 + 1, zidx1),
                     (xidx0, yidx0, zidx1)],
                    [(xidx0, yidx0, zidx0), (xidx0, yidx0 - 1, zidx0), (xidx0, yidx0 - 1, zidx1), (xidx0, yidx0, zidx1)]
                ] if self.check_path(_)
                ]

    def compute_all_z_offsets(self, threshold:float) -> np.ndarray:
        """
        Scan in the z direction, calculating offset to next lowest.

        :param threshold: There's weak support if the cumulative score is above this threshold
        :return: an array containing the z-offset for any block (other than the last)
        """
        z_offset_dict = {}
        xend = len(self.xs)
        yend = len(self.ys)
        zend = len(self.zs) - 1
        scores = np.zeros((zend, yend, xend))
        zoffsets = np.zeros((zend, yend, xend), int)
        for zi in range(zend):
            for xi, yi in itertools.product(range(xend),
                                                range(yend)):
                score, xoff, yoff, zoff = self.get_best_z_path((xi, yi, zi))
                scores[zi, yi, xi] = score
                zoffsets[zi, yi, xi] = zoff
            mask = scores[zi] > threshold
            if not np.all(mask):
                # Assign weak z to nearest strong z
                distances, (ysrc, xsrc) = distance_transform_edt(mask, return_indices=True)
                scores[zi, mask] = scores[zi, ysrc[mask], xsrc[mask]]
                zoffsets[zi, mask] = zoffsets[zi, ysrc[mask], xsrc[mask]]
        return zoffsets

    def get_best_z_path(self, k0:KEY_t):
        """
        Get the lowest scoring path from K0 to the node below it in Z

        :param k0: The node's key
        :return: the score, x-offset, y-offset and z-offset to the node below
        """
        best_score = 1000
        best_xoff = 0
        best_yoff = 0
        best_zoff = 0
        for path in self.get_node_z_paths(k0):
            score, xoff, yoff, zoff = self.get_path_score_and_offsets(path)
            if score < best_score and zoff <= 0:
                best_score = score
                best_xoff = xoff
                best_yoff = yoff
                best_zoff = zoff
        return best_score, best_xoff, best_yoff, best_zoff

    def flat_adjust_stacks(self, threshold:float):
        """
        For the flat method, what I've seen is this:
        Z offsets within-stack are similar from stack to stack, so if there's not enough evidence, use
        the consensus at that z-height.

        Z offsets of stacks increase in a steady fashion in the X direction for a given Y, e.g.
        there is an increase of C(y) in Z between (x, y, z) and (x+1, y, z). So for each Y, compute the
        median Z offset between (x, y, z) and (x+1, y, z)

        X and Y offsets are pretty much constant down the length of a z-stack.

        :param threshold:
        :return:
        """
        logging.info("Adjusting stacks based on alignments")
        off_z_y = {}
        off_z_z = {}
        #
        # Calculate the z-offset wrt to the z-stack.
        for z in range(1, self.n_z):
            all_z = []
            for x in range(self.n_x):
                for y in range(self.n_y):
                    for x in range(self.n_x):
                        score, off_x, off_y, off_z = self.get_alignment((x, y, z-1), (x, y, z))
                        if score > threshold:
                            if off_z == -1:
                                off_z = 0
                            off_z_z[x, y, z] = off_z
                            all_z.append(off_z)
            median_z = np.median(all_z) if len(all_z) > self.min_support else 0
            for x, y in itertools.product(range(self.n_x), range(self.n_y)):
                if (x, y, z) not in off_z_z:
                    off_z_z[x, y, z] = median_z
        #
        # Z constantly increases in the X direction for a given Y
        #
        for y in range(self.n_y):
            z_offs = []
            for x, z in itertools.product(range(1, self.n_x), range(self.n_z)):
                z_offs += [z_off for score, x_off, y_off, z_off in
                           self.get_alignments((x-1, y, z), (x, y, z), threshold)]
            median_z = np.median(z_offs) if len(z_offs) >= self.min_support else 0
            off_z_y[y] = median_z
        #
        # Update z offsets in z-stack
        #
        for x, y in itertools.product(range(self.n_x), range(self.n_y)):
            for z in range(1, self.n_z):
                if (x, y, z) in off_z_z:
                    self._stacks[x, y, z].z0 = self._stacks[x, y, z-1].z1 + off_z_z[x, y, z]
                else:
                    self._stacks[x, y, z].z0 = self._stacks[x, y, z-1].z1
        for y in range(self.n_y):
            off_z = 0
            for x in range(1, self.n_x):
                off_z += off_z_y[y]
                for z in range(self.n_z):
                    self._stacks[x, y, z].z0 = self._stacks[x, y, z].z0 + off_z
        #
        # calculate average X and Y offsets
        #
        x_off_xs = []
        x_off_xs_per_x = {}
        y_off_xs = []
        x_off_ys = []
        y_off_ys = []
        for x, y, z in itertools.product(range(1, self.n_x), range(self.n_y), range(self.n_z)):
            score, off_x, off_y, off_z = self.get_alignment((x-1, y, z), (x, y, z))
            if score > threshold:
                x_off_xs.append(off_x)
                if x not in x_off_xs_per_x:
                    x_off_xs_per_x[x-1] = []
                x_off_xs_per_x[x-1].append(off_x)
                y_off_xs.append(off_y)
        x_off_x = np.ones(self.n_x, int) * np.median(x_off_xs) if len(x_off_xs) > 0 else 0
        if self.loose_x:
            for x in range(self.n_x):
                if x in x_off_xs_per_x:
                    x_off_x[x] = np.median(x_off_xs_per_x[x])
        y_off_x = np.median(y_off_xs) if len(y_off_xs) > 0 else 0
        for x, y, z in itertools.product(range(self.n_x), range(1, self.n_y), range(self.n_z)):
            score, off_x, off_y, off_z = self.get_alignment((x, y-1, z), (x, y, z))
            if score > threshold:
                x_off_ys.append(off_x)
                y_off_ys.append(off_y)
        x_off_y = np.median(x_off_ys) if len(x_off_ys) > 0 else 0
        y_off_y = np.median(y_off_ys) if len(y_off_ys) > 0 else 0
        for x, y, z in itertools.product(range(self.n_x), range(self.n_y), range(self.n_z)):
            self._stacks[x, y, z].x0 = np.sum(x_off_x[:x]) - y * x_off_y
            self._stacks[x, y, z].y0 = y * y_off_y - x * y_off_x

    def get_ul_stacks(self, xidx, yidx, zidx):
        """
        Get stack and stack below in Z

        :param xidx: x index of stack
        :param yidx: y index of stack
        :param zidx: z index of stack
        :return: stack indexed by xidx, yidx, and zidx and stack below
        """
        s0 = self._stacks[xidx, yidx, zidx]
        s1 = self._stacks[xidx, yidx, zidx + 1]
        return s0, s1

    def rebase_stacks(self):
        """
        Readjust the stack offsets so that they start at 0, 0, 0
        """
        logging.info("Rebasing stacks")
        x0 = np.iinfo(np.int64).max
        y0 = np.iinfo(np.int64).max
        z0 = np.iinfo(np.int64).max
        for stack in self._stacks.values():
            x0 = min(x0, stack.x0)
            y0 = min(y0, stack.y0)
            z0 = min(z0, stack.z0)
        for stack in self._stacks.values():
            stack.x0 = stack.x0 - x0
            stack.y0 = stack.y0 - y0
            stack.z0 = stack.z0 - z0


def align_one_x(tgt_path:pathlib.Path,
                src_paths:typing.Sequence[pathlib.Path],
                x0_off:int, x1_off:int,
                y0_off:int, y1_off:int,
                z_off:int,
                dark:int,
                decimate:int) -> typing.Tuple[float, int, int, int]:
    """
    Align the target to all of the sources, returning the chosen x_offset,
    y_offset and z_offset

    :param tgt_path: The plane to be aligned
    :param src_paths: The choices of planes in the Z direction
    :param x0_off: Start looking at alignment choices in X here
    :param x1_off: End looking here
    :param y0_off: Start looking in Y here
    :param y1_off: End looking here
    :param decimate: Decimate the image by this amount (= zoom by 1/decimate)
    :return: a 4 tuple of the best alignment score, and the x, y and z offsets
    chosen
    """
    best_score, best_xoff, best_yoff, best_zoff = align_one(
        dark, decimate, align_plane_x, src_paths, tgt_path, x0_off,
        x1_off, y0_off, y1_off)
    return best_score, best_xoff, best_yoff, best_zoff + z_off


def align_one(dark, decimate, plane_fn, src_paths, tgt_path, x0_off, x1_off,
              y0_off, y1_off):
    tgt_img = imread(tgt_path)
    src_imgs = [imread(_) for _ in src_paths]
    decimations = []
    d = decimate
    while True:
        decimations.append(d)
        if d == 1:
            break
        d = d // 2
    for decimate in decimations:
        best_score = 0.0
        best_xoff = 0
        best_yoff = 0
        best_zoff = 0
        if decimate != 1:
            tgt_img_decimate = zoom(tgt_img, 1 / decimate)
            src_imgs_decimate = [zoom(_, 1/ decimate) for _ in src_imgs]
        else:
            tgt_img_decimate = tgt_img
            src_imgs_decimate = src_imgs
        for z, src_img in enumerate(src_imgs_decimate):
            best_score, best_xoff, best_yoff, best_zoff = plane_fn(
                best_score, best_xoff, best_yoff, best_zoff, dark,
                decimate, src_img, tgt_img_decimate, x0_off, x1_off, y0_off,
                y1_off, z)
        if best_score  == 0:
            break
        x0_off= max(-tgt_img.shape[1], best_xoff - decimate)
        x1_off = min(best_xoff + decimate, tgt_img.shape[1])
        y0_off = max(-tgt_img.shape[1], best_yoff - decimate)
        y1_off = min(best_yoff + decimate, tgt_img.shape[0])
    return best_score, best_xoff, best_yoff, best_zoff


def align_plane_x(best_score, best_xoff, best_yoff, best_zoff, dark,
                  decimate, src_img, tgt_img, x0_off, x1_off, y0_off,
                  y1_off, z):
    for x_off_big in range(x0_off, x1_off, decimate):
        x_off = x_off_big // decimate
        for y_off_big in range(y0_off, y1_off, decimate):
            y_off = y_off_big // decimate
            score, src_slice, tgt_slice = score_plane_x(src_img, tgt_img, x_off, y_off)
            mask = (tgt_slice > dark) & (src_slice > dark)
            if np.sum(mask) < np.sqrt(np.prod(tgt_slice.shape)):
                continue
            if score > best_score:
                best_score = score
                best_xoff = x_off_big
                best_yoff = y_off_big
                best_zoff = z
    return best_score, best_xoff, best_yoff, best_zoff


def score_plane_x(src_img, tgt_img, x_off, y_off):
    x00 = x_off
    x10 = src_img.shape[1]
    x01 = 0
    x11 = src_img.shape[1] - x_off
    if y_off > 0:
        y00 = 0
        y10 = tgt_img.shape[0] - y_off
        y01 = y_off
        y11 = tgt_img.shape[0]
    else:
        y00 = -y_off
        y10 = tgt_img.shape[0]
        y01 = 0
        y11 = tgt_img.shape[0] + y_off
    tgt_slice = tgt_img[y01:y11, x01:x11]
    src_slice = src_img[y00:y10, x00:x10]
    score = np.corrcoef(tgt_slice.flatten().astype(np.float32),
                        src_slice.flatten().astype(np.float32))[0, 1]
    return score, src_slice, tgt_slice


def align_one_y(tgt_path:pathlib.Path,
                src_paths:typing.Sequence[pathlib.Path],
                x0_off:int, x1_off:int,
                y0_off:int, y1_off:int,
                z_off:int,
                dark:int,
                decimate:int) -> typing.Tuple[float, int, int, int]:
    """
    Align the target to all of the sources, returning the chosen x_offset,
    y_offset and z_offset

    :param tgt_path: The plane to be aligned
    :param src_paths: The choices of planes in the Z direction
    :param x0_off: Start looking at alignment choices in X here
    :param x1_off: End looking here
    :param y0_off: Start looking in Y here
    :param y1_off: End looking here
    :param z_off: the z-offset of the first src_path
    :param dark: the threshold between foreground and background image intensity
    :param decimate: Decimate the image by this amount (= zoom by 1/decimate)
    :return: a 4 tuple of the best alignment score, and the x, y and z offsets
    chosen
    """
    best_score, best_xoff, best_yoff, best_zoff =  align_one(
        dark, decimate, align_plane_y,
        src_paths, tgt_path, x0_off, x1_off, y0_off, y1_off)
    return best_score, best_xoff, best_yoff, best_zoff + z_off


def align_plane_y(best_score, best_xoff, best_yoff, best_zoff, dark, decimate,
                  src_img, tgt_img, x0_off, x1_off, y0_off, y1_off, z):
    for x_off_big in range(x0_off, x1_off, decimate):
        x_off = x_off_big // decimate
        for y_off_big in range(y0_off, y1_off, decimate):
            y_off = y_off_big // decimate
            score, src_slice, tgt_slice = score_plane_y(src_img, tgt_img, x_off, y_off)
            mask = (tgt_slice > dark) & (src_slice > dark)
            if np.sum(mask) < np.sqrt(np.prod(tgt_slice.shape)):
                continue
            if score > best_score:
                best_score = score
                best_xoff = x_off_big
                best_yoff = y_off_big
                best_zoff = z
    return best_score, best_xoff, best_yoff, best_zoff


def score_plane_y(src_img, tgt_img, x_off, y_off):
    if x_off > 0:
        x00 = 0
        x10 = tgt_img.shape[1] - x_off
        x01 = x_off
        x11 = tgt_img.shape[1]
    else:
        x00 = -x_off
        x10 = tgt_img.shape[1]
        x01 = 0
        x11 = tgt_img.shape[1] + x_off
    y00 = y_off
    y10 = src_img.shape[0]
    y01 = 0
    y11 = src_img.shape[0] - y_off
    tgt_slice = tgt_img[y01:y11, x01:x11]
    src_slice = src_img[y00:y10, x00:x10]
    score = np.corrcoef(tgt_slice.flatten().astype(np.float32),
                        src_slice.flatten().astype(np.float32))[0, 1]
    return score, src_slice, tgt_slice


def align_one_z(src_paths:typing.Sequence[pathlib.Path],
                tgt_path:pathlib.Path,
                x0:int,
                x1:int,
                y0:int,
                y1:int,
                z_off:int,
                dark:int,
                decimate:int) -> typing.Tuple[float, int, int, int]:
    """
    Align one plane in the x and y direction on behalf of z

    :param src_paths: the paths to the source images
    :param tgt_path: the target image
    :param x0: start at this x offset
    :param x1: end at this x offset
    :param y0:  start at this y offset
    :param y1: end at this y offset
    :param z_off: the z-offset being tested (for bookkeeping only)
    :param dark: For counting minimum # of bright pixels, all values lower
    than this are considered background
    :param decimate: Start out by reducing the size of the image by this factor
    :return: a 4 tuple of the best score,  x offset, y offset and z offset
    """
    best_score, best_x_offset, best_y_offset, best_z_offset =\
        align_one(dark, decimate, align_plane_z, src_paths, tgt_path,
                     x0, x1, y0, y1)
    return best_score, best_x_offset, best_y_offset, best_z_offset + z_off


def align_plane_z(best_score, best_xoff, best_yoff, best_z_off, dark, decimate,
                  src_img, tgt_img, x0, x1, y0, y1,
                  z_off):
    for x_off_big in range(x0, x1, decimate):
        x_off = x_off_big // decimate
        for y_off_big in range(y0, y1, decimate):
            y_off = y_off_big // decimate
            score, src_slice, tgt_slice = score_plane_z(src_img, tgt_img, x_off, y_off)
            mask = (tgt_slice > dark) & (src_slice > dark)
            if np.sum(mask) < np.sqrt(np.prod(tgt_slice.shape)):
                continue
            if score > best_score:
                best_score = score
                best_xoff = x_off_big
                best_yoff = y_off_big
                best_z_off = z_off
    return best_score, best_xoff, best_yoff, best_z_off


def score_plane_z(src_img, tgt_img, x_off, y_off):
    if x_off > 0:
        x00 = 0
        x10 = tgt_img.shape[1] - x_off
        x01 = x_off
        x11 = tgt_img.shape[1]
    else:
        x00 = -x_off
        x10 = tgt_img.shape[1]
        x01 = 0
        x11 = tgt_img.shape[1] + x_off
    if y_off > 0:
        y00 = 0
        y10 = tgt_img.shape[0] - y_off
        y01 = y_off
        y11 = tgt_img.shape[0]
    else:
        y00 = -y_off
        y10 = tgt_img.shape[0]
        y01 = 0
        y11 = tgt_img.shape[0] + y_off
    tgt_slice = tgt_img[y01:y11, x01:x11]
    src_slice = src_img[y00:y10, x00:x10]
    score = np.corrcoef(tgt_slice.flatten().astype(np.float32),
                        src_slice.flatten().astype(np.float32))[0, 1]
    return score, src_slice, tgt_slice


if __name__ == "__main__":
    import json
    from tsv.volume import VExtent
    logging.basicConfig(level=logging.INFO)
    root = "/mnt/cephfs/SmartSPIM_CEPH/2019/20190920_14_47_43_#384_488LP35_647LP35/Ex_2_Em_2_destriped"
    so_path = "/mnt/cephfs/users/lee/2019-09-20_384/stack-offsets_ch2.json"
    x, y, z = 2573, 3731, 752
    voxel_size = [1.8, 1.8, 2.0]
    drift = AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0)
    scanner = Scanner(
        pathlib.Path(root),
        voxel_size,
        z_skip="middle",
        x_slop=30,
        y_slop=30,
        z_slop=6,
        decimate=8,
        dark=100,
        drift=drift)
    def dump_round(fd):
        json.dump(dict(
            x=dict([(",".join([str(_) for _ in k]), scanner.alignments_x[k])
                    for k in scanner.alignments_x]),
            y=dict([(",".join([str(_) for _ in k]), scanner.alignments_y[k])
                    for k in scanner.alignments_y]),
            z=dict([(",".join([str(_) for _ in k]), scanner.alignments_z[k])
                    for k in scanner.alignments_z])), fd, indent=2)

    def load_round(scanner, fd):
        d = json.load(fd)
        scanner.alignments_x = \
            dict([(tuple(int(_) for _ in k.split(",")), d["x"][k])
                  for k in d["x"]])
        scanner.alignments_y = \
            dict([(tuple(int(_) for _ in k.split(",")), d["y"][k])
                  for k in d["y"]])
        scanner.alignments_z = \
            dict([(tuple(int(_) for _ in k.split(",")), d["z"][k])
                  for k in d["z"]])
    with open(so_path) as fd:
        load_round(scanner, fd)

    scanner.calculate_next_round_parameters()
    scanner.rebase_stacks()
    volume = VExtent(x - 500, x + 500, y - 500, y + 500, z - 10, z + 10)
    img = scanner.imread(volume, np.uint16)
    tifffile.imsave("/tmp/stitched_volume.tiff", img)