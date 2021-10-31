import os
import sys
import csv
import psutil
import pathlib
import numpy as np
import pandas as pd
import pystripe_forked as pystripe
from pystripe_forked.raw import raw_imread
from scipy import stats
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_bilateral
from multiprocessing import freeze_support, Process, Queue, Manager
from queue import Empty
from math import isnan

# How to generate flat images
# https://calm.ucsf.edu/how-acquire-flat-field-correction-images

# https://stackoverflow.com/questions/50306632/multiprocessing-not-achieving-full-cpu-usage-on-dual-processor-windows-machine
os.environ["OPENBLAS_MAIN_FREE"] = "1"


def img_read(path):
    img_mem_map = None
    if path.lower().endswith(".tif") or path.lower().endswith(".tiff"):
        img_mem_map = pystripe.imread(path)
    elif path.lower().endswith(".raw"):
        img_mem_map = raw_imread(path)
    return img_mem_map


def get_img_stats(img_mem_map):
    if img_mem_map is None:
        return ['mean', 'min', 'max', 'cv', 'variance', 'std', 'skewness', 'kurtosis', 'n']
    img_mem_map = img_mem_map.flatten()
    img_nobs, (img_min, img_max), img_mean, img_variance, img_skewness, img_kurtosis = stats.describe(img_mem_map)
    img_std = np.sqrt(img_variance)
    img_cv = img_std / img_mean
    return [img_mean, img_min, img_max, img_cv, img_variance, img_std, img_skewness, img_kurtosis, img_nobs]


class MultiProcessGetImgStats(Process):
    def __init__(self, queue, img_path, tile_size):
        Process.__init__(self)
        self.daemon = True
        self.queue = queue
        self.img_path = img_path
        self.tile_size = tile_size

    def run(self):
        img_mem_map, img_stats = None, None
        try:
            img_mem_map = img_read(self.img_path)
            if img_mem_map is not None and img_mem_map.shape == self.tile_size:
                # ['mean', 'min', 'max', 'cv', 'variance', 'std', 'skewness', 'kurtosis', 'n']
                img_stats = get_img_stats(img_mem_map)
                img_stats = img_stats[0:-1]
                img_stats = [0 if isnan(x) else x for x in img_stats]
                img_stats = np.array(img_stats, dtype=np.float)
                # is_flat = self.classifier_model.predict([img_stats[0:-1]])
            elif img_mem_map.shape != self.tile_size:
                print(f"tile size mismatch for file:\n{self.img_path}")
            else:
                print(f"problem reading file:\n{self.img_path}")
        except Exception as inst:
            print(f'Process failed for {self.img_path}.')
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)
        self.queue.put([img_mem_map, img_stats])


class MultiProcessDenoiseImg(Process):
    def __init__(self, queue, img_list, img_idx, sigma_spatial=1):
        Process.__init__(self)
        self.daemon = True
        self.queue = queue
        self.img_list = img_list
        self.img_idx = img_idx
        self.sigma_spatial = sigma_spatial

    def run(self):
        img_denoised = None
        try:
            img_denoised = denoise_bilateral(self.img_list[self.img_idx], sigma_spatial=self.sigma_spatial)
        except Exception as inst:
            print(f'Denoise process failed.')
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)
        self.queue.put(img_denoised)


def img_path_generator(path):
    if path.exists():
        for root, dirs, files in os.walk(path):
            for name in files:
                name_l = name.lower()
                if name_l.endswith(".raw") or name_l.endswith(".tif") or name_l.endswith(".tiff"):
                    img_path = os.path.join(root, name)
                    yield img_path


def update_progress(percent, prefix='progress', posix=''):
    percent = int(percent)
    hash_count = percent // 10
    space_count = 10 - hash_count
    sys.stdout.write(f"\r{prefix}: [{'#' * hash_count}{' ' * space_count}] {percent}% {posix}")
    sys.stdout.flush()


def save_csv(path, list_2d):
    with open(path, 'w') as file:
        write = csv.writer(file)
        write.writerows(list_2d)


def get_flat_classifier(training_data_path, train_test=False):
    # from sklearn.linear_model import LogisticRegression
    # model = LogisticRegression(max_iter=10000)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=10,
        # criterion="entropy",
        min_samples_split=100,
        n_jobs=psutil.cpu_count(logical=True)
    )
    # from sklearn.ensemble import AdaBoostClassifier
    # model = AdaBoostClassifier(n_estimators=60)
    # from sklearn.tree import DecisionTreeClassifier
    # model = DecisionTreeClassifier(
    #     max_depth=10,
    #     min_samples_split=100
    # )
    # from sklearn.neighbors import KNeighborsClassifier
    # model = KNeighborsClassifier(n_neighbors=9)
    df = pd.read_csv(training_data_path)
    x = df[["mean", "min", "max", "cv", "variance", "std", "skewness", "kurtosis"]]
    x = x.to_numpy()
    y = np.where(df['flat'].isin(['yes']), True, False)
    if train_test:
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        log_reg = model.fit(x_train, y_train)
        print(f"Training set score: {log_reg.score(x_train, y_train) * 100:.1f}%")
        print(f"Testing  set score: {log_reg.score(x_test, y_test) * 100:.1f}%")
    else:
        log_reg = model.fit(x, y)
        print(f"Training set score: {log_reg.score(x, y) * 100:.1f}%")
    return model


def create_flat_img(
        img_source_path, flat_training_data_path, tile_size,
        max_images=1024,
        batch_size=psutil.cpu_count(),
        patience_before_skipping=None,
        skips=256,
        sigma_spatial=1,
        save_as_tiff=True):

    print()
    img_path_gen = iter(img_path_generator(img_source_path))
    img_flat_count = 0
    queue_stats = Queue()
    queue_denoise = Queue()
    manager = Manager()
    running_processes = 0
    img_flat_list, img_mean_list = [], []
    if flat_training_data_path is None:
        classifier_model = None
    else:
        classifier_model = get_flat_classifier(flat_training_data_path)

    there_is_img_to_process, first_run = True, True
    while img_flat_count < max_images and (there_is_img_to_process or running_processes > 0):
        for run in range(batch_size - running_processes):
            img_path = next(img_path_gen, None)
            if img_path is None:
                there_is_img_to_process = False
                break
            MultiProcessGetImgStats(queue_stats, img_path, tile_size).start()
            running_processes += 1

        update_progress(
            img_flat_count / max_images * 100,
            prefix=img_source_path.name,
            posix=f'flat: {img_flat_count}, reading {running_processes} images.'
        )

        img_mem_map_list = manager.list()  # a shared list to avoid copying images to the worker
        img_stats_list = []
        img_non_flat_count = 0
        future_running_processes = 0
        while running_processes > 0:
            try:  # check the queue for the optimization results then show result
                [img_mem_map, img_stats] = queue_stats.get(block=False)
                running_processes -= 1
                update_progress(
                    img_flat_count / max_images * 100,
                    prefix=img_source_path.name,
                    posix=f'flat: {img_flat_count}, images in the queue: {running_processes}                           '
                )
                if img_mem_map is not None and img_stats is not None:
                    img_mem_map_list += [img_mem_map]
                    img_stats_list += [img_stats]
                else:
                    img_non_flat_count += 1
                    if img_mem_map is None:
                        print('an image could not be loaded.')
                    else:
                        print('image stats could not be calculated for one of the images.')

                img_path = next(img_path_gen, None)
                if img_path is None:
                    there_is_img_to_process = False
                else:
                    MultiProcessGetImgStats(queue_stats, img_path, tile_size).start()
                    future_running_processes += 1
            except Empty:
                pass

        if len(img_stats_list) > 0:
            if classifier_model is not None:
                is_flat_list = classifier_model.predict(img_stats_list)
            else:
                is_flat_list = [True] * len(img_stats_list)
            for img_idx, (is_flat, img_stats) in enumerate(zip(is_flat_list, img_stats_list), start=0):
                if is_flat:
                    MultiProcessDenoiseImg(
                        queue_denoise, img_mem_map_list, img_idx, sigma_spatial=sigma_spatial,
                    ).start()
                    running_processes += 1
                    img_mean_list += [img_stats[0]]
                else:
                    img_non_flat_count += 1

            update_progress(
                img_flat_count / max_images * 100,
                prefix=img_source_path.name,
                posix=f'flat: {img_flat_count}, non-flat: {img_non_flat_count}/{batch_size}                            '
            )

            if patience_before_skipping and patience_before_skipping < img_non_flat_count:
                print(f"\nskipping {skips} files because non-flat images were more than {patience_before_skipping}.\n")
                for skip in range(skips):
                    next(img_path_gen, None)
        else:
            print("\ncould not read any images in this batch.\n")

        while running_processes > 0:
            try:  # check the queue for the optimization results then show result
                img_mem_map_denoised = queue_denoise.get(block=False)
                running_processes -= 1
                if img_mem_map_denoised is not None:
                    img_flat_list += [img_mem_map_denoised]
                    img_flat_count += 1
                    update_progress(
                        img_flat_count / max_images * 100,
                        prefix=img_source_path.name,
                        posix=f'flat: {img_flat_count}, non-flat: {img_non_flat_count}/{batch_size}                    '
                    )
            except Empty:
                pass
        running_processes = future_running_processes

    if img_flat_count > 0:
        img_flat_median = np.median(img_flat_list, axis=0)
        img_flat_denoised = denoise_bilateral(img_flat_median, sigma_spatial=sigma_spatial)
        img_flat = img_flat_denoised / np.max(img_flat_denoised)
        dark = int(round(float(np.median(img_mean_list)/np.median(img_flat)), 0))
        if save_as_tiff:
            pystripe.imsave(
                str(img_source_path.parent / (img_source_path.name + '_flat.tif')),
                img_flat,
                convert_to_8bit=False
            )
            with open(img_source_path.parent / (img_source_path.name + '_dark.txt'), "w") as f:
                f.write(str(dark))
        update_progress(
            100, prefix=img_source_path.name, posix=f'found: {img_flat_count}                                         ')
        return img_flat, dark
    else:
        print("no flat image found!")
        raise RuntimeError


if __name__ == '__main__':
    freeze_support()
    AllChannels = ["Ex_488_Em_525", "Ex_561_Em_600", "Ex_642_Em_680"]
    SourceFolder = pathlib.Path(
        r"Y:\SmartSPIM_Data\2021_10_21\20211021_11_14_46_15x_FlatImage_0_Offset_Compressed"
        # r"/mnt/f/20210907_16_56_41_SM210705_01_LS_4X_4000z"
    )

    # SourceFolder = pathlib.Path(__file__).parent
    for Channel in AllChannels:
        if SourceFolder.joinpath(Channel).exists():
            img_flat_, img_dark_ = create_flat_img(
                SourceFolder / Channel,
                None,  # r'./image_classes.csv',
                (1850, 1850),  # (1850, 1850) (1600, 2000)
                max_images=999,
                batch_size=psutil.cpu_count(logical=True),
                patience_before_skipping=None,
                skips=256,
                sigma_spatial=1,
                save_as_tiff=True
            )
            print(img_dark_)
    print()
