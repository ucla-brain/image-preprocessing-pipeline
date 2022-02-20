# from pystripe_forked.raw import raw_imsave, raw_imread
# from pystripe_forked.core import imread, imsave, _find_all_images
# from skimage.transform import resize
# from flat import update_progress
# import shutil
# import numpy as np
# import os
# import pandas as pd
# import pathlib
# from flat import get_flat_classifier
#
#
# input_path = pathlib.Path(
#     r"D:\downsample_ds_lightsheet\20210729_16_18_40_SW210318-07_R-HPC_15x_Zstep1um_50p_4ms_flat_applied_tif_8b_3bsh_ds\Ex_642_Em_680"
# )
# imgs_path = _find_all_images(input_path)
# save_path = pathlib.Path(
#     r"D:\downsample_ds_lightsheet\20210729_16_18_40_SW210318-07_R-HPC_15x_Zstep1um_50p_4ms_flat_applied_tif_8b_3bsh_ds_resized\Ex_642_Em_680"
# )
#
# i = 0
# total = len(imgs_path)
# for path in imgs_path:
#     if path.suffix == '.tif':
#         try:
#             i += 1
#             update_progress(i // total * 100)
#             img = imread(path)
#             new_path = save_path / path.relative_to(input_path)
#             if not new_path.parent.exists():
#                 new_path.parent.mkdir(parents=True)
#             if img.shape > (781, 781):
#                 img = resize(img, (781, 781), preserve_range=True, anti_aliasing=True)
#                 imsave(new_path, img.astype(np.uint8))
#             else:
#                 shutil.copy(path, new_path)
#         except Exception:
#             print(path)
#             print(Exception)
#             pass
#         # print(path)
#         # img = imread(path)
#         # new_path = str(path)[0:-3] + 'raw'
#         # print(new_path)
#         # raw_imsave(new_path, img.astype('uint16'))
#         # path.unlink()

# import sys
# from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication,
#                              QGridLayout, QLCDNumber)
#
# class Example(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.Init_UI()
#
#     def Init_UI(self):
#         self.setGeometry(750, 300, 400, 300)
#         self.setWindowTitle('QGridLayout-QLCDNumber-calculator')
#
#         grid = QGridLayout()
#         self.setLayout(grid)
#
#         self.lcd = QLCDNumber()
#         grid.addWidget(self.lcd, 0, 0, 3, 0)
#         grid.setSpacing(10)
#
#         names = ['Cls', 'Bc', '',  'Close',
#                  '7',   '8',  '9', '/',
#                  '4',   '5',  '6', '*',
#                  '1',   '2',  '3', '-',
#                  '0',   '.',  '=', '+']
#
#         positions = [(i,j) for i in range(4,9) for j in range(4,8)]
#
#         for position, name in zip(positions, names):
#             print("position=`{}`, name=`{}`".format(position, name))
#             if name == '':
#                 continue
#
#             button = QPushButton(name)
#
#             grid.addWidget(button, *position)
#
#             button.clicked.connect(self.Cli)
#
#         self.show()
#
#     def Cli(self):
#         sender = self.sender().text()
#         ls = ['/', '*', '-', '=', '+']
#         if sender in ls:
#             self.lcd.display('A')
#         else:
#             self.lcd.display(sender)
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = Example()
#     app.exit(app.exec_())

# def img_path_generator():
#     img_folder = pathlib.Path(r"F:\test")
#     try:
#         if img_folder.exists():
#             for root, dirs, files in os.walk(img_folder):
#                 for name in files:
#                     name_l = name.lower()
#                     if name_l.endswith(".raw") or name_l.endswith(".tif") or name_l.endswith(".tiff"):
#                         img_path = os.path.join(root, name)
#                         yield img_path
#     except StopIteration:
#         yield None
#
#
# generator = img_path_generator()
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))

# path = pathlib.Path(r'F:\flat_data').rglob('*.csv')
# frames = []
# for file in path:
#     print(file)
#     frames += [pd.read_csv(file)]
# pd.concat(frames).drop_duplicates().to_csv(r'F:\flat_data\image_classes.csv')
# import os
# import sys
# import csv
# import psutil
# import pathlib
# import numpy as np
# import pandas as pd
# import pystripe_forked as pystripe
# from pystripe_forked.raw import raw_imread
# from scipy import stats
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from skimage.restoration import denoise_bilateral
# from multiprocessing import freeze_support, Process, Queue
# from queue import Empty
# from time import time, sleep


# def get_flat_classifier(training_data_path, train_test=False):
#     # model = LogisticRegression(max_iter=10000)  # 83.1%
#     model = RandomForestClassifier(
#         n_estimators=10,
#         # criterion="entropy",
#         min_samples_split=60,
#         n_jobs=psutil.cpu_count(logical=True)
#     )  # 89.8%
#     # model = SVC()  # 77.7%
#     # model = MLPClassifier(
#     #     activation='tanh',
#     #     hidden_layer_sizes=30,
#     #     # batch_size=1024,
#     #     max_iter=10000,
#     #     solver='adam',
#     #     learning_rate='adaptive',
#     #     learning_rate_init=0.01,
#     #     # tol=1e-6,
#     #     early_stopping=True,
#     #     shuffle=True
#     # )  # 84-85%
#     # model = KNeighborsClassifier(n_neighbors=9)  # 84.1%
#     # model = GaussianProcessClassifier()
#     # model = DecisionTreeClassifier(
#     #     max_depth=10,
#     #     min_samples_split=100
#     # )  # 89.3%
#     # model = AdaBoostClassifier(n_estimators=60)  # 87.7%
#     # model = QuadraticDiscriminantAnalysis()  # 80.8%
#     df = pd.read_csv(training_data_path)
#     x = df[["mean", "min", "max", "cv", "variance", "std", "skewness", "kurtosis"]]
#     x = x.to_numpy()
#     y = np.where(df['flat'].isin(['yes']), True, False)
#     if train_test:
#         x_train, x_test, y_train, y_test = train_test_split(x, y)
#         log_reg = model.fit(x_train, y_train)
#         print(f"Training set score: {log_reg.score(x_test, y_test)*100:.1f}%")
#     else:
#         log_reg = model.fit(x, y)
#         print(f"Training set score: {log_reg.score(x, y) * 100:.1f}%")
#     return model

# from flat import get_flat_classifier
# model = get_flat_classifier(r'F:\flat_data\image_classes.csv', train_test=True)

# import tsv
# from tsv.convert import convert_to_2D_tif
# from multiprocessing import freeze_support
#
# if __name__ == '__main__':
#     freeze_support()
#     volume = tsv.volume.TSVVolume.load(
#         r"X:\3D_stitched\20211104_13_31_02_SM211018_02_OL_LS_15X_1000z_stitched_v4\Ex_488_Em_525_xml_import_step_5.xml")
#     convert_to_2D_tif(
#         volume,
#         r"X:\3D_stitched\20211104_13_31_02_SM211018_02_OL_LS_15X_1000z_stitched_v4\tif\img_{z:05d}.tif",
#         compression=None
#     )

# from pystripe_forked.core import read_filter_save
#
#
# read_filter_save(
#     r"D:\20211020_11_21_05_SM210705_02_4x_2000z_Compressed_stitched_v4\RES(8740x7355x3900)\090500\090500_098010\090500_098010_028000.tif",
#     r"D:\090500_098010_028000.tif",
#     (256, 256)
# )

# from distributed import Client, progress
# from multiprocessing import freeze_support, Pool, cpu_count
# from subprocess import call, run
#
#
# def worker(x):
#     return call(x, shell=True)
#
#
# if __name__ == '__main__':
#     freeze_support()
#     # run(args="echo 'hi'", shell=True)
#     # run(args="echo 'goodbye'", shell=True)
#     cmd = "echo 'hi'"
#     work = [cmd]
#     cmd = "echo 'goodbye'"
#     work += [cmd]
#     # print(work)
#     with Pool(processes=2) as pool:
#         pool.map(worker, work)
#     # p.wait()
#     # client = Client()
#     # L = client.map(inc, range(10000))
#     # # total = client.submit(sum, L)
#     # print(progress(L))

from pathlib import Path
from itertools import chain, repeat
from multiprocessing import freeze_support, Pool, cpu_count, Manager
from timeit import default_timer
import numpy as np
import os
import re
from tifffile import imread, imwrite
import subprocess


# def glob_re(pattern: str, path: Path):
#     regexp = re.compile(pattern, re.IGNORECASE)
#
#     for p in os.scandir(path):
#         if p.is_file() and regexp.search(p.name):
#             yield Path(p.path)
#         elif p.is_dir(follow_symlinks=False):
#             yield from glob_re(pattern, p.path)
#
#     # for p in path.iterdir():
#     #     if p.is_file() and regexp.search(p.suffix):
#     #         yield p
#     #     elif p.is_dir():
#     #         yield from glob_re(pattern, p)
#
#     # for p in path.rglob("*.*"):
#     #     if p.is_file() and regexp.search(p.suffix):
#     #         yield p
#
#     # walk = os.walk(path)
#     # return chain.from_iterable(
#     #     (Path(os.path.join(root, file)) for file in files if regexp.search(file)) for root, dirs, files in walk)


# def worker(p: Path):
#     print(p.rename(p.parent / p.name[0:-1]))
#
#
# if __name__ == '__main__':
#     freeze_support()
#     # input_path = Path(r"x:\Keivan")
#     input_path = Path(r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z_8b_3bsh_ds")
#     start = default_timer()
#     # with Pool(processes=61) as pool:
#     #     files = pool.imap_unordered(
#     #         worker,
#     #         glob_re(r"\.tiff$", input_path),
#     #         chunksize=512
#     #     )
#     for p in input_path.rglob('*.tiff'):
#         new_name = p.parent / p.name[0:-1]
#         if new_name.exists() and new_name.is_file():
#             new_name.unlink()
#         p.rename(new_name)
#     print(default_timer() - start)

# from numpy import rot90
# from tifffile import TiffWriter, memmap
#
# original_image = memmap('/mnt/md0/kmoradi/20210907_16_56_41_SM210705_01_LS_4X_4000z_sep_C0.tif')
# rotated = rot90(original_image, axes=(0, 1))
#
# with TiffWriter('/mnt/md0/kmoradi/20210907_16_56_41_SM210705_01_LS_4X_4000z_sep_C0_rotated.tif', bigtiff=True) as tif:
#     for i in range(rotated.shape[0]):
#         tif.write(rotated[i], photometric='minisblack')

# for file in sorted([
#   r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\092760\092760_125440\076460.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\092760\092760_153400\076430.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\099750\099750_146410\076440.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\099750\099750_160390\076450.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\099750\099750_160390\076460.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\106740\106740_146410\076430.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\113730\113730_125440\076480.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\113730\113730_132430\076460.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\113730\113730_146410\076430.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\113730\113730_153400\076410.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\113730\113730_174370\076350.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\113730\113730_181360\076330.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\120720\120720_139420\076440.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\134700\134700_132430\076450.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\148680\148680_146410\076430.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\148680\148680_153400\076410.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\148680\148680_160390\076390.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\148680\148680_174370\076340.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\148680\148680_174370\076350.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\148680\148680_181360\076250.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\155670\155670_132430\076230.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\155670\155670_146410\076200.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\155670\155670_153400\076170.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_561_Em_600\155670\155670_174370\076080.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\092760\092760_132430\076240.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\092760\092760_139420\076220.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\092760\092760_146410\076190.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\092760\092760_153400\076220.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\092760\092760_160390\076210.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\099750\099750_125440\076250.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\099750\099750_132430\076230.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\099750\099750_153400\076200.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\099750\099750_153400\076210.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\099750\099750_167380\076180.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\099750\099750_174370\076100.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\106740\106740_132430\076230.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\106740\106740_139420\076200.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\106740\106740_153400\076150.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\120720\120720_125440\076240.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\120720\120720_146410\076180.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\120720\120720_167380\076130.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\127710\127710_139420\076200.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\127710\127710_146410\076170.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\127710\127710_146410\076180.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\134700\134700_132430\076190.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\134700\134700_139420\076170.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\134700\134700_139420\076180.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\134700\134700_146410\076180.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\134700\134700_153400\076160.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\141690\141690_174370\076070.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\141690\141690_174370\076080.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\148680\148680_139420\076210.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\148680\148680_167380\076140.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\148680\148680_167380\076150.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\148680\148680_174370\076090.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\148680\148680_174370\076100.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\155670\155670_125440\076250.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\155670\155670_146410\076200.tiff"
# , r"Y:\SmartSPIM_Data\20211026_11_05_21_SM210705_02_L_HPF_LS_15x_1000z\Ex_642_Em_680\155670\155670_153400\076210.tiff"
# ]):
#     # pass
#     # print(f", r\"{file}\"")
#     try:
#         if Path(file).exists():
#             imread(file)
#         else:
#             print(file)
#     except Exception:
#         imwrite(file, np.zeros(shape=(1850, 1850), dtype=np.uint16), compression=('ZLIB', 1))

# parent = Path(r"Y:\SmartSPIM_Data\20211122_11_13_27_SA210705_02_L_HPF_LS_15X_1000z\Ex_561_Em_600\122370\122370_189330")
# dummy_data = np.zeros(shape=(1850, 1850), dtype=np.uint16)
# for i in range(0, 89990, 10):
#     file = parent / f"{i:06}.tiff"
#     if not file.exists():
#         print(file)
#         imwrite(file, dummy_data, compression=('ZLIB', 1))

# def folder_iterator(path: Path):
#     counter = 0
#     for item in path.iterdir():
#         if item.is_file():
#             counter += 1
#         elif item.is_dir():
#             print(f"{counter}, {str(item)}")
#             folder_iterator(item)
#             counter = 0
#
#
# folder_iterator(Path(r"Y:\SmartSPIM_Data\20211122_11_13_27_SA210705_02_L_HPF_LS_15X_1000z"))

# def worker(command: str):
#     result = subprocess.call(command, shell=True)
#     print(f"\nfinished:\n{command}\nresult:\n{result}\n")
#     return result


# if __name__ == '__main__':
#     freeze_support()
#     TeraStitcherPath = Path(r"./TeraStitcher_windows_avx2")
#     os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
#     os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.joinpath('pyscripts').as_posix()}"
#     work = [
#         "imaris\ImarisConvertiv.exe "
#         "--input D:\\20211005_08_30_35_NW210718_01_LS_4x_2000z_stitched_v4\\tif\\img_000000.tif "
#         "--output D:\\20211005_08_30_35_NW210718_01_LS_4x_2000z_stitched_v4\\20211005_08_30_35_NW210718_01_LS_4x_2000z.ims "
#         "--inputformat TiffSeries "
#         "--nthreads 40 "
#         "--compression 1",
#
#         "mpiexec -np 20 python -m mpi4py TeraStitcher_windows_avx2\pyscripts\paraconverter.py "
#         "--sfmt=\"TIFF (series, 2D)\" --dfmt=\"TIFF (tiled, 3D)\" --resolutions=\"012345\" --clist=0 --halve=max "
#         "--noprogressbar --sparse_data "
#         "-s=D:\\20211005_08_30_35_NW210718_01_LS_4x_2000z_stitched_v4\\tif "
#         "-d=D:\\20211005_08_30_35_NW210718_01_LS_4x_2000z_stitched_v4\\TeraFly_Ex_561_Em_600"
#     ]
#     with Pool(processes=61) as pool:
#         a = list(pool.imap_unordered(worker, work, chunksize=1))
#         print(a)
from pathlib import Path
# from multiprocessing import freeze_support, Pool, Queue, Process
# from typing import List
# from datetime import datetime
# import subprocess
# import sys
# from process_images2 import correct_path_for_cmd
# from time import time, sleep
# from queue import Empty
# from tqdm import tqdm
# import re
#
#
# class MultiProcess(Process):
#     def __init__(self, queue, command, position):
#         Process.__init__(self)
#         super().__init__()
#         self.daemon = True
#         self.queue = queue
#         self.command = command
#         self.position = position
#
#     def run(self):
#         return_code = None  # 0 == success and any other number is an error code
#         pattern = re.compile(r"(WriteProgress:)\s+(\d*.\d+)\s*$")
#         previous_percent = 0
#         try:
#             process = subprocess.Popen(
#                 self.command,
#                 stdout=subprocess.PIPE,
#                 # stderr=subprocess.PIPE,
#                 shell=True,
#                 text=True)
#             while return_code is None:
#                 return_code = process.poll()
#                 line = process.stdout.readline()
#                 m = re.match(pattern, line)
#                 if m:
#                     percent = int(float(m[2])*100)
#                     self.queue.put([percent - previous_percent, self.position, return_code, self.command])
#                     previous_percent = percent
#         except Exception as inst:
#             print(f'Process failed for {self.command}.')
#             print(type(inst))  # the exception instance
#             print(inst.args)  # arguments stored in .args
#             print(inst)
#         self.queue.put([100 if return_code == 0 else 0, self.position, return_code, self.command])
#
#
# def get_imaris_command(
#         path,
#         voxel_size_x: float,
#         voxel_size_y: float,
#         voxel_size_z: float,
#         workers: int = cpu_count()):
#
#     files = list(path.rglob("*.tif"))
#     file = files[0]
#     command = []
#     if imaris_converter.exists() and len(files) > 0:
#         print(f"{datetime.now()}: converting {path.name} to ims ... ")
#         ims_file_path = path.parent / f'{path.name}.ims'
#         command = [
#             f"{imaris_converter}" if sys.platform == "win32" else f"wine {imaris_converter}",
#             f"--input {file}",
#             f"--output {ims_file_path}",
#         ]
#         if sys.platform == "linux" and 'microsoft' in uname().release.lower():
#             command = [
#                 f'{correct_path_for_cmd(imaris_converter)}',
#                 f'--input {correct_path_for_wsl(file)}',
#                 f"--output {correct_path_for_wsl(ims_file_path)}",
#             ]
#         if len(files) > 1:
#             command += ["--inputformat TiffSeries"]
#
#         command += [
#             f"--nthreads {workers}",
#             f"--compression 1",
#             f"--voxelsize {voxel_size_x}-{voxel_size_y}-{voxel_size_z}",  # x-y-z
#             "--logprogress"
#         ]
#         print(f"\ttiff to ims conversion command:\n\t\t{' '.join(command)}\n")
#
#     else:
#         if len(files) > 0:
#             print("\tnot found Imaris View: not converting tiff to ims ... ")
#         else:
#             print("\tno tif file found to convert to ims!")
#
#     return " ".join(command)
#
#
# def main():
#     queue = Queue()
#
#     command = get_imaris_command(Path(r"D:\unstitched_deconvoluted_stitched\Ex_642_Em_680_tif_test"), 0.2, 0.2, 1.0)
#     # command = r"imaris\ImarisConvertiv.exe --input D:\unstitched_deconvoluted_stitched\Ex_642_Em_680_tif_test\img_000051.tif --output D:\unstitched_deconvoluted_stitched\Ex_642_Em_680_tif_test.ims --inputformat TiffSeries --nthreads 96 --compression 1 --voxelsize 0.2-0.2-1.0 --logprogress"
#     MultiProcess(queue, command, 0).start()
#     running_processes = 1
#     progress_bar = [tqdm(total=100, ascii=True, position=0), tqdm(total=100, ascii=True, position=1)]
#     start_time = time()
#     while running_processes > 0:
#         try:
#             [percent_addition, position, return_code, command] = queue.get()
#             if return_code is not None:
#                 if return_code > 0:
#                     print(f"Following command failed:\n\t{command}\n\treturn code: {return_code}")
#                 else:
#                     print(f"Following command succeeded:\n\t{command}")
#                 running_processes -= 1
#                 print(time() - start_time)
#             progress_bar[position].update(percent_addition)
#         except Empty:
#             sleep(1)  # waite one minutes before checking again
#
#
# if __name__ == '__main__':
#     freeze_support()
#     imaris_converter = Path(r"./imaris/ImarisConvertiv.exe")
#     main()
import shutil
installed_imaris = Path(r"C:\Program Files\Bitplane\ImarisViewer 9.8.2")
new_imaris = Path(r"D:\imaris")
new_imaris.mkdir(exist_ok=True)
for file in Path(r"./imaris").rglob("*.*"):
    path_file = installed_imaris / file.name
    if path_file.exists():
        shutil.copy(path_file, new_imaris)
    else:
        print(f"{path_file} did not exist.")