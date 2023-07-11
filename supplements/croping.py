# crop imaris file according to roi bounds defined in spreadsheet
# save to a single tiff file
import os
import shutil
import h5py
import numpy as np
import pandas as pd
import traceback
from PIL import Image
from tifffile import imwrite


def convert_32bit_to_16bit(img, right_shift=16):
    if img.dtype == 'uint32':
        img = (img >> right_shift)
        img[img > 65535] = 65535
        img = img.astype('uint16')
    else:
        print(f"\nWarning: the original image was not 32 bit. 16 bit conversion is disabled.\n")
    return img


def convert_16bit_to_8bit_fun(img, right_shift=3):
    if img.dtype == 'uint32':
        img = convert_32bit_to_16bit(img)

    if img.dtype == 'uint16':
        # bit shift then change the type to avoid floating point operations
        # img >> 8 is equivalent to img / 256
        if 0 <= right_shift <= 8:
            img = (img >> right_shift)
            img[img > 255] = 255
            img = img.astype('uint8')
        else:
            print("right shift should be between 0 and 8")
            raise RuntimeError
    else:
        print(f"\nWarning: the original image was not 16 bit. 8 bit conversion is disabled.\n")
    return img


def imwrite_lzw(path, data, convert_16bit_to_8bit=True, right_shift=3):
    img_list = []
    if convert_16bit_to_8bit:
        data = convert_16bit_to_8bit_fun(data, right_shift=right_shift)
    for img in data:
        image = Image.fromarray(img)
        img_list.append(image)

    img_list[0].save(path, compression="tiff_lzw", save_all=True, append_images=img_list[1:])


def read_ims(ims_path, zyx_offsets, zyx_extents=(-1, -1, -1), channel=0, resolution_level=0):
    """
    return subvolume(s) of data defined by zyx_offsets and zyx_extents, at
    requested resolution_level and channel. this function assumes only
    time point 0 exists
    :param ims_path: path to imaris file
    :param zyx_offsets: tuple of 3 integers
    :param zyx_extents: if negative, select till end of axis
    :param channel: list of channel ids
    :param resolution_level: required resolution level
    :return: numpy array (or list of numpy arrays of multiple channels requested)
    """
    if not os.path.isfile(ims_path):
        raise ValueError('{} does not exist'.format(ims_path))
    if len(zyx_offsets) != 3 or len(zyx_extents) != 3:
        raise ValueError('zyx_offsets and zyx_extents should be of length 3')
    ims = h5py.File(ims_path, 'r')
    try:
        multi_channel_datasets = ims[f'DataSet/ResolutionLevel {resolution_level}/TimePoint 0']
    except KeyError:
        raise KeyError(f'requested resolution level {resolution_level} does not exist')
    if channel < 0 or channel >= len(multi_channel_datasets.keys()):
        raise ValueError(f'{channel} is not a valid channel id')

    channel_dataset = multi_channel_datasets[f'Channel {channel}/Data']
    z_start, y_start, x_start = zyx_offsets
    z_stop = zyx_extents[0] + z_start if zyx_extents[0] > 0 else channel_dataset.shape[0]
    y_stop = zyx_extents[1] + y_start if zyx_extents[1] > 0 else channel_dataset.shape[1]
    x_stop = zyx_extents[2] + x_start if zyx_extents[2] > 0 else channel_dataset.shape[2]
    if z_start < 0 or y_start < 0 or x_start < 0:
        raise ValueError('offset values must be non-negative')
    if z_stop - z_start > channel_dataset.shape[0] or y_stop - y_start > channel_dataset.shape[1] or \
            x_stop - x_start > channel_dataset.shape[2]:
        raise ValueError('extent values out of range')
    # skipping range checking and assume 16bit. quick script
    img = np.zeros(shape=(z_stop - z_start, y_stop - y_start, x_stop - x_start), dtype=np.uint16)
    channel_dataset.read_direct(img, source_sel=(np.s_[z_start: z_stop, y_start: y_stop, x_start: x_stop]))
    ims.close()
    # img = convert_32bit_to_16bit(img)
    return img


def get_imaris_path(df, i):
    return os.path.join(df.loc[i, 'File Location'], df.loc[i, 'File Name'])


def get_roi_str(df, i):
    return f"{df.loc[i, 'ROI_zmin']}_{df.loc[i, 'ROI_zmax']}_" \
           f"{df.loc[i, 'ROI_ymin']}_{df.loc[i, 'ROI_ymax']}_" \
           f"{df.loc[i, 'ROI_xmin']}_{df.loc[i, 'ROI_xmax']}"


def get_zyx_offsets(df, i):
    return [df.loc[i, 'ROI_zmin'], df.loc[i, 'ROI_ymin'], df.loc[i, 'ROI_xmin']]


def get_zyx_extents(df, i):
    return [df.loc[i, 'ROI_zmax'] - df.loc[i, 'ROI_zmin'] + 1, df.loc[i, 'ROI_ymax'] - df.loc[i, 'ROI_ymin'] + 1,
            df.loc[i, 'ROI_xmax'] - df.loc[i, 'ROI_xmin'] + 1]


def get_tiff_dir(df, i):
    return os.path.join(df.loc[i, 'File Location'], 'mr')


def get_tiff_path(df, i, posix=''):
    ims_name, _ = os.path.splitext(df.loc[i, 'File Name'])
    return os.path.join(get_tiff_dir(df, i), f'{ims_name}_{get_roi_str(df, i)}_{posix}.tif')


def crop_imaris():
    sheet_name = '3D Manual Tracing Tracking Sheet.csv'
    if not os.path.isfile(sheet_name):
        raise ValueError('the spreadsheet \"3D Manual Tracing Tracking Sheet.csv\" not found')
    # make a copy of the sheet
    shutil.copy(sheet_name, '3D Manual Tracing Tracking Sheet copy.csv')
    # fill empty ROI location column as empty string
    df = pd.read_csv(sheet_name).assign(**{'ROI location': lambda x: x['ROI location'].fillna(value='')})
    # iterate dataframe, skip lines with existing ROI location
    for i in range(len(df)):
        if not os.path.isfile(get_imaris_path(df, i)):
            raise ValueError('input imaris file not found at {}'.format(get_imaris_path(df, i)))
        if os.path.isfile(get_tiff_path(df, i)):
            continue    # crop already exists
        try:
            data = read_ims(
                get_imaris_path(df, i), get_zyx_offsets(df, i), zyx_extents=get_zyx_extents(df, i),
                channel=df.loc[i, 'Channel (0 indexed: the 1st channel in the file is entered as channel 0)'])
            print('image is cropped.')
            os.makedirs(get_tiff_dir(df, i), exist_ok=True)
            print('made a new directory.')
            imwrite(get_tiff_path(df, i, posix='16bit'), data)
            print('16 bit image saved.')
            try:
                imwrite_lzw(get_tiff_path(df, i, posix='lzw_8bit'), data, convert_16bit_to_8bit=True, right_shift=3)
                print('8 bit image saved with lzw compression.')
            except Exception as e:
                print('LZW compression failed for 8-bit image. saving without compression.')
                print(e.args)
                traceback.print_exc()
                imwrite(get_tiff_path(df, i, posix='8bit'), convert_16bit_to_8bit_fun(data, right_shift=3))
                print('8 bit image saved.')
            # assert os.path.isfile(get_tiff_path(df, i))
            # update tiff crop location
            # df.loc[i, 'ROI location'] = get_tiff_path(df, i)
            print(f"cropped {df.loc[i, 'File Name']}: roi {get_roi_str(df, i)}")
        except Exception as e:
            traceback.print_exc()
            print(e.args)
            print('error making crops for {} roi {}'.format(df.loc[i, 'File Name'], get_roi_str(df, i)))
            if os.path.isfile(get_tiff_path(df, i)):
                os.remove(get_tiff_path(df, i))
    print(df.loc[0, 'ROI location'])
    df.to_csv(sheet_name, index=False)


# open cmd.exe and run python3 package installation commands. if there's technical difficulties ask a developer for help
#    - python -m pip install h5py
#    - python -m pip install pandas
#    - python -m pip install numpy
#    - python -m pip install Pillow
# place "3D Manual Tracing Tracking Sheet.csv" in the same directory as this script.
# in cmd.exe, enter the directory where the script and the spreadsheet is located,
# and run the script with command python imaris_crop_lzw.py
# the script will
# (1) read the spreadsheet and make a copy of it "3D Manual Tracing Tracking Sheet copy.csv"
# (2) for each line,
#       a. checks if the expected tiff file already exists.
#       b. if a is false, crop and make tiff file. if a is true, restart step (2)
#       c. tiff file is saved at ims_dir/mr/ims_basename_zmin_zmax_ymin_ymax_xmin_xmax.tif.
#       d. write the tiff file path in column "ROI location"
# (3) save the spreadsheet "3D Manual Tracing Tracking Sheet.csv". if upon inspection there are errors, recover the
#     spreadsheet before the script from from the copy made in (1)
# (4) error messages and successful crops will be printed in cmd window. existing crops will not be remade
if __name__ == '__main__':
    crop_imaris()
