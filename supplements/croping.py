# crop imaris file according to roi bounds defined in spreadsheet
# save to a single tiff file
import os
import shutil
import h5py
import numpy as np
import pandas as pd
from PIL import Image


def convert_32bit_to_16bit(img, right_shift=16):
    if img.dtype == 'uint32':
        img = (img >> right_shift)
        img[img > 4294967295] = 4294967295
        img = img.astype('uint16')
    else:
        print(f"\nWarning: the original image was not 32 bit. 16 bit conversion is disabled.\n")
    return img


def convert_16bit_to_8bit_fun(img, right_shift=8):
    if img.dtype == 'uint16':
        # bit shift then change the type to avoid floating point operations
        # img >> 8 is equivalent to img / 256
        if 0 < right_shift < 8:
            img = (img >> right_shift)
            img[img > 255] = 255
            img = img.astype('uint8')
        elif right_shift < 0 or right_shift > 8:
            print("right shift should be between 0 and 8")
            raise RuntimeError
        else:
            img = (img >> 8).astype('uint8')
    else:
        print(f"\nWarning: the original image was not 16 bit. 8 bit conversion is disabled.\n")
    return img


def imwrite_lzw(path, data, convert_16bit_to_8bit=True, right_shift=8):
    im_list = []
    if convert_16bit_to_8bit:
        data = convert_16bit_to_8bit_fun(data, right_shift=right_shift)
    for im in data:
        image = Image.fromarray(im)
        im_list.append(image)

    im_list[0].save(path, compression="tiff_lzw", save_all=True, append_images=im_list[1:])


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
        multi_channel_datasets = ims['DataSet/ResolutionLevel {}/TimePoint 0'.format(resolution_level)]
    except KeyError:
        raise KeyError('requested resolution level {} does not exist'.format(resolution_level))
    if channel < 0 or channel >= len(multi_channel_datasets.keys()):
        raise ValueError('{} is not a valid channel id'.format(channel))

    channel_dataset = multi_channel_datasets['Channel {}/Data'.format(channel)]
    z_start, y_start, x_start = zyx_offsets
    z_stop = zyx_extents[0] + z_start if zyx_extents[0] > 0 else channel_dataset.shape[0]
    y_stop = zyx_extents[1] + y_start if zyx_extents[1] > 0 else channel_dataset.shape[1]
    x_stop = zyx_extents[2] + x_start if zyx_extents[2] > 0 else channel_dataset.shape[2]
    if z_start < 0 or y_start < 0 or x_start < 0:
        raise ValueError('offset values must be non negative')
    if z_stop - z_start > channel_dataset.shape[0] or y_stop - y_start > channel_dataset.shape[1] or \
            x_stop - x_start > channel_dataset.shape[2]:
        raise ValueError('extent values out of range')
    # skipping range checking and assume 16bit. quick script
    img = np.zeros(shape=(z_stop - z_start, y_stop - y_start, x_stop - x_start), dtype=np.uint32)
    channel_dataset.read_direct(img, source_sel=(np.s_[z_start: z_stop, y_start: y_stop, x_start: x_stop]))
    ims.close()
    return convert_32bit_to_16bit(img)


def get_imaris_path(df, i):
    return os.path.join(df.loc[i, 'File Location'], df.loc[i, 'File Name'])


def get_roi_str(df, i):
    return '{}_{}_{}_{}_{}_{}'.format(df.loc[i, 'ROI_zmin'], df.loc[i, 'ROI_zmax'], df.loc[i, 'ROI_ymin'], df.loc[i, 'ROI_ymax'], df.loc[i, 'ROI_xmin'], df.loc[i, 'ROI_xmax'])


def get_zyx_offsets(df, i):
    return [df.loc[i, 'ROI_zmin'], df.loc[i, 'ROI_ymin'], df.loc[i, 'ROI_xmin']]


def get_zyx_extents(df, i):
    return [df.loc[i, 'ROI_zmax'] - df.loc[i, 'ROI_zmin'] + 1, df.loc[i, 'ROI_ymax'] - df.loc[i, 'ROI_ymin'] + 1,
            df.loc[i, 'ROI_xmax'] - df.loc[i, 'ROI_xmin'] + 1]


def get_tiff_dir(df, i):
    return os.path.join(df.loc[i, 'File Location'], 'mr')


def get_tiff_path(df, i):
    ims_name, _ = os.path.splitext(df.loc[i, 'File Name'])
    return os.path.join(get_tiff_dir(df, i), '{}_{}.tif'.format(ims_name, get_roi_str(df, i)))


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
            data = read_ims(get_imaris_path(df, i), get_zyx_offsets(df, i), zyx_extents=get_zyx_extents(df, i),
                            channel=df.loc[i, 'Channel (0 indexed: the 1st channel in the file is entered as channel 0)'])
            os.makedirs(get_tiff_dir(df, i), exist_ok=True)
            imwrite_lzw(get_tiff_path(df, i), data)
            assert os.path.isfile(get_tiff_path(df, i))
            # update tiff crop location
            df.loc[i, 'ROI location'] = get_tiff_path(df, i)
            print('cropped {}: roi {}'.format(df.loc[i, 'File Name'], get_roi_str(df, i)))
        except Exception as e:
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
