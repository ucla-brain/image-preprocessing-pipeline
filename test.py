from process_images import *
from pystripe.core import *
from pywt import wavelist

# tsv_volume = TSVVolume(
#         r'W:\20231127_17_38_15_NM231025_01_LS_15x_800z_stitched_1\Ex_642_Ch2_xml_import_step_5.xml',
#         alt_stack_dir=r"W:\20231127_17_38_15_NM231025_01_LS_15x_800z_destriped_tif_1\Ex_642_Ch2")
# shape: Tuple[int, int, int] = tsv_volume.volume.shape  # shape is in z y x format
# img = tsv_volume.imread(
#     VExtent(
#         tsv_volume.volume.x0, tsv_volume.volume.x1,
#         tsv_volume.volume.y0, tsv_volume.volume.y1,
#         tsv_volume.volume.z0 + shape[0] // 2, tsv_volume.volume.z0 + shape[0] // 2 + 1),
#     tsv_volume.dtype)[0]
img = imread_tif_raw_png(Path(r"C:\Users\kmoradi\Downloads")/f"test2.png")


def main(wavelet):
    img_debleach = process_img(
        img,
        exclude_dark_edges_set_them_to_zero=False,
        sigma=(250, 250),
        wavelet=wavelet,
        # bidirectional=True,
        # bleach_correction_frequency=0.0005,
        # bleach_correction_clip_min=0,
        # bleach_correction_clip_max=63,
        # log1p_normalization_needed=True,
        convert_to_8bit=True,
        bit_shift_to_right=0,
        # tile_size=img.shape,
        # d_type=uint16,
        # down_sample=(16, 16),
        # down_sample_method="mean",
        # rotate=90
    )
    save_path = Path(r"C:\Users\kmoradi\Downloads")/"test_8"
    save_path.mkdir(exist_ok=True)
    imsave_tif(save_path/f"{wavelet}.tif", img_debleach)


if __name__ == '__main__':
    freeze_support()
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['NUMEXPR_NUM_THREADS'] = '1'
    # os.environ['OMP_NUM_THREADS'] = '1'
    with ProcessPoolExecutor(max_workers=32) as executor:
        # list(executor.map(main, wavelist(kind='discrete')))
        list(executor.map(main, ["bior3.7", "bior3.9", "db8", "db9", "db10", "sym7", "sym9", "sym10"]))




