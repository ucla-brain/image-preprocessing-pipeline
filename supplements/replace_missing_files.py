from pathlib import Path

files = [
r"/data/20220725_17_41_55_SW220510_02_LS_15x_1000z_lightsheet_cleaned_tif_bitshift.g2.r2.b0_downsampled/Ex_561_Em_600/182520/182520_133870/010440.tif",
]
source = Path("/data/20220725_17_41_55_SW220510_02_LS_15x_1000z_lightsheet_cleaned_tif_bitshift.g2.r2.b0_downsampled/Ex_561_Em_600/182520/182520_133870/000010.tif").read_bytes()

for file in files:
    Path(file).write_bytes(source)
