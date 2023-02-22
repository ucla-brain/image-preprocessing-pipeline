from pathlib import Path
from pandas import read_csv
from shutil import rmtree
from math import pi


# annotation file that contains soma locations only
annotations = next(
    Path(r"/qnap/3D_stitched_LS/20220725_SW220510_02_LS_6x_1000z/Ex_488_Em_525_Terafly_Ano/Somata").glob(
        "SW220510_02_LS_6x_1000z_combined_stamp*.ano.apo"))
recut = annotations.parent / 'soma_recut'
if recut.exists():
    rmtree(recut)
recut.mkdir()
annotations_df = read_csv(annotations)
soma_radius_um = 8.0
for column in ("x", "y", "z", "volsize"):
    annotations_df[column] = annotations_df[column].round(decimals=0).astype(int)

for row in annotations_df.itertuples():
    with open(recut/f"marker_{row.x}_{row.y}_{row.z}_{int(4/3*pi * soma_radius_um**3 + 0.5)}", 'w') as soma_file:
        soma_file.write("# x,y,z,radius_um\n")
        soma_file.write(f"{row.x},{row.y},{row.z},{soma_radius_um}")
