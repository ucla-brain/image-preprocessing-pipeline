from pathlib import Path
from pandas import read_csv
from shutil import rmtree


# annotation file that contains soma locations only
annotations = next(
    Path(r"/data/20220725_17_41_55_SW220510_02_LS_6x_1000z_stitched").glob(
        "SW220406_01_LS_6x_1000z_combined_stamp*.ano.apo"))
recut = annotations.parent / 'soma_recut'
if recut.exists():
    rmtree(recut)
recut.mkdir()
annotations_df = read_csv(annotations)
soma_diam = 28
for column in ("x", "y", "z", "volsize"):
    annotations_df[column] = annotations_df[column].round(decimals=0).astype(int)

for row in annotations_df.itertuples():
    with open(recut/f"marker_{row.x}_{row.y}_{row.z}_{soma_diam}", 'w') as soma_file:  # row.volsize * 3
        soma_file.write("# x,y,z\n")
        soma_file.write(f"{row.x},{row.y},{row.z}")
