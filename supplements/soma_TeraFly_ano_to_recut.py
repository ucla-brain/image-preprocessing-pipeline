from pathlib import Path
from pandas import read_csv


# annotation file that contains soma locations only
annotations = Path(r"/qnap/3D_stitched_LS/20220725_SW220510_02_LS_6x_1000z/Ex_488_Em_525_Terafly_Ano/Somata/SW220406_01_LS_6x_1000z_combined_stamp_2022_09_01_18_18.ano.apo")
recut = annotations.parent / 'soma_recut'
recut.mkdir(exist_ok=True)
annotations_df = read_csv(annotations)
soma_diam = 1000
for column in ("x", "y", "z", "volsize"):
    annotations_df[column] = annotations_df[column].round(decimals=0).astype(int)

for row in annotations_df.itertuples():
    with open(recut/f"marker_{row.x}_{row.y}_{row.z}_{soma_diam}", 'w') as soma_file:  # row.volsize * 3
        soma_file.write("# x,y,z\n")
        soma_file.write(f"{row.x},{row.y},{row.z}")
