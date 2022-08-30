from pandas import read_csv, concat
from pathlib import Path

source = Path(r"Y:\3D_stitched_LS\20220605_SW220406_01_LS_6x_1000z\Ex_488_Em_525_TeraFly_Ano\Somata\Nnaemeka")
files = [
    r"SW220406_01_LS_6x_1000z_Neuron_0027_stamp_2022_08_19_10_01.ano.apo",
    r"SW220406_01_LS_6x_1000z_Somas_(NEW)_stamp_2022_08_26_16_48.ano.apo",
    r"SW220406_01_LS_6x_1000z_Somas_stamp_2022_08_05_11_40.ano.apo",
    r"SW220406_01_LS_6x_1000z_Somas_stamp_2022_08_26_03_29.ano.apo"
]
output = "SW220406_01_LS_6x_1000z_Somas"
df = concat([read_csv(source/file) for file in files], ignore_index=True).drop_duplicates().reset_index(drop=True)
# for column in df.columns:
#     print(column)
df["##n"] = df.index
apo_file = source/f"{output}.ano.apo"
ano_file = source/f"{output}.ano"
eswc_file = source/f"{output}.ano.eswc"
df.to_csv(apo_file, index=False)
with open(ano_file, "w") as ano:
    ano.write(f"APOFILE={apo_file.name}\n")
    ano.write(f"SWCFILE={apo_file.name[0:-3]}eswc\n")

with open(eswc_file, "w") as eswc:
    eswc.write(
        "#name undefined\n"
        "#comment terafly_annotations\n"
        "#n type x y z radius parent seg_id level mode timestamp TFresindex\n"
    )
