# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:44:03 2023

@author: YangRecon2
"""

from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

base_dir= Path(r'J:\LS_PreProcess\Camk2a_MORF3_Bcl11bKO_311_3_LH_Str-ROI\15x_z04_025_lightsheet_cleaned_bitshift.g1_tif_stitched\run-4\seeds_stamp_2023_02_27_16_44.ano.apo')
dest_dir = Path(r'C:\Users\YangRecon2\Desktop\swc_export_test')


# for each proofread neurons, extract their soma location from the swc file name
# and aggregate to a single swc file


with open(dest_dir/'6x6ROI_striatum_only_from_terafly_proofread.swc','w') as dest_file:
    cnt = 1
    for file in tqdm(base_dir.rglob('*')):
        # print(file)
        if len([l for l in open(file).readlines()]) > 0:
            # file_name = file.name.replace('.swc','').replace('tree-with-soma-xyz-','')
            file_name = file.name.replace('marker_', '')
            # print(file_name.split('-')[1])
            # first_line = open(file).readlines()[2]
            # print(first_line)

            x = file_name.split('_')[0]
            y = file_name.split('_')[1]
            z = file_name.split('_')[2]
            radii = file_name.split('_')[3]

            # x = file_name.split('-')[0]
            # y = file_name.split('-')[1]
            # z = file_name.split('-')[2]

            dest_file.write(f"{cnt} 0 {x} {y} {z} {radii} -1\n")
            cnt += 1
dest_file.close()
        

a = [[1,2,3], [4,5,6]]
b = np.array(a)
c = b[:,2]
c
d = 10-c
d    


b
b[:,2] = 10 - b[:,2]
b
