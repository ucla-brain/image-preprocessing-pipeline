LsDeconv(...
    "/panfs/dong/U01_MORF/upload/20220818_22_59_23_Camk2a_MORF3_D1Tom_TME12_1_15x_5x5_x042_y042_z100_stitched/15x/Ex_561_Em_1_tif", ...
    422, ... % dxy (nm)
    1000, ... % dz (nm)
    9,    ... % numit
    0.40, ... % NA
    1.52, ... % rf
    561,  ... % lambda_ex 488 561 642
    600,  ... % lambda_em 525 600 680
    240,  ... % fcyl
    12.0, ... % slitwidth (mm)
    0,    ... % damping percent or lambda
    0,    ... % clipval
    0,    ... % stop_criterion
    6.9,  ... % block_size_max in GB
    [1 2 3 4 5 6 7 8], ... % [1 2 3 4 5 6 7 8] gpu index in gpuDeviceTable, [0] means CPU
    4.0,  ... % signal amplification if clipval=0. clipval=1 means no amplification.
    [0.5, 0.5, 2.0], ... % x y z sigma of the 3D gaussian filter applied before deconvolution
    1,    ... % 0 not resume, 1 = resume
    1,    ... % starting block should be greater than 0 for multiGPU processing
    "/data/cache_deconvolution_yang/" ... % cache drive (optional)
);