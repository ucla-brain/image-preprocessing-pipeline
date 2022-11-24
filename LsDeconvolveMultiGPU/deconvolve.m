LsDeconv(...
    "/data/20221122_10_15_02_Camk2a_MORF3_Bcl11bKO_311_3_LH_15x_5x5_z04_2_5percent_stitched/Ex_647_Em_690_tif_bs=1", ...
    400, ... % dxy (nm)
    400, ... % dz (nm)
    9,    ... % numit
    0.40, ... % NA
    1.52, ... % rf
    647,  ... % lambda_ex 488 561 642
    690,  ... % lambda_em 525 600 680
    80,  ... % fcyl 240 for the older unaligned stage
    12.0, ... % slitwidth (mm)
    0,    ... % damping percent or lambda
    0,    ... % clipval
    0,    ... % stop_criterion
    6.3,  ... % block_size_max in GB
    [1 2 3 4 5 6 7 8], ... % [1 2 3 4 5 6 7 8] gpu index in gpuDeviceTable, [0] means CPU
    3.0,  ... % signal amplification if clipval=0. clipval=1 means no amplification.
    [0.5, 0.5, 0.5], ... % x y z sigma of the 3D gaussian filter applied before deconvolution [0.5, 0.5, 2.0]
    1,    ... % 0 not resume, 1 = resume
    1,    ... % starting block should be greater than 0 for multiGPU processing
    "/data/cache_deconvolution_yang/" ... % cache drive (optional)
);