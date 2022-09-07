LsDeconv(...
    "/data/20220818_12_24_18_SW220405_05_LS_6x_1000z_stitched/Ex_642_Em_680_tif", ...
    1000, ... % dxy (nm)
    1000, ... % dz (nm)
    9,    ... % numit
    0.40, ... % NA
    1.52, ... % rf
    642,  ... % lambda_ex 488 561 642
    680,  ... % lambda_em 525 600 680
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
    "/data/cache_deconvolution/" ... % cache drive (optional)
);