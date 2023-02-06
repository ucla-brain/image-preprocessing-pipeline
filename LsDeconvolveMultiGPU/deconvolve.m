LsDeconv(...
    "/data/20230106_17_28_41_iDISCO_test_PV_MORF_NM220810_LS_15x_1000z_stitched/Ex_561_Em_600_tif", ...
    422, ... % dxy (nm)
    1000, ... % dz (nm)
    25,   ... % numit
    0.40, ... % NA
    1.52, ... % rf
    561,  ... % lambda_ex 488 561 642 647
    600,  ... % lambda_em 525 600 680 690
    240,  ... % fcyl 240 for the older unaligned stage
    12.0, ... % slitwidth (mm)
    0,    ... % damping percent or lambda
    0,    ... % clipval
    0,    ... % stop_criterion
    4.2,  ... % block_size_max in GB
    [0], ... % [1 2 3 4 5 6 7 8] gpu index in gpuDeviceTable, [0] means CPU
    3.0,  ... % signal amplification if clipval=0. clipval=1 means no amplification.
    [0.5, 0.5, 2.0], ... % x y z sigma of the 3D gaussian filter applied before deconvolution [0.5, 0.5, 2.0]
    1,    ... % 0 not resume, 1 = resume
    1,    ... % starting block should be greater than 0 for multiGPU processing
    "/data/cache_deconvolution/" ... % cache drive (optional)
);