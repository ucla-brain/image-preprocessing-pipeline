LsDeconv(...
    "/data/Camk2a_MORF3_Bcl11bKO_311_3_LH_15x_3x3_stitched/Ex_647_Em_690_tif_bs=1_cropped", ...
    400, ... % dxy (nm)
    400, ... % dz (nm)
    3,   ... % numit
    0.40, ... % NA
    1.52, ... % rf
    647,  ... % lambda_ex 488 561 642 647
    690,  ... % lambda_em 525 600 680 690
    240,  ... % fcyl 80, 240 for the older unaligned stage
    12.0, ... % slitwidth (mm)
    0,    ... % damping percent or lambda
    0,    ... % clipval
    0,    ... % stop_criterion
    intmax('int32'),... % block_size_max=max number of elements in the block. GPU arrays on MATLAB are limited to intmax('int32') as the max size.
    [0],  ... % [1:8 zeros(1, 64)] gpu index in gpuDeviceTable, 0 means CPU
    1.0,  ... % signal amplification if clipval=0. clipval=1 means no amplification.
    [0.5, 0.5, 1.51], ... % x y z sigma of the 3D gaussian filter applied before deconvolution. filter_size = ceil(sigma * 2 + 1)
    0,    ... % 0 not resume, 1 = resume
    1,    ... % starting block should be greater than 0 for multiGPU processing
    1,    ... % 1 flip the deconvolved image upside down. 0 do not.
    "/data/Camk2a_MORF3_Bcl11bKO_311_3_LH_15x_3x3_stitched/cache_deconvolution_cropped" ... % cache drive (optional)
);