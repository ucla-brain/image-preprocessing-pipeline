LsDeconv(...
    "/data/20220605_SW220406_01_LS_6x_1000z/Ex_488_Em_525_tif", ...
    1000, ... % dxy (nm)
    1000, ... % dz (nm)
    50,   ... % numit
    0.40, ... % NA
    1.52, ... % rf
    488,  ... % lambda_ex 488 561 642 647
    525,  ... % lambda_em 525 600 680 690
    240,  ... % fcyl 80, 240 for the older unaligned stage
    12.0, ... % slitwidth (mm)
    0,    ... % damping percent or lambda
    0,    ... % clipval
    0,    ... % stop_criterion
    intmax('int32'),... % block_size_max=max number of elements in the block. GPU arrays on MATLAB are limited to intmax('int32') as the max size.
    [1:8 1:8],  ... % [1:8 zeros(1, 64)] gpu index in gpuDeviceTable, 0 means CPU
    1.0,  ... % signal amplification if clipval=0. clipval=1 means no amplification.
    [0.5, 0.5, 1], ... % x y z sigma of the 3D gaussian filter applied before deconvolution. filter_size = ceil(sigma * 4 + 1)
    [3, 3, 7], ... % filter_size
    1,    ... % 0 not resume, 1 = resume
    1,    ... % starting block should be greater than 0 for multiGPU processing
    0,    ... % 1 flip the deconvolved image upside down. 0 do not.
    "/data/20220605_SW220406_01_LS_6x_1000z/cache_deconvolution" ... % cache drive (optional)
);