LsDeconv(...
    "/data/20230624_SM230223_05_LS_15x_800z_Double_MORF_B4/test/", ...
    400,  ... % dxy (nm)
    800,  ... % dz (nm)
    25,   ... % numit
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
    [repmat(1:8, 1, 5)],  ... % [repmat(1:8, 1, 5) zeros(1, 64)] gpu index in gpuDeviceTable, 0 means CPU
    1.0,  ... % signal amplification if clipval=0. clipval=1 means no amplification.
    [1, 1, 1.5], ... % x y z sigma of the 3D gaussian filter applied before deconvolution. filter_size = ceil(sigma * 4 + 1)
    [5, 5, 9], ... % filter_size
    20,   ... % denoising and background subtraction strength [0 to 255] for 8bit and [0 to 65535] for 16bit images
    0,    ... % 0 not resume, 1 = resume
    1,    ... % starting block should be greater than 0 for multiGPU processing
    0,    ... % 1 flip the deconvolved image upside down. 0 do not.
    1,    ... % 1 convert_to_8bit, 0 keep as is
    "/data/cache_deconvolution_b4" ... % cache drive (optional)
);
