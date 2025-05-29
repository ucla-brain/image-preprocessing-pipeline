% ===============================
% build_mex.m
% ===============================
% Compile CUDA MEX files for wavedec2 and waverec2

% Set source files
src_semaphore = 'semaphore.c';
src_queue = 'queue.c';
src_loadOTFCacheMapped = 'loadOTFCacheMapped_mex.c'

%src_wavedec2 = 'wavedec2_mex.cu';
%src_waverec2 = 'waverec2_mex.cu';
%src_filter_subband = 'filter_subband_mex.cu';
%build_xml = "C:/Program Files/MATLAB/R2023a/toolbox/parallel/gpu/extern/src/mex/win64/nvcc_msvcpp2022.xml";

% Include directory for headers
%root_dir = '.';
%include_dir = './cuda_kernels';
%setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8');
%setenv('CUDA_ROOT', getenv('CUDA_PATH'));
%setenv('CUDA_TOOLKIT_ROOT_DIR', getenv('CUDA_PATH'));
%setenv('MW_ALLOW_ANY_CUDA', '1');


% Compile
mex('-O', '-v', src_semaphore);
mex('-O', '-v', src_queue);
mex('-O', '-v', src_loadOTFCacheMapped);
%mexcuda('-v', '-g', '-G', '-f', build_xml, '-R2018a', src_wavedec2,       ['-I' root_dir], ['-I' include_dir]);
%mexcuda('-v', '-g', '-G', '-f', build_xml, '-R2018a', src_waverec2,       ['-I' root_dir], ['-I' include_dir]);
%mexcuda('-v', '-g', '-G', '-f', build_xml, '-R2018a', src_filter_subband, ['-I' root_dir], ['-I' include_dir]);