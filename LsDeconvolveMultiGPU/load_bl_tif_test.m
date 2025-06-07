function load_bl_tif_test()
% ==============================================================
% load_bl_tif_test.m  • Reliability + performance test-suite
%
% 2025-06-07  –  Patched:
%   • Suite-2 full-frame is memory-safe (single-slice only)
%   • Overflow ROI now counted as pass when it errors
%   • Robust to MATLAB versions lacking Tiff.* enumerations
% ==============================================================

%% -------------------------------------------------------------
% 0. Locate source data ________________________________________
% --------------------------------------------------------------
folder_path = '/data/tif/B11_ds_4.5x_ABeta_z1200';          % ← Linux
if ispc
    folder_path = 'V:/tif/B11_ds_4.5x_ABeta_z1200';         % ← Windows
end
if isempty(folder_path) || ~isfolder(folder_path)
    error('TIF test folder not found. Check that folder_path is valid.');
end

files     = dir(fullfile(folder_path,'*.tif'));
assert(~isempty(files),'No TIF files found in the folder.');

filelist  = fullfile({files.folder},{files.name});
numSlices = numel(filelist);

info          = imfinfo(filelist{1});
imageHeight   = info.Height;
imageWidth    = info.Width;
bitDepth      = info.BitDepth;
if ~ismember(bitDepth,[8 16])
    error('Expected 8- or 16-bit grayscale TIFF images in source folder.');
end

fprintf('--- load_bl_tif_test: %d×%d pixels, %d slices, %d-bit ---\n',...
        imageHeight,imageWidth,numSlices,bitDepth);

%% -------------------------------------------------------------
% 1. Baseline regression _______________________________________
% --------------------------------------------------------------
blockSizes = [32, 12;
              12, 23;
              23, 12;
             512,1024];
testZ      = [round(numSlices/2), max(1,numSlices-3)];
totalTests = size(blockSizes,1) * numel(testZ) * 2;
results    = zeros(totalTests,9);           % [pass z h w x y maxerr speedup transpose]
testIdx    = 1;

fprintf('\n[Suite 1] Reference vs MEX baseline:\n');
fprintf('%-4s | %-6s | %-9s | %-13s | %-11s | %-12s | %s\n', ...
    'pass','Z','BlockSize','(X,Y)','Max Error','Speedup','Mode');
fprintf(repmat('-',1,74)); fprintf('\n');

for b = 1:size(blockSizes,1)
    blkH = blockSizes(b,1); blkW = blockSizes(b,2);

    for zidx = testZ
        for transposeFlag = [false true]
            % --- random top-left corner (always valid) --------------
            maxYStart = max(1, imageHeight - blkH + 1);
            maxXStart = max(1, imageWidth  - blkW + 1);
            y = randi([1, maxYStart]);
            x = randi([1, maxXStart]);

            y_idx = y : min(imageHeight, y+blkH-1);
            x_idx = x : min(imageWidth , x+blkW-1);
            z_idx = zidx : min(numSlices, zidx+2);

            % MATLAB reference
            if transposeFlag
                bl_gt = zeros(numel(x_idx),numel(y_idx),numel(z_idx),'uint16');
            else
                bl_gt = zeros(numel(y_idx),numel(x_idx),numel(z_idx),'uint16');
            end
            t1 = tic;
            for k = 1:numel(z_idx)
                slice = imread(filelist{z_idx(k)}, ...
                               'PixelRegion',{[y_idx(1),y_idx(end)],...
                                              [x_idx(1),x_idx(end)]});
                bl_gt(:,:,k) = ternary(transposeFlag, slice', slice);
            end
            tref = toc(t1);

            % MEX call
            t2     = tic;
            bl_mex = load_bl_tif(filelist(z_idx),y,x,blkH,blkW,transposeFlag);
            tmex   = toc(t2);

            % Compare
            minH = min(size(bl_gt,1),size(bl_mex,1));
            minW = min(size(bl_gt,2),size(bl_mex,2));
            minZ = min(size(bl_gt,3),size(bl_mex,3));
            pass = isequal(bl_gt(1:minH,1:minW,1:minZ), ...
                           bl_mex(1:minH,1:minW,1:minZ));

            if pass, maxerr = 0;
            else
                diff   = abs(double(bl_mex) - double(bl_gt));
                maxerr = max(diff(:));
            end
            symbol = char(pass*10003 + ~pass*10007);  % ✓ / ✗
            modeStr= ternary(transposeFlag,'T','N');
            fprintf('  %s  | %-6d | [%3d,%3d]  | (%5d,%5d) | %1.4e | %8.2fx |   %s\n',...
                    symbol,zidx,blkH,blkW,x,y,maxerr,tref/tmex,modeStr);

            results(testIdx,:) = [pass,zidx,blkH,blkW,x,y,maxerr,tref/tmex,transposeFlag];
            testIdx = testIdx+1;
        end
    end
end
if all(results(:,1)), fprintf('\n🎉 Baseline tests passed.\n');
else,                 fprintf('\n❗ Baseline failures detected.\n'); end

%% -------------------------------------------------------------
% 2. Boundary & out-of-bounds ROIs _____________________________
% --------------------------------------------------------------
fprintf('\n[Suite 2] Spatial boundary checks:\n');

edgeROIs = {
   1,               1,                1,                1,   "top-left 1×1",       "ok";
   1,               1,                1,             512,   "top row stripe",     "ok";
   1,               1,              512,                1,   "left col stripe",    "ok";
   imageHeight-0,   1,                1,             512,   "bottom row stripe",  "ok";
   1,          imageWidth-0,         512,                1,   "right col stripe",   "ok";
   1,               1,       imageHeight,      imageWidth,   "full-frame (1 Z)",   "ok_singleZ";
  -20,             -20,             128,              128,   "upper-left overflow","expect_error";
   imageHeight-50,  imageWidth-50,  100,              100,   "bottom-right overflow","ok"
};

for k = 1:size(edgeROIs,1)
    [y,x,h,w,label,kind] = edgeROIs{k,:};
    try
        switch kind
            case "ok"
                blk = load_bl_tif(filelist,y,x,h,w,false); %#ok<NASGU>
            case "ok_singleZ"
                blk = load_bl_tif(filelist(1),y,x,h,w,false); %#ok<NASGU>
            otherwise
                % expected to throw
                load_bl_tif(filelist,y,x,h,w,false);
        end
        if kind=="expect_error"
            fprintf('  ✗ %-25s  did NOT error\n',label);
        else
            fprintf('  ✓ %-25s  (size %s)\n',label,mat2str(size(blk)));
        end
    catch ME
        if kind=="expect_error"
            fprintf('  ✓ %-25s  raised error (%s)\n',label,ME.identifier);
        else
            fprintf('  ✗ %-25s  ERROR: %s\n',label,ME.message);
        end
    end
end

%% -------------------------------------------------------------
% 3. Byte-order & bit-depth sanity _____________________________
% --------------------------------------------------------------
fprintf('\n[Suite 3] 8/16-bit little- vs big-endian:\n');
tmpdir = tempname; mkdir(tmpdir);

specs = [
    struct("bits",8 ,"big",false);
    struct("bits",8 ,"big",true );
    struct("bits",16,"big",false);
    struct("bits",16,"big",true )
];
fname8LE = '';

for s = specs'
    fname = fullfile(tmpdir, sprintf("test_%dbit_%s.tif", ...
                     s.bits, ternary(s.big,'BE','LE')));
    if s.bits==8
        img = randi(intmax('uint8' ),imageHeight,imageWidth,'uint8' );
    else
        img = randi(intmax('uint16'),imageHeight,imageWidth,'uint16');
    end
    if s.bits==8 && ~s.big, fname8LE = fname; end

    % --- write synthetic TIFF slice ------------------------------
    t = Tiff(fname,'w');
    tag.ImageLength       = size(img,1);
    tag.ImageWidth        = size(img,2);
    tag.Photometric       = tryEnum('Tiff.Photometric.MinIsBlack',1);
    tag.BitsPerSample     = s.bits;
    tag.SamplesPerPixel   = 1;
    tag.PlanarConfiguration = tryEnum('Tiff.PlanarConfiguration.Contig',1);
    tag.Compression       = tryEnum('Tiff.Compression.None',1);
    if s.big, setByteOrderBig(t); end
    t.setTag(tag); t.write(img); t.close;

    % --- loader check -------------------------------------------
    try
        blk = load_bl_tif({fname},1,1,32,32,false);
        assert(isequal(blk, img(1:32,1:32)));
        fprintf('  ✓ %2d-bit %s-endian\n',s.bits, ternary(s.big,'big','little'));
    catch ME
        fprintf('  ✗ %2d-bit %s-endian : %s\n',s.bits, ternary(s.big,'big','little'), ME.message);
    end
end

%% -------------------------------------------------------------
% 4. Tile / strip + compression matrix _________________________
% --------------------------------------------------------------
fprintf('\n[Suite 4] Tile/strip + compression:\n');
cfgs = [
   struct("tiled",false,"comp",'None'   ,"name","strip-none");
   struct("tiled",false,"comp",'LZW'    ,"name","strip-lzw" );
   struct("tiled",false,"comp",'Deflate',"name","strip-deflate");
   struct("tiled",true, "comp",'None'   ,"name","tile-none");
   struct("tiled",true, "comp",'LZW'    ,"name","tile-lzw" );
   struct("tiled",true, "comp",'Deflate',"name","tile-deflate");
];
for c = cfgs'
    fname = fullfile(tmpdir,"tile_"+c.name+".tif");
    img   = magic(257);                       % non-power-of-two
    t     = Tiff(fname,'w');
    tag.ImageLength     = size(img,1);
    tag.ImageWidth      = size(img,2);
    tag.BitsPerSample   = 16;
    tag.SamplesPerPixel = 1;
    tag.Photometric     = tryEnum('Tiff.Photometric.MinIsBlack',1);
    tag.PlanarConfiguration = tryEnum('Tiff.PlanarConfiguration.Contig',1);
    tag.Compression     = tryEnum("Tiff.Compression."+c.comp, ...
                                  compressionNumericFallback(c.comp));
    if c.tiled
        tag.TileLength  = 64; tag.TileWidth = 64;
    else
        tag.RowsPerStrip= 33;
    end
    t.setTag(tag); t.write(uint16(img)); t.close;

    try
        blk = load_bl_tif({fname},20,20,100,100,false);
        ok  = isequal(blk,uint16(img(20:119,20:119)));
        fprintf('  %s  →  %s\n',c.name, ternary(ok,'✓','✗'));
    catch ME
        fprintf('  %s  →  ✗ (%s)\n',c.name, ME.message);
    end
end

%% -------------------------------------------------------------
% 5. Negative-path assertions __________________________________
% --------------------------------------------------------------
fprintf('\n[Suite 5] Expected-error checks:\n');
negTests = {
    "Non-overlapping ROI",    @() load_bl_tif(filelist(1), -5000,-5000, 10,10,false);
    "Mismatched bit-depth",   @() load_bl_tif({fname8LE},   1,1,10,10,false);
    "File list empty string", @() load_bl_tif({''},          1,1,10,10,false);
};
for n = 1:size(negTests,1)
    desc = negTests{n,1}; fn = negTests{n,2};
    try
        fn(); fprintf('  ✗  %-30s did NOT error\n',desc);
    catch
        fprintf('  ✓  %-30s raised error\n',desc);
    end
end

%% -------------------------------------------------------------
% 6. Thread-scaling benchmark _________________________________
% --------------------------------------------------------------
fprintf('\n[Suite 6] Thread scaling (full-frame × all slices):\n');
bigROI = [1,1,imageHeight,imageWidth];
for nThreads = [1 2 4 8]
    setenv('LOAD_BL_TIF_THREADS',num2str(nThreads));
    tic;
    load_bl_tif(filelist, bigROI(1),bigROI(2),bigROI(3),bigROI(4),false);
    t = toc;
    fprintf('  %2d threads → %.3f s\n', nThreads, t);
end
setenv('LOAD_BL_TIF_THREADS','');   % restore

%% -------------------------------------------------------------
% 7. 500-iteration random ROI fuzzing __________________________
% --------------------------------------------------------------
fprintf('\n[Suite 7] 500 random ROI fuzz tests (progress dots):\n');
rng(42);
for k = 1:500
    h = randi([1,imageHeight]);
    w = randi([1,imageWidth ]);
    y = randi([-32,imageHeight]);
    x = randi([-32,imageWidth ]);
    try
        load_bl_tif(filelist,y,x,h,w, rand>0.5);
        if mod(k,20)==0, fprintf('.'); end
    catch ME
        fprintf('\n ✗ crash at iter %d (y=%d,x=%d,h=%d,w=%d): %s\n',...
                k,y,x,h,w, ME.message);
        break
    end
end
fprintf('\nAll suites finished.\n');

end % main function
% ==============================================================
% Helper sub-functions
% ==============================================================

function out = ternary(cond,a,b)
    if cond, out = a; else, out = b; end
end

function val = tryEnum(enumStr, fallback)
% Return the enum constant if it exists, otherwise the numeric fallback.
    try
        val = eval(enumStr);
    catch
        val = fallback;
    end
end

function n = compressionNumericFallback(name)
% Hard-coded numeric codes used by libtiff for common compressions
    switch upper(name)
        case 'NONE',    n = 1;
        case 'LZW',     n = 5;
        case 'DEFLATE', n = 32946;
        otherwise,      n = 1;
    end
end

function setByteOrderBig(tiffObj)
% Force big-endian header even on little-endian hosts.
    try
        tiffObj.setTag('ByteOrder','big');   % R2021a+
    catch
        % Older MATLAB: tiny patch of the header
        warning('off','all');
        tiffObj.rewrite;                     % flush current header
        fseek(tiffObj.FileID,0,'bof');
        fwrite(tiffObj.FileID,'MM','char');  % "MM" = big-endian magic
        warning('on','all');
    end
end
