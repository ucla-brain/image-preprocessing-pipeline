function load_bl_tif_test()
% =================================================================
% load_bl_tif_test.m  (2025-06-07)
%
%   • Reliability & performance test-suite for load_bl_tif MEX
%   • Safe on older MATLAB versions without Tiff.ByteOrder / rewrite
%
% Run:
%   matlab -batch load_bl_tif_test
% =================================================================

%% -----------------------------------------------------------------
% 0. Locate source data _____________________________________________
% ------------------------------------------------------------------
folder_path = '/data/tif/B11_ds_4.5x_ABeta_z1200';          % ← Linux
if ispc, folder_path = 'V:/tif/B11_ds_4.5x_ABeta_z1200'; end
assert(isfolder(folder_path), ...
    'TIF test folder not found.  Edit folder_path in load_bl_tif_test.m');

files = dir(fullfile(folder_path,'*.tif'));
assert(~isempty(files),'No TIF files found in the folder.');

filelist  = fullfile({files.folder},{files.name});
numSlices = numel(filelist);

info        = imfinfo(filelist{1});
imageHeight = info.Height;
imageWidth  = info.Width;
bitDepth    = info.BitDepth;
fprintf('--- Dataset: %d×%d  |  %d slices  |  %d-bit ---\n', ...
        imageHeight,imageWidth,numSlices,bitDepth);

%% -----------------------------------------------------------------
% 1. Baseline regression ___________________________________________
% ------------------------------------------------------------------
blockSizes = [32,12; 12,23; 23,12; 512,1024];
testZ      = [round(numSlices/2), max(1,numSlices-3)];

fprintf('\n[Suite 1] Reference vs MEX baseline:\n');
fprintf('%-4s | %-6s | %-9s | %-13s | %-11s | %-12s | %s\n', ...
        'pass','Z','BlockSize','(X,Y)','Max Error','Speedup','Mode');
fprintf(repmat('-',1,74)); fprintf('\n');

for b = 1:size(blockSizes,1)
    blkH = blockSizes(b,1);  blkW = blockSizes(b,2);
    for zidx = testZ
        for transposeFlag = [false true]
            maxYStart = max(1, imageHeight - blkH + 1);
            maxXStart = max(1, imageWidth  - blkW + 1);
            y = randi([1, maxYStart]);
            x = randi([1, maxXStart]);

            y_idx = y : min(imageHeight, y+blkH-1);
            x_idx = x : min(imageWidth , x+blkW-1);
            z_idx = zidx : min(numSlices, zidx+2);

            % ── MATLAB reference ──────────────────────────────
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

            % ── MEX call ──────────────────────────────────────
            t2     = tic;
            bl_mex = load_bl_tif(filelist(z_idx), y, x, blkH, blkW, transposeFlag);
            tmex   = toc(t2);

            % ── Compare ───────────────────────────────────────
            pass   = isequaln(bl_gt, bl_mex);
            maxerr = 0;
            if ~pass
                diff   = abs(double(bl_mex) - double(bl_gt));
                maxerr = max(diff(:));
            end
            symbol = char(pass*10003 + ~pass*10007);           % ✓ / ✗
            mode   = ternary(transposeFlag,'T','N');
            fprintf('  %s  | %-6d | [%3d,%3d]  | (%5d,%5d) | %1.4e | %8.2fx |   %s\n', ...
                    symbol,zidx,blkH,blkW,x,y,maxerr,tref/tmex,mode);
        end
    end
end

%% -----------------------------------------------------------------
% 2. Spatial boundary checks _______________________________________
% ------------------------------------------------------------------
fprintf('\n[Suite 2] Spatial boundary checks:\n');
edgeROIs = {
   1,               1,                1,                1,   "top-left 1×1",          "ok";
   1,               1,                1,             512,   "top row stripe",        "ok";
   1,               1,              512,                1,   "left col stripe",       "ok";
   imageHeight,     1,                1,             512,   "bottom row stripe",     "ok";
   1,          imageWidth,         512,                1,   "right col stripe",      "ok";
   1,               1,       imageHeight,      imageWidth,   "full-frame (1 Z)",      "ok_singleZ";
  -20,             -20,             128,              128,   "upper-left overflow",   "expect_error";
   imageHeight-50,  imageWidth-50,  100,              100,   "bottom-right overflow", "ok";
};
for k = 1:size(edgeROIs,1)
    [y,x,h,w,label,kind] = edgeROIs{k,:};
    try
        switch kind
            case "ok"
                blk = load_bl_tif(filelist, y,x,h,w,false); %#ok<NASGU>
            case "ok_singleZ"
                blk = load_bl_tif(filelist(1), y,x,h,w,false); %#ok<NASGU>
            otherwise
                load_bl_tif(filelist, y,x,h,w,false);         % should error
        end
        if kind=="expect_error"
            fprintf('  ✗ %-25s did NOT error\n', label);
        else
            fprintf('  ✓ %-25s (size %s)\n', label, mat2str(size(blk)));
        end
    catch ME
        if kind=="expect_error"
            fprintf('  ✓ %-25s raised (%s)\n', label, ME.identifier);
        else
            fprintf('  ✗ %-25s ERROR: %s\n', label, ME.message);
        end
    end
end

%% -----------------------------------------------------------------
% 3. 8/16-bit little- vs big-endian ________________________________
% ------------------------------------------------------------------
fprintf('\n[Suite 3] 8/16-bit little- vs big-endian:\n');
tmpdir  = tempname; mkdir(tmpdir);
specs   = [
    struct("bits",8 ,"big",false);
    struct("bits",8 ,"big",true );
    struct("bits",16,"big",false);
    struct("bits",16,"big",true )
];
fname8LE = '';                                  % for negative test later

for s = specs'
    tagOK = true; skipped = false;
    fname = fullfile(tmpdir, sprintf("test_%dbit_%s.tif", ...
                     s.bits, ternary(s.big,'BE','LE')));
    if s.bits==8,  img = randi(intmax('uint8' ),imageHeight,imageWidth,'uint8' );
    else           img = randi(intmax('uint16'),imageHeight,imageWidth,'uint16');
    end

    t = Tiff(fname,'w');
    tag.ImageLength       = size(img,1);
    tag.ImageWidth        = size(img,2);
    tag.Photometric       = tryEnum('Tiff.Photometric.MinIsBlack',1);
    tag.BitsPerSample     = s.bits;
    tag.SamplesPerPixel   = 1;
    tag.PlanarConfiguration=tryEnum('Tiff.PlanarConfiguration.Contig',1);
    tag.Compression       = tryEnum('Tiff.Compression.None',1);
    t.setTag(tag);

    if s.big
        if ~setByteOrderBig(t)
            skipped = true;
        end
    end

    if skipped
        close(t);
        fprintf('  - skipped %2d-bit big-endian (ByteOrder unsupported)\n', s.bits);
        continue
    end

    t.write(img); close(t);

    if s.bits==8 && ~s.big, fname8LE = fname; end

    try
        blk = load_bl_tif({fname},1,1,32,32,false);
        assert(isequal(blk, img(1:32,1:32)));
        fprintf('  ✓ %2d-bit %s-endian\n', s.bits, ternary(s.big,'big','little'));
    catch ME
        fprintf('  ✗ %2d-bit %s-endian : %s\n', s.bits, ternary(s.big,'big','little'), ME.message);
    end
end

%% -----------------------------------------------------------------
% 4. Tile/strip + compression matrix _______________________________
% ------------------------------------------------------------------
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
    img   = magic(257);
    t     = Tiff(fname,'w');
    tag.ImageLength       = size(img,1);
    tag.ImageWidth        = size(img,2);
    tag.BitsPerSample     = 16;
    tag.SamplesPerPixel   = 1;
    tag.Photometric       = tryEnum('Tiff.Photometric.MinIsBlack',1);
    tag.PlanarConfiguration=tryEnum('Tiff.PlanarConfiguration.Contig',1);
    tag.Compression       = tryEnum("Tiff.Compression."+c.comp, ...
                                    compressionNumericFallback(c.comp));
    if c.tiled, tag.TileLength = 64; tag.TileWidth = 64;
    else,       tag.RowsPerStrip = 33;        end
    t.setTag(tag); t.write(uint16(img)); close(t);

    try
        blk = load_bl_tif({fname},20,20,100,100,false);
        ok  = isequal(blk, uint16(img(20:119,20:119)));
        fprintf('  %s  →  %s\n', c.name, ternary(ok,'✓','✗'));
    catch ME
        fprintf('  %s  →  ✗ (%s)\n', c.name, ME.message);
    end
end

%% -----------------------------------------------------------------
% 5. Negative-path assertions ______________________________________
% ------------------------------------------------------------------
fprintf('\n[Suite 5] Expected-error checks:\n');
negTests = {
    "Non-overlapping ROI",    @() load_bl_tif(filelist(1), -5000,-5000, 10,10,false);
    "File list empty str",    @() load_bl_tif({''},          1,1,10,10,false);
};
if ~isempty(fname8LE)
    negTests(end+1,:) = {"Mismatched bit-depth", ...
                         @() load_bl_tif({fname8LE}, 1,1,10,10,false)};
end
for n = 1:size(negTests,1)
    desc = negTests{n,1}; fn = negTests{n,2};
    try
        fn(); fprintf('  ✗ %-25s did NOT error\n', desc);
    catch
        fprintf('  ✓ %-25s raised error\n', desc);
    end
end

%% -----------------------------------------------------------------
% 6. Thread-scaling benchmark ______________________________________
% ------------------------------------------------------------------
fprintf('\n[Suite 6] Thread scaling (full-frame × all slices):\n');
bigROI = [1,1,imageHeight,imageWidth];
for nThreads = [1 2 4 8]
    setenv('LOAD_BL_TIF_THREADS',num2str(nThreads));
    tic; load_bl_tif(filelist, bigROI(1),bigROI(2),bigROI(3),bigROI(4),false); t=toc;
    fprintf('  %2d threads → %.3f s\n', nThreads, t);
end
setenv('LOAD_BL_TIF_THREADS','');

%% -----------------------------------------------------------------
% 7. Random ROI fuzzing ____________________________________________
% ------------------------------------------------------------------
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
        fprintf('\n ✗ crash at iter %d (y=%d,x=%d,h=%d,w=%d): %s\n', ...
                k,y,x,h,w, ME.message);
        break
    end
end
fprintf('\nAll suites finished.\n');
end  % main
% =================================================================
% Helper functions
% =================================================================
function out = ternary(c,a,b), out = a; if ~c, out = b; end, end

function val = tryEnum(enumStr, fallback)
    try,   val = eval(enumStr);
    catch, val = fallback; end
end

function n = compressionNumericFallback(name)
    switch upper(name)
        case 'NONE',    n = 1;
        case 'LZW',     n = 5;
        case 'DEFLATE', n = 32946;
        otherwise,      n = 1;
    end
end

function success = setByteOrderBig(tiffObj)
% Attempt to force big-endian header. Returns false if unsupported.
    success = false;
    try
        tiffObj.setTag('ByteOrder','big');     % R2021a+
        success = true;
    catch
        if ismethod(tiffObj,'rewrite')
            try
                tiffObj.rewrite;               % flush header
                fseek(tiffObj.FileID,0,'bof');
                fwrite(tiffObj.FileID,'MM','char'); % "MM" = big-endian
                success = true;
            catch, success = false; end
        end
    end
end
