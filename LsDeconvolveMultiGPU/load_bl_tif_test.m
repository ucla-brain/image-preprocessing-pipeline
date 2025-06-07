function load_bl_tif_test()
% =================================================================
% load_bl_tif_test.m   (2025-06-07, patch-3)
%
% Full reliability & performance suite for the load_bl_tif MEX.
%
%   • Fixes Suite-4 path: file-names are now char, not string
%   • Suite-6 benchmark uses a memory-safe 512×512 ROI (all slices)
%   • Gracefully skips big-endian tests when unsupported
%
% Run from a shell so the env-var thread control works:
%     matlab -batch load_bl_tif_test
% =================================================================

%% -----------------------------------------------------------------
% 0. Locate microscope data ________________________________________
% ------------------------------------------------------------------
folder_path = '/data/tif/B11_ds_4.5x_ABeta_z1200';    % ← Linux
if ispc, folder_path = 'V:/tif/B11_ds_4.5x_ABeta_z1200'; end
assert(isfolder(folder_path), ...
       'TIF test folder not found.  Edit folder_path in load_bl_tif_test.m');

files = dir(fullfile(folder_path,'*.tif'));
assert(~isempty(files),'No TIF files found in the folder.');
filelist   = fullfile({files.folder},{files.name});
numSlices  = numel(filelist);

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
        for tr = [false true]
            maxY = max(1, imageHeight - blkH + 1);
            maxX = max(1, imageWidth  - blkW + 1);
            y = randi([1, maxY]);   x = randi([1, maxX]);
            y_idx = y : min(imageHeight, y+blkH-1);
            x_idx = x : min(imageWidth , x+blkW-1);
            z_idx = zidx : min(numSlices, zidx+2);

            % MATLAB reference
            if tr
                ref = zeros(numel(x_idx),numel(y_idx),numel(z_idx),'uint16');
            else
                ref = zeros(numel(y_idx),numel(x_idx),numel(z_idx),'uint16');
            end
            t1 = tic;
            for k = 1:numel(z_idx)
                slice = imread(filelist{z_idx(k)}, ...
                         'PixelRegion',{[y_idx(1),y_idx(end)], [x_idx(1),x_idx(end)]});
                ref(:,:,k) = ternary(tr, slice', slice);
            end
            tref = toc(t1);

            % MEX call
            t2   = tic;
            mexO = load_bl_tif(filelist(z_idx), y, x, blkH, blkW, tr);
            tmex = toc(t2);

            pass   = isequaln(ref, mexO);
            maxerr = pass*0 + ~pass*max(abs(double(mexO(:))-double(ref(:))));
            fprintf('  %c  | %-6d | [%3d,%3d]  | (%5d,%5d) | %1.4e | %8.2fx |   %s\n', ...
                    char(pass*10003 + ~pass*10007), zidx, blkH, blkW, ...
                    x, y, maxerr, tref/tmex, ternary(tr,'T','N'));
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
            otherwise         % expect_error
                load_bl_tif(filelist, y,x,h,w,false);
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
fname8LE = '';                          % used later for neg-test

for s = specs'
    fname = fullfile(tmpdir, sprintf("t%db_%s.tif", ...
                     s.bits, ternary(s.big,'BE','LE')));
    if s.bits==8,  img = randi(intmax('uint8' ),512,512,'uint8' );
    else           img = randi(intmax('uint16'),512,512,'uint16');
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

    if s.big && ~setByteOrderBig(t)
        close(t); delete(fname);        % skip if unsupported
        fprintf('  - skipped %2d-bit big-endian (unsupported)\n', s.bits);
        continue
    end
    t.write(img); close(t);
    if s.bits==8 && ~s.big, fname8LE = fname; end

    try
        blk = load_bl_tif({char(fname)},1,1,32,32,false);
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
   struct("tiled",false,"comp",'None'   ,"name","strip-none"   );
   struct("tiled",false,"comp",'LZW'    ,"name","strip-lzw"    );
   struct("tiled",false,"comp",'Deflate',"name","strip-deflate");
   struct("tiled",true, "comp",'None'   ,"name","tile-none"    );
   struct("tiled",true, "comp",'LZW'    ,"name","tile-lzw"     );
   struct("tiled",true, "comp",'Deflate',"name","tile-deflate" );
];
for c = cfgs'
    fname = char(fullfile(tmpdir, ['tile_',c.name,'.tif']));  % ⇦ char!
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
    else,       tag.RowsPerStrip = 33; end
    t.setTag(tag); t.write(uint16(img)); close(t);

    try
        blk = load_bl_tif( cellstr(fname), 20,20,100,100,false );
        ok  = isequal(blk,uint16(img(20:119,20:119)));
        fprintf('  %-13s → %s\n', c.name, ternary(ok,'✓','✗'));
    catch ME
        fprintf('  %-13s → ✗ (%s)\n', c.name, ME.message);
    end
end

%% -----------------------------------------------------------------
% 5. Negative-path assertions ______________________________________
% ------------------------------------------------------------------
fprintf('\n[Suite 5] Expected-error checks:\n');
neg = {
    "Non-overlapping ROI",    @() load_bl_tif(filelist(1), -5000,-5000, 10,10,false);
    "File list empty str",    @() load_bl_tif({''},          1,1,10,10,false);
};
if ~isempty(fname8LE)
    neg(end+1,:) = {"Mismatched bit-depth", ...
                    @() load_bl_tif({fname8LE}, 1,1,10,10,false)};
end
for n = 1:size(neg,1)
    try, neg{n,2}(); fprintf('  ✗ %-22s did NOT error\n', neg{n,1});
    catch,         fprintf('  ✓ %-22s raised error\n',    neg{n,1}); end
end

%% -----------------------------------------------------------------
% 6. Thread-scaling benchmark (memory-safe) ________________________
% ------------------------------------------------------------------
fprintf('\n[Suite 6] Thread scaling (512×512 ROI × all slices):\n');
roiH = min(512,imageHeight);   % → never allocates > ≈1 GB
roiW = min(512,imageWidth );
for th = [1 2 4 8]
    setenv('LOAD_BL_TIF_THREADS',num2str(th));
    tic;
    load_bl_tif(filelist, 1,1, roiH, roiW, false);
    t = toc;
    fprintf('  %2d threads → %.3f s\n', th, t);
end
setenv('LOAD_BL_TIF_THREADS','');

%% -----------------------------------------------------------------
% 7. Random ROI fuzzing ____________________________________________
% ------------------------------------------------------------------
fprintf('\n[Suite 7] 500 random ROI fuzz tests (progress dots):\n');
rng(42);
for k = 1:500
    h = randi([1,imageHeight]);  w = randi([1,imageWidth ]);
    y = randi([-32,imageHeight]);x = randi([-32,imageWidth ]);
    try
        load_bl_tif(filelist,y,x,h,w, rand>0.5);
        if mod(k,20)==0, fprintf('.'); end
    catch ME
        fprintf('\n ✗ fuzz crash at k=%d (y=%d,x=%d,h=%d,w=%d): %s\n',...
                k,y,x,h,w,ME.message); break
    end
end
fprintf('\nAll suites finished.\n');
end  % main
% =================================================================
% Helper utilities
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

function ok = setByteOrderBig(tiffObj)
% Try to switch to big-endian header; returns true if it worked.
    ok = false;
    try
        tiffObj.setTag('ByteOrder','big'); ok = true;
    catch
        if ismethod(tiffObj,'rewrite')
            try, tiffObj.rewrite; fseek(tiffObj.FileID,0,'bof');
                 fwrite(tiffObj.FileID,'MM','char'); ok = true;
            catch, ok = false; end
        end
    end
end
