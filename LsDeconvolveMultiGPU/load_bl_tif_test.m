function load_bl_tif_test()
% ==============================================================
% load_bl_tif_test.m  (2025-06-07  •  patch-6)
%
% Comprehensive reliability & performance test-suite for the
% load_bl_tif MEX.  Works on MATLAB R2018b + with stock libtiff.
%
% Updated: Handles automatic bit-depth detection (uint8/uint16),
% adds temp cleanup, clarifies error reporting, and minor polish.
% Suite 3: Loop fixed for struct array ('ternary' error resolved).
% ==============================================================

clearvars; clc;

%% ----------------------------------------------------------------
% 0. Dataset location  --------------------------------------------
% -----------------------------------------------------------------
folder_path = '/data/tif/B11_ds_4.5x_ABeta_z1200';         % ← Linux
if ispc, folder_path = 'V:/tif/B11_ds_4.5x_ABeta_z1200'; end
assert(isfolder(folder_path), 'Edit folder_path inside load_bl_tif_test.m');

files = dir(fullfile(folder_path,'*.tif'));
assert(~isempty(files),'No TIFF files in folder_path.');
filelist  = fullfile({files.folder},{files.name});
numSlices = numel(filelist);

info          = imfinfo(filelist{1});
imageHeight   = info.Height;
imageWidth    = info.Width;
bitDepth      = info.BitDepth;
if isfield(info,'BitDepth') && info.BitDepth == 8
    dtype = 'uint8';
else
    dtype = 'uint16';
end
fprintf('--- Dataset: %d×%d ‖ %d slices ‖ %d-bit (%s) ---\n', ...
        imageHeight,imageWidth,numSlices,bitDepth,dtype);

%% ----------------------------------------------------------------
% 1. Baseline reference vs MEX  -----------------------------------
% -----------------------------------------------------------------
blockSizes = [32,12; 12,23; 23,12; 512,1024];
testZ      = [round(numSlices/2), max(1,numSlices-3)];

fprintf('\n[Suite 1] Reference vs MEX baseline:\n');
fprintf('%-4s | %-5s | %-9s | %-13s | %-11s | %-11s | %s\n', ...
        '✓/✗','Z','Block','(X,Y)','MaxErr','Speed-up','Mode');
fprintf(repmat('-',1,76)); fprintf('\n');

for b = 1:size(blockSizes,1)
    blkH = blockSizes(b,1); blkW = blockSizes(b,2);
    for zidx = testZ
        for tr = [false true]
            maxY = max(1, imageHeight - blkH + 1);
            maxX = max(1, imageWidth  - blkW + 1);
            y = randi([1,maxY]);  x = randi([1,maxX]);

            y_idx = y : min(imageHeight, y+blkH-1);
            x_idx = x : min(imageWidth , x+blkW-1);
            z_idx = zidx : min(numSlices, zidx+2);

            % MATLAB reference (handle transpose)
            sz = tr* [numel(x_idx) numel(y_idx)] + ~tr*[numel(y_idx) numel(x_idx)];
            ref = zeros([sz, numel(z_idx)], dtype); % Correct dtype
            tref = tic;
            for k = 1:numel(z_idx)
                slice = imread(filelist{z_idx(k)}, ...
                    'PixelRegion',{[y_idx(1),y_idx(end)], [x_idx(1),x_idx(end)]});
                % If tr==true, transpose XY block (to match MEX's tr flag)
                ref(:,:,k) = ternary(tr,slice',slice);
            end
            tref = toc(tref);

            % MEX
            tmex = tic;
            mexO = load_bl_tif(filelist(z_idx), y,x,blkH,blkW, tr);
            tmex = toc(tmex);

            pass   = isequaln(ref,mexO);
            if pass
                maxerr = 0;
            else
                try
                    maxerr = max(abs(double(ref(:))-double(mexO(:))));
                catch
                    maxerr = NaN;
                end
            end
            fprintf('  %c  | %-5d | [%3d,%3d] | (%5d,%5d) | %1.4e | %9.2fx | %s\n', ...
                char(pass*10003+~pass*10007), zidx, blkH,blkW, x,y, maxerr, tref/tmex, ...
                ternary(tr,'T','N'));
        end
    end
end

%% ----------------------------------------------------------------
% 2. Spatial boundary checks  -------------------------------------
% -----------------------------------------------------------------
fprintf('\n[Suite 2] Spatial boundary checks:\n');
edge = {...
   1,1,1,1,                         "top-left 1×1",        "ok";
   1,1,1,512,                       "top row stripe",      "ok";
   1,1,512,1,                       "left col stripe",     "ok";
   imageHeight,1,1,512,             "bottom row stripe",   "ok";
   1,imageWidth,512,1,              "right col stripe",    "ok";
   1,1,imageHeight,imageWidth,      "full-frame (1 Z)",    "ok_singleZ";
  -20,-20,128,128,                  "upper-left overflow", "expect_error";
   imageHeight-50,imageWidth-50,100,100, "bottom-right overflow","ok";
};
for k = 1:size(edge,1)
    [y,x,h,w,label,kind] = edge{k,:};
    try
        switch kind
            case "ok", mexP = load_bl_tif(filelist, y,x,h,w,false); %#ok<NASGU>
            case "ok_singleZ", mexP = load_bl_tif(filelist(1), y,x,h,w,false); %#ok<NASGU>
            otherwise, load_bl_tif(filelist, y,x,h,w,false); % should error
        end
        if kind=="expect_error"
            fprintf('  ✗ %-25s did NOT error\n', label);
        else
            fprintf('  ✓ %-25s (size %s)\n', label, mat2str(size(mexP)));
        end
    catch ME
        if kind=="expect_error"
            fprintf('  ✓ %-25s raised (%s)\n', label, ME.identifier);
        else
            fprintf('  ✗ %-25s ERROR: %s [%s]\n', label, ME.message, ME.identifier);
        end
    end
end

%% ----------------------------------------------------------------
% 3. Little- vs big-endian, 8-/16-bit  ----------------------------
% -----------------------------------------------------------------
fprintf('\n[Suite 3] 8/16-bit little- vs big-endian:\n');
tmpdir = tempname; mkdir(tmpdir);
cleanupObj = onCleanup(@() cleanupTempDir(tmpdir));
specs  = [ ...
   struct("bits",8 ,"big",false)
   struct("bits",8 ,"big",true )
   struct("bits",16,"big",false)
   struct("bits",16,"big",true )];
fname8LE = '';
for idx = 1:numel(specs)
    s = specs(idx);
    fname = fullfile(tmpdir, sprintf('end_%db_%s.tif',s.bits,ternary(s.big,'BE','LE')));
    img   = randi(intmax(sprintf('uint%d',s.bits)), 512,512, sprintf('uint%d',s.bits));
    t     = Tiff(fname,'w');
    tag.ImageWidth        = size(img,2);
    tag.ImageLength       = size(img,1);
    tag.BitsPerSample     = s.bits;
    tag.SamplesPerPixel   = 1;
    tag.Photometric       = tryEnum('Tiff.Photometric.MinIsBlack',1);
    tag.PlanarConfiguration=tryEnum('Tiff.PlanarConfiguration.Contig',1);
    tag.Compression       = tryEnum('Tiff.Compression.None',1);
    t.setTag(tag);
    if s.big && ~forceBigEndianHeader(t), close(t), delete(fname);
        fprintf('  - skipped %2d-bit BE (unsupported)\n',s.bits); continue
    end
    t.write(img); close(t);
    if s.bits==8 && ~s.big, fname8LE = fname; end
    try
        load_bl_tif({fname},1,1,32,32,false);
        fprintf('  ✓ %2d-bit %s-endian\n',s.bits,ternary(s.big,'big','little'));
    catch ME
        fprintf('  ✗ %2d-bit %s-endian (%s) [%s]\n',s.bits, ...
                ternary(s.big,'big','little'), ME.message, ME.identifier);
    end
end

%% ----------------------------------------------------------------
% 4. Tile/strip + compression  ------------------------------------
% -----------------------------------------------------------------
fprintf('\n[Suite 4] Tile/strip + compression:\n');
cfgs = [ ...
  struct("tiled",false,"comp",'None'   ,"name","strip-none"   )
  struct("tiled",false,"comp",'LZW'    ,"name","strip-lzw"    )
  struct("tiled",false,"comp",'Deflate',"name","strip-deflate")
  struct("tiled",true ,"comp",'None'   ,"name","tile-none"    )
  struct("tiled",true ,"comp",'LZW'    ,"name","tile-lzw"     )
  struct("tiled",true ,"comp",'Deflate',"name","tile-deflate")];
for c = cfgs
    fname = fullfile(tmpdir, ['tile_' c.name '.tif']);  % <-- safe path
    img   = cast(magic(257), dtype); % Use detected dtype
    try
        t = Tiff(fname,'w');
    catch ME
        fprintf('  %-13s → skipped (cannot create file: %s) [%s]\n', c.name, ME.message, ME.identifier);
        continue
    end
    tag.ImageWidth        = size(img,2);
    tag.ImageLength       = size(img,1);
    tag.BitsPerSample     = bitDepth;
    tag.SamplesPerPixel   = 1;
    tag.Photometric       = tryEnum('Tiff.Photometric.MinIsBlack',1);
    tag.PlanarConfiguration=tryEnum('Tiff.PlanarConfiguration.Contig',1);
    [tag.Compression,supported] = compressionTag(c.comp);
    if ~supported
        fprintf('  %-13s → skipped (compression unsupported)\n', c.name);
        close(t); delete(fname); continue
    end
    if c.tiled, tag.TileWidth = 64; tag.TileLength = 64;
    else,       tag.RowsPerStrip = 33;       end
    t.setTag(tag); t.write(img); close(t);

    try
        blk = load_bl_tif(cellstr(fname), 20,20,100,100,false);
        ok  = isequal(blk,img(20:119,20:119));
        fprintf('  %-13s → %s\n', c.name, ternary(ok,'✓','✗'));
    catch ME
        fprintf('  %-13s → ✗ (%s) [%s]\n', c.name, ME.message, ME.identifier);
    end
end

%% ----------------------------------------------------------------
% 5. Expected-error paths  ----------------------------------------
% -----------------------------------------------------------------
fprintf('\n[Suite 5] Expected-error checks:\n');
neg = {
  "Non-overlap ROI", @() load_bl_tif(filelist(1),-5000,-5000,10,10,false);
  "Empty file cell", @() load_bl_tif({''},1,1,10,10,false);
};
if ~isempty(fname8LE)
    neg(end+1,:) = {"Mismatched bit-depth",@() load_bl_tif({fname8LE},1,1,10,10,false)};
end
for n = 1:size(neg,1)
    try
        neg{n,2}();
        fprintf('  ❌ %-22s did NOT error\n',neg{n,1});
    catch ME
        fprintf('  ✔️ %-22s raised error [%s]\n',neg{n,1}, ME.identifier);
    end
end

%% ----------------------------------------------------------------
% 6. Thread-scaling micro-benchmark  ------------------------------
% -----------------------------------------------------------------
fprintf('\n[Suite 6] Thread scaling (255×255 ROI × all Z):\n');
roiH = min(256,imageHeight); roiW = min(256,imageWidth);
for th = [1 2 4 8]
    setenv('LOAD_BL_TIF_THREADS',num2str(th));
    tic; load_bl_tif(filelist,1,1,roiH,roiW,false); t=toc;
    fprintf('  %2d threads → %.3f s\n', th, t);
end
setenv('LOAD_BL_TIF_THREADS','');

%% ----------------------------------------------------------------
% 7. 500-iteration random fuzz  -----------------------------------
% -----------------------------------------------------------------
fprintf('\n[Suite 7] 500 random ROI fuzz tests (progress dots):\n');
rng(42);
numFuzz = 500;
for k = 1:numFuzz
    h = randi([1,imageHeight]);
    w = randi([1,imageWidth ]);
    y = randi([-32,imageHeight]);
    x = randi([-32,imageWidth ]);
    try
        load_bl_tif(filelist,y,x,h,w,rand>0.5);
        if mod(k,10)==0, fprintf('.'); end
    catch ME
        fprintf('\n ✗ fuzz crash (k=%d): %s [%s]\n',k,ME.message, ME.identifier); break
    end
end
fprintf('\nAll suites finished.\n');
end  % main
% ==============================================================
% Helper utilities
% ==============================================================

function o = ternary(c,a,b), if c, o = a; else, o = b; end, end

function val = tryEnum(e,fallback)
    try, val = eval(e); catch, val = fallback; end
end

function [tagVal,supported] = compressionTag(name)
    switch upper(name)
        case 'NONE',    tagVal = tryEnum('Tiff.Compression.None',1);    supported=true;
        case 'LZW',     tagVal = tryEnum('Tiff.Compression.LZW',5);     supported=hasCodec(5);
        case 'DEFLATE', tagVal = tryEnum('Tiff.Compression.Deflate',32946); supported=hasCodec(32946);
        otherwise,      tagVal = 1; supported=false;
    end
end
function tf = hasCodec(id)
    try, matlab.internal.imagesci.tifflib('getCodecName',id); tf=true;
    catch, tf=false; end
end

function ok = forceBigEndianHeader(t)
% Try to switch TIFF header to big-endian.  Returns true if it worked.
    ok = false;
    try
        t.setTag('ByteOrder','big'); ok=true; return; end %#ok<TRYNC>
    if ismethod(t,'rewrite')
        try
            t.rewrite; fseek(t.FileID,0,'bof'); fwrite(t.FileID,'MM','char'); ok=true;
        catch
            ok=false;
        end
    end
end

function cleanupTempDir(tmpdir)
    if isfolder(tmpdir)
        try
            delete(fullfile(tmpdir, '*'));
            rmdir(tmpdir);
        catch
            % If any files still locked, ignore cleanup error
        end
    end
end
