function load_bl_tif_test()
% ==============================================================
% load_bl_tif_test.m  • Complete reliability + performance suite
%
% Run from a system shell:
%     matlab -batch load_bl_tif_test
%
% (2025-06-07)  ⟶  Patched to avoid randi() domain errors when the
% requested block size is larger than the image dimension.
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
% 1. Original reliability + performance regression _____________
% --------------------------------------------------------------
blockSizes = [32, 12;
              12, 23;
              23, 12;
             512,1024];
testZ  = [round(numSlices/2), max(1,numSlices-3)];
totalTests = size(blockSizes,1) * numel(testZ) * 2;
results = zeros(totalTests,9);           % [pass, z, h, w, x, y, maxerr, speedup, transpose]
testIdx = 1;

fprintf('\n[Suite 1] Reference vs MEX baseline:\n');
fprintf('%-4s | %-6s | %-9s | %-13s | %-11s | %-12s | %s\n', ...
    'pass','Z','BlockSize','(X,Y)','Max Error','Speedup','Mode');
fprintf(repmat('-',1,74)); fprintf('\n');

for b = 1:size(blockSizes,1)
    blkH = blockSizes(b,1);
    blkW = blockSizes(b,2);

    for zidx = testZ
        for transposeFlag = [false true]
            % --- random top-left corner (guaranteed legal) ---------
            maxYStart = max(1, imageHeight - blkH + 1);
            maxXStart = max(1, imageWidth  - blkW + 1);
            y = randi([1, maxYStart]);
            x = randi([1, maxXStart]);

            y_indices = y : min(imageHeight, y+blkH-1);
            x_indices = x : min(imageWidth,  x+blkW-1);
            z_indices = zidx : min(numSlices, zidx+2);

            % MATLAB reference
            if transposeFlag
                bl_gt = zeros(numel(x_indices),numel(y_indices),numel(z_indices),'uint16');
            else
                bl_gt = zeros(numel(y_indices),numel(x_indices),numel(z_indices),'uint16');
            end
            t1 = tic;
            for k = 1:numel(z_indices)
                slice = imread(filelist{z_indices(k)}, ...
                    'PixelRegion',{[y_indices(1), y_indices(end)],...
                                   [x_indices(1), x_indices(end)]});
                bl_gt(:,:,k) = ternary(transposeFlag, slice', slice);
            end
            tref = toc(t1);

            % MEX call
            t2 = tic;
            bl_mex = load_bl_tif(filelist(z_indices),y,x,blkH,blkW,transposeFlag);
            tmex = toc(t2);

            % Compare
            minH = min(size(bl_gt,1),size(bl_mex,1));
            minW = min(size(bl_gt,2),size(bl_mex,2));
            minZ = min(size(bl_gt,3),size(bl_mex,3));
            bl_gt_c  = bl_gt (1:minH,1:minW,1:minZ);
            bl_mex_c = bl_mex(1:minH,1:minW,1:minZ);
            pass = isequal(bl_gt_c,bl_mex_c);
            if pass
                maxerr = 0;
            else
                diff   = abs(double(bl_mex_c)-double(bl_gt_c));
                maxerr = max(diff(:));
            end
            symbol = char(pass*10003 + ~pass*10007);  % ✓ / ✗
            modeStr= ternary(transposeFlag,'T','N');
            fprintf('  %s  | %-6d | [%3d,%3d]  | (%5d,%5d) | %1.4e | %8.2fx |   %s\n', ...
                    symbol,zidx,blkH,blkW,x,y,maxerr,tref/tmex,modeStr);

            results(testIdx,:) = [pass,zidx,blkH,blkW,x,y,maxerr,tref/tmex,transposeFlag];
            testIdx = testIdx+1;
        end
    end
end
if all(results(:,1)),  fprintf('\n🎉 Baseline tests passed.\n');
else,                  fprintf('\n❗ Baseline failures detected.\n'); end

%% -------------------------------------------------------------
% 2. Boundary & out-of-bounds ROIs  (unchanged) ________________
% --------------------------------------------------------------
fprintf('\n[Suite 2] Spatial boundary checks:\n');
edgeROIs = {
   1,               1,                1,                1,   "top-left 1×1";
   1,               1,                1,             512,   "top row stripe";
   1,               1,              512,                1,   "left col stripe";
   imageHeight-0,   1,                1,             512,   "bottom row stripe";
   1,          imageWidth-0,         512,                1,   "right col stripe";
   1,               1,       imageHeight,      imageWidth,   "full-frame";
  -20,             -20,             128,              128,   "upper-left overflow";
   imageHeight-50,  imageWidth-50,  100,              100,   "bottom-right overflow"
};
for k = 1:size(edgeROIs,1)
    [y,x,h,w,label] = edgeROIs{k,:};
    try
        blk = load_bl_tif(filelist,y,x,h,w,false); %#ok<NASGU>
        fprintf('  ✓ %-25s  (size %s)\n',label,mat2str(size(blk)));
    catch ME
        fprintf('  ✗ %-25s  ERROR: %s\n',label,ME.message);
    end
end

%% -------------------------------------------------------------
% 3. Byte-order & bit-depth sanity  (unchanged) ________________
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
    if s.bits==8 && ~s.big, fname8LE = fname; end   % for negative test

    % --- write synthetic TIFF slice ----------------------------------
    t = Tiff(fname,'w');
    tag.ImageLength       = size(img,1);
    tag.ImageWidth        = size(img,2);
    tag.Photometric       = Tiff.Photometric.MinIsBlack;
    tag.BitsPerSample     = s.bits;
    tag.SamplesPerPixel   = 1;
    tag.PlanarConfiguration = Tiff.PlanarConfiguration.Contig;
    tag.Compression       = Tiff.Compression.None;
    tag.Software          = 'edge-case gen';
    if s.big, setByteOrderBig(t); end
    t.setTag(tag); t.write(img); t.close;

    % --- loader check ------------------------------------------------
    try
        blk = load_bl_tif({fname},1,1,32,32,false);
        assert(isequal(blk, img(1:32,1:32)));
        fprintf('  ✓ %2d-bit %s-endian\n',s.bits, ternary(s.big,'big','little'));
    catch ME
        fprintf('  ✗ %2d-bit %s-endian : %s\n',s.bits, ternary(s.big,'big','little'), ME.message);
    end
end

%% -------------------------------------------------------------
% 4. Tile / strip + compression matrix  (unchanged) ____________
% --------------------------------------------------------------
fprintf('\n[Suite 4] Tile/strip + compression:\n');
cfgs = [
   struct("tiled",false,"comp",Tiff.Compression.None,    "name","strip-none");
   struct("tiled",false,"comp",Tiff.Compression.LZW,     "name","strip-lzw" );
   struct("tiled",false,"comp",Tiff.Compression.Deflate, "name","strip-deflate");
   struct("tiled",true, "comp",Tiff.Compression.None,    "name","tile-none");
   struct("tiled",true, "comp",Tiff.Compression.LZW,     "name","tile-lzw" );
   struct("tiled",true, "comp",Tiff.Compression.Deflate, "name","tile-deflate");
];
for c = cfgs'
    fname = fullfile(tmpdir,"tile_"+c.name+".tif");
    img   = magic(257);                   % deliberately non-power-of-two size
    t = Tiff(fname,'w');
    tag.ImageLength     = size(img,1);
    tag.ImageWidth      = size(img,2);
    tag.BitsPerSample   = 16;
    tag.SamplesPerPixel = 1;
    tag.Photometric     = Tiff.Photometric.MinIsBlack;
    tag.PlanarConfiguration = Tiff.PlanarConfiguration.Contig;
    tag.Compression     = c.comp;
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
% 5. Negative-path assertions  (unchanged) _____________________
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
        fn();
        fprintf('  ✗  %-30s did NOT error\n',desc);
    catch
        fprintf('  ✓  %-30s raised error\n',desc);
    end
end

%% -------------------------------------------------------------
% 6. Thread-scaling benchmark  (unchanged) _____________________
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
% 7. 500-iteration random ROI fuzzing  (unchanged) _____________
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

function setByteOrderBig(tiffObj)
% Force big-endian header (MATLAB default is host order).  Works down to
% R2018b using a low-level header patch if ByteOrder tag unsupported.
    try
        tiffObj.setTag('ByteOrder','big');   % R2021a+
    catch
        % Fallback: rewrite, then patch first two bytes ("II"/"MM").
        warning('off','all');
        tiffObj.rewrite;
        fseek(tiffObj.FileID,0,'bof');
        fwrite(tiffObj.FileID,'MM','char');
        warning('on','all');
    end
end
