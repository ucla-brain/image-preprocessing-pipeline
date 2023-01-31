clc; clear all;
global cSpt RunOnWindows BlankTifFile RootPathOfData PathTarget FramesizeRow FramesizeCol XYStep ZStep;
RunOnWindows = ispc;
%RootPathOfData = 'X:\3D_stitched_Bin\20221029_SM20220717_03_40X_2000z';
%PathTarget = 'X:\3D_stitched_Bin\20221029_SM20220717_03_40X_2000z_tera\Ex_561_Em_600';
RootPathOfData = '/qnap/3D_stitched_Bin/20221029_SM20220717_03_40X_2000z';
PathTarget = '/qnap/3D_stitched_Bin/20221029_SM20220717_03_40X_2000z_tera/Ex_561_Em_600';
%RootPathOfData = 'G:\sm20220216-40X_20220426_112014_100ms-44GB';
%PathTarget = [RootPathOfData '_Target'];
BlankTifFile = 'blank.tif';
FramesizeRow = 2048;
FramesizeCol = 2048;
XYStep = 2460;
ZStep = 20;
blank_im = zeros(FramesizeRow, FramesizeCol, 'uint16');
imwrite(blank_im, BlankTifFile);

if(RunOnWindows)
    cSpt = '\';
else
    cSpt = '/';
end

ZFolders = GetSubFolders([RootPathOfData cSpt ]);
YFolders = {}; XFolders = {};
for ZFolder = ZFolders
    disp (ZFolder{1});
    Ys = GetSubFolders([RootPathOfData cSpt ZFolder{1}]);
    for YFolder = Ys
        disp (YFolder{1});
        YFolders{end+1} = YFolder{1}; YFolders = unique(YFolders);
        XFiles = GetSubFolders([RootPathOfData cSpt ZFolder{1} cSpt YFolder{1}]);
        for XFile = XFiles
            SourceFile = ([RootPathOfData cSpt ZFolder{1} cSpt YFolder{1} cSpt XFile{1}]);
            disp (SourceFile);
            XFolder = XFile{1}(17:23);
            XFolders{end+1} = XFolder; XFolders = unique(XFolders);
        end
    end
end

ZFolders
YFolders
XFolders
XFolders = AdvanceSort(XFolders)

for kY = 1:size(YFolders, 2)                    % #Cols
    YFolder = YFolders{size(YFolders, 2)+1-kY}; % Source folder name. reverse sort
    sX = num2str(kY*XYStep, '%06d');                % Tartet folder name
    for kX = 1:size(XFolders, 2)                % #Rows
        XFolder = XFolders{kX};                 % Source folder name
        sY = num2str(kX*XYStep, '%06d');            % Target folder name
        TargetFolder = [PathTarget cSpt sY cSpt sY '_' sX cSpt ];
        if not(isfolder(TargetFolder))
            disp (['mkdir ' TargetFolder]);
            dos  (['mkdir ' TargetFolder]);
        end
        for kZ = 1:size(ZFolders, 2)            % #Z
            ZFolder = ZFolders{kZ};
            sZ = num2str((kZ - 1)*ZStep, '%06d');  % Target #Z, Start from 000000
            SourceFile = [RootPathOfData cSpt ZFolder cSpt YFolder cSpt ZFolder '_' YFolder '_' XFolder '.tif'];
            disp (['initial source = ' SourceFile]);
            if not(isfile(SourceFile))
                SourceFile = BlankTifFile;
            end
            %TargetFile = [PathTarget cSpt YFolder{1} cSpt XFolder cSpt ZFolder{1} '.tif'];
            TargetFile = [TargetFolder sZ '.tif'];
            if(RunOnWindows)
                if not(isfile(TargetFile))
                    disp (['copy ' SourceFile ' ' TargetFile]);
                    dos (['copy ' SourceFile ' ' TargetFile]);
                end
            else
                disp (['cp -u ' SourceFile ' ' TargetFile]);
                dos (['cp -u ' SourceFile ' ' TargetFile]);
            end
        end
    end
end


return

overlapxl = 248; %2048-(1965-165)=2048-1800=248
FramesizeRowPure = 2048;
FramesizeColPure = 2048-overlapxl;
LeftEdgeIndex = overlapxl/2;
TopEdgeIndex = 1;

for kZ = 1:size(ZFolders, 2)            % #Z
    sZ = num2str((kZ - 1)*ZStep, '%06d');
    imgStitch = zeros(FramesizeRowPure * size(XFolders, 2), FramesizeColPure * size(YFolders, 2), 'uint16');
    for col =1: size(YFolders, 2)
        sX = num2str(col*XYStep, '%06d');
        for row = 1: size(XFolders, 2)
            sY = num2str(row*XYStep, '%06d');
            filename = [PathTarget cSpt sY cSpt sY '_' sX cSpt sZ '.tif'];
            disp(['  reading ' filename]);
            img = GetRawImage(filename);
            img = img(TopEdgeIndex:TopEdgeIndex+FramesizeRowPure-1, LeftEdgeIndex:LeftEdgeIndex+FramesizeColPure-1);
            imgStitch((row-1)*FramesizeRowPure+1:FramesizeRowPure*row, (col-1)*FramesizeColPure+1:FramesizeColPure*col) = img;
            %h = imshow(img, [1, mean(img(:))*4]);
        end
    end
    h = imshow(imgStitch, [0, mean(imgStitch(:))*4]);
    tFile = ['./' ZFolders{kZ} '_overview.png'];
    saveas(gcf, tFile);
    tFile = ['./' ZFolders{kZ} '_LZW.tif'];
    SaveTif(imgStitch, tFile, 16, 1, false, true);
    %tFile = ['./Z_' sZ '.tif'];
    %imwrite(imgStitch, tFile);
    tFile = ['./' ZFolders{kZ} '_Half1.tif'];
    imwrite(imgStitch(1:end/2, :), tFile);
    tFile = ['./' ZFolders{kZ} '_Half2.tif'];
    imwrite(imgStitch(end/2+1:end, :), tFile);
end

return

function subFolders = GetSubFolders(strPath)
%strPath
list = dir(strPath);
subFolders = {};
for k = 3:size(list, 1)
    subFolders{k-2} = list(k).name;
end
%subFolders
end

function results = AdvanceSort(XFolders)
Xs=[];
for XFolder = XFolders
    Xs(end+1) = str2num(strrep(XFolder{1}, 'X', ''));
end
Xs = sort(Xs);
results = {};
for X = Xs
    results{end+1} = ['X' num2str(X, '%06d')];
end
end

function imgRaw = GetRawImage(filename)
global FramesizeRow FramesizeCol;
%filename
fin = fopen(filename, 'r');
if(fin == -1)
    imgRaw= zeros (FramesizeRow, FramesizeCol, 'uint16');
else
    imgRawBin = fread(fin, FramesizeRow * FramesizeCol, 'uint16=>uint16');
    fclose(fin);
    if(size(imgRawBin,1) == FramesizeRow*FramesizeCol)
        img = reshape(imgRawBin, FramesizeRow, FramesizeCol);
        img = img';
        imgRaw = img;
    else
        imgRaw= zeros (FramesizeRow, FramesizeCol, 'uint16');
    end
end
end

function SaveTif(imgs, TifFileName, nBit, nColor, appendMode, bigDataEnabled)
disp(['Save .Tif File: ' TifFileName]);
if (nColor == 1)
    len = size(imgs, 3);
else
    len = size(imgs, 4);
end

for k = 1:len

    if(k == 1)
        if (bigDataEnabled)
            mode = 'w8';
        else
            mode = 'w';
        end
    else
        mode = 'a';
        if(mod(k, 1000)==0) disp(['  Save frame ' num2str(k)]); end
    end
    if(appendMode)
        mode = 'a';
    end
    tiffObj = Tiff(TifFileName, mode);
    tiffObj.setTag('ImageLength', size(imgs, 1));
    tiffObj.setTag('ImageWidth', size(imgs, 2));
    tiffObj.setTag('Photometric', Tiff.Photometric.MinIsBlack);
    tiffObj.setTag('BitsPerSample', nBit);
    tiffObj.setTag('SamplesPerPixel', nColor); %1:gray, 3:color
    %tiffObj.setTag('RowsPerStrip', 64);
    tiffObj.setTag('SampleFormat', Tiff.SampleFormat.UInt);
    %tiffObj.setTag('TileWidth', 128);
    %tiffObj.setTag('TileLength', 128);
    tiffObj.setTag('Compression', Tiff.Compression.LZW);
    tiffObj.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
    tiffObj.setTag('Software', 'MATLAB');
    if (nColor == 1)
        tiffObj.write(imgs(:,:,k));
    else
        tiffObj.write(imgs(:,:,:,k));
    end
    tiffObj.close();
end
disp('Save to .Tif finished.');
end
