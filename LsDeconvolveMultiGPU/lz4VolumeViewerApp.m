function lz4VolumeViewerApp()
% lz4VolumeViewerApp
% 3-D LZ4 volume viewer with volshow + viewer3d and modern UI components.

    %==== App State
    appState.currentFullFilePath         = "";
    appState.currentDirectoryPath        = "";
    appState.currentFilePrefix           = "";
    appState.currentFileSuffix           = ".lz4";
    appState.currentFileIndex            = NaN;
    appState.currentFileIndexWidth       = 0;
    appState.originalVolumeSingle        = [];
    appState.previewScaleFactor          = 1.0;
    appState.invertDisplay               = false;
    appState.useLowerClip                = false;
    appState.lowerClipPercentile         = 0.0;
    appState.upperClipPercentile         = 99.9;
    appState.defaultLowerPercentile      = 0.1;
    appState.defaultUpperPercentile      = 99.9;
    appState.renderingStyle              = 'VolumeRendering'; % or 'MaximumIntensityProjection'
    appState.colormapName                = 'gray';
    appState.backgroundColor             = [0 0 0];
    appState.volumePanel                 = [];
    appState.viewer3dHandle              = [];   % NEW: viewer3d parent for volshow
    appState.volumeObjectHandle          = [];   % volshow Volume object
    appState.lastDisplayVolumeNormalized = [];   % for robust camera reset
    appState.statusLabel                 = [];

    %==== Build UI
    ui = buildUserInterface();
    refreshStatusText("Ready. Load an .lz4 3-D volume to begin.");
    ui.mainFigure.WindowKeyPressFcn = @(~, evt) handleGlobalKeyPress(evt);

    %---------------------- UI Builders & Handlers -------------------------
    function ui = buildUserInterface()
        ui.mainFigure = uifigure( ...
            'Name','LZ4 3D Volume Viewer', ...
            'Position',[100 100 1250 800], ...
            'Color',[0.12 0.12 0.12], ...
            'AutoResizeChildren','on');

        ui.rootGrid = uigridlayout(ui.mainFigure,[1 2], ...
            'ColumnWidth',{'fit','1x'}, 'RowHeight',{'1x'}, ...
            'ColumnSpacing',12, 'RowSpacing',12, 'Padding',12);

        % Left controls
        ui.controlPanel = uipanel(ui.rootGrid, ...
            'Title','Controls', 'BackgroundColor',[0.16 0.16 0.16]);
        ui.controlGrid = uigridlayout(ui.controlPanel,[16 2], ...
            'ColumnWidth',{'fit','1x'}, 'RowHeight',repmat({'fit'},1,16), ...
            'Padding',10, 'RowSpacing',8, 'ColumnSpacing',8);

        % Right viewer host panel + viewer3d parent for volshow
        appState.volumePanel = uipanel(ui.rootGrid, ...
            'Title','3-D View (rotate with mouse)', ...
            'BackgroundColor',[0 0 0]);

        % Create the viewer3d inside the panel (this is the *Parent* for volshow)
        % Using default interactions; we control background via viewer3d.
        appState.viewer3dHandle = viewer3d( ...
            'Parent', appState.volumePanel, ...
            'BackgroundColor', appState.backgroundColor);

        % --- File row ---
        uilabel(ui.controlGrid,'Text','LZ4 File:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        fileRow = uigridlayout(ui.controlGrid,[1 3], ...
            'ColumnWidth',{'1x',30,30}, 'RowHeight',{'fit'}, ...
            'ColumnSpacing',6, 'BackgroundColor',[0.16 0.16 0.16]);
        ui.filePathEdit = uieditfield(fileRow,'text', ...
            'Editable','off', 'Tooltip','Loaded file path');
        ui.browseButton = uibutton(fileRow,'push', ...
            'Text','…','Tooltip','Browse for .lz4 file', ...
            'ButtonPushedFcn',@(s,~) onBrowseFile());
        ui.reloadButton = uibutton(fileRow,'push', ...
            'Text','⟲','Tooltip','Reload current file', ...
            'ButtonPushedFcn',@(s,~) onReloadFile());

        % --- Navigation ---
        uilabel(ui.controlGrid,'Text','Navigate:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        navRow = uigridlayout(ui.controlGrid,[1 3], ...
            'ColumnWidth',{80,80,'1x'}, 'RowHeight',{'fit'}, ...
            'ColumnSpacing',6, 'BackgroundColor',[0.16 0.16 0.16]);
        ui.prevButton = uibutton(navRow,'push','Text','← Prev', ...
            'Tooltip','Load previous numbered block', ...
            'ButtonPushedFcn',@(s,~) onLoadPrevious());
        ui.nextButton = uibutton(navRow,'push','Text','Next →', ...
            'Tooltip','Load next numbered block', ...
            'ButtonPushedFcn',@(s,~) onLoadNext());
        jumpRow = uigridlayout(navRow,[1 3], ...
            'ColumnWidth',{65,80,'1x'}, 'ColumnSpacing',6, ...
            'BackgroundColor',[0.16 0.16 0.16]);
        uilabel(jumpRow,'Text','Index:','HorizontalAlignment','right', ...
            'FontColor',[1 1 1]);
        ui.jumpIndexEdit = uieditfield(jumpRow,'numeric', ...
            'Value',0,'Limits',[0 inf], 'RoundFractionalValues',true, ...
            'Tooltip','Numeric suffix of the file (e.g., 7 for block_0007.lz4)');
        ui.jumpIndexEdit.ValueChangedFcn = @(~,~) onJumpToIndex();
        ui.jumpButton = uibutton(jumpRow,'push','Text','Jump', ...
            'Tooltip','Load the specified index', ...
            'ButtonPushedFcn',@(s,~) onJumpToIndex());

        % --- Rendering style ---
        uilabel(ui.controlGrid,'Text','Rendering Style:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        ui.renderStyleDropdown = uidropdown(ui.controlGrid, ...
            'Items',{'VolumeRendering','MaximumIntensityProjection'}, ...
            'Value','VolumeRendering', ...
            'ValueChangedFcn',@(s,~) onRenderingStyleChanged());

        % --- Colormap ---
        uilabel(ui.controlGrid,'Text','Colormap:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        ui.colormapDropdown = uidropdown(ui.controlGrid, ...
            'Items',{'gray','parula','turbo','hot','bone','copper'}, ...
            'Value','gray', ...
            'ValueChangedFcn',@(s,~) onColormapChanged());

        % --- Background ---
        uilabel(ui.controlGrid,'Text','Background:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        ui.backgroundDropdown = uidropdown(ui.controlGrid, ...
            'Items',{'black','white','dark gray'}, ...
            'Value','black', ...
            'ValueChangedFcn',@(s,~) onBackgroundChanged());

        % --- Invert ---
        uilabel(ui.controlGrid,'Text','Polarity:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        ui.invertCheckbox = uicheckbox(ui.controlGrid, ...
            'Text','Invert brightness','Value',false, ...
            'ValueChangedFcn',@(s,~) onInvertToggled());

        % --- Preview scale ---
        uilabel(ui.controlGrid,'Text','Preview Scale:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        ui.previewDropdown = uidropdown(ui.controlGrid, ...
            'Items',{'1.0x','0.5x','0.25x'}, 'Value','1.0x', ...
            'ValueChangedFcn',@(s,~) onPreviewScaleChanged());

        % --- Auto window ---
        uilabel(ui.controlGrid,'Text','Window Preset:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        ui.autoWindowButton = uibutton(ui.controlGrid,'push', ...
            'Text','Auto (0.1–99.9%)', ...
            'ButtonPushedFcn',@(s,~) onAutoWindow());

        % --- Lower clip ---
        uilabel(ui.controlGrid,'Text','Lower Clip (%):', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        lowerRow = uigridlayout(ui.controlGrid,[1 2], ...
            'ColumnWidth',{'1x',60}, 'BackgroundColor',[0.16 0.16 0.16], ...
            'ColumnSpacing',6);
        ui.enableLowerClipCheck = uicheckbox(lowerRow,'Text','Enable', ...
            'Value',false, 'ValueChangedFcn',@(s,~) onLowerClipEnableToggled());
        ui.lowerClipEdit = uieditfield(lowerRow,'numeric', ...
            'Limits',[0 50], 'Value',appState.defaultLowerPercentile, ...
            'RoundFractionalValues',false, ...
            'ValueChangedFcn',@(s,~) onLowerClipValueChanged());

        % --- Upper clip (main brightness) ---
        uilabel(ui.controlGrid,'Text','Upper Clip (%):', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        upperRow = uigridlayout(ui.controlGrid,[1 2], ...
            'ColumnWidth',{'1x',70}, 'BackgroundColor',[0.16 0.16 0.16], ...
            'ColumnSpacing',6);
        ui.upperClipSlider = uislider(upperRow, ...
            'Limits',[50 100],'Value',appState.upperClipPercentile, ...
            'MajorTicks',50:10:100);
        ui.upperClipSlider.ValueChangedFcn = @(~,~) onUpperClipSliderChanged();
        ui.upperClipEdit = uieditfield(upperRow,'numeric', ...
            'Limits',[50 100], 'Value',appState.upperClipPercentile, ...
            'RoundFractionalValues',false, ...
            'ValueChangedFcn',@(s,~) onUpperClipEditChanged());

        % --- Camera ---
        uilabel(ui.controlGrid,'Text','Camera:', ...
            'FontWeight','bold','FontColor',[1 1 1]);
        ui.resetCameraButton = uibutton(ui.controlGrid,'push','Text','Reset Camera', ...
            'ButtonPushedFcn',@(s,~) onResetCamera());

        % --- Status
        appState.statusLabel = uilabel(ui.controlGrid, ...
            'Text','', 'FontAngle','italic', 'FontColor',[0.9 0.9 0.9], ...
            'WordWrap','on');
        appState.statusLabel.Layout.Column = [1 2];

        % Initially disable navigation and lower clip until index/window known
        setNavigationEnabled(false);
        setLowerClipControlsEnabled(false);
    end

    %------------------------------- Actions --------------------------------
    function onBrowseFile()
        try
            [f, p] = uigetfile({'*.lz4','LZ4 Volume (*.lz4)'},'Select LZ4 3-D Volume');
            if isequal(f,0), return; end
            performLoadFromFile(string(fullfile(p,f)));
        catch e
            refreshStatusText("Error while browsing/loading: " + string(e.message));
        end
    end

    function onReloadFile()
        if strlength(appState.currentFullFilePath)==0
            refreshStatusText("No file to reload. Use browse to select a file."); return;
        end
        performLoadFromFile(appState.currentFullFilePath);
    end

    function onLoadPrevious()
        if ~isfinite(appState.currentFileIndex)
            refreshStatusText("Cannot navigate: current file name has no numeric index."); return;
        end
        loadByIndexIfExists(appState.currentFileIndex - 1, "Previous");
    end

    function onLoadNext()
        if ~isfinite(appState.currentFileIndex)
            refreshStatusText("Cannot navigate: current file name has no numeric index."); return;
        end
        loadByIndexIfExists(appState.currentFileIndex + 1, "Next");
    end

    function onJumpToIndex()
        idx = ui.jumpIndexEdit.Value;
        if ~isfinite(idx)
            refreshStatusText("Enter a valid numeric index to jump."); return;
        end
        loadByIndexIfExists(idx, "Jump");
    end

    function onRenderingStyleChanged()
        appState.renderingStyle = ui.renderStyleDropdown.Value;
        updateVolumeVisualization();
    end

    function onColormapChanged()
        appState.colormapName = ui.colormapDropdown.Value;
        applyColormap();
    end

    function onBackgroundChanged()
        switch ui.backgroundDropdown.Value
            case 'black',     appState.backgroundColor = [0 0 0];
            case 'white',     appState.backgroundColor = [1 1 1];
            case 'dark gray', appState.backgroundColor = [0.15 0.15 0.15];
        end
        applyBackground();
    end

    function onInvertToggled()
        appState.invertDisplay = ui.invertCheckbox.Value;
        updateVolumeVisualization();
    end

    function onPreviewScaleChanged()
        v = ui.previewDropdown.Value;
        if strcmp(v,'1.0x'), appState.previewScaleFactor = 1.0; end
        if strcmp(v,'0.5x'), appState.previewScaleFactor = 0.5; end
        if strcmp(v,'0.25x'), appState.previewScaleFactor = 0.25; end
        updateVolumeVisualization();
    end

    function onAutoWindow()
        if isempty(appState.originalVolumeSingle)
            refreshStatusText("Load a volume before applying auto-windowing."); return;
        end
        appState.lowerClipPercentile = appState.defaultLowerPercentile;
        appState.upperClipPercentile = appState.defaultUpperPercentile;

        ui.enableLowerClipCheck.Value  = true;
        ui.lowerClipEdit.Value         = appState.lowerClipPercentile;
        ui.upperClipEdit.Value         = appState.upperClipPercentile;
        ui.upperClipSlider.Value       = appState.upperClipPercentile;
        appState.useLowerClip          = true;
        setLowerClipControlsEnabled(true);
        updateVolumeVisualization();
    end

    function onLowerClipEnableToggled()
        appState.useLowerClip = ui.enableLowerClipCheck.Value;
        setLowerClipControlsEnabled(appState.useLowerClip);
        updateVolumeVisualization();
    end

    function onLowerClipValueChanged()
        appState.lowerClipPercentile = ui.lowerClipEdit.Value;
        updateVolumeVisualization();
    end

    function onUpperClipSliderChanged()
        appState.upperClipPercentile = ui.upperClipSlider.Value;
        ui.upperClipEdit.Value = appState.upperClipPercentile;
        updateVolumeVisualization();
    end

    function onUpperClipEditChanged()
        appState.upperClipPercentile = ui.upperClipEdit.Value;
        appState.upperClipPercentile = max(50,min(100,appState.upperClipPercentile));
        ui.upperClipSlider.Value = appState.upperClipPercentile;
        updateVolumeVisualization();
    end

    function onResetCamera()
        % Try a soft reset; if unsupported, recreate viewer & re-show.
        try
            if ~isempty(appState.viewer3dHandle) && isvalid(appState.viewer3dHandle)
                reset(appState.viewer3dHandle);
                return;
            end
        catch %#ok<CTCH>
            % Fall through to hard reset
        end
        % Hard reset: recreate viewer and redraw current normalized data
        if isgraphics(appState.viewer3dHandle)
            delete(appState.viewer3dHandle);
        end
        appState.viewer3dHandle = viewer3d( ...
            'Parent', appState.volumePanel, ...
            'BackgroundColor', appState.backgroundColor);
        if ~isempty(appState.lastDisplayVolumeNormalized)
            if isgraphics(appState.volumeObjectHandle)
                delete(appState.volumeObjectHandle);
            end
            appState.volumeObjectHandle = volshow(appState.lastDisplayVolumeNormalized, ...
                'Parent', appState.viewer3dHandle, ...
                'Colormap', getColormapByName(appState.colormapName));
            applyRenderingStyle();
        end
    end

    function handleGlobalKeyPress(evt)
        if strcmpi(evt.Key,'leftarrow')
            onLoadPrevious();
        elseif strcmpi(evt.Key,'rightarrow')
            onLoadNext();
        end
    end

    %----------------------------- Core Loading -----------------------------
    function performLoadFromFile(fullPathString)
        refreshStatusText("Loading: " + fullPathString + " ..."); drawnow;
        try
            volumeSingle = load_lz4(char(fullPathString)); % must return single 3-D
        catch mexErr
            refreshStatusText("Failed to load file: " + string(mexErr.message));
            return;
        end
        if ~isa(volumeSingle,'single')
            refreshStatusText('Loaded volume must be single precision (imsingle output).'); return;
        end
        if ndims(volumeSingle) ~= 3
            refreshStatusText('Loaded data must be a 3-D array (X×Y×Z).'); return;
        end

        appState.currentFullFilePath  = fullPathString;
        ui.filePathEdit.Value         = char(fullPathString);
        [appState.currentDirectoryPath, baseName, ext] = fileparts(char(fullPathString));
        appState.currentFileSuffix    = string(ext);

        % Parse numeric index in base name (e.g., block_0007 -> 7)
        [prefix, idx, padWidth] = parseIndexedFileName(baseName);
        appState.currentFilePrefix     = string(prefix);
        appState.currentFileIndex      = idx;
        appState.currentFileIndexWidth = padWidth;

        if isfinite(idx)
            ui.jumpIndexEdit.Value = idx;
            setNavigationEnabled(true);
        else
            ui.jumpIndexEdit.Value = 0;
            setNavigationEnabled(false);
        end

        appState.originalVolumeSingle = volumeSingle;
        if isempty(appState.upperClipPercentile) || ~isfinite(appState.upperClipPercentile)
            appState.upperClipPercentile = appState.defaultUpperPercentile;
        end

        updateVolumeVisualization();

        volSize = size(appState.originalVolumeSingle);
        refreshStatusText(sprintf('Loaded %s  |  Size: %d × %d × %d  |  Index: %s', ...
            char(baseName + ext), volSize(2), volSize(1), volSize(3), ...
            iif(isfinite(appState.currentFileIndex), num2str(appState.currentFileIndex), "N/A")));
    end

    function loadByIndexIfExists(targetIndex, actionName)
        if ~isfinite(targetIndex)
            refreshStatusText(actionName + ": invalid target index."); return;
        end
        if strlength(appState.currentDirectoryPath)==0 || strlength(appState.currentFilePrefix)==0
            refreshStatusText(actionName + ": need a loaded file to infer naming pattern."); return;
        end

        candidateFile = buildIndexedFileName( ...
            appState.currentDirectoryPath, ...
            appState.currentFilePrefix, ...
            appState.currentFileIndexWidth, ...
            targetIndex, ...
            appState.currentFileSuffix);

        if isfile(candidateFile)
            performLoadFromFile(candidateFile);
        else
            refreshStatusText(actionName + ": file not found -> " + candidateFile);
        end
    end

    %--------------------------- Visualization Core -------------------------
    function updateVolumeVisualization()
        if isempty(appState.originalVolumeSingle), return; end

        % Optional downscale for interactivity
        workingVolume = appState.originalVolumeSingle;
        if appState.previewScaleFactor ~= 1.0
            try
                workingVolume = imresize3(workingVolume, appState.previewScaleFactor, 'linear');
            catch
                workingVolume = imresize3(workingVolume, appState.previewScaleFactor, 'nearest');
            end
        end

        % Percentile-based window
        [lowerVal, upperVal] = computeWindowBoundsFromPercentiles(workingVolume, ...
            appState.useLowerClip, appState.lowerClipPercentile, appState.upperClipPercentile);

        if ~(upperVal > lowerVal)
            epsVal = max(1e-6, 0.0001 * max(1.0, abs(upperVal)));
            upperVal = lowerVal + epsVal;
        end

        % Normalize to [0,1]; optional invert
        normalizedVolume = (workingVolume - lowerVal) ./ (upperVal - lowerVal);
        normalizedVolume = max(0, min(1, normalizedVolume));
        if appState.invertDisplay
            normalizedVolume = 1.0 - normalizedVolume;
        end
        appState.lastDisplayVolumeNormalized = normalizedVolume; % for robust reset

        % Create or update volshow using viewer3d as parent
        if isempty(appState.volumeObjectHandle) || ~isvalid(appState.volumeObjectHandle)
            appState.volumeObjectHandle = volshow(normalizedVolume, ...
                'Parent', appState.viewer3dHandle, ...
                'Colormap', getColormapByName(appState.colormapName));
        else
            appState.volumeObjectHandle.Data = normalizedVolume;
        end

        applyRenderingStyle();
        applyColormap();
        applyBackground();

        refreshStatusText(sprintf( ...
            'Showing %s  |  Window [%.4g, %.4g] (percentiles %s, %s)  |  Mode: %s  |  Scale: %.2fx', ...
            fileNameOr("unnamed"), lowerVal, upperVal, ...
            iif(appState.useLowerClip, sprintf('%.3f%%', appState.lowerClipPercentile), 'disabled'), ...
            sprintf('%.3f%%', appState.upperClipPercentile), ...
            appState.renderingStyle, appState.previewScaleFactor));
    end

    function applyRenderingStyle()
        if isempty(appState.volumeObjectHandle) || ~isvalid(appState.volumeObjectHandle)
            return;
        end
        % Support different releases (RenderingStyle vs Rendering)
        try
            appState.volumeObjectHandle.RenderingStyle = appState.renderingStyle;
        catch
            appState.volumeObjectHandle.Rendering = appState.renderingStyle;
        end
    end

    function [lowerVal, upperVal] = computeWindowBoundsFromPercentiles(vol, useLower, pLow, pHigh)
        numVox = numel(vol);
        if numVox > 64e6
            sampleCount = min(4e6, numVox);
            rng(1,'twister');
            idx = randi([1 numVox], sampleCount, 1, 'uint32');
            sample = vol(idx);
        else
            sample = vol(:);
        end
        if useLower
            lowerVal = prctile(sample, pLow);
        else
            lowerVal = min(sample);
        end
        upperVal = prctile(sample, pHigh);
    end

    %----------------------------- UI Helpers -------------------------------
    function applyColormap()
        if ~isempty(appState.volumeObjectHandle) && isvalid(appState.volumeObjectHandle)
            appState.volumeObjectHandle.Colormap = getColormapByName(appState.colormapName);
        end
    end

    function applyBackground()
        % Background is a property of the viewer3d (not the Volume object)
        if ~isempty(appState.viewer3dHandle) && isvalid(appState.viewer3dHandle)
            appState.viewer3dHandle.BackgroundColor = appState.backgroundColor;
        end
    end

    function setLowerClipControlsEnabled(isEnabled)
        ui.enableLowerClipCheck.Value = isEnabled;
        ui.lowerClipEdit.Enable       = iif(isEnabled,'on','off');
        appState.useLowerClip         = isEnabled;
    end

    function setNavigationEnabled(isEnabled)
        ui.prevButton.Enable = iif(isEnabled,'on','off');
        ui.nextButton.Enable = iif(isEnabled,'on','off');
        ui.jumpButton.Enable = iif(isEnabled,'on','off');
    end

    function refreshStatusText(msg)
        if isgraphics(appState.statusLabel)
            appState.statusLabel.Text = string(msg);
        else
            disp(char(msg));
        end
    end

    function out = fileNameOr(defaultName)
        if strlength(appState.currentFullFilePath) > 0
            [~,b,e] = fileparts(appState.currentFullFilePath);
            out = string(b) + string(e);
        else
            out = string(defaultName);
        end
    end

    %------------------------------ Utilities -------------------------------
    function [prefix, idx, padWidth] = parseIndexedFileName(baseName)
        tokens = regexp(baseName,'^(.*?)(\d+)$','tokens','once');
        if isempty(tokens)
            prefix   = baseName;
            idx      = NaN;
            padWidth = 0;
        else
            prefix   = tokens{1};
            digits   = tokens{2};
            idx      = str2double(digits);
            padWidth = numel(digits);
        end
    end

    function fullPath = buildIndexedFileName(dirPath, prefix, padWidth, idx, ext)
        if padWidth > 0
            numStr = sprintf(['%0',num2str(padWidth),'d'], idx);
        else
            numStr = num2str(idx);
        end
        fullPath = string(fullfile(char(dirPath), char(prefix + numStr + ext)));
    end

    function C = getColormapByName(name)
        switch lower(string(name))
            case "gray",   C = gray(256);
            case "parula", C = parula(256);
            case "turbo",  C = turbo(256);
            case "hot",    C = hot(256);
            case "bone",   C = bone(256);
            case "copper", C = copper(256);
            otherwise,     C = gray(256);
        end
    end

    function out = iif(cond, a, b)
        if cond, out = a; else, out = b; end
    end
end
