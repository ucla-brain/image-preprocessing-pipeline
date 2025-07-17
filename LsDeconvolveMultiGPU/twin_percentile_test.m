function twin_percentile_test(varargin)
%TWIN_PERCENTILE_TEST   Unit tests & in-test benchmark for twin_percentile_mex
%
%   twin_percentile_test                % default tol=1e-5 percent
%   twin_percentile_test('quick')       % fast, default tol=1e-5 percent
%   twin_percentile_test(tol)           % tol is relative percent difference
%   twin_percentile_test('quick', tol)  % both

tol = 1;
isQuick = false;
for i = 1:numel(varargin)
    if ischar(varargin{i}) && strcmpi(varargin{i}, 'quick')
        isQuick = true;
    elseif isnumeric(varargin{i}) && isscalar(varargin{i})
        tol = varargin{i};
    end
end

rng(42);
fprintf('\n== twin_percentile_mex test & benchmark ==\n');
fprintf('Tolerance for pass: %.2e%% (relative percent diff)\n', tol);

clipvals = [0.5 1 2 5];

if isQuick
    nDouble = 20;
    nSingle = 20;
    nUint   = 4;
else
    nDouble = 100;
    nSingle = 100;
    nUint   = 10;
end

sizesDouble = [5e4, 1e5, 2e5, 5e5, 1e6, 5e6];
sizesSingle = [5e4, 1e5, 2e5, 5e5, 1e6, 5e6];
sizesUint   = [1e4, 5e4, 1e5, 2e5, 5e5];

benchRows = {};
failCountDouble = 0;
failCountSingle = 0;
failCountUint = 0;

%% --- DOUBLE TESTS ---
fprintf('\nTesting double (%d cases):\n', nDouble);
for k = 1:nDouble
    sz  = sizesDouble(randi(numel(sizesDouble)));
    p   = clipvals(randi(numel(clipvals)));
    A   = randn(sz,1,'double')*50 + 100;

    t0 = tic; ref = prctile(A(:), [p 100-p], 'Method','exact'); tMAT = toc(t0);
    t0 = tic; dut = twin_percentile_mex(A, p); tMEX = toc(t0);

    percentDiff = (ref - dut) ./ (ref + dut) * 200;

    pass = all(abs(percentDiff) <= tol);
    if ~pass
        failCountDouble = failCountDouble + 1;
        if failCountDouble <= 10
            fprintf('[double FAIL #%d]  N=%g  p=%g\n', failCountDouble, sz, p);
            fprintf('    ref = [%.15g %.15g]\n', ref);
            fprintf('    dut = [%.15g %.15g]\n', dut);
            fprintf('    Δ    = [%.3e %.3e]\n', ref-dut);
            fprintf('    %%diff= [%.3e %.3e]\n', percentDiff);
        end
    end

    benchRows(end+1,:) = {'double', sz, p, tMEX, tMAT, tMAT/tMEX, pass};
end

%% --- SINGLE TESTS ---
fprintf('\nTesting single (%d cases):\n', nSingle);
for k = 1:nSingle
    sz  = sizesSingle(randi(numel(sizesSingle)));
    p   = clipvals(randi(numel(clipvals)));
    A   = single(randn(sz,1)*50 + 100);

    t0 = tic; ref = prctile(A(:), [p 100-p], 'Method','exact'); tMAT = toc(t0);
    t0 = tic; dut = twin_percentile_mex(A, p); tMEX = toc(t0);

    percentDiff = (ref - dut) ./ (ref + dut) * 200;

    pass = all(abs(percentDiff) <= tol);
    if ~pass
        failCountSingle = failCountSingle + 1;
        if failCountSingle <= 10
            fprintf('[single FAIL #%d]  N=%g  p=%g\n', failCountSingle, sz, p);
            fprintf('    ref = [%.15g %.15g]\n', ref);
            fprintf('    dut = [%.15g %.15g]\n', dut);
            fprintf('    Δ    = [%.3e %.3e]\n', ref-dut);
            fprintf('    %%diff= [%.3e %.3e]\n', percentDiff);
        end
    end

    benchRows(end+1,:) = {'single', sz, p, tMEX, tMAT, tMAT/tMEX, pass};
end

%% --- UINT16 TESTS (quick) ---
fprintf('\nTesting uint16 (%d cases):\n', nUint);
for k = 1:nUint
    sz  = sizesUint(randi(numel(sizesUint)));
    p   = clipvals(randi(numel(clipvals)));
    A   = uint16(1000*rand(sz,1));

    t0 = tic; ref = prctile(double(A(:)), [p 100-p], 'Method','exact'); tMAT = toc(t0);
    t0 = tic; dut = twin_percentile_mex(A, p); tMEX = toc(t0);

    percentDiff = (ref - dut) ./ (ref + dut) * 200;

    pass = all(abs(percentDiff) <= tol);
    if ~pass
        failCountUint = failCountUint + 1;
        if failCountUint <= 5
            fprintf('[uint16 FAIL #%d]  N=%g  p=%g\n', failCountUint, sz, p);
            fprintf('    ref = [%.15g %.15g]\n', ref);
            fprintf('    dut = [%.15g %.15g]\n', dut);
            fprintf('    Δ    = [%.3e %.3e]\n', ref-dut);
            fprintf('    %%diff= [%.3e %.3e]\n', percentDiff);
        end
    end

    benchRows(end+1,:) = {'uint16', sz, p, tMEX, tMAT, tMAT/tMEX, pass};
end

%% --- SUMMARY ---
benchTbl = cell2table(benchRows, ...
    'VariableNames', {'Type','N','p','TimeMEX','TimeMAT','Speedup','Pass'});

fprintf('\n=== BENCHMARK SUMMARY (unit test timing) ===\n');
disp(benchTbl)

fprintf('Double: %d / %d passed, Single: %d / %d passed, Uint16: %d / %d passed.\n', ...
    nDouble-failCountDouble, nDouble, nSingle-failCountSingle, nSingle, nUint-failCountUint, nUint);

if failCountDouble + failCountSingle + failCountUint == 0
    fprintf('✅  All unit tests passed within tolerance %.2e%%!\n', tol);
else
    error('Unit tests failed: %d total mismatches.', failCountDouble+failCountSingle+failCountUint);
end
end
