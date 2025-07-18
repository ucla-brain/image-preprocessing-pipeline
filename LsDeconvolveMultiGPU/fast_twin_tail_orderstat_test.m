function fast_twin_tail_orderstat_test(varargin)
% FAST_TWIN_TAIL_ORDERSTAT_TEST   Unit tests & benchmark for fast_twin_tail_orderstat MEX
%
%   fast_twin_tail_orderstat_test                % default tol=5 percent
%   fast_twin_tail_orderstat_test('quick')       % fast, default tol=5 percent
%   fast_twin_tail_orderstat_test(tol)           % set tolerance
%   fast_twin_tail_orderstat_test('quick', tol)  % both

tol = 1; % percent
isQuick = false;
for i = 1:numel(varargin)
    if ischar(varargin{i}) && strcmpi(varargin{i}, 'quick')
        isQuick = true;
    elseif isnumeric(varargin{i}) && isscalar(varargin{i})
        tol = varargin{i};
    end
end

rng(42);
fprintf('\n== fast_twin_tail_orderstat test & benchmark ==\n');
fprintf('Tolerance for pass: %.1f%% (relative percent diff)\n', tol);

clipvals = [0.01 0];

if isQuick
    nDouble = 1;
    nSingle = 1;
    nUint   = 1;
else
    nDouble = 2;
    nSingle = 2;
    nUint   = 2;
end

sizesDouble = [5e4, 2^31 - 1];
sizesSingle = [5e4, 2^31 - 1];
sizesUint   = [5e4, 2^31 - 1];

benchRows = {};
failCountDouble = 0;
failCountSingle = 0;
failCountUint = 0;

cpuDevice = 'CPU';

    function [ref, dut, tMAT, tMEX] = ref_and_dut(A, p, typeStr)
        % Helper to compute reference and DUT outputs (shape normalized)
        if strcmp(typeStr,'uint16')
            t0 = tic; ref = prctile(double(A), [p 100-p], 'all', Method='midpoint'); tMAT = toc(t0);
            t0 = tic; dut = fast_twin_tail_orderstat(A, [p 100-p]); tMEX = toc(t0);
            ref = ref(:).';  % always row
            dut = double(dut(:).');
        else
            t0 = tic; ref = prctile(A, [p 100-p], 'all', Method='midpoint'); tMAT = toc(t0);
            t0 = tic; dut = fast_twin_tail_orderstat(A, [p 100-p]); tMEX = toc(t0);
            ref = ref(:).';  % always row
            dut = dut(:).';
        end
    end

    function [failCount, benchRows] = bench_and_report(typeStr, nCases, sizes, benchRows, failCount)
        fprintf('\nTesting %s (%d cases, CPU only):\n', typeStr, nCases);
        for k = 1:nCases
            sz  = sizes(randi(numel(sizes)));
            p   = clipvals(randi(numel(clipvals)));
            switch typeStr
                case 'double', A = randn(sz,1,'double')*50 + 100;
                case 'single', A = single(randn(sz,1)*50 + 100);
                case 'uint16', A = uint16(1000*rand(sz,1));
                otherwise, error('Unsupported type');
            end
            [ref, dut, tMAT, tMEX] = ref_and_dut(A, p, typeStr);
            percentDiff = abs(ref - dut) ./ (abs(ref) + abs(dut) + eps('double')) * 200;
            pass = all(percentDiff <= tol);

            if ~pass
                failCount = failCount + 1;
                if failCount <= 10
                    fprintf('[%s-CPU FAIL #%d]  N=%g  p=%g\n', typeStr, failCount, sz, p);
                    fprintf('    ref = [%.15g %.15g]\n', ref);
                    fprintf('    dut = [%.15g %.15g]\n', dut);
                    fprintf('    Δ    = [%.3e %.3e]\n', ref - dut);
                    fprintf('    %%diff= [%.3e %.3e]\n', percentDiff);
                end
            end
            benchRows(end+1,:) = {typeStr, sz, p, tMEX, tMAT, tMAT/tMEX, pass, cpuDevice};
        end
    end

%% --- DOUBLE TESTS ---
[failCountDouble, benchRows] = bench_and_report('double', nDouble, sizesDouble, benchRows, failCountDouble);

%% --- SINGLE TESTS ---
[failCountSingle, benchRows] = bench_and_report('single', nSingle, sizesSingle, benchRows, failCountSingle);

%% --- UINT16 TESTS (quick) ---
[failCountUint, benchRows] = bench_and_report('uint16', nUint, sizesUint, benchRows, failCountUint);

%% --- SUMMARY ---
benchTbl = cell2table(benchRows, ...
    'VariableNames', {'Type','N','p','TimeMEX','TimeMAT','Speedup','Pass','Device'});

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
