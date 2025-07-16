function if_else_test_suite()
%IF_ELSE_TEST_SUITE  Robust test and benchmark suite for if_else MEX.
%
%   Tests correctness and performance for CPU, gpuArray, and includes
%   a memory-efficient version using logical indexing.

fprintf('\n======= if_else MEX TEST SUITE =======\n');

types = {'single', 'double'};
Ns = [10, 100, 1e4, 1e6, 1e7];
all_pass = true;

for t = 1:length(types)
    T = types{t};
    fprintf('\n--- Testing type: %s ---\n', T);

    for N = Ns
        fprintf('  Test size: %g ... ', N);
        cond = rand(N,1) > 0.5;
        x = cast(randn(N,1), T);
        y = cast(randn(N,1), T);

        % --- Test CPU: MEX
        out_mex = if_else(cond, x, y);
        out_ref = cond .* x + (~cond) .* y;
        pass = isequaln(out_mex, out_ref);
        all_pass = all_pass && pass;

        if ~pass
            idx_fail = find(out_mex ~= out_ref, 10);
            fprintf('FAILED (CPU)! Failed indices: '); disp(idx_fail(:)');
        else
            fprintf('PASS (CPU); ');
        end

        % --- Test CPU: logical indexing
        out_idx = logical_idx_fn(cond, x, y);
        if ~isequaln(out_mex, out_idx)
            idx_fail = find(out_mex ~= out_idx, 10);
            fprintf('FAILED (logical idx CPU)! Failed indices: '); disp(idx_fail(:)');
        else
            fprintf('PASS (logical CPU); ');
        end

        % --- Test CPU: memory-efficient version
        out_efficient = if_else_efficient(cond, x, y);
        if ~isequaln(out_mex, out_efficient)
            idx_fail = find(out_mex ~= out_efficient, 10);
            fprintf('FAILED (if_else_efficient CPU)! Failed indices: '); disp(idx_fail(:)');
        else
            fprintf('PASS (efficient CPU); ');
        end

        % --- Test GPU (if available)
        try
            gcond = gpuArray(cond); gx = gpuArray(x); gy = gpuArray(y);
            gout_mex = if_else(gcond, gx, gy);
            gout_mex = gather(gout_mex);
            if isequaln(gout_mex, out_ref)
                fprintf('PASS (GPU); ');
            else
                idx_fail = find(gout_mex ~= out_ref, 10);
                fprintf('FAILED (GPU)! Failed indices: '); disp(idx_fail(:)');
                all_pass = false;
            end
        catch
            fprintf('[No GPU toolbox/skipped] ');
        end

        fprintf('\n');
    end
end

%% Edge cases: All true/false, scalars, empty, large ND
fprintf('\n--- Testing edge cases ---\n');

assert(isequal(if_else(true, 7, 9), 7));
assert(isequal(if_else(false, 7, 9), 9));
assert(isempty(if_else(false(0,1), [], [])));
assert(isequal(if_else([true;false], [1;2], [3;4]), [1;4]));

cond = true(1,1000); x = ones(1,1000); y = zeros(1,1000);
assert(isequal(if_else(cond, x, y), x));
cond = false(1,1000);
assert(isequal(if_else(cond, x, y), y));

A = rand(5,7,2,'single'); B = rand(5,7,2,'single'); mask = rand(5,7,2)>0.2;
assert(isequaln(if_else(mask, A, B), mask.*A + (~mask).*B));

fprintf('All edge cases: PASS\n');

%% Benchmark section
fprintf('\n======= BENCHMARKING =======\n');
N = 1e8;
x = randn(N,1,'single'); 
y = randn(N,1,'single');

% Warmup
cond = rand(N,1) > 0.5;
if_else(cond, x, y);

fprintf('%-22s   %s\n', 'Method', 'Time (s)');
fprintf('-----------------------   ----------\n');

% if_else (MEX)
tm = timeit(@() if_else(rand(N,1) > 0.5, x, y));
fprintf('%-22s   %.4f\n', 'if_else (MEX)', tm);

% logical indexing
ti = timeit(@() logical_idx_fn(rand(N,1) > 0.5, x, y));
fprintf('%-22s   %.4f\n', 'logical indexing', ti);

% arithmetic
ta = timeit(@() (rand(N,1) > 0.5) .* x + (~(rand(N,1) > 0.5)) .* y);
fprintf('%-22s   %.4f\n', 'arithmetic (.*)', ta);

% memory-efficient version
te = timeit(@() if_else_efficient(rand(N,1) > 0.5, x, y));
fprintf('%-22s   %.4f\n', 'if_else_efficient', te);

% arrayfun
try
    tf = timeit(@() arrayfun(@(a,b,c)a*b+~a*c, rand(N,1) > 0.5, x, y));
    fprintf('%-22s   %.4f\n', 'arrayfun', tf);
catch
    fprintf('%-22s   N/A\n', 'arrayfun');
end


%% GPU Benchmark (No Comma, No Warning)
try
    N_gpu = 1e8; % Increase for accurate timing (e.g., 100 million)
    d = gpuDevice;
    wait(d);
    % Make cond/gcond part of the timed operation by generating them in-place

    fprintf('\n(GPU timings use N = %.0e elements)\n', N_gpu);

    tmg = gputimeit(@() if_else(gpuArray(rand(N_gpu,1) > 0.5), ...
                                gpuArray.rand(N_gpu,1,'single'), ...
                                gpuArray.rand(N_gpu,1,'single')));
    wait(d);
    fprintf('%-22s   %.4f\n', 'if_else (GPU)', tmg);

    tag = gputimeit(@() gpuArray.rand(N_gpu,1,'single') .* gpuArray.rand(N_gpu,1,'single') + ...
                          (~gpuArray.rand(N_gpu,1) > 0.5) .* gpuArray.rand(N_gpu,1,'single'));
    wait(d);
    fprintf('%-22s   %.4f\n', 'arithmetic (GPU)', tag);

    te_gpu = gputimeit(@() if_else_efficient(gpuArray(rand(N_gpu,1) > 0.5), ...
                                             gpuArray.rand(N_gpu,1,'single'), ...
                                             gpuArray.rand(N_gpu,1,'single')));
    wait(d);
    fprintf('%-22s   %.4f\n', 'if_else_efficient (GPU)', te_gpu);

    tfg = gputimeit(@() arrayfun(@(a,b,c)a*b+~a*c, ...
                                 gpuArray(rand(N_gpu,1) > 0.5), ...
                                 gpuArray.rand(N_gpu,1,'single'), ...
                                 gpuArray.rand(N_gpu,1,'single')));
    wait(d);
    fprintf('%-22s   %.4f\n', 'arrayfun (GPU)', tfg);

catch
    fprintf('No GPU detected: Skipping GPU benchmarks.\n');
end


fprintf('\n======= TESTS COMPLETED =======\n');
if all_pass
    fprintf('All tests PASSED! ✅\n');
else
    fprintf('Some tests FAILED! ❌\n');
end

end

% ------------ Helper functions --------------

function out = logical_idx_fn(cond, x, y)
    out = zeros(size(x), 'like', x);
    out(cond) = x(cond); 
    out(~cond) = y(~cond);
end

function out = if_else_efficient(cond, x, y)
%IF_ELSE_EFFICIENT  Memory-efficient conditional selection.
%   out = if_else_efficient(cond, x, y)
%
%   Equivalent to: cond .* x + (~cond) .* y
%   but with minimal memory overhead.
%
%   All inputs must be the same size, and cond must be logical.

    % Sanity checks
    if ~isequal(size(cond), size(x), size(y))
        error('All inputs must be the same size.');
    end
    if ~islogical(cond)
        error('cond must be logical.');
    end

    out = y;
    out(cond) = x(cond);
end
