function semaphore_test()
%SEMAPHORE_TEST  Exhaustive unit test for semaphore MEX
%
%   - Verifies create / wait / post / destroy semantics.
%   - Checks max-count saturation + warning.
%   - Confirms idempotency of create and destroy.
%   - Tests that destroy() cleanly wakes blocked workers.
%   - Runs under both serial and parallel contexts.

    key = 12345;                            % test key (unique per host)
    maxCount = 2;                           % initial count
    cleanup = onCleanup(@() semaphore('d', key));  %#ok<NASGU>

    fprintf('\n===== semaphore_test =====\n');

    %% 0. ensure pristine start
    silentDestroy(key);

    %% 1. CREATE (idempotent)
    fprintf('[1] create ... ');
    count = semaphore('c', key, maxCount);
    assert(count == maxCount, 'Create should return initial count');
    % recreate should be silently accepted
    semaphore('c', key, maxCount);
    disp('OK');

    %% 2. SERIAL wait / post
    fprintf('[2] wait / post non-blocking ... ');
    assert(semaphore('w', key) == maxCount-1, 'Count incorrect after 1st wait');
    assert(semaphore('w', key) == maxCount-2, 'Count incorrect after 2nd wait');
    % now at zero → next wait must block, so test in background
    disp('OK');

    %% 3. BLOCKING wait  (parfeval)
    pool = ensurePool;
    fWait = parfeval(@semaphore, 1, 'w', key);   % will block
    pause(1);                                    % give it time to hit sem_wait

    %% 4. POST should unblock the waiter
    fprintf('[3] post to unblock ... ');
    assert(semaphore('p', key) == 1, 'Post did not set count to 1');
    val = fetchOutputs(fWait);
    assert(val == 0, 'Background wait should observe count==0');
    disp('OK');

    %% 5. SATURATION WARNING
    fprintf('[4] saturation warning ... ');
    lastwarn('');                           % clear any prior warning

    % ❶ first post: count goes 0 → 1  (no warning expected)
    semaphore('p', key);
    assert(isempty(lastwarn), 'Unexpected warning before max reached');

    % ❷ second post: count goes 1 → 2 (now at max, still no warning)
    semaphore('p', key);
    assert(isempty(lastwarn), 'Unexpected warning at exact max');

    % ❸ third post: attempt to exceed max → should warn
    semaphore('p', key);
    [warnStr, warnId] = lastwarn;
    assert(~isempty(warnStr) && strcmpi(warnId, 'semaphore:post'), ...
        'Expected warning not raised when posting past max');
    disp('OK');

    %% 6. DESTROY must wake blockers
    %% 5. DESTROY must wake blockers
    fprintf('[5] destroy wakes waiters ... ');

    % --- make sure count == 0 so the next wait will block ---
    while true
        try
            c = semaphore('w', key);
            if c == 0                  % last permit consumed
                break;
            end
        catch                          % already at zero
            break;
        end
    end
    % --------------------------------------------------------

    fBlock = parfeval(@blocker, 1, key);   % will now block in semaphore('w')
    pause(0.5);                            % let it reach sem_wait

    semaphore('d', key);                  % destroy should wake & terminate
    res = fetchOutputs(fBlock);
    assert(strcmp(res, 'TERMINATED'), 'Destroy did not wake blocked waiters');
    disp('OK');

    %% 7. DESTROY idempotent + post/wait failure modes
    fprintf('[6] destroy idempotent ... ');
    silentDestroy(key);          % second destroy – no error
    silentDestroy(99999);        % on never-created key – no error
    disp('OK');

    fprintf('[7] wait after destroy throws ... ');
    try
        semaphore('w', key);
        error('wait did not throw after destroy');
    catch ME
        ok = contains(ME.identifier, 'semaphore:terminated') || ...
             contains(ME.identifier, 'semaphore:wait')       || ...
             contains(ME.identifier, 'semaphore:notfound')   || ...
             contains(ME.identifier, 'semaphore:shm_open');
        assert(ok, 'Unexpected error after destroy: %s', ME.identifier);
    end
    disp('OK');

    fprintf('\nALL semaphore tests passed ✅\n');
end

%% ------------------------------------------------------------------------
function silentDestroy(k)
% destroy without error if absent
    try, semaphore('d',k); catch, end
end

%% ------------------------------------------------------------------------
function out = blocker(k)
% Helper for section 6 – waits until destroy wakes it
    try
        semaphore('w',k);
        out = 'UNBLOCKED';
    catch
        out = 'TERMINATED';
    end
end

%% ------------------------------------------------------------------------
function pool = ensurePool
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('Processes');  % ✅ REQUIRED: process-based workers
    end
end
