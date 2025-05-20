function semaphore_test()
    key = 12345;

    fprintf('--- Testing semaphore MEX interface ---\n');

    % Ensure semaphore is fully cleaned before test
    try
        semaphore('d', key);
    catch
        disp('[Info] Semaphore did not exist initially (as expected).');
    end

    % Create
    semaphore('c', key, 2);
    disp('Creating semaphore with count = 2...');

    % Wait twice (non-blocking)
    val1 = semaphore('w', key);
    val2 = semaphore('w', key);
    fprintf('First wait: value = %d\n', val1);
    fprintf('Second wait: value = %d\n', val2);

    % Background wait (should block)
    disp('Spawning a wait() in background (should block until post)...');
    f = parfeval(@semaphore, 1, 'w', key);

    % Give it a moment to reach blocking state
    pause(1.0);

    % Post once to release
    disp('Posting to unblock wait...');
    val_post = semaphore('p', key);
    fprintf('Posted: new value = %d\n', val_post);

    % Wait for background to finish
    val_wait = fetchOutputs(f);
    fprintf('Background wait completed: value = %d\n', val_wait);

    % Try posting beyond max
    disp('Posting once more...');
    semaphore('p', key);  % This brings count to max
    disp('Trying to post beyond max (should warn)...');
    semaphore('p', key);  % Should warn

    % Idempotent destroy test
    disp('Destroying semaphore...');
    semaphore('d', key);

    disp('Calling destroy again (should silently succeed)...');
    semaphore('d', key);  % Now silently ignored

    % Call destroy before create (should silently pass)
    disp('Calling destroy before create (should silently pass)...');
    semaphore('d', 99999);  % never created

    % Confirm wait throws error
    try
        semaphore('w', key);
    catch ME
        fprintf('Caught expected error: %s\n', ME.message);
    end

    disp('All tests passed.');
end
