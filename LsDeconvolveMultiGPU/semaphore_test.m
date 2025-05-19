% semaphore_test.m
% Test script for cross-platform semaphore MEX interface

key = 12345;

fprintf('--- Testing semaphore MEX interface ---\n');

% Clean up any existing instance
try
    semaphore('d', key);
    pause(0.1);
catch
    fprintf('[Info] Semaphore did not exist initially (as expected).\n');
end

% 1. Create with initial value
fprintf('Creating semaphore with count = 2...\n');
val = semaphore('c', key, 2);
assert(val == 2);

% 2. Wait once
fprintf('Calling wait()...\n');
val = semaphore('w', key);
assert(val == 1);

% 3. Wait again
fprintf('Calling wait()...\n');
val = semaphore('w', key);
assert(val == 0);

% 4. Try wait with timeout (should block if run manually)
fprintf('Spawning a wait() in background (should block until post)...\n');
f = parfeval(@() semaphore('w', key), 1);

pause(1);  % simulate processing
fprintf('Posting to unblock wait...\n');
val = semaphore('p', key);
assert(val == 1);

wait(f);  % ensure background finishes
fprintf('Background wait completed: value = %d\n', f.OutputArguments{1});

% 5. Post again (up to max)
fprintf('Posting once more...\n');
val = semaphore('p', key);
assert(val == 2);

fprintf('Trying to post beyond max (should warn)...\n');
val = semaphore('p', key);  % This should emit a warning

% 6. Destroy
fprintf('Destroying semaphore...\n');
semaphore('d', key);

% 7. Confirm that access fails after destroy
fprintf('Expecting error on wait after destroy...\n');
try
    semaphore('w', key);
    error('Expected failure did not occur.');
catch ME
    disp(['Caught expected error: ', ME.message]);
end

fprintf('All tests passed.\n');
