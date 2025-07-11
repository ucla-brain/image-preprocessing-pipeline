% SEMAPHORE  Cross-platform shared memory semaphore for MATLAB.
%
%   This MEX function provides access to a custom, cross-platform shared memory
%   semaphore implementation. It is designed to allow synchronization across
%   independent MATLAB sessions or threads, with consistent behavior on both
%   Windows and POSIX systems.
%
%   SEMAPHORE('create', KEY, VAL)
%      Creates or reinitializes a named semaphore identified by integer KEY.
%      VAL must be a positive integer and sets the initial and maximum count.
%      If the semaphore already exists, it will be reset and any waiting threads
%      or processes will be released.
%
%   SEMAPHORE('wait', KEY)
%      Decrements (locks) the semaphore. If the semaphore count is greater than
%      zero, the function returns immediately. If the count is zero, the call
%      blocks until a 'post' occurs or the semaphore is destroyed.
%
%   SEMAPHORE('post', KEY)
%      Increments (unlocks) the semaphore. If the count is already at maximum,
%      a warning is issued but the call still succeeds. A waiting thread or
%      process (if any) will be signaled to continue.
%
%   SEMAPHORE('destroy', KEY)
%      Marks the semaphore for termination and wakes up all waiting threads or
%      processes. Cleans up all associated shared memory and synchronization
%      primitives. Safe to call even if no one is waiting.
%
%   This implementation ensures identical semantics on both Windows and Linux
%   platforms, using shared memory regions and inter-process synchronization
%   primitives native to each OS.
%
%   Example:
%      semkey = 1;
%      semaphore('create', semkey, 2);
%      semaphore('wait', semkey);
%      semaphore('post', semkey);
%
%
%   Author: Keivan Moradi
%   Date: Updated May 2025
