function varargout = queue(varargin)
%QUEUE Shared-memory FIFO queue for inter-process communication via MEX.
%
% This is a MEX function. It provides a cross-platform shared queue that
% allows multiple MATLAB processes to communicate using shared memory.
%
% === Compilation (Windows or Linux) ===
% >> mex -v queue.c
%
% === Usage ===
% Create a shared queue with initial values:
%     queue('create', key, [1 2 3])
%
% Post (enqueue) a value:
%     queue('p', key, value)
%
% Wait and pop (dequeue) a value (blocks if queue is empty):
%     value = queue('w', key)
%
% Destroy the queue and shared memory:
%     queue('d', key)
%
% === Notes ===
% - `key` must be a unique scalar (or string if implemented).
% - The queue stores doubles, FIFO (first-in, first-out).
% - Use single quotes: e.g., 'create', not "create".
% - The queue persists between MATLAB sessions until destroyed.

% This is just a placeholder for MATLAB's help system.
% The actual implementation is in queue.c (compiled MEX file).
error('The file "queue.c" must be compiled with MEX before use: mex -v queue.c');
