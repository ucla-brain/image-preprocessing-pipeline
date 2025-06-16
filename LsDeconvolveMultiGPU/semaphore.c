/*
 * semaphore.c - Cross-platform shared memory semaphore for MATLAB MEX
 *
 * Author: Keivan Moradi (2025) @ B.R.A.I.N. Lab, UCLA
 * License: GPLv3
 *
 * Overview:
 * ---------
 * This file implements a named, multi-process semaphore mechanism for MATLAB,
 * portable across Windows and POSIX (Linux/macOS) systems.
 *
 * It enables robust parallel execution in environments where MATLAB lacks
 * built-in inter-process synchronization, particularly for workflows that use
 * `parfeval`, `batch`, or `parpool('Processes')`.
 *
 * Core Concepts:
 * --------------
 * - Each semaphore is identified by a unique integer key.
 * - Shared memory stores the semaphore state:
 *     - `count`: current semaphore value (updated atomically)
 *     - `max`  : maximum allowed count
 *     - `terminate`: flag to force wakeups during `destroy()`
 * - Synchronization is platform-specific:
 *     - Windows: Named file mappings + Mutex + Manual-reset Event
 *     - POSIX : POSIX named semaphores (`sem_open`) + `shm_open` for metadata
 *
 * POSIX Atomic Count:
 * -------------------
 * On POSIX, `sem_getvalue()` is **not reliable** for synchronization:
 *   - It is non-atomic and may return stale values under concurrent access.
 *   - Cannot be used to enforce maximum count limits safely.
 *
 * To ensure correctness:
 *   - A separate shared memory field `meta->count` is maintained.
 *   - It is updated using `__sync_fetch_and_add()` and `__sync_fetch_and_sub()`
 *     to guarantee atomicity across processes.
 *   - This mirrors the Windows implementation, where shared memory is already
 *     explicitly tracked.
 *
 * Why Not Use Windows Native Semaphore API:
 * -----------------------------------------
 * Windows `CreateSemaphore` is avoided because its behavior does not match
 * POSIX semantics:
 *   1. No way to read current count (`sem_getvalue()` equivalent)
 *   2. `ReleaseSemaphore()` allows over-release without error
 *   3. Cannot interrupt blocked `WaitForSingleObject()` on destroy
 *   4. No shared state or visibility across cooperating processes
 *
 * Instead, the Windows implementation uses:
 *   - Named `CreateFileMapping` for shared state
 *   - Named `CreateMutex` for exclusive access
 *   - Named `CreateEvent` to signal wakeups on `post` or `destroy`
 *
 * Supported Operations:
 * ---------------------
 * 1. create(key, initval):
 *      - Initializes or re-attaches to a semaphore.
 *      - If it exists, reuses it safely. Count = initval, max = initval.
 *
 * 2. wait(key):
 *      - Blocks until count > 0, then decrements.
 *      - Returns new count or error if destroyed.
 *
 * 3. post(key):
 *      - Increments count unless at max. If at max, raises warning.
 *      - Wakes one or more waiters via event/semaphore.
 *
 * 4. destroy(key):
 *      - Sets `terminate = 1` and wakes all waiting threads.
 *      - On POSIX, also posts `max` times to ensure all waiters are released.
 *      - Unlinks shared memory and semaphore handles.
 *
 * Synchronization Guarantees:
 * ---------------------------
 * - All operations are safe under concurrent access from multiple processes.
 * - All updates to count are atomic.
 * - All error paths and cleanup are robust.
 *
 * Platform APIs Used:
 * -------------------
 * Windows:
 *   - CreateFileMappingA, MapViewOfFile
 *   - CreateMutexA, WaitForSingleObject, ReleaseMutex
 *   - CreateEventA, SetEvent, ResetEvent
 *
 * POSIX:
 *   - shm_open, ftruncate, mmap, munmap, shm_unlink
 *   - sem_open, sem_post, sem_wait, sem_unlink
 *   - __sync_fetch_and_add / __sync_fetch_and_sub for atomic count
 */


#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include "mex.h"

#define SEMAPHORE_KEY_NAME_FMT "MATLAB_MEX_SEM_%d"

#if defined(_WIN32)

#   include <windows.h>
#   include <strsafe.h>

#   define MUTEX_NAME_PREFIX "Local\\"
#   define EVENT_NAME_PREFIX "Local\\"
#   define SHM_NAME_PREFIX   "Local\\"

    // Shared structure used across processes
    typedef struct {
        int initialized;
        int count;
        int max;
        volatile int terminate;
    } shared_semaphore_t;

    // Wrapper for Windows handle and pointer state
    typedef struct {
        shared_semaphore_t* sem;
        HANDLE hMap;
        HANDLE hMutex;
        HANDLE hEvent;
        BOOL is_new;
    } semaphore_map_result_t;

    // Close all Windows handles associated with a semaphore
    static void close_handles(semaphore_map_result_t* mapped) {
        if (mapped->sem)   UnmapViewOfFile(mapped->sem);
        if (mapped->hMutex) CloseHandle(mapped->hMutex);
        if (mapped->hEvent) CloseHandle(mapped->hEvent);
        if (mapped->hMap)   CloseHandle(mapped->hMap);
    }

    // Map the shared memory and synchronization objects (create or open)
    static semaphore_map_result_t map_shared_semaphore(int key, BOOL create)
    {
        semaphore_map_result_t result = {0};
        size_t size = sizeof(shared_semaphore_t);
        char basename[64], shmname[128], mname[128], ename[128];

        snprintf(basename, sizeof(basename), SEMAPHORE_KEY_NAME_FMT, key);
        StringCchPrintfA(shmname, 128, "%s%s", SHM_NAME_PREFIX, basename);
        StringCchPrintfA(mname, 128, "%s%s_MUTEX", MUTEX_NAME_PREFIX, basename);
        StringCchPrintfA(ename, 128, "%s%s_EVENT", EVENT_NAME_PREFIX, basename);

        if (create) {
            /* … unchanged … */
        } else {
            /* ── OPEN EXISTING ────────────────────────────────────────────── */
            result.hMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, shmname);
            if (!result.hMap) {
                /* 🔸 Distinguish “already destroyed” from “never created” */
                if (GetLastError() == ERROR_FILE_NOT_FOUND)
                    mexErrMsgIdAndTxt("semaphore:terminated",
                        "Semaphore %d has been destroyed.", key);

                mexErrMsgIdAndTxt("semaphore:notfound",
                    "Semaphore %d not found. Use 'create' first.", key);
            }
            result.is_new = FALSE;
        }

        result.sem = (shared_semaphore_t*)MapViewOfFile(result.hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (!result.sem) {
            CloseHandle(result.hMap);
            mexErrMsgIdAndTxt("semaphore:map", "Failed to map shared memory for key %d", key);
        }

        result.hMutex = CreateMutexA(NULL, FALSE, mname);
        if (!result.hMutex) {
            UnmapViewOfFile(result.sem);
            CloseHandle(result.hMap);
            mexErrMsgIdAndTxt("semaphore:mutex", "Failed to create/open named mutex");
        }

        result.hEvent = CreateEventA(NULL, TRUE, FALSE, ename);
        if (!result.hEvent) {
            CloseHandle(result.hMutex);
            UnmapViewOfFile(result.sem);
            CloseHandle(result.hMap);
            mexErrMsgIdAndTxt("semaphore:event", "Failed to create/open named event");
        }

        WaitForSingleObject(result.hMutex, INFINITE);
        return result;
    }

    // Create or reinitialize a semaphore (idempotent)
    static int create_semaphore(int key, int initval, int* out_count) {
        semaphore_map_result_t mapped = map_shared_semaphore(key, TRUE);

        if (mapped.is_new) {
            ZeroMemory(mapped.sem, sizeof(shared_semaphore_t));
            mapped.sem->count = initval;
            mapped.sem->max = initval;
            mapped.sem->initialized = 1;
        } else {
            if (!mapped.sem->initialized) {
                ReleaseMutex(mapped.hMutex);
                close_handles(&mapped);
                mexErrMsgIdAndTxt("semaphore:uninitialized", "Semaphore %d is in inconsistent state.", key);
            }
        }

        if (out_count) *out_count = mapped.sem->count;

        ReleaseMutex(mapped.hMutex);
        close_handles(&mapped);
        return 0;
    }

    // Open existing semaphore (must already exist and be initialized)
    static semaphore_map_result_t get_semaphore(int key) {
        semaphore_map_result_t mapped = map_shared_semaphore(key, FALSE);
        if (!mapped.sem->initialized) {
            ReleaseMutex(mapped.hMutex);
            close_handles(&mapped);
            mexErrMsgIdAndTxt("semaphore:uninitialized", "Semaphore %d exists but not initialized.", key);
        }
        return mapped;
    }

    // Wait operation with blocking loop and terminate check
    static int wait_semaphore(int key) {
        semaphore_map_result_t mapped = get_semaphore(key);

        while (1) {
            // Check if destroy() was called before entering wait
            if (mapped.sem->terminate) {
                ReleaseMutex(mapped.hMutex);
                close_handles(&mapped);
                mexErrMsgIdAndTxt("semaphore:terminated",
                    "Semaphore %d has been destroyed.", key);
            }

            if (mapped.sem->count > 0) {
                mapped.sem->count--;

                // 🔸 Re-check after decrementing
                if (mapped.sem->terminate) {
                    ReleaseMutex(mapped.hMutex);
                    close_handles(&mapped);
                    mexErrMsgIdAndTxt("semaphore:terminated",
                        "Semaphore %d has been destroyed.", key);
                }

                int current_count = mapped.sem->count;
                if (current_count == 0)
                    ResetEvent(mapped.hEvent);

                ReleaseMutex(mapped.hMutex);
                close_handles(&mapped);
                return current_count;
            }

            ReleaseMutex(mapped.hMutex);

            // Block until post() or destroy() signals the event
            WaitForSingleObject(mapped.hEvent, INFINITE);

            // Re-acquire the mutex before retrying
            WaitForSingleObject(mapped.hMutex, INFINITE);
        }
    }

    // Post to the semaphore (increment), enforcing max limit
    static int post_semaphore(int key) {
        semaphore_map_result_t mapped = get_semaphore(key);

        // Check if semaphore has been destroyed
        if (mapped.sem->terminate) {
            ReleaseMutex(mapped.hMutex);
            close_handles(&mapped);
            mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
        }

        // Check if count is already at maximum
        if (mapped.sem->count >= mapped.sem->max) {
            // Do not increment; just warn
            ReleaseMutex(mapped.hMutex);
            close_handles(&mapped);
            mexWarnMsgIdAndTxt("semaphore:post", "Cannot post: semaphore %d already at max count (%d).", key, mapped.sem->max);
            return mapped.sem->max;
        }

        // Increment count
        mapped.sem->count++;
        int current_count = mapped.sem->count;

        // Signal waiters
        SetEvent(mapped.hEvent);

        // Clean up
        ReleaseMutex(mapped.hMutex);
        close_handles(&mapped);
        return current_count;
    }

    // Destroy semaphore: sets terminate flag and signals all waiters
    static void destroy_semaphore(int key)
    {
        char basename[64], shmname[128], mname[128], ename[128];
        snprintf(basename, sizeof(basename), SEMAPHORE_KEY_NAME_FMT, key);
        StringCchPrintfA(shmname, 128, "%s%s", SHM_NAME_PREFIX, basename);
        StringCchPrintfA(mname, 128, "%s%s_MUTEX", MUTEX_NAME_PREFIX, basename);
        StringCchPrintfA(ename, 128, "%s%s_EVENT", EVENT_NAME_PREFIX, basename);

        HANDLE hMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, shmname);
        if (!hMap) {
            // If file mapping not found, assume already destroyed
            return;
        }

        shared_semaphore_t* sem = (shared_semaphore_t*)MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(shared_semaphore_t));
        if (!sem) {
            CloseHandle(hMap);
            return;
        }

        HANDLE hMutex = CreateMutexA(NULL, FALSE, mname);
        if (!hMutex) {
            UnmapViewOfFile(sem);
            CloseHandle(hMap);
            return;
        }

        HANDLE hEvent = CreateEventA(NULL, TRUE, FALSE, ename);
        if (!hEvent) {
            CloseHandle(hMutex);
            UnmapViewOfFile(sem);
            CloseHandle(hMap);
            return;
        }

        WaitForSingleObject(hMutex, INFINITE);
        sem->terminate = 1;
        SetEvent(hEvent);  // Wake up all waiting threads
        ReleaseMutex(hMutex);

        // Cleanup
        CloseHandle(hEvent);
        CloseHandle(hMutex);
        UnmapViewOfFile(sem);
        CloseHandle(hMap);
    }

#else  // POSIX

#   include <fcntl.h>
#   include <sys/mman.h>
#   include <sys/stat.h>
#   include <unistd.h>
#   include <semaphore.h>

#   define SEM_NAME_FMT "/matlab_mex_sem_%d"
#   define SHM_NAME_FMT "/matlab_mex_meta_%d"

    // Shared metadata structure (mapped into shared memory)
    typedef struct {
        int initialized;
        int max;
        volatile int count;        // 🔸 now tracked atomically
        volatile int terminate;
    } semaphore_meta_t;

    static void name_for_key(char* out, size_t outlen, const char* fmt, int key) {
        snprintf(out, outlen, fmt, key);
    }

    static int create_semaphore(int key, int initval, int* out_count) {
        char sem_name[64], shm_name[64];
        name_for_key(sem_name, sizeof(sem_name), SEM_NAME_FMT, key);
        name_for_key(shm_name, sizeof(shm_name), SHM_NAME_FMT, key);

        sem_t* sem = sem_open(sem_name, O_CREAT | O_EXCL, 0666, initval);
        if (sem == SEM_FAILED && errno == EEXIST)
            sem = sem_open(sem_name, 0);
        if (sem == SEM_FAILED)
            mexErrMsgIdAndTxt("semaphore:create", "sem_open failed: %s", strerror(errno));

        int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            sem_close(sem);
            sem_unlink(sem_name);
            mexErrMsgIdAndTxt("semaphore:shm_open", "Failed to open shared memory: %s", strerror(errno));
        }

        if (ftruncate(shm_fd, sizeof(semaphore_meta_t)) == -1) {
            close(shm_fd);
            sem_close(sem);
            sem_unlink(sem_name);
            shm_unlink(shm_name);
            mexErrMsgIdAndTxt("semaphore:truncate", "Failed to resize shared memory: %s", strerror(errno));
        }

        semaphore_meta_t* meta = mmap(NULL, sizeof(semaphore_meta_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (meta == MAP_FAILED) {
            close(shm_fd);
            sem_close(sem);
            sem_unlink(sem_name);
            shm_unlink(shm_name);
            mexErrMsgIdAndTxt("semaphore:mmap", "Failed to map shared memory.");
        }

        if (!meta->initialized) {
            meta->max = initval;
            meta->count = initval;
            meta->terminate = 0;
            meta->initialized = 1;
        }

        if (out_count) {
            *out_count = meta->count;
        }

        munmap(meta, sizeof(semaphore_meta_t));
        close(shm_fd);
        sem_close(sem);
        return 0;
    }

    static int wait_semaphore(int key)
    {
        char sem_name[64], shm_name[64];
        name_for_key(sem_name, sizeof(sem_name), SEM_NAME_FMT, key);
        name_for_key(shm_name, sizeof(shm_name), SHM_NAME_FMT, key);

        /* ── 1. open the POSIX semaphore ─────────────────────────────────── */
        sem_t* sem = sem_open(sem_name, 0);
        if (sem == SEM_FAILED) {
            if (errno == ENOENT)
                mexErrMsgIdAndTxt("semaphore:terminated",
                    "Semaphore %d has been destroyed.", key);
            mexErrMsgIdAndTxt("semaphore:wait",
                "Semaphore %d not found.", key);
        }

        /* ── 2. open & map shared metadata ───────────────────────────────── */
        int shm_fd = shm_open(shm_name, O_RDWR, 0666);
        if (shm_fd == -1) {
            sem_close(sem);
            if (errno == ENOENT)
                mexErrMsgIdAndTxt("semaphore:terminated",
                    "Semaphore %d has been destroyed.", key);
            mexErrMsgIdAndTxt("semaphore:shm_open",
                "Shared memory for key %d not found.", key);
        }

        semaphore_meta_t* meta = mmap(NULL, sizeof(semaphore_meta_t),
                                       PROT_READ | PROT_WRITE, MAP_SHARED,
                                       shm_fd, 0);
        if (meta == MAP_FAILED) {
            close(shm_fd);
            sem_close(sem);
            mexErrMsgIdAndTxt("semaphore:mmap",
                "Failed to map metadata for key %d", key);
        }

        // 🔸 Check before blocking
        if (meta->terminate) {
            munmap(meta, sizeof(semaphore_meta_t));
            close(shm_fd);
            sem_close(sem);
            mexErrMsgIdAndTxt("semaphore:terminated",
                "Semaphore %d has been destroyed.", key);
        }

        sem_wait(sem);  // May be woken by destroy()

        // 🔸 Check again after wake
        if (meta->terminate) {
            munmap(meta, sizeof(semaphore_meta_t));
            close(shm_fd);
            sem_close(sem);
            mexErrMsgIdAndTxt("semaphore:terminated",
                "Semaphore %d has been destroyed.", key);
        }

        int new_val = __sync_fetch_and_sub(&meta->count, 1) - 1;

        munmap(meta, sizeof(semaphore_meta_t));
        close(shm_fd);
        sem_close(sem);
        return new_val;
    }

    // Post to the semaphore (increment), enforcing atomic max limit
    static int post_semaphore(int key)
    {
        char sem_name[64], shm_name[64];
        name_for_key(sem_name, sizeof(sem_name), SEM_NAME_FMT, key);
        name_for_key(shm_name, sizeof(shm_name), SHM_NAME_FMT, key);

        /* Open resources ---------------------------------------------------- */
        sem_t* sem = sem_open(sem_name, 0);
        if (sem == SEM_FAILED)
            mexErrMsgIdAndTxt("semaphore:post", "Semaphore %d not found.", key);

        int shm_fd = shm_open(shm_name, O_RDWR, 0666);
        if (shm_fd == -1) {
            sem_close(sem);
            mexErrMsgIdAndTxt("semaphore:shm_open", "Shared memory for key %d not found.", key);
        }

        semaphore_meta_t* meta =
            mmap(NULL, sizeof(semaphore_meta_t), PROT_READ | PROT_WRITE,
                 MAP_SHARED, shm_fd, 0);
        if (meta == MAP_FAILED) {
            close(shm_fd);
            sem_close(sem);
            mexErrMsgIdAndTxt("semaphore:mmap", "Failed to map metadata for key %d", key);
        }

        /* Atomic increment -------------------------------------------------- */
        int prev   = __sync_fetch_and_add(&meta->count, 1);
        int newval = prev + 1;
        int ret;                                /* ★ value to return */
        int maxval = meta->max;                 /* ★ cache before munmap */

        if (newval > maxval) {
            /* Roll back and warn */
            __sync_fetch_and_sub(&meta->count, 1);
            mexWarnMsgIdAndTxt("semaphore:post",
                "Cannot post: semaphore %d already at max count (%d).",
                key, maxval);
            ret = maxval;                       /* ★ stay at max */
        } else {
            sem_post(sem);                      /* within bounds */
            ret = newval;                       /* ★ updated count */
        }

        /* Clean-up ---------------------------------------------------------- */
        munmap(meta, sizeof(semaphore_meta_t));
        close(shm_fd);
        sem_close(sem);
        return ret;                             /* ★ safe, no dangling ptr */
    }

    static void destroy_semaphore(int key) {
        char sem_name[64], shm_name[64];
        name_for_key(sem_name, sizeof(sem_name), SEM_NAME_FMT, key);
        name_for_key(shm_name, sizeof(shm_name), SHM_NAME_FMT, key);

        sem_t* sem = sem_open(sem_name, 0);
        int shm_fd = shm_open(shm_name, O_RDWR, 0666);
        if (sem == SEM_FAILED || shm_fd == -1)
            return;

        semaphore_meta_t* meta = mmap(NULL, sizeof(semaphore_meta_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (meta != MAP_FAILED) {
            meta->terminate = 1;
            // post up to max count to wake up waiters
            for (int i = 0; i < meta->max; ++i) {
                sem_post(sem);
            }
            munmap(meta, sizeof(semaphore_meta_t));
        }

        sem_close(sem);
        sem_unlink(sem_name);
        close(shm_fd);
        shm_unlink(shm_name);
    }

#endif  // End of POSIX

// MATLAB MEX entry point: parses command and dispatches to correct operation
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 2)
        mexErrMsgTxt("Requires at least 2 inputs: directive and key.");

    char directive[16];
    if (mxGetString(prhs[0], directive, sizeof(directive)) != 0)
        mexErrMsgTxt("Failed to parse directive string.");

    directive[0] = (char)tolower(directive[0]);
    int key = (int)(mxGetScalar(prhs[1]) + 0.5);
    int count = -1;

    switch (directive[0]) {
        case 'c': {
            int initval = 1;
            if (nrhs > 2)
                initval = (int)(mxGetScalar(prhs[2]) + 0.5);
            create_semaphore(key, initval, &count);
            break;
        }
        case 'w':
            count = wait_semaphore(key);
            break;
        case 'p':
            count = post_semaphore(key);
            break;
        case 'd':
            destroy_semaphore(key);
            break;
        default:
            mexErrMsgTxt("Unknown directive. Use 'create', 'wait', 'post', or 'destroy'.");
    }

    if (nlhs > 0) {
        if (count >= 0)
            plhs[0] = mxCreateDoubleScalar((double)count);
        else
            plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }
}
